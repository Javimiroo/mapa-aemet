# -*- coding: utf-8 -*-
"""
PoC — QPE (Quantitative Precipitation Estimation) radar + estacions (fase 2, projecte GRAF).

Idea: partir dels compòsits de REFLECTIVITAT d'AEMET (imatges PNG acolorides que el
'radar-arxiu' ja arxiva cada ~10 min), estimar la pluja acumulada d'una finestra i
"clavar-la" a la realitat amb el biaix mitjà de les estacions (Meteocat).

Passos:
  1. Llegir el manifest del radar-arxiu i triar els frames de reflectivitat dins la finestra.
  2. Invertir la paleta de color -> dBZ (a cada píxel, el dBZ del color MÉS PROPER).
  3. dBZ -> intensitat de pluja R amb Z-R (Marshall-Palmer): Z = a*R^b  ->  R = (z/a)^(1/b),
     z = 10^(dBZ/10). Sostre de dBZ per a calamarsa.
  4. Acumular R*dt sobre els frames -> mm de radar en la finestra.
  5. BIAIX MITJÀ GLOBAL: factor = Σ(pluja estacions) / Σ(radar al píxel de l'estació),
     aplicat a tot el camp (ancora el radar a terra).
  6. Escriure un PNG acolorit (paleta de pluja) retallat a Catalunya per comparar amb Meteocat.

Ús:
    set MAPA_PASS=...   (per desxifrar les dades de les estacions publicades)
    python qpe_radar.py --finestra 6h --out qpe_6h.png
    python qpe_radar.py --finestra 24h --out qpe_24h.png

NOTA: la paleta (PALETA_DBZ) és una AEMET estàndard de PARTIDA. Si el radar-arxiu fa servir
una altra escala, passa'm el PNG original o l'script i l'ajustem — la resta del pipeline no canvia.
"""
import argparse, io, json, math, os, sys, urllib.request
from datetime import datetime, timezone, timedelta

import numpy as np
from PIL import Image

RADAR_BASE = "https://javimiroo.github.io/radar-arxiu/"
# mateixos límits que fa servir privat.html per a la capa (imatge equirectangular)
RAD_BOUNDS = ((35.5, -9.5), (44.0, 4.5))     # ((latS, lonW), (latN, lonE))
CAT_BBOX = (0.05, 3.40, 40.45, 42.95)        # lon0, lon1, lat0, lat1 (Catalunya + marge)
DADES_URL = "https://raw.githubusercontent.com/Javimiroo/mapa-aemet/dades/dades_privat.enc"

# Z-R (Marshall-Palmer estàndard). Sostre de dBZ (calamarsa) per no disparar la pluja.
ZR_A, ZR_B, DBZ_MAX = 200.0, 1.6, 53.0

# Paleta de reflectivitat AEMET (dBZ creixent -> RGB). PLACEHOLDER a calibrar amb el PNG real.
PALETA_DBZ = [
    ( 5, (0, 236, 236)), (10, (0, 160, 246)), (15, (0, 0, 246)),
    (20, (0, 255, 0)),   (25, (0, 200, 0)),   (30, (0, 144, 0)),
    (35, (255, 255, 0)), (40, (231, 192, 0)), (45, (255, 144, 0)),
    (50, (255, 0, 0)),   (55, (214, 0, 0)),   (60, (255, 0, 255)),
    (65, (150, 0, 200)),
]
# paleta de sortida (mm acumulats) — la mateixa idea que precacum al privat.html
PALETA_MM = [
    (0, (240, 248, 255)), (0.5, (190, 225, 240)), (2, (120, 190, 225)), (5, (70, 150, 215)),
    (10, (45, 110, 190)), (20, (70, 175, 95)), (35, (170, 210, 80)), (50, (245, 230, 90)),
    (75, (245, 175, 65)), (100, (235, 110, 50)), (150, (210, 40, 45)), (200, (170, 30, 110)),
]

FIN_HORES = {"1h": 1, "3h": 3, "6h": 6, "24h": 24, "dia": None}   # 'dia' = des de mitjanit local


# ----------------------------------------------------------------- utils
def _get_bytes(url, tries=4):
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "graf-qpe"})
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception as ex:  # noqa
            last = ex
    raise RuntimeError("no s'ha pogut baixar %s (%s)" % (url, last))


def _parse_iso(s):
    s = str(s).strip().replace("Z", "+00:00")
    d = datetime.fromisoformat(s)
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


# ----------------------------------------------------------------- estacions (biaix)
def estacions_pacum(finestra, password):
    """Torna [(lat, lon, mm)] de la pluja acumulada de les estacions per a la finestra,
    llegint dades_privat.enc (mateix camp 'pacum' que calcula fetch_privat.py)."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import base64
    blob = json.loads(_get_bytes(DADES_URL + "?_=" + str(int(datetime.now().timestamp()))))
    salt = base64.b64decode(blob["salt"]); iv = base64.b64decode(blob["iv"]); ct = base64.b64decode(blob["ct"])
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt,
                     iterations=blob.get("it", 200000)).derive(password.encode())
    dades = json.loads(AESGCM(key).decrypt(iv, ct, None).decode("utf-8"))
    out = []
    for e in dades.get("estacions", []):
        pac = (e.get("actual") or {}).get("pacum") or {}
        v = pac.get(finestra)
        if v is not None and e.get("lat") is not None and e.get("lon") is not None:
            out.append((float(e["lat"]), float(e["lon"]), float(v)))
    return out


# ----------------------------------------------------------------- radar
def _paleta_arrays():
    dbz = np.array([p[0] for p in PALETA_DBZ], np.float32)
    rgb = np.array([p[1] for p in PALETA_DBZ], np.float32)
    return dbz, rgb


def png_a_dbz(png_bytes, tol=42.0):
    """Inverteix la paleta: cada píxel -> dBZ del color més proper (o NaN si és fons)."""
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    a = np.asarray(im, np.float32)
    rgb, alpha = a[:, :, :3], a[:, :, 3]
    dbz_tab, rgb_tab = _paleta_arrays()
    H, W = rgb.shape[:2]
    flat = rgb.reshape(-1, 3)
    # distància a cada color de la paleta -> índex més proper
    d = np.linalg.norm(flat[:, None, :] - rgb_tab[None, :, :], axis=2)   # (N, K)
    idx = d.argmin(axis=1)
    dmin = d[np.arange(d.shape[0]), idx]
    dbz = dbz_tab[idx].astype(np.float32)
    # fons: transparent, o massa lluny de qualsevol color de pluja, o quasi blanc/gris
    maxch = flat.max(axis=1); minch = flat.min(axis=1)
    gris = (maxch - minch) < 18
    fons = (alpha.reshape(-1) < 40) | (dmin > tol) | (gris & (maxch > 150))
    dbz[fons] = np.nan
    return dbz.reshape(H, W)


def dbz_a_intensitat(dbz):
    """dBZ -> R (mm/h) amb Z-R. NaN i sota llindar -> 0."""
    d = np.where(np.isnan(dbz), -100.0, dbz)
    d = np.minimum(d, DBZ_MAX)
    z = np.power(10.0, d / 10.0)
    R = np.power(z / ZR_A, 1.0 / ZR_B)
    R[d < 5] = 0.0
    return R.astype(np.float32)


def retalla_cat(arr):
    """Retalla la graella (equirectangular RAD_BOUNDS) al bbox de Catalunya.
    Torna (sub, lon0, lon1, lat0, lat1)."""
    (latS, lonW), (latN, lonE) = RAD_BOUNDS
    H, W = arr.shape
    lon0, lon1, lat0, lat1 = CAT_BBOX
    c0 = int((lon0 - lonW) / (lonE - lonW) * W); c1 = int((lon1 - lonW) / (lonE - lonW) * W)
    r0 = int((latN - lat1) / (latN - latS) * H); r1 = int((latN - lat0) / (latN - latS) * H)
    c0, c1 = max(0, min(c0, c1)), min(W, max(c0, c1))
    r0, r1 = max(0, min(r0, r1)), min(H, max(r0, r1))
    return arr[r0:r1, c0:c1], lon0, lon1, lat0, lat1


def mostreja(sub, lon0, lon1, lat0, lat1, lat, lon):
    """Valor de 'sub' (equirectangular al bbox) al punt (lat,lon), o NaN fora."""
    H, W = sub.shape
    if not (lon0 <= lon <= lon1 and lat0 <= lat <= lat1):
        return np.nan
    c = min(W - 1, max(0, int((lon - lon0) / (lon1 - lon0) * W)))
    r = min(H - 1, max(0, int((lat1 - lat) / (lat1 - lat0) * H)))
    return sub[r, c]


def color_mm(mm):
    tab = PALETA_MM
    if mm <= tab[0][0]:
        return tab[0][1]
    if mm >= tab[-1][0]:
        return tab[-1][1]
    for i in range(len(tab) - 1):
        a, b = tab[i], tab[i + 1]
        if a[0] <= mm <= b[0]:
            t = (mm - a[0]) / (b[0] - a[0])
            return tuple(int(a[1][k] + (b[1][k] - a[1][k]) * t) for k in range(3))
    return tab[-1][1]


# ----------------------------------------------------------------- pipeline
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finestra", default="6h", choices=list(FIN_HORES.keys()))
    ap.add_argument("--out", default="qpe.png")
    ap.add_argument("--no-biaix", action="store_true", help="no aplicar el biaix d'estacions (radar cru)")
    a = ap.parse_args()
    pwd = os.environ.get("MAPA_PASS")

    print("Baixant manifest del radar-arxiu…")
    man = json.loads(_get_bytes(RADAR_BASE + "manifest.json?_=" + str(int(datetime.now().timestamp()))))
    comp = man.get("composit") or {}
    if not comp:
        raise SystemExit("el manifest no té 'composit' (reflectivitat)")
    temps = sorted((_parse_iso(t), p) for t, p in comp.items())
    tref = temps[-1][0]
    if a.finestra == "dia":
        try:
            from zoneinfo import ZoneInfo
            desde = tref.astimezone(ZoneInfo("Europe/Madrid")).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        except Exception:  # noqa
            desde = tref.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        desde = tref - timedelta(hours=FIN_HORES[a.finestra])
    frames = [(t, p) for (t, p) in temps if desde <= t <= tref]
    if len(frames) < 2:
        raise SystemExit("pocs frames de radar a la finestra (%d)" % len(frames))
    print("Finestra %s: %d frames de %s a %s UTC" % (a.finestra, len(frames),
          frames[0][0].strftime("%m-%d %H:%M"), frames[-1][0].strftime("%m-%d %H:%M")))

    # acumulació: cada frame val fins al següent (dt en minuts)
    acum = None
    for i, (t, p) in enumerate(frames):
        try:
            R = dbz_a_intensitat(png_a_dbz(_get_bytes(RADAR_BASE + p)))
        except Exception as ex:  # noqa
            print("  avis: frame %s no processat (%s)" % (p, str(ex)[:70])); continue
        dt = (frames[i + 1][0] - t).total_seconds() / 60.0 if i + 1 < len(frames) else \
             np.median([(frames[j + 1][0] - frames[j][0]).total_seconds() / 60.0 for j in range(len(frames) - 1)])
        dt = min(max(dt, 1.0), 20.0)          # protecció contra buits grans
        acum = (R * (dt / 60.0)) if acum is None else acum + R * (dt / 60.0)
        if i % 12 == 0:
            print("  ...%d/%d" % (i + 1, len(frames)))
    if acum is None:
        raise SystemExit("cap frame processat")

    sub, lon0, lon1, lat0, lat1 = retalla_cat(acum)
    print("Radar acumulat (cru) a Catalunya: màx %.1f mm · mitjana %.2f mm" % (float(np.nanmax(sub)), float(np.nanmean(sub))))

    # biaix mitjà global amb les estacions
    factor = 1.0
    if not a.no_biaix and pwd:
        try:
            ests = estacions_pacum(a.finestra if a.finestra in ("1h", "3h", "6h", "24h") else "24h", pwd)
            sg = sr = 0.0; nfit = 0
            for lat, lon, mm in ests:
                if mm is None or mm < 0.2:
                    continue
                rv = mostreja(sub, lon0, lon1, lat0, lat1, lat, lon)
                if not np.isnan(rv) and rv > 0.1:
                    sg += mm; sr += float(rv); nfit += 1
            if nfit >= 3 and sr > 0:
                factor = sg / sr
                print("Biaix mitjà global: %.2f (a partir de %d estacions amb pluja)" % (factor, nfit))
            else:
                print("Biaix: poques estacions amb pluja coincident (%d); deixo factor 1.0" % nfit)
        except Exception as ex:  # noqa
            print("  avis: biaix no aplicat (%s)" % str(ex)[:80])
    elif not pwd:
        print("Biaix: falta MAPA_PASS; escric el radar cru (sense ajust d'estacions)")
    sub = sub * factor

    # escriu PNG acolorit (mm) retallat a Catalunya
    H, W = sub.shape
    out = np.zeros((H, W, 4), np.uint8)
    for r in range(H):
        for c in range(W):
            v = sub[r, c]
            if np.isnan(v) or v < 0.1:
                continue
            out[r, c, :3] = color_mm(float(v)); out[r, c, 3] = 210
    Image.fromarray(out, "RGBA").save(a.out)
    print("Escrit %s (%dx%d) · màx corregit %.1f mm" % (a.out, W, H, float(np.nanmax(sub))))


if __name__ == "__main__":
    main()
