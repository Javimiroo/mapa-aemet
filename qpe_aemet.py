# -*- coding: utf-8 -*-
"""
PoC — QPE amb el PRODUCTE DE PLUJA d'AEMET (fase 2A, projecte GRAF).

En comptes d'invertir la reflectivitat (que arrossega clutter), fem servir el producte
de pluja d'AEMET ja calculat i filtrat: RN1.1HR (pluja acumulada de l'última hora, mm),
que ve als GeoTIFF de l'HVD/api-eltiempo amb la MATEIXA AEMET_API_KEY.

Aquest tir directe baixa l'últim RN1, el porta a la graella de Catalunya (mosaic dels
nodes vius, nanmax), opcionalment l'ajusta amb el biaix mitjà de les estacions Meteocat
(camp 'pacum' d'1 h) i escriu un PNG + estadístiques dels VALORS REALS. Serveix per
confirmar l'escala del producte abans de muntar el buffer acumulador (24h/dia/setmana).

Ús:
    set AEMET_API_KEY=...   & set MAPA_PASS=...
    python qpe_aemet.py --out qpe_aemet_1h.png
"""
import argparse, io, json, os, sys, tarfile, re, urllib.request
from datetime import datetime, timezone

import numpy as np

HVD_URL = "https://www.aemet.es/es/api-eltiempo/radar/download/echotop/ba"  # torna TOT l'HVD (inclou RN1)
PROD_TAG = "RN1.1HR"                         # pluja acumulada 1 h (mm)
CAT_BBOX = (0.05, 3.40, 40.45, 42.95)        # lon0, lon1, lat0, lat1
NX, NY = 260, 210                            # graella de sortida (~1,3 km)
DADES_URL = "https://raw.githubusercontent.com/Javimiroo/mapa-aemet/dades/dades_privat.enc"

PALETA_MM = [
    (0, (240, 248, 255)), (0.5, (190, 225, 240)), (2, (120, 190, 225)), (5, (70, 150, 215)),
    (10, (45, 110, 190)), (20, (70, 175, 95)), (35, (170, 210, 80)), (50, (245, 230, 90)),
    (75, (245, 175, 65)), (100, (235, 110, 50)), (150, (210, 40, 45)), (200, (170, 30, 110)),
]


# ----------------------------------------------------------------- baixada AEMET
def baixa_hvd(api_key):
    req = urllib.request.Request(HVD_URL, headers={"api_key": api_key, "User-Agent": "graf-qpe"})
    body = urllib.request.urlopen(req, timeout=90).read()
    if body[:2] != b"\x1f\x8b":
        raise SystemExit("la resposta d'AEMET no és un .tar.gz (%d bytes, comença %r)" % (len(body), body[:4]))
    return body


def _parse_nom(nom):
    m = re.match(r"down_([A-Z]{3})(\d{12})\.(.+)\.tiff?$", os.path.basename(nom))
    if not m:
        return None
    radar, ts, prod = m.groups()
    try:
        dt = datetime.strptime(ts, "%y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return radar, dt, prod


def extreu_rn1(tar_bytes):
    """Torna [(radar, datetime, band, bounds)] dels membres RN1.1HR MÉS RECENTS (només
    nodes vius: descarta els de mostra vells). Llegeix amb rasterio."""
    import rasterio
    from rasterio.io import MemoryFile
    tf = tarfile.open(fileobj=io.BytesIO(tar_bytes))
    cand = []
    for m in tf.getmembers():
        info = _parse_nom(m.name)
        if not info:
            continue
        radar, dt, prod = info
        if PROD_TAG in prod:
            cand.append((radar, dt, m))
    if not cand:
        raise SystemExit("cap membre %s a l'arxiu d'AEMET" % PROD_TAG)
    tmax = max(dt for (_, dt, _) in cand)
    frescos = [(r, dt, m) for (r, dt, m) in cand if (tmax - dt).total_seconds() <= 3 * 3600]
    out = []
    meta_bolcat = False
    for radar, dt, m in frescos:
        try:
            with MemoryFile(tf.extractfile(m).read()) as mf, mf.open() as ds:
                if not meta_bolcat:                 # BOLCAT de metadades del 1r node: ESCALA + taula de color
                    meta_bolcat = True
                    print("  --- METADADES del producte RN1 (node %s) ---" % radar)
                    try:
                        tg = ds.tags()
                        for k, v in tg.items():
                            print("     tag %s = %s" % (k, str(v)[:300]))
                    except Exception as ex:  # noqa
                        print("     (sense tags: %s)" % ex)
                    try:
                        cm = ds.colormap(1)
                        items = sorted(cm.items())
                        print("     colormap (valor -> RGBA), valors usats 239-254:")
                        for val in range(238, 255):
                            if val in cm:
                                print("       %d -> %s" % (val, cm[val]))
                    except Exception as ex:  # noqa
                        print("     (sense colormap: %s)" % ex)
                    print("  --- fi metadades ---")
                raw = ds.read(1)
                band = raw.astype(np.float32)
                nod = ds.nodata
                if nod is not None:
                    band[band == nod] = np.nan
                band[band >= 255] = np.nan          # 255 = farciment de buit (nodata) del producte byte
                b = ds.bounds
                out.append((radar, dt, band, (b.left, b.bottom, b.right, b.top)))
        except Exception as ex:  # noqa
            print("  avis: node %s no llegit (%s)" % (radar, str(ex)[:60]))
    print("RN1 més recent: %s UTC · %d nodes vius" % (tmax.strftime("%Y-%m-%d %H:%M"), len(out)))
    return out, tmax


# ----------------------------------------------------------------- mosaic a CAT
def a_graella(band, bounds, nx=NX, ny=NY, bbox=CAT_BBOX):
    """Remostreja (veí més proper) una banda EPSG:4326 a la graella de sortida de CAT."""
    lon0, lon1, lat0, lat1 = bbox
    left, bottom, right, top = bounds
    H, W = band.shape
    lons = np.linspace(lon0, lon1, nx)
    lats = np.linspace(lat1, lat0, ny)                      # nord primer
    cols = np.floor((lons - left) / (right - left) * W).astype(int)
    rows = np.floor((top - lats) / (top - bottom) * H).astype(int)
    out = np.full((ny, nx), np.nan, np.float32)
    okc = (cols >= 0) & (cols < W)
    okr = (rows >= 0) & (rows < H)
    if not okc.any() or not okr.any():
        return out
    C = cols[okc]
    sub = band[np.ix_(rows[okr], C)]
    out[np.ix_(okr, okc)] = sub
    return out


def mosaic_cat(nodes):
    mos = np.full((NY, NX), np.nan, np.float32)
    for (_, _, band, bounds) in nodes:
        g = a_graella(band, bounds)
        mos = np.fmax(mos, g)                               # nanmax: pren el màxim vàlid
    return mos


# ----------------------------------------------------------------- estacions (biaix)
def estacions_1h(password):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import base64
    req = urllib.request.Request(DADES_URL + "?_=" + str(int(datetime.now().timestamp())), headers={"User-Agent": "graf-qpe"})
    blob = json.loads(urllib.request.urlopen(req, timeout=60).read())
    salt = base64.b64decode(blob["salt"]); iv = base64.b64decode(blob["iv"]); ct = base64.b64decode(blob["ct"])
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=blob.get("it", 200000)).derive(password.encode())
    dades = json.loads(AESGCM(key).decrypt(iv, ct, None).decode("utf-8"))
    out = []
    for e in dades.get("estacions", []):
        v = ((e.get("actual") or {}).get("pacum") or {}).get("1h")
        if v is not None and e.get("lat") is not None:
            out.append((float(e["lat"]), float(e["lon"]), float(v)))
    return out


def mostreja(grid, lat, lon, bbox=CAT_BBOX):
    lon0, lon1, lat0, lat1 = bbox
    if not (lon0 <= lon <= lon1 and lat0 <= lat <= lat1):
        return np.nan
    i = min(NX - 1, max(0, int((lon - lon0) / (lon1 - lon0) * NX)))
    j = min(NY - 1, max(0, int((lat1 - lat) / (lat1 - lat0) * NY)))
    return grid[j, i]


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


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="qpe_aemet_1h.png")
    ap.add_argument("--no-biaix", action="store_true")
    a = ap.parse_args()
    key = os.environ.get("AEMET_API_KEY")
    pwd = os.environ.get("MAPA_PASS")
    if not key:
        raise SystemExit("falta AEMET_API_KEY")

    print("Baixant el producte de pluja d'AEMET (HVD)…")
    nodes, tmax = extreu_rn1(baixa_hvd(key))
    if not nodes:
        raise SystemExit("cap node de pluja llegit")
    mos = mosaic_cat(nodes)
    vals = mos[~np.isnan(mos)]
    print("RN1 (cru) a Catalunya: %d cel·les amb dada · màx %.2f · mitjana %.3f · p99 %.2f"
          % (vals.size, float(np.nanmax(mos)) if vals.size else 0,
             float(np.nanmean(mos)) if vals.size else 0,
             float(np.percentile(vals, 99)) if vals.size else 0))
    # DIAGNÒSTIC d'escala: quins valors byte apareixen (per saber si són mm o classes)
    if vals.size:
        u, c = np.unique(vals, return_counts=True)
        parell = sorted(zip(u.tolist(), c.tolist()), key=lambda x: -x[1])[:15]
        print("  valors presents (valor: nº cel·les), més freqüents:")
        print("   " + "  ".join("%g:%d" % (v, n) for v, n in parell))
        print("  (valors distints: %d · mínim>0: %g)" % (u.size, float(u[u > 0].min()) if (u > 0).any() else 0))

    factor = 1.0
    if not a.no_biaix and pwd:
        try:
            ests = estacions_1h(pwd)
            sg = sr = 0.0; nfit = 0
            for lat, lon, mm in ests:
                if mm is None or mm < 0.2:
                    continue
                rv = mostreja(mos, lat, lon)
                if not np.isnan(rv) and rv > 0.1:
                    sg += mm; sr += float(rv); nfit += 1
                    if nfit <= 8:
                        print("   estació %.2f,%.2f: gauge %.1f mm · radar %.1f mm" % (lat, lon, mm, rv))
            if nfit >= 3 and sr > 0:
                factor = sg / sr
                print("Biaix mitjà global (1h): %.2f (de %d estacions amb pluja)" % (factor, nfit))
            else:
                print("Biaix: poques estacions amb pluja coincident (%d)" % nfit)
        except Exception as ex:  # noqa
            print("  avis: biaix no aplicat (%s)" % str(ex)[:80])
    mos = mos * factor

    # PNG acolorit
    out = np.zeros((NY, NX, 4), np.uint8)
    for j in range(NY):
        for i in range(NX):
            v = mos[j, i]
            if np.isnan(v) or v < 0.1:
                continue
            out[j, i, :3] = color_mm(float(v)); out[j, i, 3] = 210
    try:
        from PIL import Image
        Image.fromarray(out, "RGBA").save(a.out)
        print("Escrit %s (%dx%d) · màx corregit %.2f mm" % (a.out, NX, NY, float(np.nanmax(mos)) if vals.size else 0))
    except Exception as ex:  # noqa
        print("  avis: no s'ha pogut escriure el PNG (%s)" % ex)


if __name__ == "__main__":
    main()
