# -*- coding: utf-8 -*-
"""
QPE de producció (fase 2A) — pluja acumulada radar (AEMET) + biaix d'estacions.

Cada execució (workflow cada ~15 min):
  1. Baixa l'últim RN1.1HR d'AEMET i el porta a la graella de Catalunya (mm).  [qpe_aemet]
  2. Manté un BUFFER de tiles HORARIS disjunts a --store/tiles/<YYYYMMDDHH>.npy
     (RN1 = pluja de l'última hora; en desem un cada ~hora per no comptar dues voltes).
  3. Acumula les finestres 1h/3h/6h/24h/dia/setmana i les ajusta amb el biaix mitjà de
     les estacions Meteocat (camp 'pacum' de dades_privat.enc).
  4. Escriu un ràster XIFRAT per finestra (--store/qpe_<win>.enc) + qpe.json (metadada).

El workflow fa el git de la branca 'qpe' (checkout a --store, commit i push).

Ús (al workflow):
    python qpe_prod.py --store qpe_store
"""
import argparse, base64, json, os, glob
from datetime import datetime, timezone, timedelta

import numpy as np

import qpe_aemet as A          # reutilitza baixada/mosaic/estacions ja validats
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ITER = 200000
WINDOWS = {"1h": 1, "3h": 3, "6h": 6, "24h": 24, "dia": None, "7d": 168}   # hores (None = des de mitjanit local)
TILE_MIN_SEP = 55 * 60         # segons mínims entre tiles horaris (evita duplicar la mateixa hora)
BUFFER_DIES = 8
SERIE_TILES = 72               # tiles horaris publicats a qpe_series.enc (escombrar fins ~48 h enrere amb finestres de fins a 24 h)


def xifrar(text, password):
    salt = os.urandom(16); iv = os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(password.encode())
    ct = AESGCM(key).encrypt(iv, text.encode("utf-8"), None)
    return {"v": 1, "kdf": "PBKDF2-SHA256", "it": ITER, "alg": "AES-GCM",
            "salt": base64.b64encode(salt).decode(), "iv": base64.b64encode(iv).decode(),
            "ct": base64.b64encode(ct).decode()}


# ------------------------------------------------------------------ buffer horari
def _ts_de_nom(p):
    b = os.path.basename(p).split(".")[0]
    try:
        return datetime.strptime(b, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def carrega_tiles(tdir):
    out = []
    for p in sorted(glob.glob(os.path.join(tdir, "*.npz"))):
        t = _ts_de_nom(p)
        if t is not None:
            try:
                out.append((t, np.load(p)["g"].astype(np.float32)))
            except Exception:  # noqa
                pass
    return out


def poda(tdir, tiles, ara):
    lim = ara - timedelta(days=BUFFER_DIES)
    for p in glob.glob(os.path.join(tdir, "*.npz")):
        t = _ts_de_nom(p)
        if t is not None and t < lim:
            try:
                os.remove(p)
            except OSError:
                pass
    return [(t, g) for (t, g) in tiles if t >= lim]


# ------------------------------------------------------------------ acumulació + biaix
def suma(tiles, desde, fins):
    acc = None
    for (t, g) in tiles:
        if desde < t <= fins:
            acc = g.copy() if acc is None else np.where(np.isnan(acc), 0, acc) + np.where(np.isnan(g), 0, g)
    return acc


def biaix_global(grid, ests):
    sg = sr = 0.0; n = 0
    for lat, lon, mm in ests:
        if mm is None or mm < 0.2:
            continue
        rv = A.mostreja(grid, lat, lon)
        if not np.isnan(rv) and rv > 0.1:
            sg += mm; sr += float(rv); n += 1
    return (sg / sr, n) if (n >= 3 and sr > 0) else (1.0, n)


def _frame(grid):
    """grid mm (ny,nx, NaN=sense pluja), NORD-a-dalt -> {esc,mask,d,n} SUD-primer.
    El frontend (PrecipLayer) vol row 0 = sud (com l'IDW) -> invertim verticalment."""
    grid = np.asarray(grid, np.float32)[::-1, :]
    val = (~np.isnan(grid)) & (grid >= 0.1)
    mm = np.where(val, grid, 0.0).astype(np.float32)
    mx = float(mm.max()) if val.any() else 1.0
    esc = max(0.25, mx / 250.0)
    q = np.clip(np.round(mm / esc), 0, 255).astype(np.uint8)
    idx = np.nonzero(val.ravel())[0]
    return {"esc": round(esc, 4),
            "mask": base64.b64encode(np.packbits(val.ravel().astype(np.uint8)).tobytes()).decode(),
            "d": base64.b64encode(q.ravel()[idx].tobytes()).decode(),
            "n": int(idx.size)}


def quantitza_xifra(grid, meta, password):
    """grid mm (ny,nx, NaN=sense pluja) -> payload xifrat compatible amb el frontend."""
    payload = dict(meta)
    payload.update({
        "bbox": [A.CAT_BBOX[0], A.CAT_BBOX[1], A.CAT_BBOX[2], A.CAT_BBOX[3]],
        "nx": A.NX, "ny": A.NY,
    })
    payload.update(_frame(grid))
    return xifrar(json.dumps(payload, separators=(",", ":")), password)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="qpe_store")
    a = ap.parse_args()
    key = os.environ.get("AEMET_API_KEY"); pwd = os.environ.get("MAPA_PASS")
    if not key or not pwd:
        raise SystemExit("calen AEMET_API_KEY i MAPA_PASS")
    tdir = os.path.join(a.store, "tiles")
    os.makedirs(tdir, exist_ok=True)

    print("Baixant l'últim RN1 d'AEMET…")
    nodes, tmax = A.extreu_rn1(A.baixa_hvd(key))
    mos = A.mosaic_cat(nodes)                                  # mm de l'última hora, graella CAT
    ncel = int((~np.isnan(mos)).sum())
    print("RN1 %s UTC · %d cel·les amb pluja · màx %.1f mm"
          % (tmax.strftime("%Y-%m-%d %H:%M"), ncel, float(np.nanmax(mos)) if ncel else 0.0))

    tiles = carrega_tiles(tdir)
    # desa un tile nou si ha passat prou temps des de l'últim (tiles horaris disjunts)
    darrer = max((t for (t, _) in tiles), default=None)
    if darrer is None or (tmax - darrer).total_seconds() >= TILE_MIN_SEP:
        np.savez_compressed(os.path.join(tdir, tmax.strftime("%Y%m%d%H%M") + ".npz"), g=mos.astype(np.float16))
        tiles.append((tmax, mos.astype(np.float32)))
        print("  tile horari desat: %s" % tmax.strftime("%Y-%m-%d %H:%M"))
    else:
        print("  (encara no toca tile nou; últim fa %d min)" % int((tmax - darrer).total_seconds() / 60))
    tiles = poda(tdir, tiles, tmax)

    # estacions per al biaix (pacum de dades_privat.enc)
    ests_win = {}
    try:
        ests_win = carrega_pacums(pwd)
    except Exception as ex:  # noqa
        print("  avis: no s'han pogut llegir els pacum de les estacions (%s)" % str(ex)[:70])

    # mitjanit local (per 'dia')
    try:
        from zoneinfo import ZoneInfo
        mitjanit = tmax.astimezone(ZoneInfo("Europe/Madrid")).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    except Exception:  # noqa
        mitjanit = tmax.replace(hour=0, minute=0, second=0, microsecond=0)

    meta_windows = {}
    for win, hores in WINDOWS.items():
        if win == "1h":
            grid = mos.copy()
        elif win == "dia":
            grid = suma(tiles, mitjanit, tmax)
        else:
            grid = suma(tiles, tmax - timedelta(hours=hores), tmax)
        if grid is None:
            print("  %s: sense dades encara" % win); continue
        ests = ests_win.get(win) or []
        factor, nfit = biaix_global(grid, ests)
        gridc = grid * factor
        meta = {"win": win, "obs": int(tmax.timestamp() * 1000), "biaix": round(factor, 3),
                "nfit": nfit, "generat": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")}
        enc = quantitza_xifra(gridc, meta, pwd)
        with open(os.path.join(a.store, "qpe_%s.enc" % win), "w", encoding="utf-8") as f:
            json.dump(enc, f)
        mx = float(np.nanmax(gridc)) if np.isfinite(np.nanmax(gridc)) else 0.0
        meta_windows[win] = {"biaix": round(factor, 3), "nfit": nfit, "max_mm": round(mx, 1)}
        print("  %s: biaix %.2f (%d est.) · màx %.1f mm" % (win, factor, nfit, mx))

    # ---- SÈRIE HORÀRIA ESCOMBRABLE (últims SERIE_TILES tiles) ----
    # Un únic fitxer amb els tiles horaris disjunts; el frontend suma la finestra que
    # acaba a l'hora que trie la màquina del temps. Apliquem el biaix d'1 h uniforme.
    b1h = meta_windows.get("1h", {}).get("biaix", 1.0)
    serie = sorted(tiles, key=lambda x: x[0])[-SERIE_TILES:]
    frames = []
    for (t, g) in serie:
        fr = _frame(g * b1h)
        fr["ts"] = int(t.timestamp() * 1000)
        frames.append(fr)
    if frames:
        payload = {"bbox": [A.CAT_BBOX[0], A.CAT_BBOX[1], A.CAT_BBOX[2], A.CAT_BBOX[3]],
                   "nx": A.NX, "ny": A.NY, "dt": 3600000, "biaix": round(b1h, 3),
                   "obs": int(tmax.timestamp() * 1000), "frames": frames}
        with open(os.path.join(a.store, "qpe_series.enc"), "w", encoding="utf-8") as f:
            json.dump(xifrar(json.dumps(payload, separators=(",", ":")), pwd), f)
        print("  sèrie escombrable: %d tiles (fins %s)" % (len(frames), serie[0][0].strftime("%Y-%m-%d %H:%M")))

    with open(os.path.join(a.store, "qpe.json"), "w", encoding="utf-8") as f:
        json.dump({"generat": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                   "obs": int(tmax.timestamp() * 1000), "n_tiles": len(tiles), "serie_tiles": len(frames),
                   "windows": meta_windows, "font": "AEMET RN1 (radar) + biaix Meteocat"}, f)
    print("Fet: %d tiles al buffer · %d finestres · %d tiles a la sèrie" % (len(tiles), len(meta_windows), len(frames)))


def carrega_pacums(password):
    """{win: [(lat,lon,mm)]} de les estacions, per a cada finestra (camp pacum)."""
    import urllib.request
    req = urllib.request.Request(A.DADES_URL + "?_=" + str(int(datetime.now().timestamp())), headers={"User-Agent": "graf-qpe"})
    blob = json.loads(urllib.request.urlopen(req, timeout=60).read())
    salt = base64.b64decode(blob["salt"]); iv = base64.b64decode(blob["iv"]); ct = base64.b64decode(blob["ct"])
    keyb = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=blob.get("it", ITER)).derive(password.encode())
    dades = json.loads(AESGCM(keyb).decrypt(iv, ct, None).decode("utf-8"))
    out = {w: [] for w in WINDOWS}
    for e in dades.get("estacions", []):
        pac = (e.get("actual") or {}).get("pacum") or {}
        if e.get("lat") is None:
            continue
        for w in WINDOWS:
            v = pac.get(w)
            if v is not None:
                out[w].append((float(e["lat"]), float(e["lon"]), float(v)))
    return out


if __name__ == "__main__":
    main()
