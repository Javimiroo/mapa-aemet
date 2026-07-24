# -*- coding: utf-8 -*-
"""
Converteix la sortida de WindNinja (ESRI ASCII en UTM) al format compacte i xifrat
que llig privat.html, per a pintar el camp fi d'una zona d'incendi.

WindNinja escriu dos fitxers:
    *_vel.asc  velocitat  -> EN MILLES PER HORA (comprovat empíricament)
    *_ang.asc  direcció   -> graus, d'on ve el vent

El payload resultant té la MATEIXA forma que vent_privat.enc (un sol fotograma),
així el navegador el pot descodificar amb la funció que ja té.

Ús:
    python windninja_zona.py --dir wn/zona --bbox 0.90,41.22,1.02,41.34 \
        --estacions wn/estacions --out vent_zona.enc
"""
import os, re, glob, json, math, base64, argparse
from datetime import datetime, timezone

import numpy as np
from pyproj import Transformer
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ITER = 200000
ESC = 0.1           # m/s per unitat int8 (fi: resol ~0,36 km/h, màx ~46 km/h de mitjana)
                    # abans 0,25 -> els vents fluixos de vall s'arrodonien a 0 i no es veien
MPH = 0.44704       # milles/hora -> m/s


def xifrar(text, password):
    salt = os.urandom(16); iv = os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(password.encode())
    ct = AESGCM(key).encrypt(iv, text.encode("utf-8"), None)
    return {"v": 1, "kdf": "PBKDF2-SHA256", "it": ITER, "alg": "AES-GCM",
            "salt": base64.b64encode(salt).decode(), "iv": base64.b64encode(iv).decode(),
            "ct": base64.b64encode(ct).decode()}


def llig_asc(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            h[k.lower()] = float(v)
        dat = np.loadtxt(f)
    return h, dat


def troba_parell(dirpath):
    vel = sorted(glob.glob(os.path.join(dirpath, "*_vel.asc")))
    ang = sorted(glob.glob(os.path.join(dirpath, "*_ang.asc")))
    if not vel or not ang:
        raise SystemExit("No trobe *_vel.asc i *_ang.asc a %s" % dirpath)
    return vel[-1], ang[-1]


def hora_del_nom(path):
    """dem_point_07-22-2026_1330_100m_vel.asc -> epoch ms (UTC)."""
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})_(\d{2})(\d{2})", os.path.basename(path))
    if not m:
        return int(datetime.now(timezone.utc).timestamp() * 1000)
    mo, dd, yy, hh, mi = (int(x) for x in m.groups())
    return int(datetime(yy, mo, dd, hh, mi, tzinfo=timezone.utc).timestamp() * 1000)


def utm_epsg(lon, lat):
    z = int(math.floor((lon + 180) / 6) + 1)
    return (32600 if lat >= 0 else 32700) + z


def converteix(dirpath, bbox, nx=220, ny=220, estacions=None):
    fvel, fang = troba_parell(dirpath)
    hv, vel = llig_asc(fvel)
    ha, ang = llig_asc(fang)
    if vel.shape != ang.shape:
        raise SystemExit("velocitat i direcció no tenen la mateixa graella")
    nod = hv["nodata_value"]
    rows, cols = vel.shape
    cs = hv["cellsize"]; x0 = hv["xllcorner"]; y0 = hv["yllcorner"]

    lon0, lat0, lon1, lat1 = bbox
    epsg = utm_epsg(0.5 * (lon0 + lon1), 0.5 * (lat0 + lat1))
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:%d" % epsg, always_xy=True)

    # graella de sortida en lat/lon (nord primer, com la resta del sistema)
    lons = np.linspace(lon0, lon1, nx)
    lats = np.linspace(lat1, lat0, ny)
    LON, LAT = np.meshgrid(lons, lats)
    X, Y = fwd.transform(LON, LAT)
    C = np.floor((X - x0) / cs).astype(int)
    R = (rows - 1 - np.floor((Y - y0) / cs)).astype(int)
    dins = (C >= 0) & (C < cols) & (R >= 0) & (R < rows)
    Cc = np.clip(C, 0, cols - 1); Rc = np.clip(R, 0, rows - 1)

    V = vel[Rc, Cc]; A = ang[Rc, Cc]
    val = dins & (V != nod) & (A != nod)
    sp = np.where(val, V * MPH, 0.0)                 # mph -> m/s
    rad = np.radians(np.where(val, A, 0.0))
    u = -sp * np.sin(rad)                            # component est
    v = -sp * np.cos(rad)                            # component nord

    # comprovació de seguretat: el camp ha de reproduir les estacions
    avis = comprova(estacions, fwd, vel, ang, nod, hv, rows, cols)

    idx = np.nonzero(val.ravel())[0]
    n = int(idx.size)
    if not n:
        raise SystemExit("cap cel·la vàlida dins del bbox")
    buf = np.zeros((1, n, 2), dtype=np.int8)
    buf[0, :, 0] = np.clip(np.round(u.ravel()[idx] / ESC), -127, 127)
    buf[0, :, 1] = np.clip(np.round(v.ravel()[idx] / ESC), -127, 127)

    payload = {
        "bbox": [lon0, lon1, lat0, lat1], "nx": nx, "ny": ny, "esc": ESC,
        "mask": base64.b64encode(np.packbits(val.ravel().astype(np.uint8)).tobytes()).decode(),
        "t": [hora_del_nom(fvel)], "n": n,
        "d": base64.b64encode(buf.tobytes()).decode(),
        "font": "WindNinja (massa + estacions)",
        "malla_m": cs,
        "generat": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    if avis:
        payload["avis"] = avis
    print("  camp: %dx%d (%d cel·les vàlides) · malla WindNinja %g m · vent %.1f-%.1f km/h"
          % (nx, ny, n, cs, sp[val].min() * 3.6, sp[val].max() * 3.6))
    return payload


def comprova(estdir, fwd, vel, ang, nod, hv, rows, cols):
    """Compara el camp amb les estacions: si el factor no és ~1, les unitats fallen."""
    if not estdir or not os.path.isdir(estdir):
        return None
    difs = []
    for fn in glob.glob(os.path.join(estdir, "*.csv")):
        try:
            linies = open(fn, encoding="utf-8").read().splitlines()
            if len(linies) < 2:
                continue
            c = [x.strip().strip('"') for x in linies[1].split(",")]
            lat, lon, sp_obs = float(c[3]), float(c[4]), float(c[7])   # m/s
            x, y = fwd.transform(lon, lat)
            cc = int((x - hv["xllcorner"]) / hv["cellsize"])
            rr = int(rows - 1 - (y - hv["yllcorner"]) / hv["cellsize"])
            if not (0 <= rr < rows and 0 <= cc < cols):
                continue
            v = vel[rr, cc]
            if v == nod or sp_obs < 0.5:
                continue
            difs.append((v * MPH) / sp_obs)
        except Exception:
            continue
    if not difs:
        return None
    f = float(np.median(difs))
    print("  comprovació d'unitats amb %d estacions: factor mitjà %.2f" % (len(difs), f))
    if not (0.6 <= f <= 1.6):
        av = ("ATENCIÓ: el camp no quadra amb les estacions (factor %.2f). "
              "Revisa les unitats de sortida de WindNinja." % f)
        print("  " + av)
        return av
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="carpeta amb els *_vel.asc / *_ang.asc")
    ap.add_argument("--bbox", required=True, help="lon0,lat0,lon1,lat1")
    ap.add_argument("--estacions", default=None, help="carpeta dels CSV (per a la comprovació)")
    ap.add_argument("--nx", type=int, default=220)
    ap.add_argument("--ny", type=int, default=220)
    ap.add_argument("--out", default="vent_zona.enc")
    a = ap.parse_args()
    bbox = tuple(float(x) for x in a.bbox.split(","))
    p = converteix(a.dir, bbox, a.nx, a.ny, a.estacions)
    pwd = os.environ.get("MAPA_PASS")
    if not pwd:
        raise SystemExit("Falta la variable d'entorn MAPA_PASS")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(xifrar(json.dumps(p, separators=(",", ":")), pwd), f)
    print("  escrit %s (%.2f MB)" % (a.out, os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main()
