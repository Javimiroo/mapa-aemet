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


def troba_parells(dirpath):
    """Tots els parells (vel, ang) de la carpeta, ordenats per l'hora del nom.
    Una execució normal en deixa un; una SIMULACIÓ, un per hora."""
    out = []
    for fvel in glob.glob(os.path.join(dirpath, "*_vel.asc")):
        fang = fvel[:-8] + "_ang.asc"
        if os.path.exists(fang):
            out.append((hora_del_nom(fvel), fvel, fang))
    out.sort()
    # una hora = un parell (si n'hi ha de repetits, l'últim escrit)
    uniq = {}
    for t, fv, fa in out:
        uniq[t] = (fv, fa)
    return [(t,) + uniq[t] for t in sorted(uniq)]


def mostreja_dem(dem_path, fwd, LON, LAT):
    """Elevació (m, int16) del DEM de WindNinja mostrejada a la graella lat/lon de sortida.
    Serveix al navegador per a calcular pendent i orientació a cada punt del front."""
    try:
        import rasterio
        with rasterio.open(dem_path) as ds:
            z = ds.read(1)
            X, Y = fwd.transform(LON, LAT)
            cols = np.clip(np.floor((X - ds.transform.c) / ds.transform.a).astype(int), 0, ds.width - 1)
            rows = np.clip(np.floor((Y - ds.transform.f) / ds.transform.e).astype(int), 0, ds.height - 1)
            e = z[rows, cols]
            e = np.where(e < -500, 0, e)
            return np.clip(np.round(e), -32000, 32000).astype("<i2")
    except Exception as ex:
        print("  avis: no s'ha pogut afegir el relleu al payload (%s)" % ex)
        return None


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


def mostreja_combustible(path, bbox, pas_m=20.0, maxn=900):
    """Classes de combustible (uint8) del ràster regional (combustible_prep.py) a la caixa, en una graella
    lat/lon regular d'uns 20 m (nord primer). Torna {nx, ny, gz} amb les dades comprimides en gzip+base64."""
    try:
        import gzip
        import rasterio
        lon0, lat0, lon1, lat1 = bbox
        latm = 0.5 * (lat0 + lat1)
        fx = int(min(maxn, max(8, round((lon1 - lon0) * 111320.0 * math.cos(math.radians(latm)) / pas_m))))
        fy = int(min(maxn, max(8, round((lat1 - lat0) * 111320.0 / pas_m))))
        lons = lon0 + (np.arange(fx) + 0.5) * (lon1 - lon0) / fx
        lats = lat1 - (np.arange(fy) + 0.5) * (lat1 - lat0) / fy
        LON, LAT = np.meshgrid(lons, lats)
        with rasterio.open(path) as ds:
            tf = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            X, Y = tf.transform(LON, LAT)
            c = np.floor((X - ds.transform.c) / ds.transform.a).astype(int)
            r = np.floor((Y - ds.transform.f) / ds.transform.e).astype(int)
            ok = (c >= 0) & (c < ds.width) & (r >= 0) & (r < ds.height)
            if not ok.any():
                return None
            c0, c1, r0, r1 = int(c[ok].min()), int(c[ok].max()), int(r[ok].min()), int(r[ok].max())
            from rasterio.windows import Window
            a = ds.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1))
            out = np.zeros((fy, fx), np.uint8)
            out[ok] = a[r[ok] - r0, c[ok] - c0]
            font = ds.tags().get("font", "")
        if not out.any():
            return None
        v, n = np.unique(out, return_counts=True)
        print("  combustible afegit: %dx%d · classes %s" % (fx, fy, ", ".join("%d:%d%%" % (int(a_), round(100.0 * b_ / out.size)) for a_, b_ in zip(v, n))))
        return {"nx": fx, "ny": fy, "gz": base64.b64encode(gzip.compress(out.tobytes(), 9)).decode(), "font": font}
    except Exception as ex:
        print("  avis: no s'ha pogut afegir el combustible (%s)" % ex)
        return None


def converteix(dirpath, bbox, nx=None, ny=None, estacions=None, dem=None, combustible=None):
    parells = troba_parells(dirpath)
    if not parells:
        raise SystemExit("No trobe *_vel.asc i *_ang.asc a %s" % dirpath)
    serie = len(parells) > 1
    # una simulació porta diverses hores: graella una mica més grossa perquè el fitxer no passe d'1 MB
    nx = nx or (160 if serie else 220)
    ny = ny or (160 if serie else 220)
    _t0, fvel, fang = parells[0]
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

    # comprovació de seguretat: el camp ha de reproduir les estacions (vel + direcció)
    avis, ajust = comprova(estacions, fwd, vel, ang, nod, hv, rows, cols)
    if os.path.exists(os.path.join(dirpath, "PLA_B")):
        pb = ("Règim complex: l'ajust exacte a les estacions no ha convergit; "
              "camp APROXIMAT d'una sola passada.")
        avis = pb + (" " + avis if avis else "")

    # la resta d'hores (simulació): mateixa graella; la màscara és la comuna a totes
    camps = [(u, v)]
    temps = [parells[0][0]]
    vmin, vmax = float(sp[val].min()), float(sp[val].max())
    for tk, fv, fa in parells[1:]:
        hk, velk = llig_asc(fv)
        _ha, angk = llig_asc(fa)
        if velk.shape != vel.shape or angk.shape != vel.shape:
            print("  avis: l'hora %s té una altra graella; es descarta" % os.path.basename(fv))
            continue
        Vk = velk[Rc, Cc]; Ak = angk[Rc, Cc]
        valk = dins & (Vk != nod) & (Ak != nod)
        val = val & valk
        spk = np.where(valk, Vk * MPH, 0.0)
        radk = np.radians(np.where(valk, Ak, 0.0))
        camps.append((-spk * np.sin(radk), -spk * np.cos(radk)))
        temps.append(tk)
        if valk.any():
            vmin = min(vmin, float(spk[valk].min())); vmax = max(vmax, float(spk[valk].max()))

    idx = np.nonzero(val.ravel())[0]
    n = int(idx.size)
    if not n:
        raise SystemExit("cap cel·la vàlida dins del bbox")
    buf = np.zeros((len(camps), n, 2), dtype=np.int8)
    for k, (uk, vk) in enumerate(camps):
        buf[k, :, 0] = np.clip(np.round(uk.ravel()[idx] / ESC), -127, 127)
        buf[k, :, 1] = np.clip(np.round(vk.ravel()[idx] / ESC), -127, 127)

    payload = {
        "bbox": [lon0, lon1, lat0, lat1], "nx": nx, "ny": ny, "esc": ESC,
        "mask": base64.b64encode(np.packbits(val.ravel().astype(np.uint8)).tobytes()).decode(),
        "t": temps, "n": n,
        "d": base64.b64encode(buf.tobytes()).decode(),
        "font": "WindNinja (massa + estacions)",
        "malla_m": cs,
        "generat": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    if avis:
        payload["avis"] = avis
    if ajust:
        payload["ajust"] = ajust
    dem_path = dem or os.path.join(dirpath, "dem.tif")
    if os.path.exists(dem_path):
        z = mostreja_dem(dem_path, fwd, LON, LAT)
        if z is not None:
            payload["dem"] = base64.b64encode(z.tobytes()).decode()     # int16 LE, metres, ny*nx, nord primer
            print("  relleu afegit: %d..%d m" % (int(z.min()), int(z.max())))
    comb = combustible or "combustible_20m.tif"          # ràster regional de models de combustible (si el repo en té)
    if os.path.exists(comb):
        fu = mostreja_combustible(comb, bbox)
        if fu:
            payload["fuel"] = fu
    print("  camp: %dx%d (%d cel·les vàlides) · %d hora/es · malla WindNinja %g m · vent %.1f-%.1f km/h"
          % (nx, ny, n, len(camps), cs, vmin * 3.6, vmax * 3.6))
    return payload


DIR_LLINDAR = 45.0      # error mitjà de direcció (graus) a partir del qual el camp s'avisa
VEL_MIN_DIR = 0.8       # m/s (~3 km/h): per davall, la direcció observada és soroll i no es compara


def comprova(estdir, fwd, vel, ang, nod, hv, rows, cols):
    """Compara el camp final amb les estacions observades: unitats (velocitat) i
    DIRECCIÓ (error angular circular, estació per estació). Retorna (avis, ajust):
    avis = text per al visor si algo no quadra; ajust = {n, dir_err, factor}."""
    if not estdir or not os.path.isdir(estdir):
        return None, None
    difs = []          # factors de velocitat (unitats)
    derr = []          # errors de direcció (graus)
    # cada fitxer pot dur VÀRIES estacions (format antic: totes en un únic
    # estacions.csv, una fila per estació) — es comproven totes les files
    for fn in glob.glob(os.path.join(estdir, "*.csv")):
        try:
            linies = open(fn, encoding="utf-8").read().splitlines()
        except Exception:
            continue
        for linia in linies[1:]:
            try:
                c = [x.strip().strip('"') for x in linia.split(",")]
                if len(c) < 10:
                    continue
                nomest = c[0]
                lat, lon, sp_obs, dir_obs = float(c[3]), float(c[4]), float(c[7]), float(c[9])
                x, y = fwd.transform(lon, lat)
                cc = int((x - hv["xllcorner"]) / hv["cellsize"])
                rr = int(rows - 1 - (y - hv["yllcorner"]) / hv["cellsize"])
                if not (0 <= rr < rows and 0 <= cc < cols):
                    continue
                v = vel[rr, cc]; a = ang[rr, cc]
                if v == nod or a == nod:
                    continue
                if sp_obs >= 0.5:
                    difs.append((v * MPH) / sp_obs)
                if sp_obs >= VEL_MIN_DIR:
                    d = abs((float(a) - dir_obs + 180.0) % 360.0 - 180.0)
                    derr.append(d)
                    print("  direcció %-28s obs %3d° · camp %3d° · dif %3d°"
                          % (nomest[:28], round(dir_obs), round(float(a)), round(d)))
            except Exception:
                continue
    if not difs and not derr:
        return None, None
    avis = None
    ajust = {}
    if difs:
        f = float(np.median(difs))
        ajust["factor"] = round(f, 2)
        print("  comprovació d'unitats amb %d estacions: factor mitjà %.2f" % (len(difs), f))
        if not (0.6 <= f <= 1.6):
            avis = ("ATENCIÓ: el camp s'aparta de la velocitat observada "
                    "(factor %.2f: <1 subestima, >1 sobreestima)." % f)
    if derr:
        e = float(np.median(derr))
        ajust["n"] = len(derr)
        ajust["dir_err"] = int(round(e))
        print("  comprovació de DIRECCIÓ amb %d estacions: error mitjà %d°" % (len(derr), round(e)))
        if avis is None and e > DIR_LLINDAR:
            avis = ("Règim fluix/desorganitzat: el camp s'aparta una mitjana de %d° "
                    "de les %d estacions de la caixa — direcció poc fiable." % (round(e), len(derr)))
    if avis:
        print("  ⚠ " + avis)
    return avis, (ajust or None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="carpeta amb els *_vel.asc / *_ang.asc")
    ap.add_argument("--bbox", required=True, help="lon0,lat0,lon1,lat1")
    ap.add_argument("--estacions", default=None, help="carpeta dels CSV (per a la comprovació)")
    ap.add_argument("--nx", type=int, default=None, help="per defecte 220 (160 si hi ha diverses hores)")
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--dem", default=None, help="DEM de WindNinja (per defecte <dir>/dem.tif) per afegir el relleu")
    ap.add_argument("--combustible", default=None, help="ràster de classes de combustible (per defecte ./combustible_20m.tif si existeix)")
    ap.add_argument("--out", default="vent_zona.enc")
    a = ap.parse_args()
    bbox = tuple(float(x) for x in a.bbox.split(","))
    p = converteix(a.dir, bbox, a.nx, a.ny, a.estacions, a.dem, a.combustible)
    pwd = os.environ.get("MAPA_PASS")
    if not pwd:
        raise SystemExit("Falta la variable d'entorn MAPA_PASS")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(xifrar(json.dumps(p, separators=(",", ":")), pwd), f)
    print("  escrit %s (%.2f MB)" % (a.out, os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main()
