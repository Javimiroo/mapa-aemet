# -*- coding: utf-8 -*-
"""
Camp de vents diagnòstic (mass-consistent) per a tota Catalunya a partir dels
vents observats de les estacions. Reutilitza la graella de terreny precalculada
(vent_grid.npz) i xifra el resultat amb el MATEIX esquema que fetch_privat.py
(PBKDF2-SHA256 + AES-GCM) -> vent_privat.enc, que privat.html desxifra amb la mateixa clau.

Ús des del fetcher:
    from camp_vents import escriu_vent
    escriu_vent(estacions, PASSWORD)
"""
import os, math, json, base64
from datetime import datetime, timezone
import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ITER = 200000
_HERE = os.path.dirname(os.path.abspath(__file__))
GRID_DEFAULT = os.path.join(_HERE, "vent_grid.npz")


def xifrar(text, password):
    salt = os.urandom(16); iv = os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(password.encode())
    ct = AESGCM(key).encrypt(iv, text.encode("utf-8"), None)
    return {"v": 1, "kdf": "PBKDF2-SHA256", "it": ITER, "alg": "AES-GCM",
            "salt": base64.b64encode(salt).decode(), "iv": base64.b64encode(iv).decode(),
            "ct": base64.b64encode(ct).decode()}


def _poisson(f, dx, dy):
    ny, nx = f.shape
    fh = dctn(f, type=2, norm="ortho")
    lx = 2 * (np.cos(np.pi * np.arange(nx) / nx) - 1) / dx ** 2
    ly = 2 * (np.cos(np.pi * np.arange(ny) / ny) - 1) / dy ** 2
    den = ly[:, None] + lx[None, :]; den[0, 0] = 1
    ph = fh / den; ph[0, 0] = 0
    return idctn(ph, type=2, norm="ortho")


def genera_camp(winds, grid_path=None, nxT=160, nyT=120):
    """winds: llista de dicts {lat,lon,alt,u,v} (u=est, v=nord, en m/s)."""
    g = np.load(grid_path or GRID_DEFAULT)
    H = g["H"].astype(float); sea = g["sea"].astype(bool)
    LON0, LON1, LAT0, LAT1 = [float(x) for x in g["bbox"]]
    nx = int(g["nx"]); ny = int(g["ny"]); dx = float(g["dx"]); dy = float(g["dy"])
    latc = (LAT0 + LAT1) / 2; mlon = 111320 * math.cos(math.radians(latc))
    lon = np.linspace(LON0, LON1, nx); lat = np.linspace(LAT0, LAT1, ny)
    xm = (lon - LON0) * mlon; ym = (lat - LAT0) * 111132.0

    W = [w for w in winds if w.get("u") is not None and w.get("v") is not None
         and w.get("lat") is not None and w.get("lon") is not None]
    if len(W) < 3:
        raise RuntimeError("massa poques estacions amb vent (%d)" % len(W))
    sx = np.array([(w["lon"] - LON0) * mlon for w in W])
    sy = np.array([(w["lat"] - LAT0) * 111132.0 for w in W])
    salt = np.array([w.get("alt") or 0.0 for w in W])
    su = np.array([w["u"] for w in W]); sv = np.array([w["v"] for w in W])

    Lz = 350.0
    U = np.zeros((ny, nx)); V = np.zeros((ny, nx)); Wt = np.zeros((ny, nx))
    for i in range(len(W)):
        d2 = (xm[None, :] - sx[i]) ** 2 + (ym[:, None] - sy[i]) ** 2
        w = 1.0 / (d2 + 800.0 ** 2) * np.exp(-((H - salt[i]) / Lz) ** 2)
        U += w * su[i]; V += w * sv[i]; Wt += w
    U /= Wt; V /= Wt

    gy, gx = np.gradient(H, ym, xm); gmag = np.hypot(gx, gy) + 1e-9
    sxu, syu = gx / gmag, gy / gmag
    climb = U * sxu + V * syu; defl = 0.9 * np.tanh(gmag / 0.15)
    U -= defl * climb * sxu; V -= defl * climb * syu
    expo = H - gaussian_filter(H, 3000.0 / abs(dx))
    fac = 1.0 + np.clip(expo / 300.0, -0.45, 0.7); U *= fac; V *= fac
    for _ in range(2):
        phi = _poisson(np.gradient(U, xm, axis=1) + np.gradient(V, ym, axis=0), dx, dy)
        U -= np.gradient(phi, xm, axis=1); V -= np.gradient(phi, ym, axis=0)

    # downsample a la graella de visualització (nord primer)
    us = []; vs = []; se = []
    for j2 in range(nyT):
        latq = LAT1 - j2 * (LAT1 - LAT0) / (nyT - 1)
        js = min(ny - 1, max(0, round((latq - LAT0) / ((LAT1 - LAT0) / (ny - 1)))))
        for i2 in range(nxT):
            lonq = LON0 + i2 * (LON1 - LON0) / (nxT - 1)
            isx = min(nx - 1, max(0, round((lonq - LON0) / ((LON1 - LON0) / (nx - 1)))))
            us.append(round(float(U[js, isx]), 2)); vs.append(round(float(V[js, isx]), 2))
            se.append(int(bool(sea[js, isx])))
    return {"bbox": [LON0, LON1, LAT0, LAT1], "nx": nxT, "ny": nyT, "u": us, "v": vs, "sea": se}


def _winds_de_estacions(estacions):
    out = []
    for e in estacions:
        a = e.get("actual") or {}
        vv = a.get("vv"); dv = a.get("dv")
        if vv is None or dv is None or e.get("lat") is None or e.get("lon") is None:
            continue
        sp = vv / 3.6  # km/h -> m/s
        r = math.radians(dv)
        out.append({"lat": e["lat"], "lon": e["lon"], "alt": e.get("alt") or 0.0,
                    "u": -sp * math.sin(r), "v": -sp * math.cos(r)})
    return out


def escriu_vent(estacions, password, out_file="vent_privat.enc", grid_path=None):
    winds = _winds_de_estacions(estacions)
    camp = genera_camp(winds, grid_path)
    camp["generat"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    blob = xifrar(json.dumps(camp, separators=(",", ":")), password)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(blob, f)
    return len(winds)


if __name__ == "__main__":
    # prova local: winds_cat.json = {codi: {lat,lon,alt,u,v}}
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "winds_cat.json"
    w = list(json.load(open(src, encoding="utf-8")).values())
    camp = genera_camp(w)
    print("camp OK:", camp["nx"], "x", camp["ny"], "| estacions:", len(w))
