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
import os, re, math, json, base64
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ITER = 200000
ESC = 0.25                      # m/s per unitat int8 (quantització)
TZ_LOCAL = ZoneInfo("Europe/Madrid")
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

    # correcció d'observacions (Barnes): la malla s'ajusta estació per estació
    gi = np.clip(np.round(sx / dx).astype(int), 0, nx - 1)
    gj = np.clip(np.round(sy / dy).astype(int), 0, ny - 1)
    R = 6000.0
    for _ in range(2):
        du = su - U[gj, gi]; dv = sv - V[gj, gi]
        cu = np.zeros((ny, nx)); cv = np.zeros((ny, nx)); wt = np.zeros((ny, nx))
        for i in range(len(W)):
            d2 = (xm[None, :] - sx[i]) ** 2 + (ym[:, None] - sy[i]) ** 2
            w = np.exp(-d2 / R ** 2)
            cu += w * du[i]; cv += w * dv[i]; wt += w
        U += cu / (wt + 1e-9); V += cv / (wt + 1e-9)

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


def winds_per_hora(estacions, inclou_actual=True):
    """{t_iso: [{lat,lon,alt,u,v}, ...]} a partir de l'històric de les estacions.

    IMPORTANT: l'històric només guarda les lectures en punt (:00), però la lectura
    'actual' (la que pinta la barba al mapa) sol ser la de :30. Si no l'afegíem,
    el camp de vents quedava mitja hora endarrerit respecte de les barbes i no
    coincidien (fins a 130° de diferència de direcció). Per això s'afig com a
    fotograma propi, amb la seua hora.
    """
    d = {}
    for e in estacions:
        lat, lon = e.get("lat"), e.get("lon")
        if lat is None or lon is None:
            continue
        alt = e.get("alt") or 0.0

        def _afig(t, vv, dv):
            sp = vv / 3.6            # km/h -> m/s
            r = math.radians(dv)
            d.setdefault(t, []).append({"lat": lat, "lon": lon, "alt": alt,
                                        "u": -sp * math.sin(r), "v": -sp * math.cos(r)})

        vistos = set()
        for row in (e.get("historic") or []):
            t, vv, dv = row.get("t"), row.get("vv"), row.get("dv")
            if not t or vv is None or dv is None:
                continue
            vistos.add(t)
            _afig(t, vv, dv)

        if inclou_actual:
            a = e.get("actual") or {}
            t, vv, dv = a.get("fint"), a.get("vv"), a.get("dv")
            if t and t not in vistos and vv is not None and dv is not None:
                _afig(t, vv, dv)
    return d


def empaqueta_hores(perh, hores, grid_path=None):
    """Genera els camps de les hores donades i els empaqueta (int8, només terra)."""
    camps = []; base = None
    for t in hores:
        try:
            c = genera_camp(perh[t], grid_path)
        except Exception:
            continue
        if base is None:
            base = c
        camps.append((t, c["u"], c["v"]))
    if not camps or base is None:
        return None
    sea = np.array(base["sea"], dtype=bool); land = ~sea
    land_idx = np.nonzero(land)[0]; n = int(land_idx.size)
    buf = np.zeros((len(camps), n, 2), dtype=np.int8); ts = []
    for h, (t, u, v) in enumerate(camps):
        ua = np.asarray(u, dtype=float)[land_idx]; va = np.asarray(v, dtype=float)[land_idx]
        buf[h, :, 0] = np.clip(np.round(ua / ESC), -127, 127).astype(np.int8)
        buf[h, :, 1] = np.clip(np.round(va / ESC), -127, 127).astype(np.int8)
        dt = _iso(t); ts.append(int(dt.timestamp() * 1000) if dt else 0)
    return {"bbox": base["bbox"], "nx": base["nx"], "ny": base["ny"], "esc": ESC,
            "mask": base64.b64encode(np.packbits(land.astype(np.uint8)).tobytes()).decode(),
            "t": ts, "n": n,
            "d": base64.b64encode(buf.tobytes()).decode()}


def _iso(s):
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    s = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", s)
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def escriu_vent(estacions, password, out_file="vent_privat.enc", grid_path=None, max_hores=26):
    """Camp de vent HORARI del dia d'avui (fitxer en directe). Els dies tancats
    els cobreix arxiu-vent/. Així la màquina del temps té camp a cada hora."""
    perh = winds_per_hora(estacions)
    disp = sorted(t for t, w in perh.items() if len(w) >= 5)
    if not disp:
        raise RuntimeError("sense hores amb vent")
    avui = datetime.now(TZ_LOCAL).date()
    hores = [t for t in disp if (_iso(t) or datetime.now(timezone.utc)).astimezone(TZ_LOCAL).date() == avui]
    if not hores:                       # just després de mitjanit: agafem les últimes
        hores = disp[-6:]
    hores = hores[-max_hores:]
    payload = empaqueta_hores(perh, hores, grid_path)
    if not payload:
        raise RuntimeError("no s'ha pogut generar cap camp")
    payload["generat"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    blob = xifrar(json.dumps(payload, separators=(",", ":")), password)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(blob, f)
    return len(payload["t"])


if __name__ == "__main__":
    # prova local: winds_cat.json = {codi: {lat,lon,alt,u,v}}
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "winds_cat.json"
    w = list(json.load(open(src, encoding="utf-8")).values())
    camp = genera_camp(w)
    print("camp OK:", camp["nx"], "x", camp["ny"], "| estacions:", len(w))
