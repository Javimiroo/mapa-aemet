# -*- coding: utf-8 -*-
"""
HARMONIE-AROME d'AEMET -> previsió PUNTUAL per a cada incendi (projecte GRAF).

Els mapes del model venen com a imatges de color (GeoTIFF RGBA) amb la LLEGENDA
dins (GDAL_METADATA -> Item 'ESCALA': color -> interval de valors). Aquest script:

  1. Baixa l'última passada (tar.gz) amb l'AEMET_API_KEY.
  2. De cada camp (temp, vent, ratxa, precip, núvols, llamps) descodifica la imatge
     a VALORS FÍSICS invertint la paleta: projecta el color del píxel sobre la
     "rampa" de colors de l'ESCALA i INTERPOLA -> valors quasi continus (millor que
     ajustar al punt mitjà de la classe).  La direcció del vent ve al geojson
     (direcc_viento_33) i es rasteritza per veí més proper.
  3. Retalla a Catalunya, quantitza (uint8 per camp) + gzip + xifra (AES-GCM, igual
     que la resta del sistema) a --store/harmonie_<run>.enc  (buffer de les últimes
     RUNS_BUFFER passades -> el frontend en fa banda min/mitjana/màx, "lagged ensemble").
  4. Escriu harmonie.json (metadada: passades disponibles, camps, hores, bbox).

Ús (al workflow):
    AEMET_API_KEY=... MAPA_PASS=... python harmonie_prep.py --store harmonie_store
"""
import argparse, ast, base64, gzip, io, json, os, glob, re, tarfile, urllib.request
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

BASE_URL = "https://www.aemet.es/es/api-eltiempo/modelos/download/harmonie/%s"
ITER = 200000
RUNS_BUFFER = 4                      # passades que guardem (banda de l'ensemble retardat)
CAT_BBOX = (0.05, 3.45, 40.45, 43.05)   # lon0, lon1, lat0, lat1 (retall de Catalunya)

RX_TS = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
RX_CODE = re.compile(r"T\d{2}:\d{2}:\d{2}\+00:00_(.+)\.(tif|tiff|geojson)$", re.I)

# camp lògic -> (codi del fitxer, unitat, quantització lo/scale per a uint8)
FIELDS = {
    "temp":   {"code": "11",     "unit": "°C",   "lo": -30.0, "scale": 0.35},
    "vent":   {"code": "32",     "unit": "km/h", "lo": 0.0,   "scale": 0.6},
    "ratxa":  {"code": "228",    "unit": "km/h", "lo": 0.0,   "scale": 0.7},
    "precip": {"code": "61_1HH", "unit": "mm",   "lo": 0.0,   "scale": 0.4},
    "nuvol":  {"code": "71",     "unit": "%",    "lo": 0.0,   "scale": 0.5},
    "llamps": {"code": "207",    "unit": "u",    "lo": 0.0,   "scale": 0.001},
}
DIR_CODE = "direcc_viento_33"       # geojson de punts amb ang_viento (graus)


# ------------------------------------------------------------------ xifratge
def xifrar_text(text, password):
    salt = os.urandom(16); iv = os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(password.encode())
    ct = AESGCM(key).encrypt(iv, text.encode("utf-8"), None)
    return {"v": 1, "kdf": "PBKDF2-SHA256", "it": ITER, "alg": "AES-GCM",
            "salt": base64.b64encode(salt).decode(), "iv": base64.b64encode(iv).decode(),
            "ct": base64.b64encode(ct).decode()}


# ------------------------------------------------------------------ ESCALA -> rampa
def parse_escala(gdal_meta_xml):
    """Extreu la llista [(valor_representatiu, (r,g,b))] ordenada per valor de l'Item ESCALA."""
    m = re.search(r"<Item name=\"ESCALA\">(.*?)</Item>", gdal_meta_xml, re.S)
    if not m:
        return None
    d = ast.literal_eval(m.group(1))         # dict amb cometes simples -> literal_eval
    punts = []
    for cl in d.get("Lista RGBA", []):
        v = cl.get("Valores", [])
        rgba = cl.get("RGBA", [])
        try:
            r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
        except Exception:
            continue
        lo = v[0] if len(v) >= 1 and v[0] != "" else None
        hi = v[1] if len(v) >= 2 and v[1] != "" else None
        if lo is not None and hi is not None:
            rep = (float(lo) + float(hi)) / 2.0
        elif lo is not None:
            rep = float(lo)
        elif hi is not None:
            rep = float(hi)
        else:
            continue
        punts.append((rep, (r, g, b)))
    punts.sort(key=lambda x: x[0])
    return punts or None


def inverteix_rampa(rgb, rampa):
    """rgb: array (...,3) float. rampa: [(val,(r,g,b))] ordenada. Torna valor per píxel
    projectant el color sobre la poligonal de colors i interpolant el valor (continu)."""
    vals = np.array([p[0] for p in rampa], np.float32)
    cols = np.array([p[1] for p in rampa], np.float32)           # (K,3)
    flat = rgb.reshape(-1, 3).astype(np.float32)                  # (N,3)
    N = flat.shape[0]; K = len(rampa)
    best_d = np.full(N, np.inf, np.float32)
    best_v = np.zeros(N, np.float32)
    if K == 1:
        return np.full(rgb.shape[:-1], vals[0], np.float32)
    for k in range(K - 1):
        c0 = cols[k]; c1 = cols[k + 1]
        seg = c1 - c0
        L2 = float(seg @ seg) or 1.0
        t = np.clip(((flat - c0) @ seg) / L2, 0.0, 1.0)          # (N,)
        proj = c0 + t[:, None] * seg
        d = np.linalg.norm(flat - proj, axis=1)
        v = vals[k] + t * (vals[k + 1] - vals[k])
        upd = d < best_d
        best_d[upd] = d[upd]; best_v[upd] = v[upd]
    return best_v.reshape(rgb.shape[:-1]), best_d.reshape(rgb.shape[:-1])


# ------------------------------------------------------------------ descàrrega + decode
def baixa(area, api_key):
    req = urllib.request.Request(BASE_URL % area, headers={"api_key": api_key, "User-Agent": "graf-harmonie"})
    body = urllib.request.urlopen(req, timeout=180).read()
    if body[:2] != b"\x1f\x8b":
        raise SystemExit("resposta d'AEMET no és .tar.gz (%d bytes, %r)" % (len(body), body[:6]))
    return body


def _codi(nom):
    m = RX_CODE.search(os.path.basename(nom)); return m.group(1) if m else None


def _hora(nom):
    m = RX_TS.search(nom)
    return datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc) if m else None


def bounds_de_tags(tags, W, H):
    sc = tags.get(33550); tp = tags.get(33922); tr = tags.get(34264)
    if tr:
        t = list(tr); left = t[3]; top = t[7]; sx = t[0]; sy = -t[5]
    elif sc and tp:
        sx, sy = float(sc[0]), float(sc[1]); i, j, X, Y = float(tp[0]), float(tp[1]), float(tp[3]), float(tp[4])
        left = X - i * sx; top = Y + j * sy
    else:
        return None
    return (left, top - H * sy, left + W * sx, top, sx, sy)


def finestra_cat(bounds, W, H):
    """Índexs de píxel (col0,col1,row0,row1) i bbox real del retall de Catalunya."""
    left, bottom, right, top, sx, sy = bounds
    lon0, lon1, lat0, lat1 = CAT_BBOX
    c0 = max(0, int((lon0 - left) / sx)); c1 = min(W, int(np.ceil((lon1 - left) / sx)))
    r0 = max(0, int((top - lat1) / sy)); r1 = min(H, int(np.ceil((top - lat0) / sy)))
    bb = (left + c0 * sx, left + c1 * sx, top - r1 * sy, top - r0 * sy)   # lon0,lon1,lat0,lat1 reals
    return c0, c1, r0, r1, bb, sx, sy


def decodifica(tar_bytes):
    """Torna (run_id, base_iso, hores[iso], camps{nom: (ny,nx,nh) float}, dirgrid, bbox, nx, ny)."""
    from PIL import Image
    tf = tarfile.open(fileobj=io.BytesIO(tar_bytes))
    per_codi = defaultdict(list)          # codi -> [(hora, membre)]
    geoj = defaultdict(list)
    for m in tf.getmembers():
        c = _codi(m.name); h = _hora(m.name)
        if not c or not h:
            continue
        (geoj if m.name.lower().endswith("geojson") else per_codi)[c].append((h, m))

    # hores de referència: les del camp temperatura (11)
    ref = sorted(h for h, _ in per_codi.get("11", []))
    if not ref:
        raise SystemExit("cap camp de temperatura (11) a l'arxiu")
    hores = ref
    nh = len(hores)
    run_cycle = (hores[0] - timedelta(hours=1))
    run_cycle = run_cycle.replace(hour=(run_cycle.hour // 6) * 6, minute=0, second=0)
    run_id = run_cycle.strftime("%Y%m%d%H")

    # finestra CAT a partir del 1r tif de temp
    temp_membres = dict(per_codi["11"])
    im0 = Image.open(io.BytesIO(tf.extractfile(temp_membres[hores[0]]).read()))
    W, Hh = im0.size
    bnd = bounds_de_tags(getattr(im0, "tag_v2", {}), W, Hh)
    if not bnd:
        raise SystemExit("no s'han pogut llegir els bounds del GeoTIFF")
    c0, c1, r0, r1, bbox, sx, sy = finestra_cat(bnd, W, Hh)
    nx, ny = c1 - c0, r1 - r0

    # cache d'ESCALA per codi (mateixa a totes les hores)
    escala = {}

    def rampa_de(cod, im):
        if cod in escala:
            return escala[cod]
        gm = getattr(im, "tag_v2", {}).get(42112, "")
        escala[cod] = parse_escala(gm)
        return escala[cod]

    camps = {}
    for nom, cfg in FIELDS.items():
        cod = cfg["code"]
        membres = dict(per_codi.get(cod, []))
        if not membres:
            print("  avis: camp %s (%s) absent" % (nom, cod)); continue
        out = np.zeros((ny, nx, nh), np.float32)
        rampa = None
        for hi, h in enumerate(hores):
            m = membres.get(h)
            if m is None:
                continue
            im = Image.open(io.BytesIO(tf.extractfile(m).read()))
            if rampa is None:
                rampa = rampa_de(cod, im)
            arr = np.array(im.convert("RGBA"))[r0:r1, c0:c1, :]     # (ny,nx,4)
            rgb = arr[:, :, :3]; al = arr[:, :, 3]
            if rampa:
                v, _d = inverteix_rampa(rgb, rampa)
            else:
                v = np.zeros((ny, nx), np.float32)
            # transparent => sota el llindar (precip/núvols/llamps = 0)
            if nom in ("precip", "nuvol", "llamps"):
                v = np.where(al < 40, 0.0, v)
            out[:, :, hi] = v
        camps[nom] = out
        print("  %-7s %s -> min %.1f màx %.1f (mitj %.1f)" % (nom, cod, float(out.min()), float(out.max()), float(out.mean())))

    # direcció del vent (graus) del geojson -> veí més proper a cada cel·la
    dirgrid = np.full((ny, nx, nh), np.nan, np.float32)
    dmem = dict(geoj.get(DIR_CODE, []))
    if dmem:
        lon0, lon1, lat0, lat1 = bbox
        lons = lon0 + (np.arange(nx) + 0.5) * (lon1 - lon0) / nx
        lats = lat1 - (np.arange(ny) + 0.5) * (lat1 - lat0) / ny
        GX, GY = np.meshgrid(lons, lats)
        for hi, h in enumerate(hores):
            m = dmem.get(h)
            if m is None:
                continue
            try:
                g = json.load(tf.extractfile(m))
            except Exception:
                continue
            pts = []; ang = []
            for f in g.get("features", []):
                co = (f.get("geometry") or {}).get("coordinates")
                a = (f.get("properties") or {}).get("ang_viento")
                if co and a is not None:
                    pts.append((co[0], co[1])); ang.append(float(a))
            if not pts:
                continue
            pts = np.array(pts); ang = np.array(ang, np.float32)
            # nearest point (n cel·les × m punts, m~1000: fem-ho per blocs simples)
            idx = np.zeros((ny, nx), np.int32)
            for j in range(ny):
                d = (pts[:, 0][None, :] - GX[j][:, None]) ** 2 + (pts[:, 1][None, :] - GY[j][:, None]) ** 2
                idx[j] = d.argmin(axis=1)
            dirgrid[:, :, hi] = ang[idx]

    return run_id, hores[0].isoformat(), [h.isoformat() for h in hores], camps, dirgrid, bbox, nx, ny


# ------------------------------------------------------------------ empaquetat + xifrat
def quantitza(camp, lo, scale):
    q = np.clip(np.round((camp - lo) / scale), 0, 255).astype(np.uint8)
    return q


def empaqueta_run(run_id, base_iso, hores, camps, dirgrid, bbox, nx, ny, password):
    fields = {}
    for nom, cfg in FIELDS.items():
        if nom not in camps:
            continue
        q = quantitza(camps[nom], cfg["lo"], cfg["scale"])          # (ny,nx,nh)
        raw = np.transpose(q, (2, 0, 1)).tobytes()                  # hora-major, nord->sud, oest->est
        fields[nom] = {"lo": cfg["lo"], "scale": cfg["scale"], "unit": cfg["unit"],
                       "data": base64.b64encode(gzip.compress(raw, 6)).decode()}
    # direcció: uint8 = round(deg/2) (0..179), 255 = sense dada
    dq = np.where(np.isnan(dirgrid), 255, np.clip(np.round((dirgrid % 360) / 2.0), 0, 179)).astype(np.uint8)
    rawd = np.transpose(dq, (2, 0, 1)).tobytes()
    fields["dir"] = {"scale": 2.0, "unit": "°", "nodata": 255,
                     "data": base64.b64encode(gzip.compress(rawd, 6)).decode()}
    payload = {"run": run_id, "base": base_iso, "bbox": list(bbox), "nx": nx, "ny": ny,
               "hores": hores, "fields": fields,
               "generat": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return xifrar_text(json.dumps(payload, separators=(",", ":")), password)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="harmonie_store")
    ap.add_argument("--area", default="PB")
    a = ap.parse_args()
    key = os.environ.get("AEMET_API_KEY"); pwd = os.environ.get("MAPA_PASS")
    if not key or not pwd:
        raise SystemExit("calen AEMET_API_KEY i MAPA_PASS")
    os.makedirs(a.store, exist_ok=True)

    print("Baixant HARMONIE-AROME (%s)…" % a.area)
    run_id, base_iso, hores, camps, dirgrid, bbox, nx, ny = decodifica(baixa(a.area, key))
    print("Passada %s · %d hores · retall CAT %dx%d bbox %s" % (run_id, len(hores), nx, ny, tuple(round(x, 3) for x in bbox)))

    enc = empaqueta_run(run_id, base_iso, hores, camps, dirgrid, bbox, nx, ny, pwd)
    fp = os.path.join(a.store, "harmonie_%s.enc" % run_id)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(enc, f)
    print("  escrit %s (%.2f MB)" % (os.path.basename(fp), os.path.getsize(fp) / 1e6))

    # buffer: conserva només les últimes RUNS_BUFFER passades
    runs = sorted(re.findall(r"harmonie_(\d{10})\.enc", " ".join(os.path.basename(p) for p in glob.glob(os.path.join(a.store, "harmonie_*.enc")))))
    for old in runs[:-RUNS_BUFFER]:
        try:
            os.remove(os.path.join(a.store, "harmonie_%s.enc" % old))
        except OSError:
            pass
    runs = sorted(re.findall(r"harmonie_(\d{10})\.enc", " ".join(os.path.basename(p) for p in glob.glob(os.path.join(a.store, "harmonie_*.enc")))))

    with open(os.path.join(a.store, "harmonie.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": runs, "darrera": run_id, "nx": nx, "ny": ny, "bbox": list(bbox),
                   "hores": len(hores), "camps": {k: FIELDS[k]["unit"] for k in FIELDS} | {"dir": "°"},
                   "generat": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                   "font": "AEMET HARMONIE-AROME (descodificació de la graella)"}, f)
    print("Fet: %d passades al buffer (%s)" % (len(runs), ", ".join(runs)))


if __name__ == "__main__":
    main()
