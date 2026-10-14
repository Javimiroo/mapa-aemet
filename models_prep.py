# -*- coding: utf-8 -*-
"""
MODELS NUMÈRICS addicionals (ICON-EU del DWD i AROME de Météo-France) -> MATEIX format que HARMONIE
(projecte GRAF). El frontend (privat.html) reutilitza tot (superfície, fletxes, acumulacions, explorador
del punt, tauler d'incendi) només canviant el model.

Sortida (a --store, branca 'models'):
  <model>_<run>.enc   payload xifrat (AES-GCM) IDÈNTIC al de harmonie_prep.py:
                      {run, base, bbox, nx, ny, hores[iso], fields{nom:{lo,scale,unit,data(gzip+b64 uint8)}}, model, font}
                      Camps: temp °C, vent km/h, ratxa km/h, precip mm/1h, nuvol %, hr % (humitat relativa, nova) i
                      dir (uint8 deg/2, 255 = sense dada) en la MATEIXA convenció que AEMET: sentit CAP ON BUFA
                      (el frontend hi suma 180 per tenir la direcció meteorològica "d'on ve").
  models.json         índex: {"models": {"icon": {runs, darrera, nx, ny, bbox, hores, camps, font, generat}, "arome": {...}}}
                      (s'actualitza NOMÉS l'entrada del model processat; la resta es conserva).

Models:
  icon   ICON-EU (DWD open data, sense clau). GRIB2 regular lat-lon 0,0625° (~7 km), passades 00/06/12/18 UTC,
         horari fins a +78 h (després cada 3 h; no els usem). https://opendata.dwd.de/weather/nwp/icon-eu/grib/<HH>/<var>/
         Variables: t_2m, relhum_2m, u_10m, v_10m, vmax_10m (ratxa màx de l'última hora), tot_prec (acumulat des de
         l'inici -> es diferencia a horari), clct.
  arome  AROME 0,025° (Météo-France, portail-api.meteofrance.fr, subscripció gratuïta a l'API AROME). Servei WCS:
         GetCapabilities (catàleg de cobertures i passades) + un GetCoverage (GRIB) per variable i hora amb subset lat/lon.
         Autenticació OAuth2 client_credentials (METEOFRANCE_CLIENT_ID + METEOFRANCE_CLIENT_SECRET) o clau
         METEOFRANCE_API_KEY. Pla gratuït ~50 peticions/min -> pausa entre peticions.
         PENDENT DE VALIDAR en viu amb credencials: els noms de cobertura es descobreixen al catàleg amb patrons
         (veure AROME_COV) i es registren al log; si algun no casa, ajustar AROME_COV.

Ús (workflow models.yml):
    MAPA_PASS=... python models_prep.py --model icon  --store models_store
    MAPA_PASS=... METEOFRANCE_CLIENT_ID=... METEOFRANCE_CLIENT_SECRET=... python models_prep.py --model arome --store models_store
Requisits: numpy, cryptography, eccodes (pip, porta la llibreria), requests (AROME).
"""
import argparse, base64, bz2, glob, gzip, json, os, re, sys, time, urllib.request
from datetime import datetime, timezone, timedelta

import numpy as np

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ITER = 200000
RUNS_BUFFER = 4
CAT_BBOX = (0.05, 3.45, 40.45, 43.05)        # lon0, lon1, lat0, lat1 (retall de Catalunya, igual que HARMONIE)
UA = "graf-models (Bombers GRAF)"

# camp lògic -> unitat i quantització uint8 (MATEIXOS lo/scale que harmonie_prep.py + 'hr')
FIELDS = {
    "temp":   {"unit": "°C",   "lo": -30.0, "scale": 0.35},
    "vent":   {"unit": "km/h", "lo": 0.0,   "scale": 0.6},
    "ratxa":  {"unit": "km/h", "lo": 0.0,   "scale": 0.7},
    "precip": {"unit": "mm",   "lo": 0.0,   "scale": 0.4},
    "nuvol":  {"unit": "%",    "lo": 0.0,   "scale": 0.5},
    "hr":     {"unit": "%",    "lo": 0.0,   "scale": 0.4},
}


# ------------------------------------------------------------------ xifratge (idèntic a la resta del sistema)
def xifrar_text(text, password):
    salt = os.urandom(16); iv = os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(password.encode())
    ct = AESGCM(key).encrypt(iv, text.encode("utf-8"), None)
    return {"v": 1, "kdf": "PBKDF2-SHA256", "it": ITER, "alg": "AES-GCM",
            "salt": base64.b64encode(salt).decode(), "iv": base64.b64encode(iv).decode(),
            "ct": base64.b64encode(ct).decode()}


# ------------------------------------------------------------------ GRIB -> graella nord-primer + retall CAT
def grib_a_matriu(msg_bytes):
    """Un missatge GRIB (bytes) -> (vals (nj,ni) NORD->SUD, lat_nord, lon_oest, dlat, dlon)."""
    import eccodes as ec
    h = ec.codes_new_from_message(msg_bytes)
    try:
        ni, nj = ec.codes_get(h, "Ni"), ec.codes_get(h, "Nj")
        v = np.array(ec.codes_get_values(h), dtype=np.float32).reshape(nj, ni)
        la1 = float(ec.codes_get(h, "latitudeOfFirstGridPointInDegrees"))
        la2 = float(ec.codes_get(h, "latitudeOfLastGridPointInDegrees"))
        lo1 = float(ec.codes_get(h, "longitudeOfFirstGridPointInDegrees"))
        dlat = abs(float(ec.codes_get(h, "jDirectionIncrementInDegrees")))
        dlon = abs(float(ec.codes_get(h, "iDirectionIncrementInDegrees")))
        if int(ec.codes_get(h, "jScansPositively")):          # sud->nord: girem perquè la fila 0 siga el nord
            v = v[::-1]
            la1 = max(la1, la2)
        if lo1 > 180:
            lo1 -= 360
        try:
            miss = float(ec.codes_get(h, "missingValue"))
            v = np.where(v == miss, np.nan, v)
        except Exception:
            pass
    finally:
        ec.codes_release(h)
    return v, la1, lo1, dlat, dlon


def retalla_cat(v, lat_n, lon_w, dlat, dlon):
    """Retalla la matriu (nord-primer) a CAT_BBOX. Torna (sub, bbox(lon0,lon1,lat0,lat1))."""
    lon0, lon1, lat0, lat1 = CAT_BBOX
    nj, ni = v.shape
    c0 = max(0, int(np.floor((lon0 - lon_w) / dlon))); c1 = min(ni, int(np.ceil((lon1 - lon_w) / dlon)) + 1)
    r0 = max(0, int(np.floor((lat_n - lat1) / dlat))); r1 = min(nj, int(np.ceil((lat_n - lat0) / dlat)) + 1)
    sub = v[r0:r1, c0:c1]
    # el centre de cel·la (r,c) és (lat_n - r*dlat, lon_w + c*dlon): la caixa va de mitja cel·la a mitja cel·la
    bbox = (lon_w + c0 * dlon - dlon / 2, lon_w + (c1 - 1) * dlon + dlon / 2,
            lat_n - (r1 - 1) * dlat - dlat / 2, lat_n - r0 * dlat + dlat / 2)
    return sub, bbox


def uv_a_vent(u, v):
    """u,v m/s -> (velocitat km/h, heading 'cap on bufa' en graus, com l'ang_viento d'AEMET)."""
    spd = np.hypot(u, v) * 3.6
    dir_from = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0        # direcció meteorològica (d'on ve)
    heading = (dir_from + 180.0) % 360.0                                # convenció del payload (= AEMET)
    return spd, heading


# ------------------------------------------------------------------ ICON-EU (DWD)
ICON_BASE = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/%02d/%s/"
ICON_FILE = "icon-eu_europe_regular-lat-lon_single-level_%s_%03d_%s.grib2.bz2"
ICON_VARS = ["t_2m", "relhum_2m", "u_10m", "v_10m", "vmax_10m", "tot_prec", "clct"]
ICON_HMAX = 78                                   # últim pas horari (després va cada 3 h)


def _get(url, timeout=120, intents=4):
    for i in range(intents):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as ex:  # noqa
            if "404" in str(ex):
                return None
            time.sleep(4 * (i + 1))
    return None


def icon_passades(hmax):
    """Passades (run_id AAAAMMDDHH) que tenen el pas hmax de t_2m publicat, de la més recent a la més antiga."""
    runs = []
    for hh in (0, 6, 12, 18):
        html = _get(ICON_BASE % (hh, "t_2m"), timeout=60)
        if not html:
            continue
        for rid in set(re.findall(rb"single-level_(\d{10})_%03d_T_2M" % hmax, html)):
            runs.append(rid.decode())
    return sorted(set(runs), reverse=True)


def icon_decodifica(run_id, hmax, ja_tenim):
    """Baixa i descodifica una passada ICON-EU. Torna (hores[iso], camps{nom:(ny,nx,nh)}, dirgrid, bbox, nx, ny)."""
    hh = int(run_id[8:10]); base = datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    passos = list(range(0, hmax + 1))             # el pas 0 només cal per a tot_prec (base de l'acumulat)
    raw = {}                                       # var -> {pas: (sub, bbox)}
    geo = None
    for var in ICON_VARS:
        raw[var] = {}
        for p in passos:
            if p == 0 and var != "tot_prec":
                continue
            url = ICON_BASE % (hh, var) + ICON_FILE % (run_id, p, var.upper())
            b = _get(url)
            if b is None:
                print("  avis: falta %s pas %03d" % (var, p)); continue
            try:
                v, la, lo, dla, dlo = grib_a_matriu(bz2.decompress(b))
            except Exception as ex:  # noqa
                print("  avis: no es pot descodificar %s pas %03d (%s)" % (var, p, str(ex)[:60])); continue
            sub, bbox = retalla_cat(v, la, lo, dla, dlo)
            if geo is None:
                geo = bbox
            raw[var][p] = sub
        print("  %-9s %d passos" % (var, len(raw[var])))
    if not raw["t_2m"]:
        raise SystemExit("ICON: cap pas de t_2m")
    ny, nx = next(iter(raw["t_2m"].values())).shape
    hores_idx = [p for p in range(1, hmax + 1) if p in raw["t_2m"]]
    nh = len(hores_idx)
    hores = [(base + timedelta(hours=p)).isoformat() for p in hores_idx]

    def pila(var, conv=lambda x: x):
        out = np.full((ny, nx, nh), np.nan, np.float32)
        for k, p in enumerate(hores_idx):
            if p in raw[var]:
                out[:, :, k] = conv(raw[var][p])
        return out

    camps = {}
    camps["temp"] = pila("t_2m", lambda x: x - 273.15)
    camps["hr"] = pila("relhum_2m")
    camps["nuvol"] = pila("clct")
    camps["ratxa"] = pila("vmax_10m", lambda x: x * 3.6)
    U = pila("u_10m"); V = pila("v_10m")
    spd, head = uv_a_vent(U, V)
    camps["vent"] = spd
    dirgrid = np.where(np.isnan(spd), np.nan, head).astype(np.float32)
    # tot_prec = acumulat des de l'inici -> horari per diferència (pas p menys pas p-1)
    pr = np.zeros((ny, nx, nh), np.float32)
    for k, p in enumerate(hores_idx):
        a = raw["tot_prec"].get(p); b = raw["tot_prec"].get(p - 1)
        if a is not None and b is not None:
            pr[:, :, k] = np.clip(a - b, 0, None)
        elif a is not None and p == 1:
            pr[:, :, k] = np.clip(a, 0, None)
    camps["precip"] = pr
    for nom in camps:
        camps[nom] = np.nan_to_num(camps[nom], nan=0.0)
    return hores, camps, dirgrid, geo, nx, ny


# ------------------------------------------------------------------ AROME (Météo-France, WCS)
AROME_BASE = os.environ.get("AROME_BASE", "https://public-api.meteofrance.fr/public/arome/1.0/wcs/MF-NWP-HIGHRES-AROME-0025-FRANCE-WCS")
AROME_TOKEN_URL = os.environ.get("AROME_TOKEN_URL", "https://portail-api.meteofrance.fr/token")
AROME_PAUSA = float(os.environ.get("AROME_PAUSA", "1.3"))
AROME_HMAX = 48
# camp -> (patró del nom de cobertura al GetCapabilities, sufix de període, subset d'alçada, conversió)
# Els noms exactes es confirmen al catàleg; el patró casa l'inici del coverageId (abans de la data de passada).
AROME_COV = {
    "precip": {"pat": r"TOTAL_PRECIPITATION__GROUND_OR_WATER_SURFACE___",             "per": "_PT1H", "height": None,  "conv": lambda x: x},
    "temp":   {"pat": r"TEMPERATURE__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND___",           "per": "",      "height": 2,     "conv": lambda x: x - 273.15},
    "hr":     {"pat": r"RELATIVE_HUMIDITY__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND___",     "per": "",      "height": 2,     "conv": lambda x: x},
    "u":      {"pat": r"U_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND___",   "per": "",      "height": 10,    "conv": lambda x: x},
    "v":      {"pat": r"V_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND___",   "per": "",      "height": 10,    "conv": lambda x: x},
    "ratxa":  {"pat": r"WIND_SPEED_GUST__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND___",       "per": "_PT1H", "height": 10,    "conv": lambda x: x * 3.6},
    "nuvol":  {"pat": r"TOTAL_CLOUD_COVER__GROUND_OR_WATER_SURFACE___",                "per": "",      "height": None,  "conv": lambda x: x},
}
AROME_DOMINI = (38.0, -3.0, 45.0, 6.0)          # S, W, N, E del subset demanat (Catalunya amb marge)


def arome_sessio():
    import requests
    s = requests.Session()
    cid, csec = os.environ.get("METEOFRANCE_CLIENT_ID", "").strip(), os.environ.get("METEOFRANCE_CLIENT_SECRET", "").strip()
    if cid and csec:
        r = s.post(AROME_TOKEN_URL, data={"grant_type": "client_credentials"}, auth=(cid, csec), timeout=30)
        if r.status_code != 200:
            raise SystemExit("AROME: no s'ha pogut obtenir el token OAuth2 (HTTP %d). Revisa METEOFRANCE_CLIENT_ID/SECRET i la subscripció a l'API AROME." % r.status_code)
        hdr = {"Authorization": "Bearer " + r.json()["access_token"]}
    else:
        k = os.environ.get("METEOFRANCE_API_KEY", "").strip()
        if not k:
            raise SystemExit("AROME: calen METEOFRANCE_CLIENT_ID + METEOFRANCE_CLIENT_SECRET (OAuth2) o METEOFRANCE_API_KEY")
        hdr = {"apikey": k}
    hdr["User-Agent"] = UA
    return s, hdr


def arome_get(s, url, params, hdr, intents=6, timeout=90):
    import requests
    for i in range(intents):
        try:
            r = s.get(url, params=params, headers=hdr, timeout=timeout)
        except requests.RequestException:
            time.sleep(5 * (i + 1)); continue
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        if r.status_code in (401, 403):
            raise SystemExit("AROME: credencials rebutjades (HTTP %d)." % r.status_code)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(10 * (i + 1)); continue
        print("  AROME HTTP %d: %s" % (r.status_code, r.text[:160])); return None
    return None


def arome_cataleg(s, hdr):
    """Llig el GetCapabilities i torna {camp: {run_txt: coverageId}} + llista de passades (recent primer)."""
    r = arome_get(s, AROME_BASE + "/GetCapabilities", {"service": "WCS", "version": "2.0.1", "language": "eng"}, hdr, timeout=180)
    if r is None:
        raise SystemExit("AROME: no es pot llegir el catàleg (GetCapabilities)")
    ids = set(re.findall(r"<wcs:CoverageId>([^<]+)</wcs:CoverageId>", r.text)) or set(re.findall(r"<CoverageId>([^<]+)</CoverageId>", r.text))
    cat = {}
    for camp, cfg in AROME_COV.items():
        rx = re.compile("^" + cfg["pat"] + r"(\d{4}-\d{2}-\d{2}T\d{2}\.\d{2}\.\d{2}Z)" + re.escape(cfg["per"]) + "$")
        cat[camp] = {}
        for cid in ids:
            m = rx.match(cid)
            if m:
                cat[camp][m.group(1)] = cid
        print("  catàleg %-7s %d passades%s" % (camp, len(cat[camp]), "" if cat[camp] else "  <-- cap cobertura casa el patró; revisa AROME_COV"))
    runs = sorted(set(cat["precip"].keys()) | set(cat["temp"].keys()), reverse=True)
    return cat, runs


def arome_decodifica(s, hdr, cat, run_txt, hmax):
    run = datetime.strptime(run_txt, "%Y-%m-%dT%H.%M.%SZ").replace(tzinfo=timezone.utc)
    S_, W_, N_, E_ = AROME_DOMINI
    raw = {c: {} for c in AROME_COV}
    geo = None
    for camp, cfg in AROME_COV.items():
        cid = cat[camp].get(run_txt)
        if not cid:
            print("  avis: %s sense cobertura per a la passada %s" % (camp, run_txt)); continue
        for h in range(1, hmax + 1):
            t = (run + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
            params = [("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"), ("format", "application/wmo-grib"),
                      ("coverageId", cid), ("subset", "time(%s)" % t), ("subset", "lat(%g,%g)" % (S_, N_)), ("subset", "long(%g,%g)" % (W_, E_))]
            if cfg["height"] is not None:
                params.append(("subset", "height(%d)" % cfg["height"]))
            r = arome_get(s, AROME_BASE + "/GetCoverage", params, hdr)
            time.sleep(AROME_PAUSA)
            if r is None or not r.content.startswith(b"GRIB"):
                break
            try:
                v, la, lo, dla, dlo = grib_a_matriu(r.content)
            except Exception as ex:  # noqa
                print("  avis: GRIB il·legible %s h+%d (%s)" % (camp, h, str(ex)[:60])); break
            sub, bbox = retalla_cat(v, la, lo, dla, dlo)
            if geo is None:
                geo = bbox
            raw[camp][h] = cfg["conv"](sub)
        print("  %-7s %d hores" % (camp, len(raw[camp])))
    if len(raw["temp"]) < 12:
        return None
    ny, nx = next(iter(raw["temp"].values())).shape
    hores_idx = sorted(raw["temp"].keys())
    nh = len(hores_idx)
    hores = [(run + timedelta(hours=h)).isoformat() for h in hores_idx]

    def pila(camp):
        out = np.full((ny, nx, nh), np.nan, np.float32)
        for k, h in enumerate(hores_idx):
            if h in raw[camp]:
                out[:, :, k] = raw[camp][h]
        return out

    camps = {"temp": pila("temp"), "hr": pila("hr"), "nuvol": pila("nuvol"), "ratxa": pila("ratxa"), "precip": pila("precip")}
    U, V = pila("u"), pila("v")
    spd, head = uv_a_vent(U, V)
    camps["vent"] = spd
    dirgrid = np.where(np.isnan(spd), np.nan, head).astype(np.float32)
    for nom in camps:
        camps[nom] = np.nan_to_num(camps[nom], nan=0.0)
    return run.strftime("%Y%m%d%H"), hores, camps, dirgrid, geo, nx, ny


# ------------------------------------------------------------------ empaquetat (idèntic a harmonie_prep.py)
def quantitza(camp, lo, scale):
    return np.clip(np.round((camp - lo) / scale), 0, 255).astype(np.uint8)


def empaqueta_run(model, font, run_id, hores, camps, dirgrid, bbox, nx, ny, password):
    fields = {}
    for nom, cfg in FIELDS.items():
        if nom not in camps:
            continue
        q = quantitza(camps[nom], cfg["lo"], cfg["scale"])
        raw = np.transpose(q, (2, 0, 1)).tobytes()                  # hora-major, nord->sud, oest->est
        fields[nom] = {"lo": cfg["lo"], "scale": cfg["scale"], "unit": cfg["unit"],
                       "data": base64.b64encode(gzip.compress(raw, 6)).decode()}
    dq = np.where(np.isnan(dirgrid), 255, np.clip(np.round((dirgrid % 360) / 2.0), 0, 179)).astype(np.uint8)
    rawd = np.transpose(dq, (2, 0, 1)).tobytes()
    fields["dir"] = {"scale": 2.0, "unit": "°", "nodata": 255, "data": base64.b64encode(gzip.compress(rawd, 6)).decode()}
    payload = {"run": run_id, "base": hores[0], "bbox": list(bbox), "nx": nx, "ny": ny, "hores": hores, "fields": fields,
               "model": model, "font": font, "dir_conv": "heading (cap on bufa, com AEMET)",
               "generat": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return xifrar_text(json.dumps(payload, separators=(",", ":")), password)


def escriu_index(store, model, info):
    fp = os.path.join(store, "models.json")
    try:
        idx = json.load(open(fp, encoding="utf-8"))
    except Exception:
        idx = {"models": {}}
    idx.setdefault("models", {})[model] = info
    idx["generat"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(idx, f)


def runs_al_buffer(store, model):
    runs = sorted(re.findall(r"%s_(\d{10})\.enc" % model, " ".join(os.path.basename(p) for p in glob.glob(os.path.join(store, "%s_*.enc" % model)))))
    for old in runs[:-RUNS_BUFFER]:
        try:
            os.remove(os.path.join(store, "%s_%s.enc" % (model, old)))
        except OSError:
            pass
    return sorted(re.findall(r"%s_(\d{10})\.enc" % model, " ".join(os.path.basename(p) for p in glob.glob(os.path.join(store, "%s_*.enc" % model)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["icon", "arome"], required=True)
    ap.add_argument("--store", default="models_store")
    ap.add_argument("--run", help="força una passada (AAAAMMDDHH); per defecte l'última completa")
    ap.add_argument("--hmax", type=int, default=None, help="últim pas horari (ICON 78, AROME 48)")
    ap.add_argument("--forca", action="store_true", help="torna a processar encara que la passada ja estiga al buffer")
    a = ap.parse_args()
    pwd = os.environ.get("MAPA_PASS")
    if not pwd:
        raise SystemExit("cal MAPA_PASS")
    os.makedirs(a.store, exist_ok=True)
    ja = set(runs_al_buffer(a.store, a.model))

    if a.model == "icon":
        hmax = a.hmax or ICON_HMAX
        font = "DWD ICON-EU (open data, 0,0625°)"
        runs = [a.run] if a.run else icon_passades(hmax)
        if not runs:
            raise SystemExit("ICON: cap passada completa publicada")
        run_id = runs[0]
        if run_id in ja and not a.forca:
            print("ICON: la passada %s ja és al buffer; res a fer" % run_id); return
        print("ICON-EU: baixant passada %s (fins a +%d h)…" % (run_id, hmax))
        hores, camps, dirgrid, bbox, nx, ny = icon_decodifica(run_id, hmax, ja)
    else:
        hmax = a.hmax or AROME_HMAX
        font = "Météo-France AROME (0,025°, WCS)"
        s, hdr = arome_sessio()
        cat, runs = arome_cataleg(s, hdr)
        if a.run:
            runs = [datetime.strptime(a.run, "%Y%m%d%H").strftime("%Y-%m-%dT%H.00.00Z")]
        res = None
        for run_txt in runs[:4]:
            rid = datetime.strptime(run_txt, "%Y-%m-%dT%H.%M.%SZ").strftime("%Y%m%d%H")
            if rid in ja and not a.forca:
                print("AROME: la passada %s ja és al buffer; res a fer" % rid); return
            print("AROME: baixant passada %s…" % rid)
            res = arome_decodifica(s, hdr, cat, run_txt, hmax)
            if res:
                break
            print("  passada incompleta; provo l'anterior")
        if not res:
            raise SystemExit("AROME: cap passada completa")
        run_id, hores, camps, dirgrid, bbox, nx, ny = res

    print("Passada %s · %d hores · retall CAT %dx%d bbox %s" % (run_id, len(hores), nx, ny, tuple(round(x, 3) for x in bbox)))
    for nom in camps:
        c = camps[nom]; print("  %-7s min %.1f màx %.1f mitj %.1f" % (nom, float(c.min()), float(c.max()), float(c.mean())))
    enc = empaqueta_run(a.model, font, run_id, hores, camps, dirgrid, bbox, nx, ny, pwd)
    fp = os.path.join(a.store, "%s_%s.enc" % (a.model, run_id))
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(enc, f)
    print("  escrit %s (%.2f MB)" % (os.path.basename(fp), os.path.getsize(fp) / 1e6))
    runs = runs_al_buffer(a.store, a.model)
    escriu_index(a.store, a.model, {"runs": runs, "darrera": run_id, "nx": nx, "ny": ny, "bbox": list(bbox), "hores": len(hores),
                                     "camps": {k: FIELDS[k]["unit"] for k in camps} | {"dir": "°"}, "font": font,
                                     "generat": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")})
    print("Fet: %d passades de %s al buffer (%s)" % (len(runs), a.model, ", ".join(runs)))


if __name__ == "__main__":
    main()
