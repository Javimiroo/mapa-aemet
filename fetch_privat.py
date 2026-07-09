# -*- coding: utf-8 -*-
"""
Baixa AEMET + Meteocat (XEMA), fusiona en un sol conjunt (camp 'font'),
calcula màx/mín del dia i XIFRA el resultat amb la contrasenya -> dades_privat.enc

Pensat per a la GitHub Action (cada 6 h) i també per provar-lo en local.
Variables d'entorn (a la Action venen de secrets; en local usa els valors per defecte):
    AEMET_API_KEY   clau d'AEMET
    METEOCAT_KEY    clau de Meteocat
    MAPA_PASS       contrasenya del mapa privat

Meteocat: límit 750 consultes/mes. Es baixen 11 variables (1 consulta cadascuna):
T, HR i el vent (velocitat/direcció/ratxa) a 2 m, 6 m i 10 m, perquè cada estació
mesura el vent a una altura diferent. ~11 consultes per execució -> amb 2
actualitzacions automàtiques/dia són ~660/mes (queda marge per a les manuals).
Les coordenades es guarden en 'meteocat_estacions.json' (només el primer cop).
"""

import base64
import json
import math
import os
import ssl
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# --- claus: SEMPRE per variable d'entorn (a la Action venen de secrets) ---
# Per provar en local, defineix-les abans d'executar, p. ex. (Windows CMD):
#   set AEMET_API_KEY=...   & set METEOCAT_KEY=...   & set MAPA_PASS=graf-prova
def _need(nom):
    v = os.environ.get(nom)
    if not v:
        raise SystemExit("Falta la variable d'entorn %s (secret a GitHub o 'set' en local)." % nom)
    return v

AEMET_API_KEY = _need("AEMET_API_KEY")
METEOCAT_KEY = _need("METEOCAT_KEY")
PASSWORD = _need("MAPA_PASS")

AEMET_BASE = "https://opendata.aemet.es/opendata/api"
MC_BASE = "https://api.meteo.cat"
PROV_CAT = {"BARCELONA", "GIRONA", "LLEIDA", "TARRAGONA"}
ITER = 200000
EST_FILE = "meteocat_estacions.json"

# variables Meteocat: codi -> (camp, factor)  (factor 3.6 = m/s -> km/h)
# El vent es mesura a 2/6/10 m segons l'estació, i cada altura té codis diferents.
# Baixem les TRES altures i les fusionem al mateix camp (vv/dv/vmax): així cada estació
# omple el vent amb l'altura que tinga. L'ordre posa 10 m l'ÚLTIM perquè, si una estació
# tingués més d'una altura al mateix instant, preval el de 10 m (s'escriu després).
#   10 m: 30/31/50 · 6 m: 48/49/53 · 2 m: 46/47/56  (font: metadades XEMA, Meteocat)
MC_VARS = {
    32: ("ta", 1.0), 33: ("hr", 1.0),
    46: ("vv", 3.6), 47: ("dv", 1.0), 56: ("vmax", 3.6),   # vent a 2 m
    48: ("vv", 3.6), 49: ("dv", 1.0), 53: ("vmax", 3.6),   # vent a 6 m
    30: ("vv", 3.6), 31: ("dv", 1.0), 50: ("vmax", 3.6),   # vent a 10 m (preferent)
}

_SSL = ssl.create_default_context()


# ============================ utilitats ============================
def _num(v, factor=1.0):
    try:
        return round(float(v) * factor, 1)
    except (TypeError, ValueError):
        return None


def punt_rosada(ta, hr):
    """Punt de rosada (°C) a partir de temperatura i humitat relativa (Magnus)."""
    try:
        t = float(ta); h = float(hr)
        if h <= 0:
            return None
        a, b = 17.625, 243.04
        g = math.log(h / 100.0) + (a * t) / (b + t)
        return round((b * g) / (a - g), 1)
    except (TypeError, ValueError):
        return None


def _get(url, headers, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            raw = urllib.request.urlopen(req, timeout=60, context=_SSL).read()
            try:
                return json.loads(raw.decode("utf-8"))
            except UnicodeDecodeError:
                return json.loads(raw.decode("latin-1"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(20); continue
            raise
    raise RuntimeError("massa reintents (429): " + url)


# ============================ AEMET ============================
def aemet(endpoint):
    meta = _get(AEMET_BASE + endpoint + "?api_key=" + AEMET_API_KEY,
                {"User-Agent": "graf", "Accept": "application/json"})
    if meta.get("estado") != 200:
        raise RuntimeError("AEMET estado=%s" % meta.get("estado"))
    return _get(meta["datos"], {"User-Agent": "graf"})


def estacions_aemet():
    inv = aemet("/valores/climatologicos/inventarioestaciones/todasestaciones")
    cat = {}
    for e in inv:
        prov = (e.get("provincia") or "").strip().upper()
        if prov in PROV_CAT:
            cat[e["indicativo"]] = prov.capitalize()

    obs = aemet("/observacion/convencional/todas")
    per = {}
    for r in obs:
        if r.get("idema") in cat:
            per.setdefault(r["idema"], []).append(r)

    out = []
    for idema, regs in per.items():
        regs.sort(key=lambda x: x.get("fint") or "")
        hist = []
        for r in regs:
            hist.append({
                "t": r.get("fint"),
                "ta": _num(r.get("ta")), "tamax": _num(r.get("tamax")), "tamin": _num(r.get("tamin")),
                "hr": _num(r.get("hr")), "vv": _num(r.get("vv"), 3.6), "vmax": _num(r.get("vmax"), 3.6),
                "dv": _num(r.get("dv")), "prec": _num(r.get("prec")),
                "pres": _num(r.get("pres_nmar") if r.get("pres_nmar") is not None else r.get("pres")),
                "tpr": _num(r.get("tpr")),
            })
        ult = regs[-1]

        def last(k):
            for r in reversed(regs):
                if r.get(k) is not None:
                    return r.get(k)
            return None

        mx = [h["tamax"] if h["tamax"] is not None else h["ta"] for h in hist if h["tamax"] is not None or h["ta"] is not None]
        mn = [h["tamin"] if h["tamin"] is not None else h["ta"] for h in hist if h["tamin"] is not None or h["ta"] is not None]
        out.append({
            "idema": idema, "nom": (ult.get("ubi") or idema).strip(), "provincia": cat.get(idema, ""),
            "font": "AEMET", "lat": ult.get("lat"), "lon": ult.get("lon"), "alt": ult.get("alt"),
            "actual": {
                "fint": ult.get("fint"), "ta": _num(last("ta")), "tamax": _num(last("tamax")),
                "tamin": _num(last("tamin")), "tamax_dia": max(mx) if mx else None,
                "tamin_dia": min(mn) if mn else None, "n_hores": len(hist), "hr": _num(last("hr")),
                "vv": _num(last("vv"), 3.6), "vmax": _num(last("vmax"), 3.6), "dv": _num(last("dv")),
                "dmax": _num(last("dmax")), "prec": _num(last("prec")),
                "pres": _num(last("pres_nmar") if last("pres_nmar") is not None else last("pres")),
                "tpr": _num(last("tpr")),
            },
            "historic": hist,
        })
    return out


# ============================ Meteocat (XEMA) ============================
def mc_get(path):
    return _get(MC_BASE + path, {"X-Api-Key": METEOCAT_KEY, "User-Agent": "graf"})


def meteocat_metadades():
    """Coordenades de les estacions XEMA. Es cacheja en fitxer (no gasta cada volta)."""
    if os.path.exists(EST_FILE):
        with open(EST_FILE, encoding="utf-8") as f:
            return json.load(f)
    est = mc_get("/xema/v1/estacions/metadades")
    meta = {}
    for e in est:
        c = e.get("coordenades") or {}
        meta[e["codi"]] = {
            "nom": e.get("nom", e["codi"]),
            "lat": c.get("latitud"), "lon": c.get("longitud"), "alt": e.get("altitud"),
            "provincia": ((e.get("provincia") or {}).get("nom") or ""),
        }
    with open(EST_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return meta


def estacions_meteocat():
    meta = meteocat_metadades()
    hui = datetime.now(timezone.utc)

    # acumulem per estació: camp -> llista de (data, valor)
    dat = {}   # codi -> {camp: [(data,valor)]}
    for code, (camp, factor) in MC_VARS.items():
        try:
            resp = mc_get("/xema/v1/variables/mesurades/%d/%04d/%02d/%02d"
                          % (code, hui.year, hui.month, hui.day))
        except Exception as ex:  # noqa
            print("  avis: variable %d no baixada (%s)" % (code, ex))
            continue
        for el in resp:
            st = el.get("codi")
            vs = el.get("variables") or []
            if not vs:
                continue
            for lect in (vs[0].get("lectures") or []):
                v = lect.get("valor")
                if v is None:
                    continue
                dat.setdefault(st, {}).setdefault(camp, []).append((lect.get("data"), _num(v, factor)))

    out = []
    for st, camps in dat.items():
        m = meta.get(st)
        if not m or m.get("lat") is None:
            continue

        # historic per hora (lectures en punt, minut :00), fusionant variables per timestamp
        hores = {}
        for camp, parells in camps.items():
            for data, val in parells:
                if data and data[11:16].endswith(":00"):
                    hores.setdefault(data, {"t": data})[camp] = val
        historic = [hores[k] for k in sorted(hores)]
        for row in historic:                       # punt de rosada calculat (sense consulta extra)
            row["tpr"] = punt_rosada(row.get("ta"), row.get("hr"))

        def latest(camp):
            arr = camps.get(camp)
            if not arr:
                return None
            arr = [p for p in arr if p[1] is not None]
            if not arr:
                return None
            return max(arr, key=lambda p: p[0] or "")[1]

        ta_all = [v for (_, v) in camps.get("ta", []) if v is not None]
        fint = None
        for camp in ("ta", "hr", "vv"):
            arr = camps.get(camp)
            if arr:
                fint = max(fint or "", max(p[0] or "" for p in arr))
        out.append({
            "idema": "MC_" + st, "nom": m["nom"], "provincia": m["provincia"], "font": "Meteocat",
            "lat": m["lat"], "lon": m["lon"], "alt": m["alt"],
            "actual": {
                "fint": fint, "ta": latest("ta"), "tamax": None, "tamin": None,
                "tamax_dia": max(ta_all) if ta_all else None, "tamin_dia": min(ta_all) if ta_all else None,
                "n_hores": len(historic), "hr": latest("hr"), "vv": latest("vv"),
                "vmax": latest("vmax"), "dv": latest("dv"), "dmax": None, "prec": None,
                "pres": None, "tpr": punt_rosada(latest("ta"), latest("hr")),
            },
            "historic": historic,
        })
    return out


# ============================ xifratge ============================
def xifrar(text, password):
    salt = os.urandom(16); iv = os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(password.encode())
    ct = AESGCM(key).encrypt(iv, text.encode("utf-8"), None)
    return {"v": 1, "kdf": "PBKDF2-SHA256", "it": ITER, "alg": "AES-GCM",
            "salt": base64.b64encode(salt).decode(), "iv": base64.b64encode(iv).decode(),
            "ct": base64.b64encode(ct).decode()}


def desxifrar(blob, password):
    salt = base64.b64decode(blob["salt"]); iv = base64.b64decode(blob["iv"]); ct = base64.b64decode(blob["ct"])
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=blob.get("it", ITER)).derive(password.encode())
    return json.loads(AESGCM(key).decrypt(iv, ct, None).decode("utf-8"))


# ============================ acumulació de l'històric ============================
DIES_HISTORIC = 6
OUT_FILE = "dades_privat.enc"


def _parse_t(t):
    if not t:
        return None
    s = t.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if len(s) >= 5 and s[-5] in "+-" and s[-3] != ":":   # +0000 -> +00:00
        s = s[:-2] + ":" + s[-2:]
    try:
        d = datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception:
        return None


def carrega_historic_previ():
    """{idema: [rows historic]} de l'execució anterior (si es pot desxifrar)."""
    if not os.path.exists(OUT_FILE):
        return {}
    try:
        with open(OUT_FILE, encoding="utf-8") as f:
            prev = desxifrar(json.load(f), PASSWORD)
        return {e["idema"]: e.get("historic", []) for e in prev.get("estacions", [])}
    except Exception as ex:  # noqa
        print("  avis: no s'ha pogut llegir l'històric previ (%s)" % ex)
        return {}


def acumula(estacions, previ):
    ara = datetime.now(timezone.utc)
    tall = ara - timedelta(days=DIES_HISTORIC)
    t24 = ara - timedelta(hours=24)
    for e in estacions:
        byt = {}
        for row in previ.get(e["idema"], []) + e.get("historic", []):
            t = row.get("t")
            if t:
                byt[t] = row                       # el nou (afegit després) sobreescriu el vell
        rows = []
        for t in sorted(byt):
            d = _parse_t(t)
            if d and d >= tall:
                rows.append(byt[t])
        e["historic"] = rows
        # màx/mín de les últimes 24 h sobre l'històric acumulat
        mx = [r[k] for r in rows if (_parse_t(r.get("t")) or ara) >= t24
              for k in ("tamax", "ta") if r.get(k) is not None]
        mn = [r[k] for r in rows if (_parse_t(r.get("t")) or ara) >= t24
              for k in ("tamin", "ta") if r.get(k) is not None]
        if mx:
            e["actual"]["tamax_dia"] = round(max(mx), 1)
        if mn:
            e["actual"]["tamin_dia"] = round(min(mn), 1)
        e["actual"]["n_hores"] = len(rows)


# ============================ principal ============================
def main():
    print("Baixant AEMET...")
    a = estacions_aemet()
    print("  AEMET:", len(a), "estacions")
    print("Baixant Meteocat (XEMA)...")
    m = estacions_meteocat()
    print("  Meteocat:", len(m), "estacions")

    estacions = sorted(a + m, key=lambda e: e["nom"])
    print("Acumulant històric (fins a %d dies)..." % DIES_HISTORIC)
    acumula(estacions, carrega_historic_previ())
    nh = [e["actual"]["n_hores"] for e in estacions] or [0]
    print("  hores/estació -> min %d · màx %d · mitjana %d" % (min(nh), max(nh), sum(nh)//len(nh)))
    dades = {
        "generat": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "font": "AEMET (OpenData) + Meteocat (XEMA) - dades © Servei Meteorologic de Catalunya",
        "n_estacions": len(estacions),
        "n_aemet": len(a), "n_meteocat": len(m),
        "estacions": estacions,
    }
    blob = xifrar(json.dumps(dades, ensure_ascii=False, separators=(",", ":")), PASSWORD)
    with open("dades_privat.enc", "w", encoding="utf-8") as f:
        json.dump(blob, f)
    print("OK -> dades_privat.enc  (%d estacions: %d AEMET + %d Meteocat)" % (len(estacions), len(a), len(m)))
    print("Contrasenya usada:", PASSWORD)


if __name__ == "__main__":
    main()
