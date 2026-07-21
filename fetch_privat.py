# -*- coding: utf-8 -*-
"""
Baixa AEMET + Meteocat (XEMA), fusiona en un sol conjunt (camp 'font'),
calcula màx/mín del dia i XIFRA el resultat amb la contrasenya -> dades_privat.enc

Pensat per a la GitHub Action (cada 6 h) i també per provar-lo en local.
Variables d'entorn (a la Action venen de secrets; en local usa els valors per defecte):
    AEMET_API_KEY   clau d'AEMET
    METEOCAT_KEY    clau de Meteocat
    MAPA_PASS       contrasenya del mapa privat

Meteocat: límit 750 consultes/mes. Es baixen 11 variables (T, HR i vent
velocitat/direcció/ratxa a 2/6/10 m). Cost per actualització:
  - normalment 11 consultes (només HUI).
  - 22 consultes NOMÉS el primer cop del dia (baixa també AHIR per tancar la
    frontera de dia UTC; després ja el tenim i no es repeteix).
Així la majoria d'actualitzacions (incloses les manuals) costen la meitat. Amb ~1
actualització automàtica/dia queda marge per a manuals dins de la quota. Les
coordenades es guarden en 'meteocat_estacions.json' (1r cop). IMPORTANT: 750/mes és
molt just; si es puja (Meteocat institucional per a Bombers), es pot pujar la freq.
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
from zoneinfo import ZoneInfo   # per arxivar per DIA LOCAL (com ho veu l'equip)

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
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            raw = urllib.request.urlopen(req, timeout=60, context=_SSL).read()
            try:
                return json.loads(raw.decode("utf-8"))
            except UnicodeDecodeError:
                return json.loads(raw.decode("latin-1"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:
                time.sleep(20); continue          # límit de peticions
            if 500 <= e.code < 600:
                time.sleep(5); continue           # error temporal del servidor (AEMET peta sovint)
            raise                                 # 4xx "de veritat" (401/403/404...): no insistim
        except urllib.error.URLError as e:
            last = e; time.sleep(5); continue     # problema de xarxa/temps d'espera
    raise RuntimeError("massa reintents (%s): %s" % (last, url))


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


def estacions_meteocat(baixa_ahir=True):
    meta = meteocat_metadades()
    ara = datetime.now(timezone.utc)
    # HUI es baixa sempre. AHIR (UTC) només si encara NO en tenim la nit: així es
    # tanca la frontera de dia una sola volta i la resta d'actualitzacions costen
    # la meitat de consultes (només hui). Meteocat serveix per dia UTC i la vesprada
    # local (20-24h) cau al dia UTC anterior; per això cal cobrir-lo el 1r cop.
    dates = [ara - timedelta(days=1), ara] if baixa_ahir else [ara]

    # acumulem per estació: camp -> llista de (data, valor)
    dat = {}   # codi -> {camp: [(data,valor)]}
    for code, (camp, factor) in MC_VARS.items():
        for dref in dates:
            try:
                resp = mc_get("/xema/v1/variables/mesurades/%d/%04d/%02d/%02d"
                              % (code, dref.year, dref.month, dref.day))
            except Exception as ex:  # noqa
                print("  avis: variable %d dia %s no baixada (%s)" % (code, dref.date(), ex))
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


def carrega_estacions_previ():
    """{idema: estació completa} de l'execució anterior. Serveix per REUTILITZAR les
    estacions d'una font (AEMET o Meteocat) si eixa font falla, i que no desapareguen."""
    if not os.path.exists(OUT_FILE):
        return {}
    try:
        with open(OUT_FILE, encoding="utf-8") as f:
            prev = desxifrar(json.load(f), PASSWORD)
        return {e["idema"]: e for e in prev.get("estacions", [])}
    except Exception as ex:  # noqa
        print("  avis: no s'ha pogut llegir l'anterior complet (%s)" % ex)
        return {}


def te_ahir_complet(previ):
    """True si l'històric previ ja té la NIT d'ahir (UTC) de Meteocat coberta (hi ha
    alguna lectura d'ahir a les 23 h UTC o més). Si és així, NO cal tornar a baixar
    ahir: així la majoria d'actualitzacions només baixen hui (la meitat de consultes)."""
    ahir = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    for idema, rows in previ.items():
        if not str(idema).startswith("MC_"):        # només mirem estacions Meteocat
            continue
        for r in rows:
            d = _parse_t(r.get("t"))
            if d and d.date() == ahir and d.hour >= 23:
                return True
    return False


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


# ============================ arxiu històric (per dia, congelat) ============================
TZ_LOCAL = ZoneInfo("Europe/Madrid")
ARXIU_DIR = "arxiu"


def _dia_local(t):
    """Data LOCAL (Europe/Madrid) d'un timestamp, com AAAA-MM-DD (o None)."""
    d = _parse_t(t)
    return d.astimezone(TZ_LOCAL).date().isoformat() if d else None


def arxiva(estacions):
    """Escriu un fitxer xifrat per cada DIA LOCAL ja tancat (>=2 dies) que encara no
    estiga arxivat. Cada dia es guarda UNA sola vegada (es congela) i s'actualitza
    l'índex. Així conservem tot l'històric sense carregar-ho a l'operativa diària."""
    os.makedirs(ARXIU_DIR, exist_ok=True)
    avui = datetime.now(timezone.utc).astimezone(TZ_LOCAL).date()
    # Arxivem dies < HUI (ahir i anteriors). Ahir ja és complet perquè cada execució
    # baixa ahir+hui de Meteocat, així que es pot congelar sense esperar 2 dies.
    limit = avui

    dies = set()
    for e in estacions:
        for r in e.get("historic", []):
            dk = _dia_local(r.get("t"))
            if dk:
                dies.add(dk)

    nous = 0
    for dk in sorted(dies):
        try:
            d_date = datetime.fromisoformat(dk).date()
        except ValueError:
            continue
        if d_date >= limit:               # massa recent (encara operatiu / pot canviar)
            continue
        path = os.path.join(ARXIU_DIR, dk + ".enc")
        if os.path.exists(path):          # ja arxivat -> es congela, no es reescriu
            continue
        ests = []
        for e in estacions:
            rows = [r for r in e.get("historic", []) if _dia_local(r.get("t")) == dk]
            if not rows:
                continue
            ests.append({
                "idema": e["idema"], "nom": e.get("nom"), "provincia": e.get("provincia"),
                "font": e.get("font"), "lat": e.get("lat"), "lon": e.get("lon"), "alt": e.get("alt"),
                "historic": rows,
            })
        if not ests:
            continue
        dia_obj = {
            "dia": dk,
            "generat": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "n_estacions": len(ests), "estacions": ests,
        }
        blob = xifrar(json.dumps(dia_obj, ensure_ascii=False, separators=(",", ":")), PASSWORD)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f)
        nous += 1
        print("  arxivat %s (%d estacions)" % (dk, len(ests)))

    # índex (llista de dies disponibles) -> pla, per poblar el calendari
    disponibles = sorted(fn[:-4] for fn in os.listdir(ARXIU_DIR) if fn.endswith(".enc"))
    with open(os.path.join(ARXIU_DIR, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"dies": disponibles,
                   "actualitzat": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")},
                  f, ensure_ascii=False)
    print("  arxiu: %d dies nous · %d dies en total" % (nous, len(disponibles)))


# ============================ principal ============================
def main():
    # Carreguem l'anterior una sola volta (per reutilitzar dades si una font falla).
    prev_full = carrega_estacions_previ()
    previ = {k: v.get("historic", []) for k, v in prev_full.items()}

    aemet_prev = [v for v in prev_full.values() if v.get("font") == "AEMET"]
    mc_prev    = [v for v in prev_full.values() if v.get("font") == "Meteocat"]

    print("Baixant AEMET...")
    try:
        a = estacions_aemet()
    except Exception as ex:  # AEMET és inestable: si falla, no estavellem tot
        a = None
        print("  AVÍS: AEMET ha fallat (%s)." % str(ex)[:90])
    if not a:                # excepció o resposta buida -> mantenim les d'abans
        a = aemet_prev
        print("  AEMET: %d estacions%s" % (len(a), "  (anteriors, no actualitzat)" if aemet_prev else ""))
    else:
        print("  AEMET: %d estacions" % len(a))

    baixa_ahir = not te_ahir_complet(previ)
    print("Baixant Meteocat (XEMA) [%s]..." % ("ahir+hui" if baixa_ahir else "només hui"))
    try:
        m = estacions_meteocat(baixa_ahir=baixa_ahir)
    except Exception as ex:  # el mateix per a Meteocat
        m = None
        print("  AVÍS: Meteocat ha fallat (%s)." % str(ex)[:90])
    if not m:
        m = mc_prev
        print("  Meteocat: %d estacions%s" % (len(m), "  (anteriors, no actualitzat)" if mc_prev else ""))
    else:
        print("  Meteocat: %d estacions" % len(m))

    if not a and not m:
        raise SystemExit("Cap font ha respost i no hi ha dades prèvies; no s'escriu res.")

    estacions = sorted(a + m, key=lambda e: e["nom"])
    print("Acumulant històric (fins a %d dies)..." % DIES_HISTORIC)
    acumula(estacions, previ)
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

    # --- camp de vents diagnòstic (capa privada) ---
    try:
        from camp_vents import escriu_vent
        nv = escriu_vent(estacions, PASSWORD)
        print("OK -> vent_privat.enc  (camp de vents, %d estacions amb vent)" % nv)
    except Exception as ex:  # mai ha de bloquejar l'actualització operativa
        print("  AVIS: camp de vents no generat (%s)" % str(ex)[:120])

    print("Arxivant històric per dies...")
    try:
        arxiva(estacions)
    except Exception as ex:  # l'arxiu no ha de bloquejar mai l'actualització operativa
        print("  avis: arxiu no completat (%s)" % ex)

    # --- arxiu HORARI del camp de vents dels dies ja tancats ---
    try:
        from arxiu_vent import backfill
        backfill(PASSWORD, max_dies=3)
    except Exception as ex:
        print("  avis: arxiu de vent no completat (%s)" % str(ex)[:100])


if __name__ == "__main__":
    main()
