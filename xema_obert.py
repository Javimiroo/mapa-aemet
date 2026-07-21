# -*- coding: utf-8 -*-
"""
XEMA per DADES OBERTES (portal analisi.transparenciacatalunya.cat, Socrata).

Mateixes dades que l'API de Meteocat (mateixos codis d'estació i de variable,
hores en UTC, base semi-horària) però SENSE QUOTA mensual.

Documentació del camp de temps (oficial): "Data i Hora Inicial de la lectura
(els registres es troben etiquetats per davant). L'hora es facilita en Temps
Universal (T.U.)"  -> UTC, igual que l'API.

Retorna la MATEIXA estructura que el bloc de descàrrega de estacions_meteocat():
    {codi_estacio: {camp: [(data_iso, valor_convertit), ...], ...}, ...}
amb data_iso en format 'AAAA-MM-DDTHH:MMZ' (idèntic al de l'API).
"""
import json, ssl, urllib.request, urllib.parse
from datetime import timedelta

BASE = "https://analisi.transparenciacatalunya.cat/resource"
DS_LECTURES = "nzvn-apee"      # dades mesurades de la XEMA
DS_ESTACIONS = "yqwd-vj5e"     # metadades d'estacions
LIMIT = 50000                  # màxim per petició de Socrata
APP_TOKEN = None               # opcional: puja els límits de ritme (gratuït)
_CTX = ssl.create_default_context()


def _soql(dataset, params, timeout=90):
    url = BASE + "/" + dataset + ".json?" + urllib.parse.urlencode(params)
    hdr = {"User-Agent": "graf", "Accept": "application/json"}
    if APP_TOKEN:
        hdr["X-App-Token"] = APP_TOKEN
    req = urllib.request.Request(url, headers=hdr)
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _dia(d):
    return d.date() if hasattr(d, "date") else d


def descarrega(dates, mc_vars, num, verbose=True):
    """Baixa les variables de mc_vars per als dies indicats.
    mc_vars: {codi_variable: (camp, factor)}   ·   num: funció _num(valor, factor)
    """
    codes = list(mc_vars.keys())
    inlist = "(" + ",".join("'%d'" % c for c in codes) + ")"
    # tmp[estacio][camp][codi_variable] = [(t, v)]  -> per respectar l'ordre de mc_vars
    tmp = {}
    nfiles = 0
    for dref in dates:
        d0 = _dia(dref); d1 = d0 + timedelta(days=1)
        where = ("codi_variable IN %s AND data_lectura >= '%sT00:00:00' "
                 "AND data_lectura < '%sT00:00:00'" % (inlist, d0, d1))
        offset = 0
        while True:
            rows = _soql(DS_LECTURES, {
                "$select": "codi_estacio,codi_variable,data_lectura,valor_lectura",
                "$where": where, "$order": "data_lectura",
                "$limit": LIMIT, "$offset": offset})
            if not rows:
                break
            for r in rows:
                v = r.get("valor_lectura")
                t = r.get("data_lectura")
                st = r.get("codi_estacio")
                cv = r.get("codi_variable")
                if v is None or not t or not st or cv is None:
                    continue
                try:
                    code = int(cv)
                except (TypeError, ValueError):
                    continue
                cf = mc_vars.get(code)
                if not cf:
                    continue
                camp, factor = cf
                # '2026-07-21T12:00:00.000' -> '2026-07-21T12:00Z' (com l'API)
                tiso = t[:16] + "Z"
                tmp.setdefault(st, {}).setdefault(camp, {}).setdefault(code, []).append((tiso, num(v, factor)))
                nfiles += 1
            if len(rows) < LIMIT:
                break
            offset += LIMIT
    # aplanem respectant l'ordre de mc_vars (2 m, 6 m, 10 m) com feia l'API
    dat = {}
    for st, camps in tmp.items():
        for camp, percodi in camps.items():
            acc = []
            for code in codes:
                if code in percodi:
                    acc.extend(percodi[code])
            dat.setdefault(st, {})[camp] = acc
    if verbose:
        print("  Dades Obertes: %d files, %d estacions, dies %s"
              % (nfiles, len(dat), ", ".join(str(_dia(d)) for d in dates)))
    return dat


def metadades_obertes():
    """Metadades d'estacions (reserva, per no dependre gens de l'API)."""
    rows = _soql(DS_ESTACIONS, {"$limit": 5000})
    meta = {}
    for e in rows:
        codi = e.get("codi_estacio") or e.get("codi")
        if not codi:
            continue
        def _f(*noms):
            for n in noms:
                if e.get(n) not in (None, ""):
                    try:
                        return float(e[n])
                    except (TypeError, ValueError):
                        return None
            return None
        meta[codi] = {
            "nom": e.get("nom_estacio") or e.get("nom") or codi,
            "lat": _f("latitud"), "lon": _f("longitud"), "alt": _f("altitud"),
            "provincia": e.get("nom_provincia") or e.get("provincia") or "",
        }
    return meta


if __name__ == "__main__":
    from datetime import datetime, timezone
    MC = {32: ("ta", 1.0), 33: ("hr", 1.0), 30: ("vv", 3.6), 31: ("dv", 1.0), 50: ("vmax", 3.6)}
    def num(v, f=1.0):
        try: return round(float(v) * f, 1)
        except (TypeError, ValueError): return None
    ara = datetime.now(timezone.utc)
    d = descarrega([ara], MC, num)
    st = sorted(d)[0]
    print("exemple estació", st, "->", {k: len(v) for k, v in d[st].items()})
