# -*- coding: utf-8 -*-
"""
Baixa l'observació d'AEMET (Catalunya), calcula màx/mín reals de 24 h i
escriu 'dades.json' (obert, sense contrasenya).

Pensat per executar-se dins d'una GitHub Action cada hora. Llig la clau
d'AEMET per variable d'entorn:
    AEMET_API_KEY  -> la clau d'AEMET

(El nom del fitxer es manté per no haver de tocar el workflow; ara genera
dades en clar, no xifrades.)
"""

import json
import os
import ssl
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

API_KEY = os.environ["AEMET_API_KEY"]
BASE_URL = "https://opendata.aemet.es/opendata/api"
PROV_CAT = {"BARCELONA", "GIRONA", "LLEIDA", "TARRAGONA"}

_SSL = ssl.create_default_context()


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                               "Accept": "application/json"})
    raw = urllib.request.urlopen(req, timeout=45, context=_SSL).read()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("latin-1"))


def _aemet(endpoint, tries=5):
    for i in range(tries):
        try:
            meta = _get_json(BASE_URL + endpoint + "?api_key=" + API_KEY)
            if meta.get("estado") != 200:
                raise RuntimeError("AEMET estado=%s" % meta.get("estado"))
            return _get_json(meta["datos"])
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(20)
                continue
            raise
    raise RuntimeError("AEMET 429 persistent")


def _num(v):
    try:
        return round(float(v), 1)
    except (TypeError, ValueError):
        return None


def _kmh(ms):
    try:
        return round(float(ms) * 3.6, 1)
    except (TypeError, ValueError):
        return None


def construir_dades():
    inv = _aemet("/valores/climatologicos/inventarioestaciones/todasestaciones")
    cat = {}
    for e in inv:
        prov = (e.get("provincia") or "").strip().upper()
        if prov in PROV_CAT:
            cat[e["indicativo"]] = prov.capitalize()

    obs = _aemet("/observacion/convencional/todas")
    per_est = {}
    for r in obs:
        if r.get("idema") in cat:
            per_est.setdefault(r["idema"], []).append(r)

    estacions = []
    for idema, regs in per_est.items():
        regs.sort(key=lambda x: x.get("fint") or "")
        historic = []
        for r in regs:
            historic.append({
                "t": r.get("fint"),
                "ta": _num(r.get("ta")),
                "tamax": _num(r.get("tamax")),
                "tamin": _num(r.get("tamin")),
                "hr": _num(r.get("hr")),
                "vv": _kmh(r.get("vv")),
                "vmax": _kmh(r.get("vmax")),
                "dv": _num(r.get("dv")),
                "prec": _num(r.get("prec")),
                "pres": _num(r.get("pres_nmar") if r.get("pres_nmar") is not None
                             else r.get("pres")),
                "tpr": _num(r.get("tpr")),
            })
        ult = regs[-1]

        def _last(key):
            for r in reversed(regs):
                if r.get(key) is not None:
                    return r.get(key)
            return None

        _mx = [h["tamax"] if h["tamax"] is not None else h["ta"]
               for h in historic if h["tamax"] is not None or h["ta"] is not None]
        _mn = [h["tamin"] if h["tamin"] is not None else h["ta"]
               for h in historic if h["tamin"] is not None or h["ta"] is not None]

        estacions.append({
            "idema": idema,
            "nom": (ult.get("ubi") or idema).strip(),
            "provincia": cat.get(idema, ""),
            "lat": ult.get("lat"),
            "lon": ult.get("lon"),
            "alt": ult.get("alt"),
            "actual": {
                "fint": ult.get("fint"),
                "ta": _num(_last("ta")),
                "tamax": _num(_last("tamax")),
                "tamin": _num(_last("tamin")),
                "tamax_dia": max(_mx) if _mx else None,
                "tamin_dia": min(_mn) if _mn else None,
                "n_hores": len(historic),
                "hr": _num(_last("hr")),
                "vv": _kmh(_last("vv")),
                "vmax": _kmh(_last("vmax")),
                "dv": _num(_last("dv")),
                "dmax": _num(_last("dmax")),
                "prec": _num(_last("prec")),
                "pres": _num(_last("pres_nmar") if _last("pres_nmar") is not None
                             else _last("pres")),
                "tpr": _num(_last("tpr")),
            },
            "historic": historic,
        })

    estacions.sort(key=lambda e: e["nom"])
    return {
        "generat": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "font": "AEMET OpenData - observacio convencional (temps real)",
        "n_estacions": len(estacions),
        "estacions": estacions,
    }


def main():
    dades = construir_dades()
    with open("dades.json", "w", encoding="utf-8") as f:
        json.dump(dades, f, ensure_ascii=False, separators=(",", ":"))
    print("OK: %s estacions a dades.json (%s)" %
          (dades["n_estacions"], dades["generat"]))


if __name__ == "__main__":
    main()
