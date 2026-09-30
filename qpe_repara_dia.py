# -*- coding: utf-8 -*-
"""
REPARACIÓ D'UN DIA PASSAT del QPE amb els blocs de 6 h del VISOR d'AEMET (projecte GRAF).

Quan el workflow del QPE no ha corregut algunes hores, els RN1 d'eixes hores s'han perdut. Però el
visor públic d'AEMET conserva ~3 dies dels blocs de 6 h de cada radar regional:
   https://www.aemet.es/es/api-eltiempo/radar/imagen-radar/RNN/<RADAR><AAMMDDHHMMSS>.RNN.6HR_CAPPI.png
(hora UTC del FINAL del bloc: 00/06/12/18). Aquest script:
  1. Baixa el paquet HVD actual (amb la clau) i llig del GeoTIFF RNN.6HR del radar triat els LÍMITS
     lon/lat i la LLEGENDA (ESCALA), i també la seua imatge en mm.
  2. AUTOCALIBRA LA PROJECCIÓ: el PNG del visor està en WEB MERCATOR (files no lineals en latitud) i
     el GeoTIFF en EPSG:4326. Baixa el PNG del MATEIX bloc que el GeoTIFF i compara les dues maneres
     de mapejar-lo (lineal vs Mercator) contra el GeoTIFF: es queda amb la que millor casa. (Sense
     açò la pluja quedava desplaçada desenes de km cap al sud.)
  3. Elimina els tiles de reparació (gap=1) anteriors del dia (per si venien mal mapejats).
  4. Per a cada bloc de 6 h que cobreix el dia (UTC 00,06,12,18 i 00 de l'endemà), baixa el PNG,
     el descodifica a mm, el porta a la graella de Catalunya amb la projecció calibrada i aplica
     qpe_prod.repara_finestra (tiles de reparació a les hores buides; suma del bloc == bloc AEMET).
  5. Desmarca el dia de dia/_final.json perquè qpe_prod.py el regenere i el finalitze amb biaix d'estacions.

Ús (al workflow qpe_repara.yml, dins del clone de la branca 'qpe'):
    AEMET_API_KEY=... python qpe_repara_dia.py --dia 2026-09-29 --store qpe_store [--radar GLD]
"""
import argparse, glob, io, json, os, tarfile, urllib.request
from datetime import datetime, timezone, timedelta

import numpy as np

import qpe_aemet as A
import qpe_prod as Q

VISOR = "https://www.aemet.es/es/api-eltiempo/radar/imagen-radar/RNN/%s%s.RNN.6HR_CAPPI.png"


def tif_referencia(tar_bytes, radar):
    """Del paquet HVD: (bounds, escala, (W,H), band_mm, datetime) del GeoTIFF RNN.6HR més recent del radar."""
    import rasterio
    from rasterio.io import MemoryFile
    tf = tarfile.open(fileobj=io.BytesIO(tar_bytes))
    cand = []
    for m in tf.getmembers():
        info = A._parse_nom(m.name)
        if info and info[0] == radar and A.PROD_6H in info[2]:
            cand.append((info[1], m))
    if not cand:
        raise SystemExit("el paquet HVD no porta cap %s del radar %s (node caigut?)" % (A.PROD_6H, radar))
    cand.sort(key=lambda x: x[0])
    dt, m = cand[-1]
    with MemoryFile(tf.extractfile(m).read()) as mf, mf.open() as ds:
        b = ds.bounds
        esc = A.escala_de_ds(ds)
        if not esc:
            raise SystemExit("el GeoTIFF %s no porta llegenda ESCALA" % m.name)
        rgba = ds.read()
        band = A.rgba_a_mm(rgba, escala=esc) if rgba.shape[0] >= 4 else None
        return (b.left, b.bottom, b.right, b.top), esc, (ds.width, ds.height), band, dt


def png_a_mm(url, escala):
    from PIL import Image
    req = urllib.request.Request(url, headers={"User-Agent": "graf-qpe-repara"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    im = Image.open(io.BytesIO(raw)).convert("RGBA")
    arr = np.array(im)                                  # (H,W,4)
    rgba = np.transpose(arr, (2, 0, 1))                 # (4,H,W) com rasterio
    return A.rgba_a_mm(rgba, escala=escala), im.size


def _score(a, b):
    """Semblança entre dos mapes de pluja: -RMSE sobre les cel·les on algun dels dos té pluja
    (més alt = millor). NaN si no hi ha prou pluja per decidir."""
    a = np.where(np.isnan(a), 0.0, a).ravel(); b = np.where(np.isnan(b), 0.0, b).ravel()
    m = (a > 0.1) | (b > 0.1)
    if m.sum() < 30:
        return float("nan")
    return -float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def calibra_projeccio(radar, bounds, escala, tif_band, t_tif):
    """Baixa el PNG del visor del MATEIX bloc que el GeoTIFF i tria el mapeig (lineal 4326 vs Mercator)
    que millor casa amb el GeoTIFF sobre la graella CAT. Torna (funció_mapeig, nom, corrs)."""
    ref = A.a_graella(tif_band, bounds)                          # el GeoTIFF és 4326 -> files lineals
    idt = t_tif.strftime("%y%m%d%H%M%S")
    try:
        png_band, mida = png_a_mm(VISOR % (radar, idt), escala)
    except Exception as ex:  # noqa
        print("  calibració: no s'ha pogut baixar el PNG del bloc %s (%s); assumisc MERCATOR" % (idt, str(ex)[:50]))
        return A.a_graella_mercator, "mercator (per defecte)", {}
    g_lin = A.a_graella(png_band, bounds)
    g_mer = A.a_graella_mercator(png_band, bounds)
    c = {"lineal": _score(g_lin, ref), "mercator": _score(g_mer, ref)}
    fmt = lambda v: ("RMSE %.2f mm" % -v) if v == v else "n/d"
    print("  calibració (bloc %s, PNG %dx%d) vs GeoTIFF -> lineal: %s · mercator: %s"
          % (t_tif.strftime("%d/%m %H:%M"), mida[0], mida[1], fmt(c["lineal"]), fmt(c["mercator"])))
    ok_lin, ok_mer = (c["lineal"] == c["lineal"]), (c["mercator"] == c["mercator"])
    if not ok_lin and not ok_mer:
        print("  calibració: sense pluja suficient per comparar; assumisc MERCATOR (és la projecció del visor)")
        return A.a_graella_mercator, "mercator (per defecte)", c
    if ok_lin and (not ok_mer or c["lineal"] > c["mercator"]):
        return A.a_graella, "lineal (4326)", c
    return A.a_graella_mercator, "mercator (3857)", c


def elimina_gap_tiles(tdir, t_ini, t_fi):
    """Esborra els tiles de REPARACIÓ (gap=1) amb hora dins [t_ini, t_fi] (per refer-los ben mapejats)."""
    n = 0
    for p in glob.glob(os.path.join(tdir, "*.npz")):
        t = Q._ts_de_nom(p)
        if t is None or not (t_ini <= t <= t_fi):
            continue
        try:
            z = np.load(p)
            if "gap" in z.files and int(z["gap"]) == 1:
                os.remove(p); n += 1
        except Exception:  # noqa
            pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dia", required=True, help="dia UTC a reparar, AAAA-MM-DD")
    ap.add_argument("--store", default="qpe_store")
    ap.add_argument("--radar", default="GLD", help="radar regional del visor (GLD = Barcelona-Gelida)")
    a = ap.parse_args()
    key = os.environ.get("AEMET_API_KEY")
    if not key:
        raise SystemExit("cal AEMET_API_KEY")
    tdir = os.path.join(a.store, "tiles"); os.makedirs(tdir, exist_ok=True)
    d0 = datetime.strptime(a.dia, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print("Llegint el GeoTIFF %s del radar %s (HVD)…" % (A.PROD_6H, a.radar))
    bounds, escala, mida_tif, tif_band, t_tif = tif_referencia(A.baixa_hvd(key), a.radar)
    print("  bounds lon/lat %s · tif %dx%d · %d classes · hora %s" % (tuple(round(x, 3) for x in bounds), mida_tif[0], mida_tif[1], len(escala), t_tif.strftime("%d/%m %H:%M")))

    mapeja, nom_proj, _ = calibra_projeccio(a.radar, bounds, escala, tif_band, t_tif)
    print("  projecció triada per als PNG del visor: %s" % nom_proj)

    # neteja reparacions anteriors del dia (de -6 h a +30 h del dia UTC) per refer-les
    n_del = elimina_gap_tiles(tdir, d0 - timedelta(hours=6), d0 + timedelta(hours=30))
    print("  tiles de reparació anteriors esborrats: %d" % n_del)
    tiles = Q.carrega_tiles(tdir)
    print("  tiles al buffer: %d" % len(tiles))

    fins = [d0 + timedelta(hours=h) for h in (0, 6, 12, 18, 24)]
    for fi in fins:
        idt = fi.strftime("%y%m%d%H%M%S")
        try:
            band, mida = png_a_mm(VISOR % (a.radar, idt), escala)
        except Exception as ex:  # noqa
            print("  bloc fins %s: no baixat (%s)" % (fi.strftime("%d/%m %H:%M"), str(ex)[:60])); continue
        mos6 = mapeja(band, bounds)                           # (NY,NX) mm, NaN = sense pluja
        nval = int((~np.isnan(mos6)).sum())
        print("  bloc (%s, %s]: %d cel·les amb pluja · màx %.1f mm" % ((fi - timedelta(hours=6)).strftime("%d/%m %H:%M"), fi.strftime("%H:%M"), nval, float(np.nanmax(mos6)) if nval else 0.0))
        tiles = Q.repara_finestra(tdir, tiles, mos6, fi, etiqueta="bloc " + fi.strftime("%d/%m %H"))

    # desmarca el dia (local) de _final.json perquè qpe_prod el regenere amb els tiles reparats
    try:
        from zoneinfo import ZoneInfo
        tzl = ZoneInfo("Europe/Madrid")
    except Exception:  # noqa
        tzl = timezone.utc
    claus = {(d0 + timedelta(hours=h)).astimezone(tzl).strftime("%Y%m%d") for h in (0, 12, 23)}
    fpath = os.path.join(a.store, "dia", "_final.json")
    try:
        final = set(json.load(open(fpath)).get("final", []))
    except Exception:  # noqa
        final = set()
    abans = len(final)
    final -= claus
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    json.dump({"final": sorted(final)}, open(fpath, "w"))
    print("Dies desmarcats per regenerar: %s (%d -> %d finalitzats). Executa qpe_prod.py per refer finestres i arxiu diari." % (", ".join(sorted(claus)), abans, len(final)))


if __name__ == "__main__":
    main()
