# -*- coding: utf-8 -*-
"""
REPARACIÓ D'UN DIA PASSAT del QPE amb els blocs de 6 h del VISOR d'AEMET (projecte GRAF).

Quan el workflow del QPE no ha corregut algunes hores, els RN1 d'eixes hores s'han perdut. Però el
visor públic d'AEMET conserva ~3 dies dels blocs de 6 h de cada radar regional:
   https://www.aemet.es/es/api-eltiempo/radar/imagen-radar/RNN/<RADAR><AAMMDDHHMMSS>.RNN.6HR_CAPPI.png
(hora UTC del FINAL del bloc: 00/06/12/18). Aquest script:
  1. Baixa el paquet HVD actual (amb la clau) i llig del GeoTIFF RNN.6HR del radar triat els LÍMITS
     georeferenciats i la LLEGENDA (ESCALA) — el PNG del visor és el mateix ràster renderitzat.
  2. Per a cada bloc de 6 h que cobreix el dia (UTC 00,06,12,18 del dia i 00 de l'endemà), baixa el
     PNG del visor, el descodifica a mm i el porta a la graella de Catalunya.
  3. Aplica la mateixa reparació que en directe (qpe_prod.repara_finestra): tiles de reparació a
     les hores buides amb el dèficit repartit. Invariant: suma(tiles del bloc) == bloc de 6 h.
  4. Desmarca el dia de dia/_final.json perquè el següent run de qpe_prod.py el torne a generar
     (ja reparat) i el finalitze amb el biaix d'estacions.

Ús (al workflow qpe_repara.yml, dins del clone de la branca 'qpe'):
    AEMET_API_KEY=... python qpe_repara_dia.py --dia 2026-09-29 --store qpe_store [--radar GLD]
"""
import argparse, io, json, os, tarfile, urllib.request
from datetime import datetime, timezone, timedelta

import numpy as np

import qpe_aemet as A
import qpe_prod as Q

VISOR = "https://www.aemet.es/es/api-eltiempo/radar/imagen-radar/RNN/%s%s.RNN.6HR_CAPPI.png"


def bounds_i_escala(tar_bytes, radar):
    """Del paquet HVD: (bounds, escala, (W,H)) del GeoTIFF RNN.6HR del radar demanat."""
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
    m = cand[-1][1]
    with MemoryFile(tf.extractfile(m).read()) as mf, mf.open() as ds:
        b = ds.bounds
        esc = A.escala_de_ds(ds)
        if not esc:
            raise SystemExit("el GeoTIFF %s no porta llegenda ESCALA" % m.name)
        return (b.left, b.bottom, b.right, b.top), esc, (ds.width, ds.height)


def png_a_mm(url, escala):
    from PIL import Image
    req = urllib.request.Request(url, headers={"User-Agent": "graf-qpe-repara"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    im = Image.open(io.BytesIO(raw)).convert("RGBA")
    arr = np.array(im)                                  # (H,W,4)
    rgba = np.transpose(arr, (2, 0, 1))                 # (4,H,W) com rasterio
    return A.rgba_a_mm(rgba, escala=escala), im.size


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

    print("Llegint límits i llegenda del GeoTIFF %s (HVD)…" % A.PROD_6H)
    bounds, escala, mida_tif = bounds_i_escala(A.baixa_hvd(key), a.radar)
    print("  radar %s · bounds %s · tif %dx%d · %d classes" % (a.radar, tuple(round(x, 3) for x in bounds), mida_tif[0], mida_tif[1], len(escala)))

    tiles = Q.carrega_tiles(tdir)
    print("  tiles al buffer: %d" % len(tiles))
    # blocs que cobreixen el dia (UTC) + el que acaba a les 00 de l'endemà (cobreix 18-24 UTC = fins 02 h local)
    fins = [d0 + timedelta(hours=h) for h in (0, 6, 12, 18, 24)]
    for fi in fins:
        idt = fi.strftime("%y%m%d%H%M%S")
        url = VISOR % (a.radar, idt)
        try:
            band, mida = png_a_mm(url, escala)
        except Exception as ex:  # noqa
            print("  bloc fins %s: no baixat (%s)" % (fi.strftime("%d/%m %H:%M"), str(ex)[:60])); continue
        if mida != mida_tif:
            print("  avis: PNG %dx%d != tif %dx%d (es mapeja per proporció)" % (mida[0], mida[1], mida_tif[0], mida_tif[1]))
        mos6 = A.a_graella(band, bounds)                   # (NY,NX) mm, NaN = sense pluja
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
