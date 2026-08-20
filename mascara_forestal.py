# -*- coding: utf-8 -*-
"""
Genera la MÀSCARA DE TERRENY FORESTAL de Catalunya (cremable: bosc/matoll/pastura)
a partir d'ESA WorldCover 2021 (10 m). Es corre UNA sola vegada (la cobertura del
sòl no canvia); el resultat 'mascara_forestal.json' es publica a l'arrel del repo
(GitHub Pages) i el frontend (privat.html -> enForest) el mostreja per cada llamp.

Classes WorldCover cremables triades: 10 arbrat, 20 matoll, 30 pastura.
S'exclouen: 40 cultiu, 50 urbà, 60 roca/nu, 70 neu, 80 aigua, 90 aiguamoll, 95, 100.

Requisits: pip install rasterio numpy   (necessita internet; llig els COG per /vsicurl)
Ús:   python mascara_forestal.py
"""
import base64, json
import numpy as np
import rasterio
from rasterio import Affine
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

BBOX = [0.10, 3.35, 40.50, 42.90]     # lon0, lon1, lat0, lat1 (Catalunya + marge)
RES_DEG = 0.0045                      # ~500 m per cel·la
FRAC_MIN = 0.30                       # cel·la = forestal si >=30% de píxels cremables
CREMABLE = (10, 20, 30)               # arbrat, matoll, pastura
BASE = "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_%s_Map.tif"


def tiles_bbox(bbox):
    lo0, lo1, la0, la1 = bbox
    out = []
    for lat in range(int(np.floor(la0/3))*3, int(np.floor(la1/3))*3+1, 3):
        for lon in range(int(np.floor(lo0/3))*3, int(np.floor(lo1/3))*3+1, 3):
            ns = "N%02d" % lat if lat >= 0 else "S%02d" % (-lat)
            ew = "E%03d" % lon if lon >= 0 else "W%03d" % (-lon)
            out.append(ns + ew)
    return out


def main():
    lo0, lo1, la0, la1 = BBOX
    nx = int(round((lo1 - lo0) / RES_DEG))
    ny = int(round((la1 - la0) / RES_DEG))
    dst_tr = from_bounds(lo0, la0, lo1, la1, nx, ny)     # nord-a-dalt (row 0 = nord)
    frac = np.zeros((ny, nx), np.float32)                # fracció de cremable per cel·la

    for t in tiles_bbox(BBOX):
        url = BASE % t
        try:
            with rasterio.open(url) as src:
                win = src.window(lo0, la0, lo1, la1).intersection(Window(0, 0, src.width, src.height))
                if win.width < 1 or win.height < 1:
                    print("  %s: sense solapament" % t); continue
                tw = max(1, int(win.width * 10.0 / (RES_DEG * 111320)))   # decima a ~RES
                th = max(1, int(win.height * 10.0 / (RES_DEG * 111320)))
                arr = src.read(1, window=win, out_shape=(th, tw), resampling=Resampling.nearest)
                burn = np.isin(arr, CREMABLE).astype(np.float32)
                st = src.window_transform(win) * Affine.scale(win.width / tw, win.height / th)
                part = np.zeros((ny, nx), np.float32)
                reproject(burn, part, src_transform=st, src_crs="EPSG:4326",
                          dst_transform=dst_tr, dst_crs="EPSG:4326", resampling=Resampling.average)
                frac = np.maximum(frac, part)
                print("  %s: OK (%d x %d px llegits)" % (t, tw, th))
        except Exception as ex:  # noqa
            print("  %s: no disponible (%s)" % (t, str(ex)[:70]))

    mask_nord = (frac >= FRAC_MIN)                        # nord-a-dalt
    mask = np.flipud(mask_nord)                           # -> SUD-primer (com espera el frontend)
    packed = np.packbits(mask.ravel().astype(np.uint8))
    obj = {"bbox": BBOX, "nx": nx, "ny": ny,
           "mask": base64.b64encode(packed.tobytes()).decode(),
           "nota": "ESA WorldCover 2021 · cremable=arbrat/matoll/pastura · frac>=%.2f" % FRAC_MIN}
    with open("mascara_forestal.json", "w", encoding="utf-8") as f:
        json.dump(obj, f)
    print("Fet: mascara_forestal.json  (%d x %d cel·les · %.1f%% forestal)"
          % (nx, ny, 100.0 * mask.mean()))


if __name__ == "__main__":
    main()
