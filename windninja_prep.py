# -*- coding: utf-8 -*-
"""
Prepara les entrades per a WindNinja d'una caixa qualsevol:
  - DEM en UTM (GeoTIFF) a partir de tessel·les d'elevació globals
  - un CSV per estació amb el vent observat (format point initialization)
  - fitxers de configuració per al CLI

Ús:
    python windninja_prep.py --bbox 0.68,41.08,1.18,41.45 --out wn --res 50 --mesh 100

Les dades d'estacions venen de Dades Obertes (sense quota). Si no n'hi ha,
el treball encara pot fer les proves de vent mitjà (domainAverage).
"""
import os, io, math, json, argparse, urllib.request, shutil
from datetime import datetime, timezone, timedelta

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

TILE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


# ------------------------------------------------------------------ DEM
def _lonlat2px(lon, lat, z):
    n = 2 ** z * 256
    x = (lon + 180.0) / 360.0 * n
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def baixa_dem(bbox, zoom=12):
    """Mosaic d'elevació (EPSG:3857) que cobreix el bbox (lon0,lat0,lon1,lat1)."""
    lon0, lat0, lon1, lat1 = bbox
    x0, y1 = _lonlat2px(lon0, lat0, zoom)
    x1, y0 = _lonlat2px(lon1, lat1, zoom)
    xt0, xt1 = int(x0 // 256), int(x1 // 256)
    yt0, yt1 = int(y0 // 256), int(y1 // 256)
    W = (xt1 - xt0 + 1) * 256
    H = (yt1 - yt0 + 1) * 256
    big = np.zeros((H, W), np.float32)
    n = 0
    for ty in range(yt0, yt1 + 1):
        for tx in range(xt0, xt1 + 1):
            url = TILE.format(z=zoom, x=tx, y=ty)
            try:
                raw = urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": "graf"}), timeout=30).read()
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                a = np.asarray(im, np.float32)
                elev = a[:, :, 0] * 256 + a[:, :, 1] + a[:, :, 2] / 256 - 32768
                big[(ty - yt0) * 256:(ty - yt0) * 256 + 256,
                    (tx - xt0) * 256:(tx - xt0) * 256 + 256] = elev
                n += 1
            except Exception as ex:
                print("  avis: tessel·la %d/%d/%d no baixada (%s)" % (zoom, tx, ty, str(ex)[:60]))
    world = 2 * math.pi * 6378137.0
    res = world / (256 * 2 ** zoom)
    shift = world / 2
    left = xt0 * 256 * res - shift
    top = shift - yt0 * 256 * res
    print("  DEM: %d tessel·les, mosaic %dx%d, elev %.0f..%.0f m" % (n, W, H, big.min(), big.max()))
    return big, Affine(res, 0, left, 0, -res, top)


def escriu_dem_utm(big, transform, bbox, path, res_m=50):
    """Reprojecta a UTM (zona automàtica) i retalla al bbox."""
    lon0, lat0, lon1, lat1 = bbox
    zona = int(math.floor((0.5 * (lon0 + lon1) + 180) / 6) + 1)
    epsg = 32600 + zona if 0.5 * (lat0 + lat1) >= 0 else 32700 + zona
    src_crs = CRS.from_epsg(3857)
    dst_crs = CRS.from_epsg(epsg)
    H, W = big.shape
    l, t = transform.c, transform.f
    r, b = l + W * transform.a, t + H * transform.e
    dt, dw, dh = calculate_default_transform(src_crs, dst_crs, W, H, l, b, r, t, resolution=res_m)
    dem = np.empty((dh, dw), "float32")
    reproject(big, dem, src_transform=transform, src_crs=src_crs,
              dst_transform=dt, dst_crs=dst_crs, resampling=Resampling.bilinear)
    with rasterio.open(path, "w", driver="GTiff", height=dh, width=dw, count=1,
                       dtype="float32", crs=dst_crs, transform=dt, nodata=-9999,
                       compress="deflate") as ds:
        ds.write(dem, 1)
    print("  DEM UTM%d: %s  (%dx%d, %g m)" % (zona, path, dw, dh, res_m))
    return epsg


# ------------------------------------------------------------------ estacions
HDR = ["Station_Name", "Coord_Sys(PROJCS,GEOGCS)", "Datum(WGS84,NAD83,NAD27)",
       "Lat/YCoord", "Lon/XCoord", "Height", "Height_Units(meters,feet)",
       "Speed", "Speed_Units(mph,kph,mps,kts)", "Direction(degrees)",
       "Temperature", "Temperature_Units(F,C)", "Cloud_Cover(%)",
       "Radius_of_Influence", "Radius_of_Influence_Units(miles,feet,meters,km)", "datetime"]


def _num(v, f=1.0):
    try:
        return round(float(v) * f, 1)
    except (TypeError, ValueError):
        return None


def estacions_csv(bbox, outdir, meta_path="meteocat_estacions.json"):
    """Un CSV per estació amb l'última observació. Retorna (n, datetime_iso)."""
    lon0, lat0, lon1, lat1 = bbox
    try:
        from xema_obert import descarrega
    except Exception as ex:
        print("  avis: xema_obert no disponible (%s)" % ex)
        return 0, None
    if not os.path.exists(meta_path):
        print("  avis: no trobe %s" % meta_path)
        return 0, None
    meta = json.load(open(meta_path, encoding="utf-8"))
    box = {k: v for k, v in meta.items()
           if v.get("lat") is not None and lon0 <= v["lon"] <= lon1 and lat0 <= v["lat"] <= lat1}
    if not box:
        print("  avis: cap estació dins del bbox")
        return 0, None

    VARS = {32: ("ta", 1.0),
            46: ("vv", 3.6), 47: ("dv", 1.0),
            48: ("vv", 3.6), 49: ("dv", 1.0),
            30: ("vv", 3.6), 31: ("dv", 1.0)}
    ara = datetime.now(timezone.utc)
    dat = descarrega([ara - timedelta(days=1), ara], VARS, _num, verbose=False)

    os.makedirs(outdir, exist_ok=True)
    for fn in os.listdir(outdir):
        if fn.endswith(".csv"):
            os.remove(os.path.join(outdir, fn))

    def q(s):
        return '"' + str(s) + '"'

    n = 0
    tmax = None
    for codi, m in sorted(box.items()):
        camps = dat.get(codi) or {}
        vv, dv, ta = camps.get("vv") or [], camps.get("dv") or [], camps.get("ta") or []
        if not vv or not dv:
            continue
        t_vv, v_vv = max(vv, key=lambda p: p[0])
        dd = dict(dv).get(t_vv)
        if dd is None:
            continue
        tt = dict(ta).get(t_vv)
        if tt is None:
            tt = 20.0
        sp = (v_vv or 0) / 3.6                      # km/h -> m/s
        dtiso = t_vv[:16].replace(" ", "T") + ":00Z" if len(t_vv) == 16 else t_vv
        if not dtiso.endswith("Z"):
            dtiso = t_vv
        tmax = max(tmax or dtiso, dtiso)
        nom = (m.get("nom") or codi).split(" - ")[0].replace(",", "")
        row = [nom, "GEOGCS", "WGS84", "%.5f" % m["lat"], "%.5f" % m["lon"], "10", "meters",
               "%.1f" % sp, "mps", "%d" % round(dd), "%.1f" % tt, "C", "0", "-1", "km", dtiso]
        with open(os.path.join(outdir, "%s.csv" % codi), "w", encoding="utf-8", newline="\n") as f:
            f.write(",".join(q(h) for h in HDR) + "\n")
            f.write(",".join(q(c) for c in row) + "\n")
        n += 1
    print("  estacions: %d CSV a %s (última obs. %s)" % (n, outdir, tmax))
    return n, tmax


# ------------------------------------------------------------------ cfg
BASE_CFG = """num_threads = 1
elevation_file = /data/dem.tif
input_wind_height = 10.0
units_input_wind_height = m
output_wind_height = 10.0
units_output_wind_height = m
vegetation = brush
mesh_resolution = {mesh}
units_mesh_resolution = m
write_ascii_output = true
"""

DOM = """initialization_method = domainAverageInitialization
input_speed = 6.0
input_speed_units = mps
input_direction = 300
"""


def escriu_proves(out, mesh, dem_src, n_est, dtiso):
    proves = []

    def prep(nom, extra):
        d = os.path.join(out, nom)
        os.makedirs(d, exist_ok=True)
        shutil.copy(dem_src, os.path.join(d, "dem.tif"))
        with open(os.path.join(d, "run.cfg"), "w", encoding="utf-8") as f:
            f.write(BASE_CFG.format(mesh=mesh) + extra)
        proves.append(nom)

    prep("t1_domini_massa", DOM)
    prep("t2_domini_moment", DOM + "momentum_flag = true\nnumber_of_iterations = 300\n")
    if n_est:
        est_dst = os.path.join(out, "t3_punts_massa", "estacions")
        os.makedirs(os.path.dirname(est_dst), exist_ok=True)
        if os.path.isdir(est_dst):
            shutil.rmtree(est_dst)
        shutil.copytree(os.path.join(out, "estacions"), est_dst)
        extra = ("initialization_method = pointInitialization\n"
                 "wx_station_filename = /data/estacions\n")
        prep("t3_punts_massa", extra)
    with open(os.path.join(out, "PROVES.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(proves))
    print("  proves preparades: %s" % ", ".join(proves))
    return proves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", default="0.68,41.08,1.18,41.45", help="lon0,lat0,lon1,lat1")
    ap.add_argument("--out", default="wn")
    ap.add_argument("--zoom", type=int, default=12)
    ap.add_argument("--res", type=float, default=50.0, help="resolució del DEM (m)")
    ap.add_argument("--mesh", type=float, default=100.0, help="mesh_resolution de WindNinja (m)")
    a = ap.parse_args()
    bbox = tuple(float(x) for x in a.bbox.split(","))
    os.makedirs(a.out, exist_ok=True)
    print("Preparant WindNinja per al bbox", bbox)
    big, tr = baixa_dem(bbox, a.zoom)
    dem = os.path.join(a.out, "dem.tif")
    escriu_dem_utm(big, tr, bbox, dem, a.res)
    n, dtiso = estacions_csv(bbox, os.path.join(a.out, "estacions"))
    escriu_proves(a.out, a.mesh, dem, n, dtiso)
    print("Fet.")


if __name__ == "__main__":
    main()
