# -*- coding: utf-8 -*-
"""
Arxiu HORARI del camp de vents: per a cada dia ja arxivat (arxiu/AAAA-MM-DD.enc)
recalcula el camp de vent de cada hora a partir de l'històric de les estacions i
el guarda comprimit i xifrat a arxiu-vent/AAAA-MM-DD.enc.

Format del payload (abans de xifrar):
  dia   : "AAAA-MM-DD"
  bbox, nx, ny : graella de visualització (igual que vent_privat.enc)
  esc   : escala de quantització (m/s per unitat int8)
  mask  : base64 d'una màscara de bits (1=terra) sobre nx*ny, ordre nord-primer
  t     : llista d'epoch ms (UTC) de cada hora disponible
  n     : nombre de cel·les de terra
  d     : base64 d'un buffer int8 de mida len(t)*n*2  (u,v intercalats)

Ús:
    from arxiu_vent import backfill
    backfill(PASSWORD)              # processa els dies que falten
    backfill(PASSWORD, max_dies=0)  # sense límit (backfill complet)
"""
import os, re, math, json, base64
from datetime import datetime, timezone
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from camp_vents import genera_camp, xifrar, ITER

ESC = 0.25          # m/s per unitat int8  (±31,75 m/s)
ARXIU_DIR = "arxiu"
OUT_DIR = "arxiu-vent"


def desxifrar(blob, password):
    salt = base64.b64decode(blob["salt"]); iv = base64.b64decode(blob["iv"]); ct = base64.b64decode(blob["ct"])
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt,
                     iterations=blob.get("it", ITER)).derive(password.encode())
    return json.loads(AESGCM(key).decrypt(iv, ct, None).decode("utf-8"))


def _iso(s):
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    s = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", s)
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def winds_per_hora(estacions):
    """{t_iso: [ {lat,lon,alt,u,v}, ... ]} a partir de l'històric de les estacions."""
    d = {}
    for e in estacions:
        lat, lon = e.get("lat"), e.get("lon")
        if lat is None or lon is None:
            continue
        alt = e.get("alt") or 0.0
        for row in (e.get("historic") or []):
            t, vv, dv = row.get("t"), row.get("vv"), row.get("dv")
            if not t or vv is None or dv is None:
                continue
            sp = vv / 3.6            # km/h -> m/s
            r = math.radians(dv)
            d.setdefault(t, []).append({"lat": lat, "lon": lon, "alt": alt,
                                        "u": -sp * math.sin(r), "v": -sp * math.cos(r)})
    return d


def genera_dia(dades_dia, grid_path=None):
    """Retorna el payload compacte d'un dia, o None si no hi ha prou dades."""
    perh = winds_per_hora(dades_dia.get("estacions") or [])
    hores = sorted(t for t, w in perh.items() if len(w) >= 5)
    if not hores:
        return None
    camps = []
    base = None
    for t in hores:
        try:
            c = genera_camp(perh[t], grid_path)
        except Exception:
            continue
        if base is None:
            base = c
        camps.append((t, c["u"], c["v"]))
    if not camps or base is None:
        return None

    sea = np.array(base["sea"], dtype=bool)
    land = ~sea
    land_idx = np.nonzero(land)[0]
    n = int(land_idx.size)
    buf = np.zeros((len(camps), n, 2), dtype=np.int8)
    ts = []
    for h, (t, u, v) in enumerate(camps):
        ua = np.asarray(u, dtype=float)[land_idx]
        va = np.asarray(v, dtype=float)[land_idx]
        buf[h, :, 0] = np.clip(np.round(ua / ESC), -127, 127).astype(np.int8)
        buf[h, :, 1] = np.clip(np.round(va / ESC), -127, 127).astype(np.int8)
        dt = _iso(t)
        ts.append(int(dt.timestamp() * 1000) if dt else 0)
    return {
        "bbox": base["bbox"], "nx": base["nx"], "ny": base["ny"], "esc": ESC,
        "mask": base64.b64encode(np.packbits(land.astype(np.uint8)).tobytes()).decode(),
        "t": ts, "n": n,
        "d": base64.b64encode(buf.tobytes()).decode(),
    }


def dies_arxiu(arxiu_dir=ARXIU_DIR):
    if not os.path.isdir(arxiu_dir):
        return []
    return sorted(fn[:-4] for fn in os.listdir(arxiu_dir) if fn.endswith(".enc"))


def dies_fets(out_dir=OUT_DIR):
    if not os.path.isdir(out_dir):
        return set()
    return set(fn[:-4] for fn in os.listdir(out_dir) if fn.endswith(".enc"))


def actualitza_index(out_dir=OUT_DIR):
    dies = sorted(fn[:-4] for fn in os.listdir(out_dir) if fn.endswith(".enc"))
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"dies": dies, "actualitzat": datetime.now(timezone.utc).isoformat(timespec="seconds")}, f)
    return dies


def backfill(password, arxiu_dir=ARXIU_DIR, out_dir=OUT_DIR, max_dies=3, grid_path=None):
    """Genera l'arxiu horari de vent per als dies que encara no el tenen.
    max_dies=0 -> sense límit."""
    os.makedirs(out_dir, exist_ok=True)
    fets = dies_fets(out_dir)
    pendents = [d for d in dies_arxiu(arxiu_dir) if d not in fets]
    if max_dies:
        pendents = pendents[:max_dies]
    if not pendents:
        print("  arxiu de vent: al dia (res pendent)")
        return 0
    ok = 0
    for dia in pendents:
        try:
            with open(os.path.join(arxiu_dir, dia + ".enc"), "r", encoding="utf-8") as f:
                blob = json.load(f)
            dades = desxifrar(blob, password)
            payload = genera_dia(dades, grid_path)
            if not payload:
                print("  %s: sense dades de vent suficients, s'omet" % dia)
                continue
            payload["dia"] = dia
            enc = xifrar(json.dumps(payload, separators=(",", ":")), password)
            with open(os.path.join(out_dir, dia + ".enc"), "w", encoding="utf-8") as f:
                json.dump(enc, f)
            ok += 1
            print("  %s: %d hores, %d cel·les de terra" % (dia, len(payload["t"]), payload["n"]))
        except Exception as ex:
            print("  %s: ERROR (%s)" % (dia, str(ex)[:100]))
    if ok:
        actualitza_index(out_dir)
    return ok


if __name__ == "__main__":
    import sys
    pwd = os.environ.get("MAPA_PASS")
    if not pwd:
        raise SystemExit("Falta la variable d'entorn MAPA_PASS")
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print("Backfill de l'arxiu de vent (max_dies=%s)..." % (lim or "sense límit"))
    n = backfill(pwd, max_dies=lim)
    print("Fet: %d dies generats" % n)
