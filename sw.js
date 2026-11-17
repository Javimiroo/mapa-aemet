// =====================================================================================
//  PYROMET · service worker (PWA)
//  - Fa l'app INSTAL·LABLE (pantalla d'inici, pantalla completa).
//  - Guarda l'última còpia de l'app i de les dades vives perquè es puga obrir amb mala cobertura
//    o sense connexió (es mostra l'última situació descarregada).
//  Estratègies:
//    · app (mateix origen)                  -> XARXA primer (sempre l'última versió); si no hi ha xarxa, la còpia guardada
//    · dades vives (raw.githubusercontent)   -> XARXA primer; si falla, l'última còpia (l'arxiu històric NO es guarda)
//    · llibreries i fonts (CDN)              -> CÒPIA primer (no canvien)
//    · autenticació (Workers) i tessel·les   -> sempre xarxa, mai en còpia
//  Per forçar que tothom reba una versió nova de la lògica de còpia, puja el número de VERSIO.
// =====================================================================================
const VERSIO = 'pyromet-v1';
const SHELL = ['./privat.html', './manifest.webmanifest', './icon-192.png', './icon-512.png', './comarques.geojson', './catalunya.geojson'];
const MAX_DADES = 40;                                   // fitxers de dades vives que es conserven (els més recents)
const NO_GUARDAR = /\/(arxiu|arxiu-vent|llamps_dia)\/|\/dia\/\d{8}\.enc/;   // arxiu històric: massa gros i immutable

self.addEventListener('install', e => {
  e.waitUntil(caches.open(VERSIO + '-app').then(c => Promise.all(SHELL.map(u => c.add(u).catch(() => null)))).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys()
    .then(ks => Promise.all(ks.filter(k => k.startsWith('pyromet-') && !k.startsWith(VERSIO)).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
const clau = u => { const x = new URL(u); x.searchParams.delete('_'); return x.toString(); };   // sense el paràmetre anti-cau
async function retalla(c, max) { const ks = await c.keys(); for (let i = 0; i < ks.length - max; i++) await c.delete(ks[i]); }
async function xarxaPrimer(req, nom, max, init) {
  const c = await caches.open(nom), k = clau(req.url);
  try {
    const r = await fetch(init ? new Request(req.url, init) : req);
    if (r && r.ok) { await c.put(k, r.clone()); if (max) retalla(c, max); }
    return r;
  } catch (e) {
    const m = await c.match(k);
    if (m) return m;
    throw e;
  }
}
async function copiaPrimer(req, nom) {
  const c = await caches.open(nom), m = await c.match(req.url);
  if (m) return m;
  const r = await fetch(req);
  if (r && (r.ok || r.type === 'opaque')) c.put(req.url, r.clone());
  return r;
}
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const u = new URL(req.url);
  if (u.hostname.endsWith('.workers.dev') || u.hostname.startsWith('auth.')) return;              // autenticació: mai en còpia
  if (u.origin === self.location.origin) {                                                          // l'app: xarxa primer, revalidant
    e.respondWith(xarxaPrimer(req, VERSIO + '-app', 0, { cache: 'no-cache' }));
    return;
  }
  if (u.hostname === 'raw.githubusercontent.com') {                                                // dades
    if (NO_GUARDAR.test(u.pathname)) return;
    e.respondWith(xarxaPrimer(req, VERSIO + '-dades', MAX_DADES));
    return;
  }
  if (/(^|\.)(unpkg\.com|jsdelivr\.net|cdnjs\.cloudflare\.com|googleapis\.com|gstatic\.com)$/.test(u.hostname)) {
    e.respondWith(copiaPrimer(req, VERSIO + '-libs'));
    return;
  }
  // tessel·les del mapa, radar, APIs externes: xarxa directa
});
