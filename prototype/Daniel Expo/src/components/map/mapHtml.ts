/* One Leaflet page for the clinic map, shared by the native host (react-native-webview) and the web
   host (an <iframe>). Same tiles, pins and behaviour as prototype/Daniel/js/app.js.

   Host -> map: window.smileMap.update({ you, clinics, active }) / window.smileMap.focus(id)
     (native calls them through injectJavaScript; web posts { smilemap: <command> } messages).
   Map -> host: { type: "ready" } and { type: "clinic", id } through ReactNativeWebView.postMessage
     on native, or parent.postMessage({ smilemapOut: ... }) on web. */

export type MapClinic = { id: string; n: number; name: string; addr: string; km: number; best: boolean; lat: number; lng: number };
export type MapState = { you: { lat: number; lng: number }; clinics: MapClinic[]; active: string | null };

export const MAP_HTML = `<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  html, body, #map { margin: 0; height: 100%; background: #ece2d9; }
  .leaflet-container { font-family: "Outfit", system-ui, sans-serif; background: #ece2d9; }
  .leaflet-popup-content-wrapper { border-radius: 16px; }
  .leaflet-div-icon { background: none; border: 0; }
  .leaflet-tile-pane { filter: grayscale(.9) sepia(.18) brightness(1.04) contrast(.9); }
  .pin { width: 40px; height: 40px; box-sizing: border-box; border-radius: 50% 50% 50% 4px; transform: rotate(-45deg);
    display: grid; place-items: center; background: #121110; color: #fff; box-shadow: 0 6px 14px rgba(0,0,0,.25); border: 3px solid #fff; }
  .pin b { transform: rotate(45deg); font-size: 15px; font-weight: 600; }
  .pin.is-best { background: #efbf9f; color: #171513; }
  .pin.is-active { background: #e07a3f; color: #fff; }
  .you-dot { width: 24px; height: 24px; box-sizing: border-box; border-radius: 50%; background: #e07a3f; border: 4px solid #fff;
    box-shadow: 0 0 0 8px rgba(224,122,63,.25); }
</style>
</head><body><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
(function () {
  function send(msg) {
    if (window.ReactNativeWebView) window.ReactNativeWebView.postMessage(JSON.stringify(msg));
    else if (window.parent !== window) window.parent.postMessage({ smilemapOut: msg }, "*");
  }
  var esc = function (t) { return String(t).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };
  var map = L.map("map", { scrollWheelZoom: false, zoomControl: true }).setView([1.3801, 103.849], 12);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
  var you = null, markers = {}, active = null;

  function fitAll() {
    var pts = Object.keys(markers).map(function (k) { return markers[k].getLatLng(); });
    if (you) pts.push(you.getLatLng());
    if (pts.length) map.fitBounds(L.latLngBounds(pts), { padding: [40, 40], maxZoom: 14 });
  }
  function paintActive() {
    Object.keys(markers).forEach(function (k) {
      var el = markers[k].getElement(), pin = el && el.querySelector(".pin");
      if (pin) pin.classList.toggle("is-active", k === active);
    });
  }
  function update(st) {
    var icon = L.divIcon({ className: "", html: '<div class="you-dot"></div>', iconSize: [22, 22], iconAnchor: [11, 11] });
    if (you) you.setLatLng([st.you.lat, st.you.lng]);
    else you = L.marker([st.you.lat, st.you.lng], { icon: icon, title: "You", zIndexOffset: 1000 }).addTo(map).bindPopup("You are here");
    Object.keys(markers).forEach(function (k) { markers[k].remove(); });
    markers = {};
    active = st.active;
    st.clinics.forEach(function (c) {
      var ic = L.divIcon({
        className: "",
        html: '<div class="pin ' + (c.best ? "is-best " : "") + (c.id === active ? "is-active" : "") + '"><b>' + c.n + "</b></div>",
        iconSize: [40, 40], iconAnchor: [6, 40], popupAnchor: [14, -40],
      });
      markers[c.id] = L.marker([c.lat, c.lng], { icon: ic, title: c.name }).addTo(map)
        .bindPopup("<strong>" + esc(c.name) + "</strong><br>" + esc(c.addr) + "<br>" + c.km.toFixed(1) + " km away")
        .on("click", function () { focus(c.id); send({ type: "clinic", id: c.id }); });
    });
    map.invalidateSize();
    fitAll();
  }
  function focus(id) {
    active = id;
    var m = markers[id];
    if (m) { map.flyTo(m.getLatLng(), 14, { duration: 0.6 }); m.openPopup(); }
    paintActive();
  }
  window.smileMap = { update: update, focus: focus };
  window.addEventListener("message", function (e) {
    var d = e.data;
    if (!d || !d.smilemap) return;
    if (d.smilemap.type === "update") update(d.smilemap.state);
    else if (d.smilemap.type === "focus") focus(d.smilemap.id);
  });
  window.addEventListener("resize", function () { map.invalidateSize(); });
  send({ type: "ready" });
})();
</script></body></html>`;
