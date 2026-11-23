# SmileCheck SG: Expo + React Native version

The [SmileCheck SG web prototype](../Daniel/README.md) rebuilt as one Expo app. The same code runs:

- **on the web**, served by `server.py` at http://127.0.0.1:8713/ (or `npx expo start --web` while developing), and
- **on a phone in Expo Go**, connected to the same screening server over Wi-Fi.

Screening works the same way as in the web prototype. `server.py` imports `prototype/Daniel/server.py`, which imports Daniel's demo `app.py`, so both front ends run the same models, thresholds, video handling and result combining: mouth gate, then the issue gate, then the YOLO detector, on every photo and every video frame. The "Try a sample" chips still show simulated results.

## Run it

You need Node.js 20 or later, and the Python requirements from the web prototype:

```bash
pip install -r models/classifier/demo/Daniel/Codex/requirements.txt
cd "prototype/Daniel Expo"
npm install
```

### On the web

```bash
cd "prototype/Daniel Expo"
npx expo export -p web          # builds the web app into dist/ (rerun after code changes)
python server.py                # or double-click run.bat, which does both steps
```

Then open http://127.0.0.1:8713/. Other devices on your Wi-Fi can open the network address the server prints, such as `http://192.168.1.5:8713`.

The web URL options from the original page still work: `?demo=calculus` (also `caries`, `discolouration`, `ulcer`, `hypodontia`, `normal`, `check`) jumps straight to a sample result, and `?classifier=off` or `?classifier=on` overrides the server's issue-classifier setting.

### On a phone (Expo Go)

1. Install **Expo Go** from the App Store or Play Store. The app targets Expo SDK 58, so use an Expo Go version that supports SDK 58.
2. Start the screening server: `python "prototype/Daniel Expo/server.py"`. It listens on your local network (see [Network access](#network-access)).
3. In another terminal, start the dev server: `cd "prototype/Daniel Expo"` then `npx expo start`.
4. Scan the QR code: on Android, from Expo Go; on iPhone, with the Camera app. The phone and the computer must be on the same Wi-Fi.

The app finds the screening server by itself. It takes the computer's address from the Expo dev server and uses port 8713 on it. The footer's **Screening server** panel shows whether it is connected. If it says *Not reachable*, type the address `server.py` printed (for example `192.168.1.5:8713`) and tap **Connect**. The same panel switches the issue classifier on or off for a phone, like `?classifier=` does on the web.

`npx expo start --web` also works while you develop. That page runs on port 8081 and calls the screening server on port 8713 of the same host.

To bake a server address into a build, set `EXPO_PUBLIC_API_URL` (for example `EXPO_PUBLIC_API_URL=http://192.168.1.5:8713`) before running `npx expo start` or `npx expo export`.

## Network access

`prototype/Daniel/server.py` only listens on 127.0.0.1. This app's `server.py` listens on all network interfaces (`0.0.0.0`), so that phones can reach it. That means anyone on the same network can also send it photos to screen.

- On Windows, allow Python through the firewall for **private** networks the first time it asks.
- To keep it on this computer only, set `PROTO_HOST=127.0.0.1` (PowerShell: `$env:PROTO_HOST = "127.0.0.1"`). Expo Go can't connect then, but the web app still works.
- Browser calls are only accepted from `localhost` and private LAN addresses (`10.x`, `172.16–31.x`, `192.168.x`). Expo Go sends no `Origin` header, so this check doesn't apply to it.

Every setting from the web prototype also works here: `PROTO_CLASSIFIER`, `PROTO_PORT`, `VIDEO_MAX_FRAMES`, `VIDEO_FLAG_FRACTION`, `VIDEO_MAX_MB`, `DEMO_WEIGHTS`, `DEMO_MOUTH_GATE` and `DEMO_DETECTOR`. If you change `PROTO_PORT`, the app's automatic address won't match, so type the new address in the **Screening server** panel or set `EXPO_PUBLIC_API_URL`.

## Differences from the web prototype

| | Web prototype | This app |
| --- | --- | --- |
| Adding photos and videos | Upload, camera, drag and drop, paste | Upload and camera everywhere. Drag and drop and paste work on the web build only. |
| Photo pre-check (too dark, washed out, blurry) | In the browser | In the browser on the web build. On a phone, every photo goes to the server, and its mouth gate still rejects unusable ones. |
| Clinic map | Leaflet on the page | The same Leaflet map and tiles, inside a WebView on phones and an `<iframe>` on the web |
| Area picker | `<select>` | A pop-up list |
| Server address | Fixed to the page's server | Detected automatically, and can be changed in the footer |
| Scroll-in animations on cards | Yes | Not ported. The floating hero teeth, scan line, ring and bars still animate. |

## Files

| File | What it holds |
| --- | --- |
| `server.py` | Serves `dist/` and `prototype/Daniel/server.py`'s API on the LAN |
| `run.bat` | Installs packages and builds the web app if needed, then starts `server.py` (Windows) |
| `App.tsx` | App state and flows: consent, media, analyse, samples, location, scrolling, web paste and drop |
| `src/screening.ts` | `predict()` and `fromModel()`: the client side of a screening, ported from `js/app.js` |
| `src/api.ts` | Finding the server, and reading `/api/predict`'s NDJSON stream through `XMLHttpRequest`, because React Native's `fetch` can't stream |
| `src/data.ts`, `src/art.ts` | Content and SVG illustrations, ported from `js/data.js` and `js/art.js` |
| `src/theme.ts` | Colours, neumorphic shadows and fonts from `css/styles.css` |
| `src/icons.tsx` | Lucide icons by the names the web page used |
| `src/platform/media*.ts` | Video thumbnails and the photo pre-check: `.web.ts` for browsers, `.ts` for phones |
| `src/components/` | Header, hero, footer, the screening, understanding and clinic sections, and the result widgets |
| `src/components/map/` | One Leaflet page (`mapHtml.ts`), hosted by `ClinicMap.tsx` (WebView) and `ClinicMap.web.tsx` (iframe) |

`src/data.ts` and `src/art.ts` are copies of the web prototype's content. If you change wording, clinics or drawings there, change them here too.

## Prototype data to replace

The same placeholders as the web prototype:

- The three "CHAS dental clinic" entries are samples.
- Clinic coordinates are approximate, and the HealthHub link goes to the homepage.
- The cost estimator is a placeholder.
