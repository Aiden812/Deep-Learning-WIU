/* Talks to the screening server (prototype/Daniel Expo/server.py, which reuses the endpoints of
   prototype/Daniel/server.py): POST /api/predict streams NDJSON, GET /api/meta describes the model. */
import Constants from "expo-constants";
import { Platform } from "react-native";

export const API_PORT = "8713";

/* Where the screening server is, in order:
   1. EXPO_PUBLIC_API_URL, if set when the app was bundled.
   2. Web: the exported build is only ever served by server.py, so it uses its own origin (whatever
      PROTO_PORT is). Under the Expo dev server (`npx expo start --web`, __DEV__), port 8713 on the
      same host.
   3. Expo Go: port 8713 on the computer running `npx expo start`. Its LAN address comes from the
      dev server's hostUri, so the phone reaches the model server on the same Wi-Fi.
   The footer's "Screening server" field can change it at run time. */
export function defaultApiBase(): string {
  const env = process.env.EXPO_PUBLIC_API_URL;
  if (env) return env.replace(/\/+$/, "");
  if (Platform.OS === "web" && typeof window !== "undefined") {
    const { protocol, hostname, port } = window.location;
    if (!__DEV__ || port === API_PORT) return "";
    return `${protocol}//${hostname || "127.0.0.1"}:${API_PORT}`;
  }
  const hostUri = Constants.expoConfig?.hostUri ?? (Constants as any).expoGoConfig?.debuggerHost ?? "";
  const host = String(hostUri).replace(/^[a-z]+:\/\//i, "").split("/")[0].split(":")[0];
  return `http://${host || "127.0.0.1"}:${API_PORT}`;
}

/* Turns whatever the user typed ("192.168.1.5", "192.168.1.5:8713", "http://...") into a base URL. */
export function normaliseBase(input: string): string {
  let s = input.trim().replace(/\/+$/, "");
  if (!s) return defaultApiBase();
  if (!/^https?:\/\//i.test(s)) s = `http://${s}`;
  if (!/:\d+$/.test(s.replace(/^https?:\/\//i, ""))) s = `${s}:${API_PORT}`;
  return s;
}

/* What to show for a base URL: "" means "this page's own server". */
export function describeBase(base: string): string {
  if (base) return base;
  return Platform.OS === "web" && typeof window !== "undefined" ? window.location.origin : "this server";
}

export async function fetchMeta(base: string): Promise<any> {
  const r = await fetch(`${base}/api/meta`);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

export async function checkHealth(base: string, timeoutMs = 4000): Promise<boolean> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(`${base}/api/health`, { signal: ctrl.signal });
    return r.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(t);
  }
}

/* /api/predict answers with NDJSON: a start line, one progress line per photo or frame, then the
   result. React Native's fetch can't read a body as a stream, so this uses XMLHttpRequest, whose
   responseText grows with each progress event on both native and web. */
export function streamPredict(
  base: string,
  form: FormData,
  onProgress?: (done: number, total: number | null) => void,
): Promise<any> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    let seen = 0, buf = "", result: any = null, failed: Error | null = null;

    const pump = () => {
      if (xhr.status && xhr.status >= 400) return;
      const text = xhr.responseText || "";
      buf += text.slice(seen);
      seen = text.length;
      let i: number;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (!line) continue;
        let msg: any;
        try { msg = JSON.parse(line); } catch { continue; }
        if (msg.type === "start" || msg.type === "progress") onProgress?.(msg.done || 0, msg.total ?? null);
        else if (msg.type === "result") result = msg;
        else if (msg.type === "error") failed = new Error(msg.detail);
      }
    };

    xhr.open("POST", `${base}/api/predict`);
    xhr.onprogress = pump;
    xhr.onreadystatechange = () => { if (xhr.readyState === 3) pump(); };
    xhr.onload = () => {
      if (xhr.status >= 400) {
        let detail = xhr.statusText || `HTTP ${xhr.status}`;
        try { detail = JSON.parse(xhr.responseText).detail || detail; } catch { /* not JSON */ }
        return reject(new Error(detail));
      }
      pump();
      if (buf.trim()) { buf += "\n"; pump(); }
      if (failed) return reject(failed);
      if (!result) return reject(new Error("The server closed the connection without a result."));
      resolve(result);
    };
    xhr.onerror = () => reject(new Error(`Can't reach the screening server at ${base || "this page's server"}.`));
    xhr.ontimeout = () => reject(new Error("The screening server took too long to answer."));
    xhr.send(form);
  });
}
