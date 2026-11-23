/* Web media helpers, using the browser's <video> and <canvas> like prototype/Daniel/js/app.js. */

const loadImage = (src: string) => new Promise<HTMLImageElement>((res, rej) => {
  const i = new Image();
  i.onload = () => res(i);
  i.onerror = rej;
  i.src = src;
});

/* A still at 0.1 s, as a JPEG data URL. Null if the browser can't decode the video. */
export function videoThumbnail(uri: string): Promise<string | null> {
  return new Promise((resolve) => {
    const v = document.createElement("video");
    let done = false;
    const finish = (out: string | null) => {
      if (done) return;
      done = true;
      v.removeAttribute("src");
      v.load();
      resolve(out);
    };
    v.muted = true;
    v.playsInline = true;
    v.preload = "auto";
    v.onloadeddata = () => { v.currentTime = Math.min(0.1, (v.duration || 1) / 2); };
    v.onseeked = () => {
      try {
        const side = 320, s = Math.min(1, side / Math.max(v.videoWidth, v.videoHeight));
        const c = document.createElement("canvas");
        c.width = Math.max(1, Math.round(v.videoWidth * s));
        c.height = Math.max(1, Math.round(v.videoHeight * s));
        c.getContext("2d")!.drawImage(v, 0, 0, c.width, c.height);
        finish(c.toDataURL("image/jpeg", 0.8));
      } catch {
        finish(null);
      }
    };
    v.onerror = () => finish(null);
    setTimeout(() => finish(null), 8000);
    v.src = uri;
  });
}

/* Rejects photos the model should not guess on: too dark, washed out, flat or blurry.
   Returns a plain-language reason, or null when the photo is usable. */
function photoQuality(d: Uint8ClampedArray, S: number): string | null {
  const lum: number[] = [];
  for (let i = 0; i < d.length; i += 4) lum.push(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
  const mean = lum.reduce((a, b) => a + b, 0) / lum.length;
  const sd = Math.sqrt(lum.reduce((a, b) => a + (b - mean) ** 2, 0) / lum.length);
  let edges = 0;
  for (let y = 0; y < S; y++) for (let x = 1; x < S; x++) edges += Math.abs(lum[y * S + x] - lum[y * S + x - 1]);
  edges /= S * (S - 1);
  if (mean < 45) return "It's too dark. Try near a window or turn on a light.";
  if (mean > 238) return "It's too bright or washed out. Avoid pointing at a lamp.";
  if (sd < 14 || edges < 3) return "We can't see any teeth. Pull your lips back and get closer.";
  return null;
}

/* photoQuality() on a 32 px copy. Photos the browser can't decode (e.g. HEIC) go to the server as they are. */
export async function photoQualityIssue(uri: string): Promise<string | null> {
  try {
    const img = await loadImage(uri), S = 32;
    const c = document.createElement("canvas");
    c.width = c.height = S;
    const ctx = c.getContext("2d")!;
    ctx.drawImage(img, 0, 0, S, S);
    return photoQuality(ctx.getImageData(0, 0, S, S).data, S);
  } catch {
    return null;
  }
}
