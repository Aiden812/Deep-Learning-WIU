/* Screening logic, ported from prototype/Daniel/js/app.js (predict, fromModel, samples).
   The server runs Daniel's pipeline from models/classifier/demo/Daniel:
   mouth gate -> issue gate (p(issue) vs its tuned threshold) -> YOLO detector on flagged photos. */
import { Image, Platform } from "react-native";
import { streamPredict } from "./api";
import { CLASS_ORDER, CONDITIONS, type ConditionKey } from "./data";
import type { IconName } from "./icons";
import { photoQualityIssue } from "./platform/media";

export type MediaKind = "image" | "video";
export type MediaItem = {
  id: number;
  kind: MediaKind;
  uri: string;            // file:// on native, blob: or data: on web
  thumb: string | null;   // still image for the grid (the photo itself, or a video frame)
  name: string;
  mime: string;
  file?: Blob;            // web only: what FormData uploads
};
export type Labelled = MediaItem & { label: string };

export type Note = { icon: IconName; bold: string; text: string };
export type DetBox = { class: string; display_name: string; conf: number; confident: boolean; box: [number, number, number, number] };
export type DetView = { src: string; w: number; h: number; boxes: DetBox[] };
export type Bar = { key: ConditionKey; value: number };
export type Ring = { value: number; title?: string; text?: (what: string, s: string) => string };

export type Result = {
  key: ConditionKey;
  what?: string;
  plural?: boolean;
  demo?: boolean;
  photoUrl?: string | null;
  view?: DetView | null;
  kicker?: string;
  ring?: Ring;
  bars?: Bar[];
  barsTitle?: string;
  lowConf?: string | null;
  notes?: Note[];
  probs?: Record<ConditionKey, number>;
};
export type Unclear = { unclear: true; media: Labelled; title?: string; why: string };

export const LOW_CONFIDENCE = 0.55;
export const VIDEO_EXT = /\.(mp4|mov|m4v|webm|avi|mkv|3gp)$/i;

export function kindOf(mime: string | undefined | null, name = ""): MediaKind | null {
  const t = mime || "";
  if (t.startsWith("image/")) return "image";
  if (t.startsWith("video/") || VIDEO_EXT.test(name)) return "video";
  if (/\.(jpe?g|png|gif|webp|heic|heif|bmp)$/i.test(name)) return "image";
  return null;
}

// "Photo", "Video" when there is one of a kind; "Photo 1", "Photo 2"... otherwise.
export function labelled(list: MediaItem[]): Labelled[] {
  const total = { image: 0, video: 0 }, seen = { image: 0, video: 0 };
  list.forEach((m) => total[m.kind]++);
  return list.map((m) => {
    const n = ++seen[m.kind], word = m.kind === "image" ? "Photo" : "Video";
    return { ...m, label: total[m.kind] > 1 ? `${word} ${n}` : word };
  });
}

// "2 photos and 1 video"
export function describe(list: MediaItem[]): string {
  const n = { image: 0, video: 0 };
  list.forEach((m) => n[m.kind]++);
  return ([["image", "photo"], ["video", "video"]] as const).filter(([k]) => n[k])
    .map(([k, w]) => `${n[k]} ${w}${n[k] > 1 ? "s" : ""}`).join(" and ");
}

// Six-class checkpoint names (if the server runs with DEMO_WEIGHTS=...best.pt) -> this app's keys.
const MODEL_KEYS: Record<string, ConditionKey> = {
  calculus: "calculus", caries: "caries", hypodontia: "hypodontia",
  mouth_ulcer: "ulcer", normal: "normal", tooth_discoloration: "discolouration",
};
// YOLO detector classes -> this app's keys.
export const DET_KEYS: Record<string, ConditionKey> = {
  Calculus: "calculus", Caries: "caries", Hypodontia: "hypodontia",
  Mouth_Ulcer: "ulcer", Tooth_Discoloration: "discolouration",
};
const DET_ORDER = CLASS_ORDER.filter((k) => k !== "normal");

// Mouth-gate outcomes, in plain words: [title when nothing could be checked, reason for one skipped item].
const GATE_REASONS: Record<string, [string, string]> = {
  no_mouth: ["We can't see a mouth", "it doesn't show a mouth or teeth."],
  mouth_not_found: ["We can't find your mouth", "we couldn't find the mouth. Face the camera with your mouth open."],
  mouth_too_small: ["Your mouth is too small in the frame", "the mouth is too small. Move the camera closer."],
  mouth_closed: ["Your mouth looks closed", "the mouth looks closed. Open wide or smile so your teeth show."],
  unreadable: ["We couldn't open this photo", "we couldn't open this file. Try a JPG or PNG."],
  unreadable_video: ["We couldn't open this video", "we couldn't read this video. Try an MP4 or MOV."],
};
const RETAKE_TIP = "Take a close-up of your teeth with your lips pulled back, or a front-facing shot with your mouth open.";
const cap = (t: string) => t.charAt(0).toUpperCase() + t.slice(1);
const mostCommon = (counts: Record<string, number>) => Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0];

export const imageSize = (uri: string) => new Promise<{ w: number; h: number }>((res, rej) =>
  Image.getSize(uri, (w, h) => res({ w, h }), rej));

/* The file part FormData uploads: a Blob on web, a { uri, name, type } descriptor on native. */
async function filePart(m: MediaItem): Promise<any> {
  if (Platform.OS !== "web") return { uri: m.uri, name: m.name, type: m.mime };
  return m.file ?? (await (await fetch(m.uri)).blob());
}

// Drops photos that are clearly unusable here first (web only), then streams the rest to the
// server, which runs every photo and every video frame through the pipeline and combines them.
export async function predict(
  base: string,
  list: Labelled[],
  classifier: string | null,
  onProgress?: (done: number, total: number | null) => void,
): Promise<Result | Unclear> {
  const send: Labelled[] = [], skipped: (Labelled & { why: string })[] = [];
  for (const m of list) {
    const why = m.kind === "image" ? await photoQualityIssue(m.uri) : null;
    if (why) skipped.push({ ...m, why });
    else send.push(m);
  }
  if (!send.length) return { unclear: true, media: skipped[0], why: skipped[0].why };

  const form = new FormData();
  for (const m of send) {
    const name = m.name || (m.kind === "video" ? "video.mp4" : "photo.png");
    const part = await filePart(m);
    if (Platform.OS === "web") form.append("files", part, name);
    else form.append("files", part);
  }
  if (classifier) form.append("classifier", classifier);
  let out: any;
  try {
    out = await streamPredict(base, form, onProgress);
  } catch (err) {
    console.error("Prediction failed:", err);
    const msg = err instanceof Error ? err.message : "";
    return {
      unclear: true, media: send[0], title: "We couldn't check this",
      why: `The screening service isn't responding. Please try again in a moment.${msg ? ` (${msg})` : ""}`,
    };
  }
  return fromModel(out, send, skipped);
}

// Turns the server's combined result into what the result card draws.
// `send` is what went to the server (same order as out.sources); `skipped` failed the quick check.
async function fromModel(out: any, send: Labelled[], skipped: (Labelled & { why: string })[]): Promise<Result | Unclear> {
  const reasonOf = (counts: Record<string, number>) => (GATE_REASONS[mostCommon(counts || {}) ?? ""] || [])[1] || "it couldn't be checked.";

  if (out.rejected) {
    const counts: Record<string, number> = {};
    out.sources.forEach((s: any) => Object.entries(s.reasons as Record<string, number>).forEach(([k, v]) => (counts[k] = (counts[k] || 0) + v)));
    const [title, reason] = GATE_REASONS[mostCommon(counts) ?? ""] || ["We can't read this", "it couldn't be checked."];
    const why = send.length === 1 && send[0].kind === "image" ? `${cap(reason)} ${RETAKE_TIP}`
      : `None of ${send.length > 1 ? "them" : "the frames"} could be checked. Mostly, ${reason} ${RETAKE_TIP}`;
    return { unclear: true, media: send[0], title, why };
  }

  // Notes on photos skipped here or by the mouth gate, and a line per video.
  const notes: Note[] = skipped.map((m) => ({ icon: "image-off", bold: m.label, text: ` skipped. ${m.why}` }));
  out.sources.forEach((s: any) => {
    const m = send[s.index];
    if (!s.accepted) {
      notes.push({ icon: s.kind === "video" ? "video-off" : "image-off", bold: m.label, text: ` skipped: ${reasonOf(s.reasons)}` });
    } else if (s.kind === "video") {
      const flagged = out.task === "binary" && s.flagged_frames ? `, ${s.flagged_frames} worth a check` : "";
      notes.push({ icon: "film", bold: m.label, text: `: ${s.accepted} of ${s.frames} frames checked${flagged}.` });
    }
  });

  const used: Labelled[] = out.sources.filter((s: any) => s.accepted).map((s: any) => send[s.index]);
  const nImg = used.filter((m) => m.kind === "image").length, nVid = used.length - nImg;
  const res: Result = {
    key: "check",
    notes,
    plural: used.length > 1,
    what: used.length === 1 ? `your ${nImg ? "photo" : "video"}`
      : !nVid ? `your ${nImg} photos` : !nImg ? `your ${nVid} videos` : `your photos and video${nVid > 1 ? "s" : ""}`,
  };

  const findings = (out.findings || []).filter((f: any) => DET_KEYS[f.class]);
  const top = findings[0];
  const shown = out.photos.find((p: any) => p.index === out.show_photo);
  const shownSrc = send[shown.source];
  // A healthy verdict draws no boxes (without the classifier, YOLO's weak boxes would otherwise show).
  res.view = out.task === "binary" && !out.flagged ? null : await detectionView(shown);
  res.photoUrl = shown.image || (shownSrc.kind === "image" ? shownSrc.uri : shownSrc.thumb);

  if (out.task === "binary") {
    // With the classifier off, p is the strongest YOLO box confidence, not a calibrated p(issue).
    const p: number = out.p_issue, detOnly = out.classifier === false;
    if (!out.flagged) {
      res.key = "normal";
      res.kicker = "Our screening says";
      res.ring = {
        value: 1 - p, title: "Looks healthy", text: detOnly
          ? (what) => `Nothing in ${what} was outlined clearly enough to flag. A dentist must confirm.`
          : (what, s) => `How likely ${what} show${s} none of the five conditions. A dentist must confirm.`,
      };
      res.bars = [{ key: "normal", value: 1 - p }, { key: "check", value: p }];
      if (p >= 0.5) res.lowConf = "It's a close call. Try a brighter, closer shot, or ask a dentist at your next check-up.";
    } else {
      res.key = top ? DET_KEYS[top.class] : "check";
      res.kicker = top?.confident ? "Looks most like" : top ? "Might be" : "Our screening says";
      res.ring = {
        value: p, title: "Needs a check", text: detOnly
          ? (what) => `How strongly the clearest finding in ${what} was outlined.`
          : (what, s) => `How likely ${what} show${s} something a dentist should look at.`,
      };
      if (out.detector_ran) {
        // Strongest outline per condition, pooled over the photos and frames.
        const conf: Record<string, number> = Object.fromEntries(findings.map((f: any) => [DET_KEYS[f.class], f.conf]));
        res.bars = DET_ORDER.map((k) => ({ key: k, value: conf[k] || 0 }));
        res.barsTitle = "How strongly each condition was outlined";
      }
      if (!top) res.lowConf = "We couldn't tell which condition it is. A dentist should take a look.";
      else if (!top.confident) res.lowConf = "We're not sure what this is. The dashed outlines are only best guesses, and it may be nothing. A check-up will tell.";
    }
  } else {
    // Six-class checkpoint: averaged probabilities, abstain below its threshold.
    res.key = MODEL_KEYS[out.class_name] || "check";
    res.bars = Object.entries(out.probs as Record<string, number>)
      .map(([n, v]) => ({ key: MODEL_KEYS[n], value: v })).filter((b) => b.key);
    res.ring = { value: out.confidence };
    if (out.abstain) res.lowConf = "The model isn't sure. Try a brighter, closer shot.";
  }
  return res;
}

// The detector's image of the shown photo with its boxes (confident ones, or the best guesses
// when nothing is confident), drawn as an SVG so the boxes stay aligned at any size.
async function detectionView(photo: any): Promise<DetView | null> {
  const det = photo?.detection;
  if (!det || !det.boxes.length || !det.image) return null;
  const boxes: DetBox[] = det.n_confident ? det.boxes.filter((b: DetBox) => b.confident) : det.boxes;
  try {
    const { w, h } = await imageSize(det.image);
    return { src: det.image, w, h, boxes };
  } catch {
    return null;
  }
}

/* ---------------- Samples (simulated) ---------------- */
function mulberry(a: number) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeProbs(top: ConditionKey, topP: number, rand: () => number) {
  const rest = CLASS_ORDER.filter((k) => k !== top);
  const w = rest.map(() => rand() + 0.15);
  const sum = w.reduce((a, b) => a + b, 0);
  const probs = { [top]: topP } as Record<ConditionKey, number>;
  rest.forEach((k, i) => (probs[k] = (w[i] / sum) * (1 - topP)));
  return probs;
}

export function demoResult(key: ConditionKey): Result {
  const seedRand = mulberry(key.length * 977 + key.charCodeAt(0));
  return { key, probs: makeProbs(key, 0.72 + seedRand() * 0.18, seedRand), demo: true, photoUrl: null };
}

export const isCondition = (k: string | null | undefined): k is ConditionKey => !!k && k in CONDITIONS;
