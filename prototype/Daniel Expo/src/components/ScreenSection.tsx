/* "Screen your teeth": consent, adding photos and videos, and the result card. */
import { useState, type Ref } from "react";
import { Image, Pressable, StyleSheet, View, type HostInstance } from "react-native";
import { condition, heroTeeth } from "../art";
import { CLASS_ORDER, CONDITIONS, type ConditionKey } from "../data";
import { Icon } from "../icons";
import { LOW_CONFIDENCE, type Labelled, type Result } from "../screening";
import { C, R, SHADOW } from "../theme";
import { BottomFade, Bars, DetectionView, ProgressBar, Ring, ScanFrame } from "./ResultParts";
import { Art, Btn, Card, Dot, SectionHead, Switch, T, isWeb, open, type Layout } from "./ui";

export type ResultMode =
  | { mode: "empty" }
  | { mode: "loading"; uri: string | null; withProgress: boolean; done: number; total: number | null }
  | { mode: "unclear"; media: Labelled; title: string; why: string }
  | { mode: "body"; result: Result };

type Props = {
  layout: Layout;
  consent: boolean;
  nudged: boolean;
  onConsent: (on: boolean) => void;
  media: Labelled[];
  hint: string;
  dragOver: boolean;
  onUpload: () => void;
  onCamera: () => void;
  onRemove: (id: number) => void;
  busy: boolean;
  onAnalyse: () => void;
  onDemo: (k: ConditionKey) => void;
  view: ResultMode;
  onRetake: () => void;
  onReset: () => void;
  resultRef: Ref<HostInstance>;
  consentRef: Ref<HostInstance>;
};

export function ScreenSection(p: Props) {
  const { layout } = p;
  const twoCol = layout.wide;
  return (
    <View>
      <SectionHead icon="scan-line" title="Screen your teeth" narrow={layout.narrow} />
      <View style={{ flexDirection: twoCol ? "row" : "column", gap: 32, alignItems: twoCol ? "flex-start" : "stretch" }}>
        <View style={twoCol ? { flex: 1, minWidth: 0 } : undefined}><UploadCard {...p} /></View>
        <View style={twoCol ? { flex: 1, minWidth: 0, alignSelf: "stretch" } : undefined} ref={p.resultRef}><ResultCard {...p} /></View>
      </View>
      <View style={s.safety}>
        <Icon name="shield-alert" color={C.ink2} />
        <T size={15} color={C.ink2} style={{ flex: 1 }}>
          A screening, not a diagnosis. Swollen face, heavy bleeding or injury?{" "}
          <T w={600} size={15} color={C.u4} onPress={() => open("tel:995")} accessibilityRole="link">Call 995</T>
        </T>
      </View>
    </View>
  );
}

/* ---------- Upload card ---------- */
function UploadCard(p: Props) {
  const tile = p.layout.narrow ? 72 : 84;
  return (
    <Card narrow={p.layout.narrow} style={{ gap: 20 }}>
      <Pressable ref={p.consentRef} accessibilityRole="switch" accessibilityState={{ checked: p.consent }}
        onPress={() => p.onConsent(!p.consent)} style={s.consent}>
        <Switch on={p.consent} nudged={p.nudged} />
        <T size={15} color={C.ink2} style={{ flex: 1 }}>
          Use my photos and videos <T w={500} size={15}>for this screening only</T>. Deleted right after. PDPA
        </T>
      </Pressable>

      <View style={[s.dropzone, p.dragOver && { backgroundColor: C.peach }]}>
        <T w={500} size={22} style={{ letterSpacing: -0.2 }}>{isWeb ? "Drop, paste or upload" : "Add photos or videos"}</T>
        <T size={14} color={C.ink3} style={{ textAlign: "center" }}>{p.hint}</T>
        <View style={[s.dzButtons, !p.consent && { opacity: 0.55 }]}>
          <Btn label="Upload" icon="upload" onPress={p.onUpload} />
          <Btn label="Camera" icon="camera" variant="soft" onPress={p.onCamera} />
        </View>
      </View>

      {p.media.length > 0 && (
        <View style={s.media} accessibilityLabel="Added photos and videos">
          {p.media.map((m) => (
            <View key={m.id} style={[s.mediaItem, { width: tile, height: tile }]}>
              {m.thumb ? <Image source={{ uri: m.thumb }} style={StyleSheet.absoluteFill} resizeMode="cover" accessibilityLabel={m.label} />
                : <View style={[StyleSheet.absoluteFill, s.center]}><Icon name="video" color={C.onDark2} size={26} /></View>}
              {m.kind === "video" && <View style={[s.mediaBtn, { left: 6, bottom: 6, borderRadius: 9 }]}><Icon name="video" size={14} color="#fff" /></View>}
              <Pressable accessibilityRole="button" accessibilityLabel={`Remove ${m.label}`} onPress={() => p.onRemove(m.id)}
                style={[s.mediaBtn, { top: 6, right: 6 }]} hitSlop={6}>
                <Icon name="x" size={14} color="#fff" />
              </Pressable>
            </View>
          ))}
          <Pressable accessibilityRole="button" accessibilityLabel="Add more" onPress={p.onUpload}
            style={[s.mediaAdd, { width: tile, height: tile }]}>
            <Icon name="plus" size={24} color={C.ink2} />
          </Pressable>
        </View>
      )}

      <Btn label="Analyse" icon="sparkles" size="lg" block disabled={!p.media.length || p.busy} onPress={p.onAnalyse} />

      <View style={s.demoRow}>
        <T size={14} color={C.ink3}>Or try a sample:</T>
        {CLASS_ORDER.map((k) => (
          <Pressable key={k} onPress={() => p.onDemo(k)} accessibilityRole="button"
            style={({ hovered }: any) => [s.demoChip, hovered && { backgroundColor: C.peach }]}>
            <Dot color={CONDITIONS[k].color} />
            <T size={13}>{CONDITIONS[k].name}</T>
          </Pressable>
        ))}
      </View>
    </Card>
  );
}

/* ---------- Result card ---------- */
const EMPTY_ART = heroTeeth();

function ResultCard(p: Props) {
  const v = p.view;
  const minH = p.layout.wide ? 460 : 300;
  return (
    <Card narrow={p.layout.narrow} style={{ minHeight: p.layout.wide ? "100%" : undefined }}>
      <View accessibilityLiveRegion="polite" style={{ flex: 1 }}>
        {v.mode === "empty" && (
          <View style={[s.empty, { minHeight: minH }]}>
            <Art xml={EMPTY_ART} style={{ width: "100%", height: p.layout.narrow ? 180 : 260 }} />
            <T w={500} size={24}>Your result shows here</T>
          </View>
        )}
        {v.mode === "loading" && (
          <View style={[s.centered, { minHeight: minH }]}>
            <ScanFrame uri={v.uri} />
            <T w={500} size={24}>Looking closely…</T>
            {v.withProgress && <ProgressBar done={v.done} total={v.total} />}
          </View>
        )}
        {v.mode === "unclear" && (
          <View style={[s.centered, { minHeight: minH }]}>
            <View style={s.unclearPhoto}>
              {v.media.thumb ? <Image source={{ uri: v.media.thumb }} style={StyleSheet.absoluteFill} resizeMode="cover" accessibilityLabel="What you added" /> : null}
            </View>
            <T w={500} size={24} style={{ textAlign: "center" }}>{v.title}</T>
            <T color={C.ink2} style={{ textAlign: "center" }}>{v.why}</T>
            <View style={s.tips} accessibilityLabel="Photo tips">
              {([["sun", "Bright light"], ["smile", "Lips back"], ["focus", "Close up"]] as const).map(([icon, text]) => (
                <View key={text} style={s.tip}>
                  <View style={s.tipIcon}><Icon name={icon} size={16} /></View>
                  <T size={14} color={C.ink2}>{text}</T>
                </View>
              ))}
            </View>
            <Btn label="Try again" icon="camera" onPress={p.onRetake} style={{ alignSelf: "center" }} />
          </View>
        )}
        {v.mode === "body" && <ResultBody result={v.result} layout={p.layout} onReset={p.onReset} />}
      </View>
    </Card>
  );
}

function ResultBody({ result: res, layout, onReset }: { result: Result; layout: Layout; onReset: () => void }) {
  const [foldOpen, setFoldOpen] = useState(false);
  const c = CONDITIONS[res.key];
  const bars = res.bars || CLASS_ORDER.map((k) => ({ key: k, value: res.probs?.[k] ?? 0 }));
  const ringVal = res.ring ? res.ring.value : res.probs?.[res.key] ?? 0;
  const pct = Math.round(ringVal * 100);
  const what = res.demo ? "this sample" : res.what || "your photo";
  const ringText = res.ring?.text ? res.ring.text(what, res.plural ? "" : "s")
    : `How closely ${what} match${res.plural ? "" : "es"}. A dentist must confirm.`;
  const low = res.lowConf ?? (ringVal < LOW_CONFIDENCE ? "The model isn't sure. Try a brighter, closer shot." : null);
  const healthy = res.key === "normal";
  const isArt = !res.photoUrl && !res.view;
  const notes = res.notes || [];

  return (
    <View style={{ gap: 22 }}>
      <View style={[s.resultHero, { aspectRatio: layout.narrow ? 4 / 3 : 16 / 10 }, isArt && { backgroundColor: C.peach }]}>
        {res.view ? <View style={StyleSheet.absoluteFill}><DetectionView view={res.view} /></View>
          : res.photoUrl ? <Image source={{ uri: res.photoUrl }} style={StyleSheet.absoluteFill} resizeMode="cover" accessibilityLabel="Your uploaded photo" />
          : <Art xml={condition(res.key)} style={s.heroArt} />}
        {res.demo && (
          <View style={s.simTag}><Icon name="flask-conical" size={14} color="#fff" /><T size={13} color="#fff">Sample · simulated</T></View>
        )}
        <View style={[s.overlay, isArt && { top: 0, justifyContent: "flex-end", maxWidth: layout.narrow ? "64%" : "52%" }]}>
          {!isArt && <BottomFade />}
          <T size={15} color={isArt ? C.ink : "#fff"} style={{ opacity: 0.85 }}>{res.kicker || "Looks most like"}</T>
          <T size={layout.narrow ? 34 : 44} color={isArt ? C.ink : "#fff"} style={{ letterSpacing: -1.2, lineHeight: layout.narrow ? 36 : 46 }}>{c.name}</T>
          <T color={isArt ? C.ink : "#fff"} style={{ marginTop: 4, opacity: 0.85 }}>{c.aka}</T>
        </View>
      </View>

      <View style={s.ringRow}>
        <Ring pct={pct} color={c.color} label={`${pct} percent: ${res.ring?.title || "likelihood"}`} />
        <View style={{ flex: 1 }}>
          <T w={600} size={17}>{res.ring?.title || "Likelihood"}</T>
          <T size={14} color={C.ink2}>{ringText}</T>
        </View>
      </View>

      {low && (
        <View style={s.lowConf}><Icon name="image-off" /><T size={15} style={{ flex: 1 }}>{low}</T></View>
      )}

      {notes.length > 0 && (
        <View style={{ gap: 6 }}>
          {notes.map((n, i) => (
            <View key={i} style={{ flexDirection: "row", gap: 8, alignItems: "flex-start" }}>
              <View style={{ marginTop: 3 }}><Icon name={n.icon} size={16} color={C.ink2} /></View>
              <T size={14} color={C.ink2} style={{ flex: 1 }}><T w={500} size={14}>{n.bold}</T>{n.text}</T>
            </View>
          ))}
        </View>
      )}

      {!healthy && bars.length > 0 && (
        <View style={s.probs}><Bars bars={bars} top={res.key} title={res.barsTitle} narrow={layout.narrow} /></View>
      )}
      {healthy && (
        <View style={s.probs}>
          <Pressable onPress={() => setFoldOpen(!foldOpen)} accessibilityRole="button" accessibilityState={{ expanded: foldOpen }}
            style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
            <View style={{ transform: [{ rotate: foldOpen ? "0deg" : "-90deg" }] }}><Icon name="chevron-down" size={16} /></View>
            <T w={500}>See all likelihoods</T>
          </Pressable>
          {foldOpen && <View style={{ marginTop: 10 }}><Bars bars={bars} top={res.key} narrow={layout.narrow} /></View>}
        </View>
      )}

      <Btn label="Delete & start again" icon="trash-2" variant="soft" size="sm" onPress={onReset} />
    </View>
  );
}

const s = StyleSheet.create({
  safety: { flexDirection: "row", alignItems: "flex-start", gap: 10, marginTop: 24 },
  consent: { flexDirection: "row", alignItems: "flex-start", gap: 14 },
  dropzone: { alignItems: "center", gap: 10, paddingVertical: 32, paddingHorizontal: 18, borderRadius: R.md, backgroundColor: C.surface, boxShadow: SHADOW.inset },
  dzButtons: { flexDirection: "row", flexWrap: "wrap", gap: 12, justifyContent: "center", marginTop: 8 },
  media: { flexDirection: "row", flexWrap: "wrap", gap: 12 },
  mediaItem: { borderRadius: 18, overflow: "hidden", backgroundColor: C.dark, boxShadow: SHADOW.raiseSm },
  mediaBtn: { position: "absolute", width: 26, height: 26, borderRadius: 13, alignItems: "center", justifyContent: "center", backgroundColor: "rgba(18,17,16,0.72)" },
  mediaAdd: { borderRadius: 18, alignItems: "center", justifyContent: "center", backgroundColor: C.surface, boxShadow: SHADOW.insetSm },
  center: { alignItems: "center", justifyContent: "center" },
  demoRow: { flexDirection: "row", flexWrap: "wrap", alignItems: "center", gap: 8 },
  demoChip: { flexDirection: "row", alignItems: "center", gap: 6, paddingVertical: 6, paddingHorizontal: 12, borderRadius: 999, backgroundColor: C.surface2 },
  empty: { alignItems: "center", justifyContent: "center", gap: 12, backgroundColor: C.peach, borderRadius: R.md, overflow: "hidden", paddingBottom: 32 },
  centered: { alignItems: "center", justifyContent: "center", gap: 12 },
  unclearPhoto: { width: 160, height: 160, borderRadius: R.md, overflow: "hidden", backgroundColor: C.surface2, boxShadow: `0 0 0 4px ${C.u3}` },
  tips: { flexDirection: "row", flexWrap: "wrap", justifyContent: "center", columnGap: 22, rowGap: 10 },
  tip: { flexDirection: "row", alignItems: "center", gap: 8 },
  tipIcon: { width: 34, height: 34, borderRadius: 12, alignItems: "center", justifyContent: "center", backgroundColor: C.peach },
  resultHero: { borderRadius: R.md, overflow: "hidden", backgroundColor: C.dark, width: "100%" },
  heroArt: { position: "absolute", right: "6%", bottom: 0, height: "92%", aspectRatio: 120 / 140 },
  simTag: {
    position: "absolute", top: 14, left: 14, flexDirection: "row", alignItems: "center", gap: 6,
    paddingVertical: 6, paddingHorizontal: 12, borderRadius: 999, backgroundColor: "rgba(18,17,16,0.72)", zIndex: 2,
  },
  overlay: { position: "absolute", left: 0, right: 0, bottom: 0, paddingTop: 48, paddingHorizontal: 24, paddingBottom: 22 },
  ringRow: { flexDirection: "row", alignItems: "center", gap: 16 },
  lowConf: { flexDirection: "row", alignItems: "center", gap: 10, paddingVertical: 12, paddingHorizontal: 16, borderRadius: R.sm, backgroundColor: C.peach },
  probs: { padding: 18, borderRadius: R.md, boxShadow: SHADOW.inset },
});
