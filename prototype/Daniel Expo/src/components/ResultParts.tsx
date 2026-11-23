/* Pieces of the result card: likelihood ring, bars, detector view, scan animation. */
import { useEffect, useRef, useState } from "react";
import { AccessibilityInfo, Animated, Easing, Image, StyleSheet, View } from "react-native";
import Svg, { Circle, Defs, Image as SvgImage, LinearGradient, Rect, Stop, Text as SvgText, G } from "react-native-svg";
import { CONDITIONS, type ConditionKey } from "../data";
import { DET_KEYS, type Bar, type DetView } from "../screening";
import { C, FONT, R, SHADOW } from "../theme";
import { Dot, T } from "./ui";

function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    AccessibilityInfo.isReduceMotionEnabled().then(setReduced).catch(() => {});
    const sub = AccessibilityInfo.addEventListener("reduceMotionChanged", setReduced);
    return () => sub.remove();
  }, []);
  return reduced;
}

/* ---------- Ring (the site's conic-gradient ring) ---------- */
export function Ring({ pct, color, label }: { pct: number; color: string; label: string }) {
  const [v, setV] = useState(0);
  const reduced = useReducedMotion();
  useEffect(() => {
    if (reduced) { setV(pct); return; }
    let raf = 0;
    const t0 = Date.now(), dur = 900;
    const step = () => {
      const k = Math.min(1, (Date.now() - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      setV(Math.round(pct * e));
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [pct, reduced]);

  const size = 84, sw = 9, r = (size - sw) / 2, circ = 2 * Math.PI * r;
  return (
    <View accessibilityRole="image" accessibilityLabel={label} style={{ width: size, height: size }}>
      <Svg width={size} height={size} style={StyleSheet.absoluteFill}>
        <Circle cx={size / 2} cy={size / 2} r={r} stroke={C.surface2} strokeWidth={sw} fill="none" />
        <Circle cx={size / 2} cy={size / 2} r={r} stroke={color} strokeWidth={sw} fill="none"
          strokeDasharray={`${(circ * v) / 100} ${circ}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} />
      </Svg>
      <View style={[s.ringInner]}>
        <T w={600} size={22} style={{ letterSpacing: -0.4, lineHeight: 26 }}>{v}%</T>
      </View>
    </View>
  );
}

/* ---------- Bars ---------- */
function BarFill({ value, color, grow }: { value: number; color: string; grow?: boolean }) {
  const w = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    w.setValue(0);
    Animated.timing(w, { toValue: value, duration: 800, easing: Easing.bezier(0.2, 0.8, 0.2, 1), useNativeDriver: false }).start();
  }, [value]);
  return (
    <View style={[s.barTrack, grow && { flex: 1 }]}>
      <Animated.View style={[s.barFill, { backgroundColor: color, width: w.interpolate({ inputRange: [0, 100], outputRange: ["0%", "100%"] }) }]} />
    </View>
  );
}

export function Bars({ bars, top, title, narrow }: { bars: Bar[]; top: ConditionKey; title?: string; narrow: boolean }) {
  const sorted = bars.slice().sort((a, b) => b.value - a.value);
  return (
    <View style={{ gap: 10 }} accessibilityLabel={title || "Likelihood of each condition"}>
      {title ? <T size={14} color={C.ink2}>{title}</T> : null}
      {sorted.map(({ key, value }) => {
        const c = CONDITIONS[key], v = Math.round(value * 100), isTop = key === top;
        return (
          <View key={key} style={narrow ? { gap: 6 } : s.probRow}>
            <View style={[s.probLabel, !narrow && { width: 156 }]}>
              <Dot color={c.color} />
              <T w={isTop ? 500 : 400} size={15} style={{ flex: 1 }} numberOfLines={1}>{c.name}</T>
              {narrow && <T size={15} color={isTop ? C.ink : C.ink2} style={s.probVal}>{v}%</T>}
            </View>
            <BarFill value={v} color={c.color} grow={!narrow} />
            {!narrow && <T size={15} color={isTop ? C.ink : C.ink2} style={[s.probVal, { width: 48 }]}>{v}%</T>}
          </View>
        );
      })}
    </View>
  );
}

/* ---------- Detector view ---------- */
// The whole photo (no cropping) so the boxes line up, drawn in one SVG.
export function DetectionView({ view }: { view: DetView }) {
  const { w, h } = view;
  const fs = Math.round(Math.max(w, h) * 0.032), pad = fs * 0.4, th = fs * 1.5, sw = Math.max(w, h) / 170;
  const names = [...new Set(view.boxes.map((b) => CONDITIONS[DET_KEYS[b.class]]?.name || b.display_name))].join(", ");
  return (
    <Svg width="100%" height="100%" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet"
      accessibilityLabel={`Your photo with outlined areas: ${names}`}>
      <SvgImage href={{ uri: view.src }} x={0} y={0} width={w} height={h} />
      {[...view.boxes].reverse().map((b, i) => { // strongest drawn last, on top
        const key = DET_KEYS[b.class];
        const col = key ? CONDITIONS[key].color : "#475569";
        const [x1, y1, x2, y2] = [b.box[0] * w, b.box[1] * h, b.box[2] * w, b.box[3] * h];
        const text = `${key ? CONDITIONS[key].name : b.display_name} ${Math.round(b.conf * 100)}%`;
        const tw = text.length * fs * 0.55 + pad * 2;
        const tx = Math.max(0, Math.min(x1, w - tw));
        const ty = y1 >= th ? y1 - th : y1;
        return (
          <G key={i}>
            <Rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} rx={fs * 0.3} fill="none" stroke={col}
              strokeWidth={sw} strokeDasharray={b.confident ? undefined : `${sw * 2.4} ${sw * 1.7}`} />
            <Rect x={tx} y={ty} width={tw} height={th} rx={fs * 0.3} fill={col} />
            <SvgText x={tx + pad} y={ty + th * 0.72} fontSize={fs} fill="#ffffff" fontFamily={FONT[500]} fontWeight="500">{text}</SvgText>
          </G>
        );
      })}
    </Svg>
  );
}

/* Dark fade under the result name (the site's linear-gradient overlay). */
export function BottomFade() {
  return (
    <Svg width="100%" height="100%" preserveAspectRatio="none" style={StyleSheet.absoluteFill}>
      <Defs>
        <LinearGradient id="fade" x1="0" y1="1" x2="0" y2="0">
          <Stop offset="0" stopColor="#121110" stopOpacity={0.82} />
          <Stop offset="1" stopColor="#121110" stopOpacity={0} />
        </LinearGradient>
      </Defs>
      <Rect x="0" y="0" width="100%" height="100%" fill="url(#fade)" />
    </Svg>
  );
}

/* ---------- Scanning frame ---------- */
export function ScanFrame({ uri }: { uri: string | null }) {
  const t = useRef(new Animated.Value(0)).current;
  const [h, setH] = useState(240);
  useEffect(() => {
    const loop = Animated.loop(Animated.sequence([
      Animated.timing(t, { toValue: 1, duration: 1400, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
      Animated.timing(t, { toValue: 0, duration: 1400, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
    ]));
    loop.start();
    return () => loop.stop();
  }, []);
  return (
    <View style={s.scanFrame} onLayout={(e) => setH(e.nativeEvent.layout.height)}>
      {uri ? <Image source={{ uri }} style={[StyleSheet.absoluteFill, { opacity: 0.85 }]} resizeMode="cover" /> : null}
      <Animated.View style={[s.scanLine, { transform: [{ translateY: t.interpolate({ inputRange: [0, 1], outputRange: [h * 0.04, h * 0.94] }) }] }]} />
    </View>
  );
}

export function ProgressBar({ done, total }: { done: number; total: number | null }) {
  const indet = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    if (total) return;
    const loop = Animated.loop(Animated.sequence([
      Animated.timing(indet, { toValue: 1, duration: 1200, useNativeDriver: false }),
      Animated.timing(indet, { toValue: 0, duration: 1200, useNativeDriver: false }),
    ]));
    loop.start();
    return () => loop.stop();
  }, [total]);
  const pct = total ? Math.min(100, (done / total) * 100) : 0;
  const fmt = (n: number) => n.toLocaleString("en-SG");
  const text = !done ? "Reading frames…" : total ? `${fmt(Math.min(done, total))} of ${fmt(total)} checked` : `${fmt(done)} checked`;
  return (
    <View style={{ width: "100%", maxWidth: 320, gap: 8, alignItems: "center" }}>
      <View style={[s.barTrack, { height: 8, width: "100%", boxShadow: SHADOW.insetSm }]}>
        {total
          ? <View style={[s.barFill, { backgroundColor: C.dark, width: `${pct}%` }]} />
          : <Animated.View style={[s.barFill, { backgroundColor: C.dark, width: "30%",
              left: indet.interpolate({ inputRange: [0, 1], outputRange: ["-3%", "70%"] }) }]} />}
      </View>
      <T size={14} color={C.ink2} style={{ fontVariant: ["tabular-nums"] }}>{text}</T>
    </View>
  );
}

const s = StyleSheet.create({
  ringInner: {
    position: "absolute", top: 9, left: 9, right: 9, bottom: 9, borderRadius: 999,
    backgroundColor: C.surface, boxShadow: SHADOW.insetSm, alignItems: "center", justifyContent: "center",
  },
  probRow: { flexDirection: "row", alignItems: "center", gap: 12 },
  probLabel: { flexDirection: "row", alignItems: "center", gap: 12 },
  probVal: { textAlign: "right", fontVariant: ["tabular-nums"] },
  barTrack: { height: 10, borderRadius: 999, backgroundColor: C.surface2, overflow: "hidden" },
  barFill: { height: "100%", borderRadius: 999 },
  scanFrame: { width: "100%", maxWidth: 320, aspectRatio: 4 / 3, borderRadius: R.md, overflow: "hidden", backgroundColor: C.dark },
  scanLine: {
    position: "absolute", left: 0, right: 0, top: 0, height: 3, backgroundColor: C.peach2,
    boxShadow: "0 0 18px 4px rgba(239,191,159,0.7)",
  },
});
