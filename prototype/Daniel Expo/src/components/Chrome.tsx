/* Header, hero and footer. */
import { useEffect, useMemo, useRef, useState } from "react";
import { Animated, Easing, Pressable, StyleSheet, TextInput, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { SvgXml } from "react-native-svg";
import { describeBase } from "../api";
import { BRAND_TOOTH, heroCrowns } from "../art";
import { Icon } from "../icons";
import { C, FONT, MAXW, SHADOW } from "../theme";
import { Btn, Chip, T, open, type Layout } from "./ui";

export type SectionId = "screen" | "understand" | "act";

/* ---------- Header ---------- */
export function Header({ layout, hasResult, active, onNav, onHeight }: {
  layout: Layout; hasResult: boolean; active: SectionId | null; onNav: (id: SectionId | "top") => void; onHeight: (h: number) => void;
}) {
  const insets = useSafeAreaInsets();
  const links: { id: SectionId; label: string; mini: string; icon: "scan-line" | "book-open" | "map-pin" }[] = [
    { id: "screen", label: "Screen", mini: "Screen", icon: "scan-line" },
    ...(hasResult ? [
      { id: "understand" as const, label: "Understand", mini: "Meaning", icon: "book-open" as const },
      { id: "act" as const, label: "Clinics & care", mini: "Clinics", icon: "map-pin" as const },
    ] : []),
  ];
  const pad = layout.narrow ? 16 : Math.min(40, Math.max(16, layout.width * 0.03));
  return (
    <View style={[s.topbar, { paddingTop: insets.top + 12, paddingHorizontal: pad }]} onLayout={(e) => onHeight(e.nativeEvent.layout.height)}>
      <View style={s.topRow}>
        <Pressable accessibilityRole="link" accessibilityLabel="SmileCheck SG home" onPress={() => onNav("top")} style={s.brand}>
          <View style={s.brandMark}><SvgXml xml={BRAND_TOOTH} width={20} height={22} /></View>
          <T w={600} size={20} numberOfLines={1} style={{ letterSpacing: -0.2 }}>
            SmileCheck <T w={400} size={20} color={C.ink3}>SG</T>
          </T>
        </Pressable>
        {layout.wide && (
          <View style={s.navChips}>
            {links.map((l) => <Chip key={l.id} label={l.label} icon={l.icon} active={active === l.id} onPress={() => onNav(l.id)} />)}
          </View>
        )}
        <Pressable accessibilityRole="link" accessibilityLabel="Dental emergency: call 995" onPress={() => open("tel:995")}
          style={[s.sos, !layout.wide && { marginLeft: "auto" }]}>
          <Icon name="siren" size={18} color="#fff" />
          <T w={600} size={16} color="#fff">{layout.narrow ? "995" : "Emergency 995"}</T>
        </Pressable>
      </View>
      {!layout.wide && (
        <View style={s.navMini}>
          {links.map((l) => (
            <Pressable key={l.id} onPress={() => onNav(l.id)}
              style={[s.miniLink, { boxShadow: active === l.id ? SHADOW.insetSm : SHADOW.raiseSm }]}>
              <T w={500} size={14}>{l.mini}</T>
            </Pressable>
          ))}
        </View>
      )}
    </View>
  );
}

/* ---------- Hero ---------- */
function FloatingCrown({ xml, delay }: { xml: string; delay: number }) {
  const t = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    const loop = Animated.loop(Animated.sequence([
      Animated.timing(t, { toValue: 1, duration: 3000, easing: Easing.inOut(Easing.sin), useNativeDriver: true }),
      Animated.timing(t, { toValue: 0, duration: 3000, easing: Easing.inOut(Easing.sin), useNativeDriver: true }),
    ]));
    const timer = setTimeout(() => loop.start(), delay);
    return () => { clearTimeout(timer); loop.stop(); };
  }, []);
  return (
    <Animated.View pointerEvents="none" style={[StyleSheet.absoluteFill, { transform: [{ translateY: t.interpolate({ inputRange: [0, 1], outputRange: [0, -10] }) }] }]}>
      <SvgXml xml={xml} width="100%" height="100%" />
    </Animated.View>
  );
}

export function HeroArt({ style }: { style?: any }) {
  const crowns = useMemo(() => heroCrowns(), []);
  return (
    <View style={[{ overflow: "hidden" }, style]} pointerEvents="none">
      {crowns.map((xml, i) => <FloatingCrown key={i} xml={xml} delay={i * 1000} />)}
    </View>
  );
}

export function Hero({ layout, onStart }: { layout: Layout; onStart: () => void }) {
  const hp = layout.narrow ? 28 : Math.min(64, Math.max(28, layout.width * 0.05));
  const h1 = layout.narrow ? 44 : Math.min(80, Math.max(46, layout.width * 0.056));
  const stacked = layout.narrow;
  return (
    <View style={[s.hero, { padding: hp, borderRadius: stacked ? 32 : 40, flexDirection: stacked ? "column" : "row", minHeight: stacked ? 0 : 500 }]}>
      {stacked && <HeroArt style={{ height: 260, marginTop: -hp, marginHorizontal: -hp }} />}
      <View style={{ flex: stacked ? undefined : 1.25, justifyContent: "center", zIndex: 1 }}>
        <T w={300} size={h1} color={C.onDark} accessibilityRole="header" style={{ letterSpacing: -h1 * 0.045, lineHeight: h1 * 1.02 }}>
          {/* Two unbroken lines on wide screens, like the site's nowrap spans. */}
          {stacked ? "Check your teeth in a minute" : "Check your teeth\nin a minute"}
        </T>
        <T size={layout.narrow ? 17 : 20} color={C.onDark2} style={{ maxWidth: 420, marginTop: 24 }}>
          Add a photo or video of your teeth. See what it looks like, how urgent it is, and where to go.
        </T>
        <View style={{ flexDirection: "row", marginTop: 32 }}>
          <Btn label="Start screening" icon="camera" variant="light" onPress={onStart} />
        </View>
      </View>
      {!stacked && <HeroArt style={{ flex: 1, alignSelf: "stretch", marginVertical: -hp, marginRight: -hp, minHeight: 320 }} />}
    </View>
  );
}

/* ---------- Footer (with the screening-server settings) ---------- */
export type ServerStatus = "checking" | "ok" | "down";
export type ClassifierChoice = null | "on" | "off";

export function Footer({ layout, fairText, apiBase, status, onServer, classifier, onClassifier }: {
  layout: Layout; fairText: string; apiBase: string; status: ServerStatus;
  onServer: (input: string) => void; classifier: ClassifierChoice; onClassifier: (c: ClassifierChoice) => void;
}) {
  const insets = useSafeAreaInsets();
  const [draft, setDraft] = useState(apiBase);
  useEffect(() => setDraft(apiBase), [apiBase]);
  const pad = layout.narrow ? 16 : Math.min(40, Math.max(16, layout.width * 0.03));
  const cols = layout.narrow ? 1 : 3;
  const blocks = [
    { icon: "shield-check" as const, title: "Your data (PDPA)", text: "Photos and videos are checked by a screening server on this computer and kept only for that check. Videos are briefly saved to a temporary file to read their frames, then deleted. Nothing is uploaded, sold or used for training." },
    { icon: "scale" as const, title: "Fair and honest", text: fairText },
    { icon: "book-marked" as const, title: "Sources", text: "MOH Dental Fee Benchmarks 2025 · CHAS dental subsidies (1 Oct 2025) · CPF MediSave · HPB & NDCS oral health guides · Kaggle Oral Image Dataset" },
  ];
  const statusText = { checking: "Checking…", ok: "Connected", down: "Not reachable" }[status];
  const statusColor = { checking: C.onDark2, ok: C.u1, down: C.u3 }[status];
  return (
    <View style={[s.footer, { paddingHorizontal: pad, paddingTop: layout.narrow ? 40 : 72, paddingBottom: 32 + insets.bottom }]}>
      <View style={[s.inner, { flexDirection: cols > 1 ? "row" : "column", gap: 32 }]}>
        {blocks.map((b) => (
          <View key={b.title} style={{ flex: cols > 1 ? 1 : undefined }}>
            <View style={s.footerTitle}><Icon name={b.icon} color={C.onDark} /><T w={500} size={17} color={C.onDark}>{b.title}</T></View>
            <T size={15} color={C.onDark2}>{b.text}</T>
          </View>
        ))}
      </View>

      <View style={[s.inner, s.server]}>
        <View style={s.footerTitle}>
          <Icon name="server" color={C.onDark} />
          <T w={500} size={17} color={C.onDark}>Screening server</T>
          <View style={[s.statusPill, { borderColor: statusColor }]}>
            <View style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: statusColor }} />
            <T size={13} color={statusColor}>{statusText}</T>
          </View>
        </View>
        <T size={14} color={C.onDark2}>
          Runs the models (prototype/Daniel Expo/server.py). On a phone, use the address of the computer running it, on the same Wi-Fi.
        </T>
        <View style={[s.serverRow, layout.narrow && { flexDirection: "column", alignItems: "stretch" }]}>
          <TextInput
            value={draft}
            onChangeText={setDraft}
            onSubmitEditing={() => onServer(draft)}
            placeholder={describeBase("")}
            placeholderTextColor={C.ink3}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
            accessibilityLabel="Screening server address"
            style={[s.input, !layout.narrow && { flex: 1 }]}
          />
          <Btn label="Connect" icon="locate-fixed" variant="light" size="sm" onPress={() => onServer(draft)} />
        </View>
        <View style={[s.serverRow, { flexWrap: "wrap" }]}>
          <T size={14} color={C.onDark2}>Issue classifier:</T>
          {([[null, "Server default"], ["on", "On"], ["off", "Off (YOLO only)"]] as const).map(([v, label]) => (
            <Pressable key={label} onPress={() => onClassifier(v)} accessibilityRole="button" accessibilityState={{ selected: classifier === v }}
              style={[s.segment, classifier === v && { backgroundColor: C.onDark }]}>
              <T size={13} w={500} color={classifier === v ? C.ink : C.onDark2}>{label}</T>
            </Pressable>
          ))}
        </View>
      </View>

      <View style={[s.inner, { marginTop: 56 }]}>
        <T w={300} size={layout.narrow ? 40 : Math.min(96, layout.width * 0.07)} color={C.onDark}
          style={{ letterSpacing: -2, lineHeight: layout.narrow ? 44 : Math.min(100, layout.width * 0.075) }}>
          SmileCheck SG
        </T>
        <T size={15} color={C.onDark2} style={{ marginTop: 12 }}>· IT2226 prototype · screening aid only</T>
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  topbar: { backgroundColor: "rgba(241,233,226,0.94)", paddingBottom: 12, zIndex: 10, gap: 10 },
  topRow: { flexDirection: "row", alignItems: "center", gap: 16 },
  brand: { flexDirection: "row", alignItems: "center", gap: 10, flexShrink: 1 },
  brandMark: { width: 38, height: 38, borderRadius: 12, backgroundColor: C.dark, alignItems: "center", justifyContent: "center" },
  navChips: { flexDirection: "row", gap: 8, marginLeft: "auto" },
  sos: { flexDirection: "row", alignItems: "center", gap: 8, paddingVertical: 10, paddingHorizontal: 16, borderRadius: 999, backgroundColor: C.u4 },
  navMini: { flexDirection: "row", gap: 6 },
  miniLink: { flex: 1, alignItems: "center", paddingVertical: 8, paddingHorizontal: 14, borderRadius: 999, backgroundColor: C.surface },
  hero: { backgroundColor: C.dark, overflow: "hidden", gap: 24 },
  footer: { marginTop: 100, backgroundColor: C.dark, borderTopLeftRadius: 40, borderTopRightRadius: 40 },
  inner: { width: "100%", maxWidth: MAXW, alignSelf: "center" },
  footerTitle: { flexDirection: "row", alignItems: "center", gap: 10, marginBottom: 8, flexWrap: "wrap" },
  server: { marginTop: 40, padding: 20, borderRadius: 24, backgroundColor: C.dark2, gap: 12 },
  statusPill: { flexDirection: "row", alignItems: "center", gap: 6, borderWidth: 1, borderRadius: 999, paddingHorizontal: 10, paddingVertical: 2 },
  serverRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  input: {
    minWidth: 0, borderRadius: 999, backgroundColor: C.dark3, color: C.onDark, paddingHorizontal: 18, paddingVertical: 12,
    fontFamily: FONT[400], fontSize: 15,
  },
  segment: { paddingVertical: 6, paddingHorizontal: 12, borderRadius: 999, backgroundColor: C.dark3 },
});
