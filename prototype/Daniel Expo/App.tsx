/* SmileCheck SG, Expo + React Native port of prototype/Daniel (one codebase for Expo Go and web).
   Photos and videos go to the screening server (server.py next to this file), which runs Daniel's
   mouth gate + issue gate + YOLO detector on every photo and video frame. The sample chips show
   simulated results. */
import {
  Outfit_300Light, Outfit_400Regular, Outfit_500Medium, Outfit_600SemiBold, useFonts,
} from "@expo-google-fonts/outfit";
import * as ImagePicker from "expo-image-picker";
import * as Location from "expo-location";
import { StatusBar } from "expo-status-bar";
import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { ActivityIndicator, Platform, ScrollView, StyleSheet, View, type HostInstance, type ScrollViewInstance } from "react-native";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { checkHealth, defaultApiBase, fetchMeta, normaliseBase } from "./src/api";
import { FAVICON } from "./src/art";
import { ActSection, type Filter, type Place } from "./src/components/ActSection";
import { Footer, Header, Hero, type ClassifierChoice, type SectionId, type ServerStatus } from "./src/components/Chrome";
import { ScreenSection, type ResultMode } from "./src/components/ScreenSection";
import { UnderstandSection } from "./src/components/UnderstandSection";
import { NarrowProvider, isWeb, useLayout } from "./src/components/ui";
import { AREAS, DEFAULT_LOCATION, type ConditionKey } from "./src/data";
import type { IconName } from "./src/icons";
import { videoThumbnail } from "./src/platform/media";
import {
  demoResult, describe, isCondition, kindOf, labelled, predict, type MediaItem, type Result,
} from "./src/screening";
import { C, MAXW } from "./src/theme";

const DEFAULT_FAIR = "The model learned from public photo datasets, not Singapore patients, so it can be wrong, especially in poor light. It can't see gum disease. We never suggest medicine.";
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
const webParam = (k: string) => (isWeb && typeof window !== "undefined" ? new URLSearchParams(window.location.search).get(k) : null);

export default function App() {
  const [fontsLoaded] = useFonts({ Outfit_300Light, Outfit_400Regular, Outfit_500Medium, Outfit_600SemiBold });
  if (!fontsLoaded) {
    return <View style={[styles.root, { alignItems: "center", justifyContent: "center" }]}><ActivityIndicator color={C.ink} /></View>;
  }
  return <SafeAreaProvider><SmileCheck /></SafeAreaProvider>;
}

function SmileCheck() {
  const layout = useLayout();

  /* ---------------- Server ---------------- */
  const [apiBase, setApiBase] = useState(defaultApiBase);
  const [status, setStatus] = useState<ServerStatus>("checking");
  // ?classifier=off skips the issue classifier (YOLO alone decides); ?classifier=on forces it on.
  // Without it the server's PROTO_CLASSIFIER setting applies. On a phone, the footer sets it.
  const [classifier, setClassifier] = useState<ClassifierChoice>(() => {
    const p = webParam("classifier");
    return p === "on" || p === "off" ? p : null;
  });
  const [fairText, setFairText] = useState(DEFAULT_FAIR);

  useEffect(() => {
    let live = true;
    setStatus("checking");
    checkHealth(apiBase).then((ok) => live && setStatus(ok ? "ok" : "down"));
    // Footer copy from the checkpoint's own test report.
    fetchMeta(apiBase).then((m) => {
      const imgs = m.metrics?.images;
      if (live && m.task === "binary" && imgs) {
        const n = (Object.values(imgs) as number[]).reduce((a, b) => a + b, 0);
        setFairText(`The model learned from ${n.toLocaleString("en-SG")} photos from public datasets, not Singapore patients, so it can be wrong, especially in poor light. It can't see gum disease. We never suggest medicine.`);
      }
    }).catch(() => { /* server not running: keep the default copy */ });
    return () => { live = false; };
  }, [apiBase]);

  /* ---------------- Web page setup ---------------- */
  useEffect(() => {
    if (!isWeb) return;
    document.title = "SmileCheck SG";
    let link = document.querySelector<HTMLLinkElement>("link[rel='icon']");
    if (!link) { link = document.createElement("link"); link.rel = "icon"; document.head.appendChild(link); }
    link.href = FAVICON;
    document.body.style.backgroundColor = C.bg;
  }, []);

  /* ---------------- Consent + media ---------------- */
  const [consent, setConsent] = useState(false);
  const [nudged, setNudged] = useState(false);
  const [media, setMedia] = useState<MediaItem[]>([]);
  const nextId = useRef(1);
  const list = useMemo(() => labelled(media), [media]);

  const addItems = useCallback(async (items: Omit<MediaItem, "id" | "thumb">[]) => {
    const withThumbs = await Promise.all(items.map(async (m) => ({
      ...m, id: nextId.current++, thumb: m.kind === "image" ? m.uri : await videoThumbnail(m.uri),
    })));
    setMedia((cur) => [...cur, ...withThumbs]);
  }, []);

  const revoke = (m: MediaItem) => { if (isWeb && m.uri.startsWith("blob:")) URL.revokeObjectURL(m.uri); };
  const removeMedia = (id: number) => setMedia((cur) => {
    const gone = cur.find((m) => m.id === id);
    if (gone) revoke(gone);
    return cur.filter((m) => m.id !== id);
  });
  const clearMedia = () => setMedia((cur) => { cur.forEach(revoke); return []; });

  const fromAssets = (assets: ImagePicker.ImagePickerAsset[]) => assets.map((a, i) => {
    const file = (a as any).file as File | undefined;
    const name = a.fileName || file?.name || `${a.type === "video" ? "video" : "photo"}-${Date.now()}-${i}.${a.type === "video" ? "mp4" : "jpg"}`;
    const mime = a.mimeType || file?.type || (a.type === "video" ? "video/mp4" : "image/jpeg");
    const kind = a.type === "video" || a.type === "pairedVideo" ? "video" : kindOf(mime, name) || "image";
    return { kind, uri: a.uri, name, mime, file } as Omit<MediaItem, "id" | "thumb">;
  });

  const consentRef = useRef<HostInstance>(null);
  const flashConsent = () => {
    setNudged(true);
    scrollToRef(consentRef, 0.35);
  };

  const pickerOptions: ImagePicker.ImagePickerOptions = {
    mediaTypes: ["images", "videos"],
    quality: 0.92, // JPEG out, so the server never gets HEIC
    preferredAssetRepresentationMode: ImagePicker.UIImagePickerPreferredAssetRepresentationMode.Compatible,
  };

  const onUpload = async () => {
    if (!consent) return flashConsent();
    if (Platform.OS === "ios") await ImagePicker.requestMediaLibraryPermissionsAsync().catch(() => null);
    const r = await ImagePicker.launchImageLibraryAsync({ ...pickerOptions, allowsMultipleSelection: true, selectionLimit: 0 });
    if (!r.canceled) addItems(fromAssets(r.assets));
  };

  const onCamera = async () => {
    if (!consent) return flashConsent();
    if (!isWeb) {
      const perm = await ImagePicker.requestCameraPermissionsAsync();
      if (!perm.granted) return;
    }
    const r = await ImagePicker.launchCameraAsync(pickerOptions);
    if (!r.canceled) addItems(fromAssets(r.assets));
  };

  // Web: paste anywhere (screenshots, copied photos, copied files) and drop onto the page.
  const [dragOver, setDragOver] = useState(false);
  const consentLive = useRef(consent);
  consentLive.current = consent;
  useEffect(() => {
    if (!isWeb) return;
    const take = (files: File[]) => {
      const ok = files.filter((f) => kindOf(f.type, f.name));
      if (!ok.length) return false;
      if (!consentLive.current) { flashConsent(); return true; }
      addItems(ok.map((f) => ({ kind: kindOf(f.type, f.name)!, uri: URL.createObjectURL(f), name: f.name || "pasted.png", mime: f.type, file: f })));
      return true;
    };
    const onPaste = (e: ClipboardEvent) => { if (take([...(e.clipboardData?.files || [])])) e.preventDefault(); };
    const onOver = (e: DragEvent) => { e.preventDefault(); if (consentLive.current) setDragOver(true); };
    const onLeave = (e: DragEvent) => { if (!e.relatedTarget) setDragOver(false); };
    const onDrop = (e: DragEvent) => { e.preventDefault(); setDragOver(false); take([...(e.dataTransfer?.files || [])]); };
    document.addEventListener("paste", onPaste);
    document.addEventListener("dragover", onOver);
    document.addEventListener("dragleave", onLeave);
    document.addEventListener("drop", onDrop);
    return () => {
      document.removeEventListener("paste", onPaste);
      document.removeEventListener("dragover", onOver);
      document.removeEventListener("dragleave", onLeave);
      document.removeEventListener("drop", onDrop);
    };
  }, [addItems]);

  const isTouch = !isWeb || (typeof window !== "undefined" && window.matchMedia?.("(pointer: coarse)").matches);
  const pasteKey = isWeb && /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘V" : "Ctrl+V";
  const hint = !consent ? "Turn on the switch above first"
    : media.length ? `${describe(media)} added`
    : isTouch ? "Photos or videos of your teeth" : `Photos or videos · ${pasteKey} to paste`;

  /* ---------------- Analyse ---------------- */
  const [view, setView] = useState<ResultMode>({ mode: "empty" });
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const resultRef = useRef<HostInstance>(null);

  const showLoading = (uri: string | null, withProgress: boolean) => {
    setView({ mode: "loading", uri, withProgress, done: 0, total: null });
    setTimeout(() => scrollToResult(), 50);
  };
  const setProgress = (done: number, total: number | null) =>
    setView((v) => (v.mode === "loading" ? { ...v, done, total } : v));

  const showResult = (res: Result) => {
    setResult(res);
    setView({ mode: "body", result: res });
    setActiveClinic(null);
  };

  const onAnalyse = async () => {
    if (!list.length || busy) return;
    setBusy(true);
    const first = list.find((m) => m.kind === "image") || list[0];
    showLoading(first.thumb, list.some((m) => m.kind === "video"));
    try {
      const cls = classifier === "on" ? "on" : classifier === "off" ? "off" : null;
      const [res] = await Promise.all([predict(apiBase, list, cls, setProgress), wait(1200)]);
      if ("unclear" in res) {
        setResult(null);
        setView({ mode: "unclear", media: res.media, title: res.title || "We can't read this photo", why: res.why });
      } else showResult(res);
    } finally {
      setBusy(false);
    }
  };

  const runDemo = async (key: ConditionKey, animate: boolean) => {
    if (animate) {
      setView({ mode: "loading", uri: null, withProgress: false, done: 0, total: null });
      setTimeout(() => scrollToResult(), 50);
      await wait(1100);
    }
    showResult(demoResult(key));
  };

  const onRetake = () => {
    clearMedia();
    setView({ mode: "empty" });
    if (consent) (isTouch && !isWeb ? onCamera : onUpload)();
  };

  const onReset = () => {
    clearMedia();
    setResult(null);
    setView({ mode: "empty" });
    scrollToSection("screen");
  };

  useEffect(() => {
    const demo = webParam("demo");
    if (isCondition(demo)) runDemo(demo, false);
  }, []);

  /* ---------------- Clinics + location ---------------- */
  const [location, setLocation] = useState<Place>({ ...DEFAULT_LOCATION, shared: false });
  const [areaLabel, setAreaLabel] = useState(DEFAULT_LOCATION.label);
  const [mapNote, setMapNote] = useState<{ icon: IconName; text: string }>({ icon: "info", text: `Distances from ${DEFAULT_LOCATION.label}.` });
  const [filter, setFilter] = useState<Filter>("best");
  const [activeClinic, setActiveClinic] = useState<string | null>(null);
  const [focusSeq, setFocusSeq] = useState(0);

  const onArea = (label: string) => {
    const a = AREAS.find((x) => x.label === label)!;
    setAreaLabel(label);
    setLocation({ lat: a.lat, lng: a.lng, label: a.label, shared: false });
    setMapNote({ icon: "info", text: `Distances from ${a.label}.` });
  };

  const onLocate = async () => {
    setMapNote({ icon: "loader", text: "Finding you…" });
    try {
      const perm = await Location.requestForegroundPermissionsAsync();
      if (!perm.granted) throw new Error("denied");
      const pos = await Promise.race([
        Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced }),
        wait(10000).then(() => { throw new Error("timeout"); }),
      ]);
      setLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude, label: "you", shared: true });
      setMapNote({ icon: "locate-fixed", text: "Distances from your location (never stored)." });
    } catch {
      setMapNote({ icon: "info", text: `Couldn't get your location. Showing distances from ${DEFAULT_LOCATION.label}.` });
    }
  };

  const focusClinic = (id: string) => { setActiveClinic(id); setFocusSeq((n) => n + 1); };

  /* ---------------- Scrolling + section nav ---------------- */
  const scrollRef = useRef<ScrollViewInstance>(null);
  const contentRef = useRef<HostInstance>(null);
  const sectionRefs = { screen: useRef<HostInstance>(null), understand: useRef<HostInstance>(null), act: useRef<HostInstance>(null) };
  const sectionY = useRef<Record<SectionId, number>>({ screen: 0, understand: 0, act: 0 });
  const scrollY = useRef(0);
  const [viewportH, setViewportH] = useState(800);
  const [, setHeaderH] = useState(70);
  const [activeSection, setActiveSection] = useState<SectionId | null>(null);

  const measureY = (ref: RefObject<HostInstance | null>, cb: (y: number) => void) => {
    const node = ref.current, root = contentRef.current;
    if (!node || !root) return;
    node.measureLayout(root, (_x: number, y: number) => cb(y), () => {});
  };

  const scrollToSection = (id: SectionId | "top") => {
    if (id === "top") return scrollRef.current?.scrollTo({ y: 0, animated: true });
    measureY(sectionRefs[id], (y) => scrollRef.current?.scrollTo({ y: Math.max(0, y - 16), animated: true }));
  };

  function scrollToRef(ref: RefObject<HostInstance | null>, at = 0) {
    measureY(ref, (y) => scrollRef.current?.scrollTo({ y: Math.max(0, y - viewportH * at), animated: true }));
  }

  // Bring the result card into view unless it is already near the top of the screen.
  function scrollToResult() {
    measureY(resultRef, (y) => {
      const top = y - scrollY.current;
      if (top < 20 || top > viewportH * 0.5) scrollRef.current?.scrollTo({ y: Math.max(0, y - 16), animated: true });
    });
  }

  const remeasureSections = () => {
    (Object.keys(sectionRefs) as SectionId[]).forEach((id) => measureY(sectionRefs[id], (y) => (sectionY.current[id] = y)));
  };

  const onScroll = (y: number) => {
    scrollY.current = y;
    const mid = y + viewportH * 0.45;
    const ids: SectionId[] = result ? ["screen", "understand", "act"] : ["screen"];
    let cur: SectionId | null = null;
    ids.forEach((id) => { if (sectionY.current[id] && mid >= sectionY.current[id]) cur = id; });
    if (cur !== activeSection) setActiveSection(cur);
  };

  // The explainer and clinic sections appear with a result: re-measure them for the nav highlight.
  useEffect(() => { if (result) setTimeout(remeasureSections, 100); }, [result]);

  const gutter = layout.narrow ? 16 : Math.min(40, Math.max(16, layout.width * 0.03));
  const sectionGap = layout.narrow ? 64 : Math.min(120, Math.max(64, layout.width * 0.09));

  return (
    <NarrowProvider value={layout.narrow}>
      <View style={styles.root} onLayout={(e) => setViewportH(e.nativeEvent.layout.height)}>
        <StatusBar style="dark" />
        <Header layout={layout} hasResult={!!result} active={activeSection} onNav={scrollToSection} onHeight={setHeaderH} />
        <ScrollView
          ref={scrollRef}
          style={{ flex: 1 }}
          scrollEventThrottle={64}
          onScroll={(e) => onScroll(e.nativeEvent.contentOffset.y)}
          keyboardShouldPersistTaps="handled"
        >
          <View ref={contentRef} collapsable={false}>
            <View style={[styles.main, { paddingHorizontal: gutter }]} onLayout={remeasureSections}>
              <Hero layout={layout} onStart={() => scrollToSection("screen")} />

              <View ref={sectionRefs.screen} collapsable={false} style={{ paddingTop: sectionGap }} onLayout={remeasureSections}>
                <ScreenSection
                  layout={layout}
                  consent={consent}
                  nudged={nudged}
                  onConsent={(on) => { setConsent(on); setNudged(false); }}
                  media={list}
                  hint={hint}
                  dragOver={dragOver}
                  onUpload={onUpload}
                  onCamera={onCamera}
                  onRemove={removeMedia}
                  busy={busy}
                  onAnalyse={onAnalyse}
                  onDemo={(k) => runDemo(k, true)}
                  view={view}
                  onRetake={onRetake}
                  onReset={onReset}
                  resultRef={resultRef}
                  consentRef={consentRef}
                />
              </View>

              {/* The explainer and clinic sections (and their nav links) only exist once there is a result. */}
              {result && (
                <>
                  <View ref={sectionRefs.understand} collapsable={false} style={{ paddingTop: sectionGap }} onLayout={remeasureSections}>
                    <UnderstandSection layout={layout} conditionKey={result.key} />
                  </View>
                  <View ref={sectionRefs.act} collapsable={false} style={{ paddingTop: sectionGap }} onLayout={remeasureSections}>
                    <ActSection
                      layout={layout}
                      conditionKey={result.key}
                      location={location}
                      areaLabel={areaLabel}
                      onArea={onArea}
                      onLocate={onLocate}
                      mapNote={mapNote}
                      filter={filter}
                      onFilter={setFilter}
                      activeClinic={activeClinic}
                      focusSeq={focusSeq}
                      onFocusClinic={focusClinic}
                    />
                  </View>
                </>
              )}
            </View>

            <Footer
              layout={layout}
              fairText={fairText}
              apiBase={apiBase}
              status={status}
              onServer={(input) => setApiBase(normaliseBase(input))}
              classifier={classifier}
              onClassifier={setClassifier}
            />
          </View>
        </ScrollView>
      </View>
    </NarrowProvider>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },
  main: { width: "100%", maxWidth: MAXW, alignSelf: "center", paddingTop: 8 },
});
