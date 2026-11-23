/* Web clinic map: the shared Leaflet page in an <iframe> (react-native-webview has no web support). */
import { useEffect, useRef, useState } from "react";
import { StyleSheet, View } from "react-native";
import { C, R, SHADOW } from "../../theme";
import type { ClinicMapProps } from "./ClinicMap";
import { MAP_HTML } from "./mapHtml";

export default function ClinicMap({ state, focusId, focusSeq, onPick, height }: ClinicMapProps) {
  const ref = useRef<HTMLIFrameElement>(null);
  const [ready, setReady] = useState(false);
  const post = (cmd: object) => ref.current?.contentWindow?.postMessage({ smilemap: cmd }, "*");
  const pick = useRef(onPick);
  pick.current = onPick;

  useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      if (e.source !== ref.current?.contentWindow || !e.data?.smilemapOut) return;
      const msg = e.data.smilemapOut;
      if (msg.type === "ready") setReady(true);
      else if (msg.type === "clinic") pick.current(msg.id);
    };
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
  }, []);

  useEffect(() => { if (ready) post({ type: "update", state }); }, [ready, state]);
  useEffect(() => { if (ready && focusId) post({ type: "focus", id: focusId }); }, [ready, focusSeq]);

  return (
    <View style={[s.wrap, { height }]}>
      <iframe
        ref={ref}
        title="Map of clinics"
        srcDoc={MAP_HTML}
        style={{ border: 0, width: "100%", height: "100%", display: "block" }}
      />
    </View>
  );
}

const s = StyleSheet.create({
  wrap: { borderRadius: R.md, overflow: "hidden", backgroundColor: C.surface2, boxShadow: SHADOW.inset },
});
