/* Native clinic map: the shared Leaflet page in a WebView. The web host is ClinicMap.web.tsx. */
import { useEffect, useRef, useState } from "react";
import { StyleSheet, View } from "react-native";
import { WebView } from "react-native-webview";
import { C, R, SHADOW } from "../../theme";
import { MAP_HTML, type MapState } from "./mapHtml";

export type ClinicMapProps = { state: MapState; focusId: string | null; focusSeq: number; onPick: (id: string) => void; height: number };

export default function ClinicMap({ state, focusId, focusSeq, onPick, height }: ClinicMapProps) {
  // WebView 14 is typed as a function component without `ref`; its instance has injectJavaScript.
  const ref = useRef<{ injectJavaScript: (js: string) => void } | null>(null);
  const refProp = { ref } as object;
  const [ready, setReady] = useState(false);
  const run = (js: string) => ref.current?.injectJavaScript(`try{${js}}catch(e){};true;`);

  useEffect(() => { if (ready) run(`window.smileMap.update(${JSON.stringify(state)})`); }, [ready, state]);
  useEffect(() => { if (ready && focusId) run(`window.smileMap.focus(${JSON.stringify(focusId)})`); }, [ready, focusSeq]);

  return (
    <View style={[s.wrap, { height }]} accessibilityLabel="Map of clinics">
      <WebView
        {...refProp}
        source={{ html: MAP_HTML, baseUrl: "https://smilecheck.sg.local/" }}
        originWhitelist={["*"]}
        style={s.web}
        scrollEnabled={false}
        nestedScrollEnabled
        setSupportMultipleWindows={false}
        onMessage={(e) => {
          try {
            const msg = JSON.parse(e.nativeEvent.data);
            if (msg.type === "ready") setReady(true);
            else if (msg.type === "clinic") onPick(msg.id);
          } catch { /* ignore */ }
        }}
      />
    </View>
  );
}

const s = StyleSheet.create({
  wrap: { borderRadius: R.md, overflow: "hidden", backgroundColor: C.surface2, boxShadow: SHADOW.inset },
  web: { flex: 1, backgroundColor: C.surface2 },
});
