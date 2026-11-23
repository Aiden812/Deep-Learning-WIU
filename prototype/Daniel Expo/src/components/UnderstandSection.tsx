/* "What this means": what it is, how urgent, how it happens, what happens if left alone. */
import { StyleSheet, View } from "react-native";
import { condition } from "../art";
import { CONDITIONS, URGENCY_COLORS, URGENCY_LABELS, type ConditionKey } from "../data";
import { Icon } from "../icons";
import { C, R, SHADOW } from "../theme";
import { Art, Card, CardTitle, SectionHead, T, type Layout } from "./ui";

export function UnderstandSection({ layout, conditionKey: key }: { layout: Layout; conditionKey: ConditionKey }) {
  const c = CONDITIONS[key];
  const n = c.flow.length;
  const u = c.urgency;
  const twoCol = layout.wide;
  const row = (a: React.ReactNode, b: React.ReactNode, weights: [number, number]) => twoCol
    ? <View style={{ flexDirection: "row", gap: 32 }}><View style={{ flex: weights[0], minWidth: 0 }}>{a}</View><View style={{ flex: weights[1], minWidth: 0 }}>{b}</View></View>
    : <>{a}{b}</>;

  return (
    <View>
      <SectionHead icon="book-open" title={key === "normal" ? "What this means" : `About ${c.name.toLowerCase()}`} narrow={layout.narrow} />
      <View style={{ gap: 32 }}>
        {row(
          <Card narrow={layout.narrow} style={[s.what, !layout.narrow && { flexDirection: "row", alignItems: "center" }, twoCol && { flex: 1 }]}>
            <View style={s.whatArt}><Art xml={condition(key)} style={{ width: layout.narrow ? 120 : 123, aspectRatio: 120 / 140 }} /></View>
            <T size={layout.narrow ? 20 : 23} style={{ lineHeight: layout.narrow ? 27 : 31, letterSpacing: -0.2, flex: layout.narrow ? undefined : 1 }}>{c.what}</T>
          </Card>,
          <Card narrow={layout.narrow} style={twoCol && { flex: 1 }}>
            <CardTitle>How urgent?</CardTitle>
            <View style={s.meter}>
              {[0, 1, 2, 3].map((i) => (
                <View key={i} style={[s.meterSeg, i < u.level ? { backgroundColor: URGENCY_COLORS[u.level - 1] } : { boxShadow: SHADOW.insetSm }]} />
              ))}
            </View>
            <View style={[s.meter, { height: undefined, marginTop: 8 }]}>
              {URGENCY_LABELS.map((l, i) => (
                <T key={l} size={13} w={i === u.level - 1 ? 600 : 400} color={i === u.level - 1 ? C.ink : C.ink3} style={{ flex: 1 }}>{l}</T>
              ))}
            </View>
            <View style={s.verdict}>
              <View style={[s.urgencyIcon, { backgroundColor: URGENCY_COLORS[u.level - 1] }]}><Icon name={u.icon} size={28} color="#fff" /></View>
              <View style={{ flex: 1 }}>
                <T w={500} size={22} style={{ lineHeight: 26, letterSpacing: -0.2 }}>{u.title}</T>
                <T size={15} color={C.ink2} style={{ marginTop: 4 }}>{u.detail}</T>
              </View>
            </View>
          </Card>,
          [1.15, 1],
        )}
        {row(
          <Card narrow={layout.narrow} style={twoCol && { flex: 1 }}>
            <CardTitle>{c.headings?.causes || "How it happens"}</CardTitle>
            <View style={{ gap: 14 }}>
              {c.causes.map((x) => (
                <View key={x.text} style={s.cause}>
                  <View style={s.causeIcon}><Icon name={x.icon} size={22} /></View>
                  <T style={{ flex: 1 }}>{x.text}</T>
                </View>
              ))}
            </View>
          </Card>,
          <Card narrow={layout.narrow} style={twoCol && { flex: 1 }}>
            <CardTitle>{c.headings?.flow || "If left alone"}</CardTitle>
            <View>
              {c.flow.map((st, i) => {
                const sev = st.sev ?? (n === 1 ? 0 : Math.round((i / (n - 1)) * 3));
                const col = key === "normal" ? C.u1 : URGENCY_COLORS[sev];
                const last = i === n - 1;
                return (
                  <View key={st.label} style={[s.flowItem, !last && { paddingBottom: 18 }]}>
                    {!last && <View style={s.flowLine} />}
                    <View style={[s.flowDot, { backgroundColor: col }]}><Icon name={st.icon} size={22} color="#fff" /></View>
                    <View style={{ flex: 1, paddingTop: 2 }}>
                      <T w={500} style={{ lineHeight: 21 }}>{st.label}</T>
                      <T size={14} color={C.ink3}>{st.time}</T>
                    </View>
                  </View>
                );
              })}
            </View>
          </Card>,
          [1.15, 1],
        )}
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  what: { backgroundColor: C.peach, gap: 24 },
  whatArt: { width: 150, height: 150, borderRadius: R.md, overflow: "hidden", backgroundColor: C.surface, alignItems: "center", justifyContent: "flex-end" },
  meter: { flexDirection: "row", gap: 6, height: 14 },
  meterSeg: { flex: 1, borderRadius: 999, backgroundColor: C.surface2 },
  verdict: { flexDirection: "row", alignItems: "center", gap: 16, marginTop: 20 },
  urgencyIcon: { width: 60, height: 60, borderRadius: 20, alignItems: "center", justifyContent: "center" },
  cause: { flexDirection: "row", alignItems: "center", gap: 14 },
  causeIcon: { width: 48, height: 48, borderRadius: 16, alignItems: "center", justifyContent: "center", backgroundColor: C.surface, boxShadow: SHADOW.raiseSm },
  flowItem: { flexDirection: "row", gap: 14 },
  flowLine: { position: "absolute", left: 22.5, top: 50, bottom: 2, width: 3, borderRadius: 3, backgroundColor: C.surface2 },
  flowDot: { width: 48, height: 48, borderRadius: 16, alignItems: "center", justifyContent: "center" },
});
