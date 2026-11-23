/* "Get it checked": clinic map and list, area picker, home-care products. */
import { useMemo, useState } from "react";
import { Modal, Pressable, ScrollView, StyleSheet, View } from "react-native";
import { product } from "../art";
import { AREAS, CLINICS, CLINIC_TYPES, CONDITIONS, PRODUCTS, type ClinicType, type ConditionKey } from "../data";
import { Icon, type IconName } from "../icons";
import { C, R, SHADOW } from "../theme";
import ClinicMap from "./map/ClinicMap";
import type { MapState } from "./map/mapHtml";
import { Art, Btn, Card, CardTitle, Chip, SectionHead, T, open, type Layout } from "./ui";

export type Filter = "best" | "all" | ClinicType;
export type Place = { lat: number; lng: number; label: string; shared: boolean };

type Props = {
  layout: Layout;
  conditionKey: ConditionKey;
  location: Place;
  areaLabel: string;
  onArea: (label: string) => void;
  onLocate: () => void;
  mapNote: { icon: IconName; text: string };
  filter: Filter;
  onFilter: (f: Filter) => void;
  activeClinic: string | null;
  focusSeq: number;
  onFocusClinic: (id: string) => void;
};

function haversine(a: { lat: number; lng: number }, b: { lat: number; lng: number }) {
  const R = 6371, r = Math.PI / 180;
  const dLat = (b.lat - a.lat) * r, dLng = (b.lng - a.lng) * r;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

// The single best match is the nearest clinic of the type that suits this result best;
// everything else follows by distance.
function clinicsForView(key: ConditionKey, loc: Place, filter: Filter) {
  const types = CONDITIONS[key].clinicTypes;
  let list = CLINICS.map((c) => ({ ...c, km: haversine(loc, c), best: false })).sort((a, b) => a.km - b.km);
  const bestId = list.find((c) => c.type === types[0])?.id;
  list.forEach((c) => (c.best = c.id === bestId));
  if (filter === "best") list = list.filter((c) => types.includes(c.type));
  else if (filter !== "all") list = list.filter((c) => c.type === filter);
  return list.sort((a, b) => (Number(b.best) - Number(a.best)) || (a.km - b.km)).slice(0, 5);
}

export function ActSection(p: Props) {
  const { layout } = p;
  const [areaOpen, setAreaOpen] = useState(false);
  const c = CONDITIONS[p.conditionKey];
  const list = useMemo(() => clinicsForView(p.conditionKey, p.location, p.filter), [p.conditionKey, p.location, p.filter]);
  const far = list.length > 0 && list.every((x) => x.km > 8);
  const mapState: MapState = useMemo(() => ({
    you: { lat: p.location.lat, lng: p.location.lng },
    clinics: list.map((x, i) => ({ id: x.id, n: i + 1, name: x.name, addr: x.addr, km: x.km, best: x.best, lat: x.lat, lng: x.lng })),
    active: p.activeClinic,
  }), [list, p.location]);
  const twoCol = layout.wide;
  const filters: { k: Filter; label: string; icon: IconName }[] = [
    { k: "best", label: "Best for you", icon: "sparkles" },
    { k: "all", label: "All", icon: "list" },
    ...(Object.entries(CLINIC_TYPES) as [ClinicType, (typeof CLINIC_TYPES)[ClinicType]][]).map(([k, t]) => ({ k, label: t.label, icon: t.icon })),
  ];

  const mapCard = (
    <Card narrow={layout.narrow} style={{ gap: 16, flex: twoCol ? 1 : undefined }}>
      <View style={s.mapBar}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ flexGrow: 0, margin: -6 }} contentContainerStyle={{ gap: 8, padding: 6 }}>
          {filters.map((f) => <Chip key={f.k} label={f.label} icon={f.icon} small dark active={p.filter === f.k} onPress={() => p.onFilter(f.k)} />)}
        </ScrollView>
        <View style={s.mapTools}>
          <Pressable onPress={() => setAreaOpen(true)} style={s.areaPick} accessibilityRole="button" accessibilityLabel={`Choose your area, now ${p.areaLabel}`}>
            <Icon name="map-pin" size={16} />
            <T w={500} size={14}>Area</T>
            <View style={s.areaValue}><T size={14}>{p.areaLabel}</T><Icon name="chevron-down" size={14} /></View>
          </Pressable>
          <Btn label="Use my location" icon="locate-fixed" variant="soft" size="sm" onPress={p.onLocate} />
        </View>
      </View>
      <ClinicMap state={mapState} focusId={p.activeClinic} focusSeq={p.focusSeq} onPick={p.onFocusClinic}
        height={layout.narrow ? 340 : 520} />
      <View style={s.note}><Icon name={p.mapNote.icon} size={16} color={C.ink3} /><T size={14} color={C.ink3} style={{ flex: 1 }}>{p.mapNote.text}</T></View>
    </Card>
  );

  const clinicCard = (
    <Card narrow={layout.narrow} style={{ flex: twoCol ? 1 : undefined }}>
      <CardTitle>Nearby clinics</CardTitle>
      {far && (
        <View style={s.far}><Icon name="map-pin-off" size={18} /><T size={14} style={{ flex: 1 }}>No sample clinics near this area yet.</T></View>
      )}
      <View style={{ gap: 14 }}>
        {list.map((x, i) => {
          const t = CLINIC_TYPES[x.type];
          const dest = encodeURIComponent(`${x.sample ? "CHAS dental clinic " : x.name + " "}${x.addr} Singapore`);
          const active = p.activeClinic === x.id;
          return (
            <Pressable key={x.id} onPress={() => p.onFocusClinic(x.id)} accessibilityLabel={`${x.name}, ${x.km.toFixed(1)} kilometres. Show on map`}
              style={({ hovered }: any) => [s.clinic, (hovered || active) && { transform: [{ translateY: -2 }] },
                active && { boxShadow: `0 0 0 3px ${C.u3}, ${SHADOW.raiseSm}` }]}>
              <View style={[s.clinicNum, x.best && { backgroundColor: C.peach2 }]}>
                <T w={600} color={x.best ? C.ink : "#fff"}>{i + 1}</T>
              </View>
              <View style={{ flex: 1, minWidth: 0 }}>
                <T w={500} style={{ lineHeight: 21 }}>{x.name}</T>
                <View style={s.clinicMeta}>
                  <T size={13} color={C.ink3}>{x.km.toFixed(1)} km away</T>
                  {x.best && <View style={[s.tag, { backgroundColor: C.peach }]}><T size={13} w={500}>Best match</T></View>}
                  <View style={s.tag}><T size={13} color={C.ink2}>{t.tag}</T></View>
                </View>
                {x.best && (
                  <T size={14} style={{ marginTop: 6 }}>
                    Closest {x.type === "chas" ? t.label : t.label.toLowerCase()} for {c.procedure.toLowerCase()}
                  </T>
                )}
                <View style={s.clinicActions}>
                  <Btn label="Directions" icon="navigation" size="sm" variant={x.best ? "dark" : "soft"}
                    href={`https://www.google.com/maps/dir/?api=1&destination=${dest}`} />
                  {x.type === "polyclinic" && <LinkText label="Book via HealthHub" url="https://www.healthhub.sg/" />}
                  {x.sample && <LinkText label="Find on CHAS" url="https://www.chas.sg/clinic-locator" />}
                </View>
              </View>
            </Pressable>
          );
        })}
      </View>
      <Pressable onPress={() => open("https://www.chas.sg/clinic-locator")} accessibilityRole="link" style={s.more}>
        <Icon name="search" /><T w={500}>All CHAS dental clinics</T><Icon name="external-link" />
      </Pressable>
    </Card>
  );

  return (
    <View>
      <SectionHead icon="map-pin" title="Get it checked" narrow={layout.narrow} />
      <View style={{ flexDirection: twoCol ? "row" : "column", gap: 32 }}>
        {twoCol ? <View style={{ flex: 1.5, minWidth: 0 }}>{mapCard}</View> : mapCard}
        {twoCol ? <View style={{ flex: 1, minWidth: 0 }}>{clinicCard}</View> : clinicCard}
      </View>

      <Card narrow={layout.narrow} style={{ marginTop: 32 }}>
        <CardTitle>Home care that helps</CardTitle>
        <View style={s.products}>
          {c.products.map((pr) => {
            const item = PRODUCTS[pr.id];
            const q = encodeURIComponent(item.search + " Singapore");
            return (
              <Pressable key={pr.id} onPress={() => open(`https://www.google.com/search?tbm=shop&q=${q}`)} accessibilityRole="link"
                style={({ hovered }: any) => [s.product, { flexBasis: layout.narrow ? "100%" : 240 }, hovered && { transform: [{ translateY: -2 }] }]}>
                <View style={s.productArt}><Art xml={product(pr.id)} style={{ width: 58, height: 58 }} /></View>
                <View style={{ flex: 1 }}>
                  <T w={500} style={{ lineHeight: 21 }}>{item.name}</T>
                  <T size={14} color={C.ink2} style={{ marginTop: 2 }}>{pr.why}</T>
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 4, marginTop: 6 }}>
                    <T w={500} size={14}>Find in shops</T><Icon name="arrow-up-right" size={16} />
                  </View>
                </View>
              </Pressable>
            );
          })}
        </View>
      </Card>

      <View style={s.soon}>
        <Icon name="wallet" color={C.ink2} />
        <T size={15} color={C.ink2}>Cost estimate after CHAS & MediSave</T>
        <View style={s.soonPill}><T w={500} size={13}>Coming soon</T></View>
      </View>

      <Modal visible={areaOpen} transparent animationType="fade" onRequestClose={() => setAreaOpen(false)}>
        <Pressable style={s.backdrop} onPress={() => setAreaOpen(false)}>
          <Pressable style={s.sheet} onPress={() => {}}>
            <T w={500} size={20} style={{ marginBottom: 8 }}>Choose your area</T>
            <ScrollView style={{ maxHeight: 420 }}>
              {AREAS.map((a) => (
                <Pressable key={a.label} onPress={() => { p.onArea(a.label); setAreaOpen(false); }}
                  style={[s.areaRow, a.label === p.areaLabel && { backgroundColor: C.peach }]}>
                  <T w={a.label === p.areaLabel ? 500 : 400}>{a.label}</T>
                  {a.label === p.areaLabel && <Icon name="check" />}
                </Pressable>
              ))}
            </ScrollView>
          </Pressable>
        </Pressable>
      </Modal>
    </View>
  );
}

function LinkText({ label, url }: { label: string; url: string }) {
  return (
    <Pressable onPress={() => open(url)} accessibilityRole="link" style={{ flexDirection: "row", alignItems: "center", gap: 4, paddingVertical: 8, paddingHorizontal: 4 }}>
      <T w={500} size={14}>{label}</T><Icon name="arrow-up-right" size={16} />
    </Pressable>
  );
}

const s = StyleSheet.create({
  mapBar: { gap: 12 },
  mapTools: { flexDirection: "row", flexWrap: "wrap", gap: 10 },
  areaPick: {
    flexDirection: "row", alignItems: "center", gap: 8, paddingVertical: 6, paddingRight: 6, paddingLeft: 14,
    borderRadius: 999, backgroundColor: C.surface, boxShadow: SHADOW.raiseSm,
  },
  areaValue: { flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: C.surface2, borderRadius: 999, paddingVertical: 6, paddingHorizontal: 10 },
  note: { flexDirection: "row", alignItems: "center", gap: 8 },
  far: { flexDirection: "row", alignItems: "center", gap: 10, marginBottom: 14, padding: 12, borderRadius: R.sm, backgroundColor: C.peach },
  clinic: { flexDirection: "row", alignItems: "flex-start", gap: 12, padding: 14, borderRadius: R.sm, backgroundColor: C.surface, boxShadow: SHADOW.raiseSm },
  clinicNum: { width: 40, height: 40, borderRadius: 14, alignItems: "center", justifyContent: "center", backgroundColor: C.dark },
  clinicMeta: { flexDirection: "row", flexWrap: "wrap", alignItems: "center", columnGap: 10, rowGap: 6, marginTop: 4 },
  tag: { paddingVertical: 1, paddingHorizontal: 9, borderRadius: 999, backgroundColor: C.surface2 },
  clinicActions: { flexDirection: "row", flexWrap: "wrap", alignItems: "center", gap: 8, marginTop: 10 },
  more: { flexDirection: "row", alignItems: "center", gap: 8, marginTop: 18 },
  products: { flexDirection: "row", flexWrap: "wrap", gap: 18 },
  product: {
    flexGrow: 1, flexDirection: "row", alignItems: "center", gap: 14, padding: 14,
    borderRadius: R.md, backgroundColor: C.surface, boxShadow: SHADOW.raiseSm,
  },
  productArt: { width: 76, height: 76, borderRadius: 20, alignItems: "center", justifyContent: "center", backgroundColor: C.peach },
  soon: { flexDirection: "row", flexWrap: "wrap", alignItems: "center", gap: 10, marginTop: 24 },
  soonPill: { paddingVertical: 4, paddingHorizontal: 12, borderRadius: 999, backgroundColor: C.peach },
  backdrop: { flex: 1, backgroundColor: "rgba(18,17,16,0.45)", alignItems: "center", justifyContent: "center", padding: 16 },
  sheet: { width: "100%", maxWidth: 380, backgroundColor: C.surface, borderRadius: R.lg, padding: 20, boxShadow: SHADOW.raise },
  areaRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingVertical: 12, paddingHorizontal: 14, borderRadius: R.sm },
});
