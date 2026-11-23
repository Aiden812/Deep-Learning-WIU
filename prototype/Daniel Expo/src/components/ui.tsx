/* Building blocks matching prototype/Daniel/css/styles.css (.card, .btn, .chip, .switch, ...). */
import { createContext, useContext, type ReactNode } from "react";
import {
  Linking, Platform, Pressable, StyleSheet, Text, View, useWindowDimensions,
  type StyleProp, type TextProps, type TextStyle, type ViewStyle,
} from "react-native";
import { SvgXml } from "react-native-svg";
import { Icon, type IconName } from "../icons";
import { BP, C, FONT, R, SHADOW, type Weight } from "../theme";

/* ---------- Layout breakpoints ---------- */
export type Layout = { width: number; wide: boolean; narrow: boolean };
export function useLayout(): Layout {
  const { width } = useWindowDimensions();
  return { width, wide: width > BP.wide, narrow: width <= BP.narrow };
}
const NarrowCtx = createContext(false);
export const NarrowProvider = NarrowCtx.Provider;

/* ---------- Text ---------- */
export function T({ w = 400, size, color = C.ink, style, ...rest }:
  TextProps & { w?: Weight; size?: number; color?: string; style?: StyleProp<TextStyle> }) {
  const narrow = useContext(NarrowCtx);
  const fs = size ?? (narrow ? 16 : 17);
  return <Text {...rest} style={[{ fontFamily: FONT[w], fontSize: fs, lineHeight: Math.round(fs * 1.5), color }, style]} />;
}

/* ---------- Links ---------- */
export const open = (url: string) => { Linking.openURL(url).catch(() => {}); };

/* ---------- Card ---------- */
export function Card({ children, style, narrow }: { children: ReactNode; style?: StyleProp<ViewStyle>; narrow?: boolean }) {
  return <View style={[s.card, { padding: narrow ? 20 : 30 }, style]}>{children}</View>;
}

export function CardTitle({ children }: { children: ReactNode }) {
  return <T w={500} size={22} style={{ letterSpacing: -0.2, marginBottom: 16, lineHeight: 28 }}>{children}</T>;
}

/* ---------- Section heading ---------- */
export function SectionHead({ icon, title, narrow }: { icon: IconName; title: string; narrow: boolean }) {
  const box = narrow ? 52 : 64;
  return (
    <View style={s.sectionHead}>
      <View style={[s.sectionIcon, { width: box, height: box, borderRadius: narrow ? 18 : 22 }]}>
        <Icon name={icon} size={narrow ? 24 : 28} />
      </View>
      <T w={400} size={narrow ? 30 : 44} style={{ letterSpacing: -1.2, lineHeight: narrow ? 34 : 48, flexShrink: 1 }}>{title}</T>
    </View>
  );
}

/* ---------- Buttons ---------- */
type BtnVariant = "dark" | "light" | "soft" | "muted";
export function Btn({ label, icon, onPress, variant = "dark", size = "md", block, disabled, href, style }: {
  label: string; icon?: IconName; onPress?: () => void; variant?: BtnVariant; size?: "sm" | "md" | "lg";
  block?: boolean; disabled?: boolean; href?: string; style?: StyleProp<ViewStyle>;
}) {
  const v = VARIANTS[disabled && variant === "dark" ? "muted" : variant];
  const pad = size === "sm" ? [10, 16, 15] : size === "lg" ? [18, 26, 18] : [14, 22, 16];
  const press = href ? () => open(href) : onPress;
  return (
    <Pressable
      accessibilityRole={href ? "link" : "button"}
      accessibilityState={{ disabled: !!disabled }}
      onPress={disabled ? undefined : press}
      style={({ pressed, hovered }: any) => [
        s.btn, { paddingVertical: pad[0], paddingHorizontal: pad[1], backgroundColor: v.bg },
        v.shadow && { boxShadow: pressed ? SHADOW.insetSm : v.shadow },
        hovered && !disabled && v.hover && { backgroundColor: v.hover },
        pressed && !disabled && { transform: [{ scale: 0.98 }] },
        disabled && variant !== "dark" && { opacity: 0.55 },
        block && { alignSelf: "stretch" },
        style,
      ]}
    >
      {icon && <Icon name={icon} size={pad[2] + 2} color={v.fg} />}
      <T w={500} size={pad[2]} color={v.fg} numberOfLines={1}>{label}</T>
    </Pressable>
  );
}
const VARIANTS: Record<BtnVariant, { bg: string; fg: string; hover?: string; shadow?: string }> = {
  dark: { bg: C.dark, fg: C.onDark, hover: C.dark3 },
  light: { bg: C.onDark, fg: C.ink, hover: "#ffffff" },
  soft: { bg: C.surface, fg: C.ink, shadow: SHADOW.raiseSm },
  muted: { bg: C.surface, fg: C.ink3, shadow: SHADOW.raiseSm },
};

/* ---------- Chips ---------- */
export function Chip({ label, icon, active, onPress, dark, small }: {
  label: string; icon?: IconName; active?: boolean; onPress?: () => void; dark?: boolean; small?: boolean;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ selected: !!active }}
      onPress={onPress}
      style={[
        s.chip, small && { paddingVertical: 8, paddingHorizontal: 14 },
        dark && active ? { backgroundColor: C.dark, boxShadow: "none" } : { boxShadow: active ? SHADOW.insetSm : SHADOW.raiseSm },
      ]}
    >
      {icon && <Icon name={icon} size={16} color={dark && active ? C.onDark : C.ink} />}
      <T w={500} size={small ? 14 : 15} color={dark && active ? C.onDark : C.ink} numberOfLines={1}>{label}</T>
    </Pressable>
  );
}

export function Dot({ color, size = 10 }: { color: string; size?: number }) {
  return <View style={{ width: size, height: size, borderRadius: size / 2, backgroundColor: color }} />;
}

/* ---------- Switch (the consent toggle) ---------- */
export function Switch({ on, nudged }: { on: boolean; nudged?: boolean }) {
  return (
    <View style={[s.switch, { backgroundColor: on ? C.u1 : C.surface2 },
      { boxShadow: nudged ? `0 0 0 4px ${C.u3}, ${SHADOW.insetSm}` : SHADOW.insetSm }]}>
      <View style={[s.knob, { transform: [{ translateX: on ? 22 : 0 }] }]} />
    </View>
  );
}

/* ---------- SVG art box ---------- */
export function Art({ xml, style }: { xml: string; style?: StyleProp<ViewStyle> }) {
  return (
    <View style={style} pointerEvents="none">
      <SvgXml xml={xml} width="100%" height="100%" />
    </View>
  );
}

export const isWeb = Platform.OS === "web";

const s = StyleSheet.create({
  card: { backgroundColor: C.surface, borderRadius: R.lg, boxShadow: SHADOW.raise },
  sectionHead: { flexDirection: "row", alignItems: "center", gap: 18, marginBottom: 28 },
  sectionIcon: { alignItems: "center", justifyContent: "center", backgroundColor: C.surface, boxShadow: SHADOW.raiseSm },
  btn: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 10, borderRadius: 999, alignSelf: "flex-start" },
  chip: { flexDirection: "row", alignItems: "center", gap: 8, paddingVertical: 10, paddingHorizontal: 16, borderRadius: 999, backgroundColor: C.surface },
  switch: { width: 52, height: 30, borderRadius: 999, padding: 4 },
  knob: { width: 22, height: 22, borderRadius: 11, backgroundColor: C.surface, boxShadow: "2px 2px 5px rgba(0,0,0,0.18)" },
});
