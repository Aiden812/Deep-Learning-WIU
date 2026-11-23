/* Design tokens from prototype/Daniel/css/styles.css: soft neumorphic UI on warm cream, dark hero. */

export const C = {
  bg: "#f1e9e2",
  surface: "#f6f0ea",
  surface2: "#ece2d9",
  peach: "#f6d8c3",
  peach2: "#efbf9f",
  ink: "#171513",
  ink2: "#5d5650",
  ink3: "#6b625b",
  dark: "#121110",
  dark2: "#1f1d1b",
  dark3: "#2c2926",
  onDark: "#f6f0ea",
  onDark2: "#b9b0a8",

  calculus: "#d9a441",
  caries: "#7a4b33",
  discolouration: "#c98b4f",
  ulcer: "#d9576a",
  hypodontia: "#6c84c4",
  normal: "#4f9f7a",
  check: "#c7772e",

  u1: "#4f9f7a",
  u2: "#d9a441",
  u3: "#e07a3f",
  u4: "#d64545",
};

// RN 0.88 (New Architecture) and react-native-web both take CSS box-shadow strings, inset included.
export const SHADOW = {
  raise: "10px 10px 24px rgba(150, 126, 106, 0.22), -10px -10px 24px rgba(255, 255, 255, 0.7)",
  raiseSm: "5px 5px 12px rgba(150, 126, 106, 0.22), -5px -5px 12px rgba(255, 255, 255, 0.75)",
  inset: "inset 5px 5px 12px rgba(150, 126, 106, 0.26), inset -5px -5px 12px rgba(255, 255, 255, 0.85)",
  insetSm: "inset 3px 3px 7px rgba(150, 126, 106, 0.24), inset -3px -3px 7px rgba(255, 255, 255, 0.85)",
};

export const R = { lg: 32, md: 24, sm: 16 };

// Outfit is loaded per weight, so each weight is its own family name.
export const FONT = {
  300: "Outfit_300Light",
  400: "Outfit_400Regular",
  500: "Outfit_500Medium",
  600: "Outfit_600SemiBold",
} as const;
export type Weight = keyof typeof FONT;

// Breakpoints from the site's media queries.
export const BP = { wide: 1080, narrow: 760 };
export const MAXW = 1240;
