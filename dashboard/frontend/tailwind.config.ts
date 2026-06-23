import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        "group-a":        "#3a5f83",
        "group-a-light":  "#e8eff6",
        "group-b":        "#4a7c6f",
        "group-b-light":  "#e8f4f0",
        brand: {
          DEFAULT: "#1a6b5a",
          hover:   "#155749",
          light:   "#e8f5f1",
        },
        surface:        "#ffffff",
        base:           "#f7f7f4",
        sidebar:        "#fafaf8",
        "border-light": "#eeeeea",
      },
      borderRadius: {
        card: "10px",
      },
      boxShadow: {
        card:    "0 1px 4px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
        "card-md": "0 2px 8px rgba(0,0,0,0.07), 0 1px 3px rgba(0,0,0,0.04)",
        drawer:  "6px 0 28px rgba(0,0,0,0.10)",
      },
    },
  },
  plugins: [],
};

export default config;
