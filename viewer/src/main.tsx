import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import type { MoiraiData } from "./types";

declare global {
  interface Window {
    __MOIRAI_DATA__: MoiraiData | string;
  }
}

function loadData(): MoiraiData {
  const raw = window.__MOIRAI_DATA__;
  if (typeof raw === "string") {
    return JSON.parse(raw) as MoiraiData;
  }
  return raw;
}

const data = loadData();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App data={data} />
  </StrictMode>,
);
