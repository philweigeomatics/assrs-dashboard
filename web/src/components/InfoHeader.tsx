/**
 * Everything the Streamlit page spread over two rows of big metric cards —
 * price, daily_basic and the six status cards — in one compact strip.
 *
 * The signal chips carry the same flags as the Streamlit cards (squeeze,
 * accumulation, bull, bear, 箱体, regime + ADX), read from ta_payload's
 * build_signals, which also fixes the MACD金叉/死叉 column names the Streamlit
 * header had wrong.
 */

import type { Analysis } from "../lib/types";
import { fixed, moveClass, signed, yi } from "../lib/format";

function Stat({ k, v, cls = "" }: { k: string; v: string; cls?: string }) {
  return (
    <div className="flex flex-col leading-tight">
      <span className="label">{k}</span>
      <span className={`font-mono tnum text-[14px] ${cls}`}>{v}</span>
    </div>
  );
}

type Tone = "up" | "down" | "warn" | "calm" | "info";

const TONE: Record<Tone, string> = {
  up: "bg-[#fdecec] text-up",          // bullish, Chinese red
  down: "bg-[#e8f5ec] text-down",      // bearish, green
  warn: "bg-[#fff3e0] text-brand-ink",
  calm: "bg-elevated text-ink-mute",
  info: "bg-[#e6f0fb] text-cyan",
};

function Chip({ label, value, tone, title }: { label: string; value: string; tone: Tone; title?: string }) {
  return (
    <span title={title} className={`inline-flex items-center gap-1.5 rounded-full px-2.5 h-7 text-[12.5px] ${TONE[tone]}`}>
      <span className="opacity-70">{label}</span>
      <span className="font-semibold">{value}</span>
    </span>
  );
}

export function InfoHeader({ data }: { data: Analysis }) {
  const h = data.header;
  const sg = data.signals;
  const box = sg.box;

  const regimeTone: Tone =
    sg.regime === "High Volatility" ? "down" : sg.regime === "Low Volatility" ? "info" : "calm";
  const regimeCn =
    sg.regime === "High Volatility" ? "高波动" : sg.regime === "Low Volatility" ? "低波动" : "正常";

  let boxChip: { value: string; tone: Tone; title?: string } = { value: "无", tone: "calm" };
  if (box) {
    if (box.kind === "BOX") {
      const tone: Tone =
        box.status === "BREAKOUT" || box.status === "AT_SUPPORT" ? "up"
          : box.status === "BREAKDOWN" || box.status === "AT_RESISTANCE" ? "down"
            : "warn";
      boxChip = {
        value: `${box.status_cn} ${fixed(box.bot)}–${fixed(box.top)}`,
        tone,
        title: `幅度 ${fixed(box.height_pct, 1)}% · 位置 ${box.position != null ? Math.round(box.position * 100) : "—"}% · 触及 上${box.touches_top}/下${box.touches_bot} · ${box.sessions}日 · 质量 ${fixed(box.quality)}`,
      };
    } else {
      boxChip = { value: box.status_cn, tone: "calm", title: `净漂移 ${signed(box.drift_pct, 1, "%")} · R² ${fixed(box.r2)} · 不是箱体` };
    }
  }

  return (
    <div className="card px-4 py-3 flex flex-col gap-2.5">
      <div className="flex flex-wrap items-end gap-x-6 gap-y-2">
        <div className="flex items-baseline gap-2 mr-2">
          <span className="text-[20px] font-semibold">{data.name}</span>
          <span className="font-mono tnum text-ink-mute">{data.ticker}</span>
        </div>
        <div className="flex items-baseline gap-2">
          <span className={`font-mono tnum text-[22px] font-semibold ${moveClass(h.change_pct)}`}>{fixed(h.close)}</span>
          <span className={`font-mono tnum text-[14px] ${moveClass(h.change_pct)}`}>{signed(h.change_pct, 2, "%")}</span>
          <span className="label">{h.date}</span>
        </div>
        <Stat k="总市值" v={yi(h.total_mv_yi)} />
        <Stat k="流通市值" v={yi(h.circ_mv_yi)} />
        <Stat k="PE(TTM)" v={fixed(h.pe_ttm)} />
        <Stat k="PB" v={fixed(h.pb)} />
        <Stat k="换手率" v={h.turnover_rate == null ? "—" : `${fixed(h.turnover_rate)}%`} />
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <Chip label="挤压" value={sg.squeeze ? "收紧" : "宽松"} tone={sg.squeeze ? "warn" : "calm"} title="布林带宽度处于低分位，波动收敛" />
        <Chip label="吸筹" value={sg.accumulation ? "进行中" : "无"} tone={sg.accumulation ? "up" : "calm"} />
        <Chip label="多头" value={sg.bull.length ? sg.bull.join(" · ") : "无"} tone={sg.bull.length ? "up" : "calm"} />
        <Chip label="空头" value={sg.bear.length ? sg.bear.join(" · ") : "无"} tone={sg.bear.length ? "down" : "calm"} />
        <Chip label="箱体" value={boxChip.value} tone={boxChip.tone} title={boxChip.title} />
        <Chip label="波动状态" value={regimeCn} tone={regimeTone} />
        <Chip label="ADX" value={`${fixed(sg.adx, 1)}${sg.adx_pattern ? ` · ${sg.adx_pattern}` : ""}`} tone="calm" />
      </div>
    </div>
  );
}
