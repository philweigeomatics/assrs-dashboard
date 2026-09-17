/**
 * Stock identity, daily_basic figures and the signal summary — the Streamlit
 * page's two rows of metric cards, restacked as a side panel beside the chart.
 *
 * The signal chips carry the same flags as the Streamlit cards (squeeze,
 * accumulation, bull, bear, 箱体, regime, ADX), from ta_payload's
 * build_signals — which also fixes the MACD金叉/死叉 column names the Streamlit
 * header had wrong.
 */

import type { Analysis, MarketCode } from "../lib/types";
import { WatchlistButton } from "./WatchlistButton";
import { dash, fixed, money, moveClass, signed, yi } from "../lib/format";

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-[3px]">
      <span className="text-[12px] text-ink-mute shrink-0">{k}</span>
      <span className="font-mono tnum text-[13px]">{v}</span>
    </div>
  );
}

type Tone = "up" | "down" | "warn" | "calm" | "info";

const TONE: Record<Tone, string> = {
  up: "bg-[#fdecec] text-up",
  down: "bg-[#e8f5ec] text-down",
  warn: "bg-[#fff3e0] text-brand-ink",
  calm: "bg-elevated text-ink-mute",
  info: "bg-[#e6f0fb] text-cyan",
};

function Chip({ label, value, tone, title }: { label: string; value: string; tone: Tone; title?: string }) {
  return (
    <span title={title}
      className={`inline-flex items-center gap-1 rounded-full px-2 py-[3px] text-[11.5px] leading-tight ${TONE[tone]}`}>
      <span className="opacity-70">{label}</span>
      <span className="font-semibold">{value}</span>
    </span>
  );
}

/**
 * Which market, and — the part that actually matters — which way round the
 * colours run.
 *
 * Switching between an A-share and a US chart flips the meaning of every
 * candle on screen, and nothing else on the page says so. A green candle is
 * a good day here and a bad day there; without this you have to remember
 * which stock you are looking at to read the chart at all.
 */
function MarketBadge({ market, upIsRed }: { market: MarketCode; upIsRed: boolean }) {
  const name = market === "CN" ? "A股" : market === "US" ? "美股" : "加股";
  return (
    <span
      title={upIsRed
        ? "中国市场惯例：红色代表上涨，绿色代表下跌"
        : "北美市场惯例：绿色代表上涨，红色代表下跌（与A股相反）"}
      className="ml-auto shrink-0 flex items-center gap-1 rounded-md bg-sunken px-1.5 py-0.5 text-[11px]"
    >
      <span className="text-ink-dim">{name}</span>
      <span className={upIsRed ? "text-up" : "text-down"}>▲{upIsRed ? "红" : "绿"}</span>
      <span className={upIsRed ? "text-down" : "text-up"}>▼{upIsRed ? "绿" : "红"}</span>
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
        title: `幅度 ${fixed(box.height_pct, 1)}% · 位置 ${box.position != null ? Math.round(box.position * 100) : "—"}%`
          + ` · 触及 上${box.touches_top}/下${box.touches_bot} · ${box.sessions}日 · 质量 ${fixed(box.quality)}`,
      };
    } else {
      boxChip = {
        value: box.status_cn, tone: "calm",
        title: `净漂移 ${signed(box.drift_pct, 1, "%")} · R² ${fixed(box.r2)} · 不是箱体`,
      };
    }
  }

  return (
    <section className="card p-3 flex flex-col gap-2">
      <div className="flex items-baseline gap-2">
        <span className="text-[17px] font-semibold truncate">{data.name}</span>
        <span className="font-mono tnum text-[12px] text-ink-mute">{data.ticker}</span>
        <MarketBadge market={data.market} upIsRed={data.up_is_red} />
      </div>
      <WatchlistButton ticker={data.ticker} market={data.market} />

      <div className="flex items-baseline gap-2">
        <span className={`font-mono tnum text-[26px] font-semibold ${moveClass(h.change_pct, data.up_is_red)}`}>
          {fixed(h.close)}
        </span>
        <span className={`font-mono tnum text-[13px] ${moveClass(h.change_pct, data.up_is_red)}`}>
          {signed(h.change_pct, 2, "%")}
        </span>
        <span className="label ml-auto">{h.date}</span>
      </div>

      <div className="border-t border-line pt-1">
        <Stat k="总市值" v={money(h.market_cap, data.currency, data.currency_symbol)} />
        <Stat k="流通市值"
          v={h.circ_mv_yi != null ? yi(h.circ_mv_yi) : dash} />
        <Stat k="PE (TTM)" v={fixed(h.pe_ttm)} />
        <Stat k="PB" v={fixed(h.pb)} />
        <Stat k="换手率" v={h.turnover_rate == null ? "—" : `${fixed(h.turnover_rate)}%`} />
      </div>

      <div className="border-t border-line pt-2 flex flex-wrap gap-1">
        <Chip label="挤压" value={sg.squeeze ? "收紧" : "宽松"} tone={sg.squeeze ? "warn" : "calm"}
          title="布林带宽度处于低分位，波动收敛" />
        <Chip label="吸筹" value={sg.accumulation ? "进行中" : "无"} tone={sg.accumulation ? "up" : "calm"} />
        <Chip label="多头" value={sg.bull.length ? sg.bull.join("·") : "无"} tone={sg.bull.length ? "up" : "calm"} />
        <Chip label="空头" value={sg.bear.length ? sg.bear.join("·") : "无"} tone={sg.bear.length ? "down" : "calm"} />
        <Chip label="箱体" value={boxChip.value} tone={boxChip.tone} title={boxChip.title} />
        <Chip label="波动" value={regimeCn} tone={regimeTone} />
        <Chip label="ADX" value={`${fixed(sg.adx, 1)}${sg.adx_pattern ? ` ${sg.adx_pattern}` : ""}`} tone="calm" />
      </div>
    </section>
  );
}
