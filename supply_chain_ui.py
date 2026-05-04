"""
Supply Chain Knowledge Graph — API + Visualization module.

Responsibilities:
  - Call the DeepSeek API to generate a graph JSON for a given ticker.
  - Render the graph as an interactive D3.js force-directed network inside Streamlit.

Designed to be imported from any Streamlit page (watchlist, stock analysis, etc.).
All DB interactions are delegated to data_manager.
"""

import json

import requests
import streamlit.components.v1 as components

import data_manager  # noqa: F401  (available to callers for convenience)

# ── DeepSeek API config ───────────────────────────────────────────────────────

_ENDPOINT = "https://api.deepseek.com/chat/completions"
_MODEL    = "deepseek-chat"

_SYSTEM_PROMPT = """\
You are an elite quantitative supply chain analyst specialising in Chinese A-Shares.
Your task is to map the core physical products and the downstream macroeconomic sectors \
for a given stock.

Return ONLY a raw, valid JSON object. Do not include any conversational text, \
explanations, or markdown code fences. Start your response directly with { and end with }.

Follow this exact schema:
{
  "ticker": "String — the ticker provided",
  "company_name": "String — official English or Pinyin name / 公司中文名称  (separated by ' / ')",
  "products": [
    "Array of 3–6 strings, each formatted as: 'English Name / 中文名称'",
    "Example: 'Glass Fiber / 玻璃纤维'"
  ],
  "macro_sectors": [
    "Array of 3–6 strings, each formatted as: 'English Name / 行业中文名称'",
    "Example: 'Electric Vehicles / 新能源汽车'"
  ],
  "links": [
    {"source": "must match a string in products exactly", "target": "must match a string in macro_sectors exactly"}
  ]
}

Rules:
1. products must be specific, tangible items or services the company manufactures.
2. macro_sectors must be broad downstream industries that consume those products.
3. links may only connect a product to a macro_sector — never product→product.
4. source and target values must be spelled exactly as they appear in their arrays.
5. Every value (not key) must contain both English and Chinese separated by ' / '.
"""


def _deepseek_api_key() -> str:
    from api_config import _get_secret
    return _get_secret("DEEPSEEK_API_KEY")


def generate_supply_chain_graph(ticker: str, company_name: str) -> dict:
    """
    Call DeepSeek to produce a supply chain knowledge graph for *ticker*.

    Returns the parsed graph dict.
    Raises RuntimeError with a user-friendly message on any failure.
    """
    try:
        api_key = _deepseek_api_key()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    user_msg = (
        f"Generate the supply chain knowledge graph for this Chinese A-share company:\n"
        f"Ticker: {ticker}\n"
        f"Company Name: {company_name}"
    )

    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 1200,
    }

    try:
        resp = requests.post(
            _ENDPOINT,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            timeout=45,
        )
        resp.raise_for_status()
    except requests.Timeout:
        raise RuntimeError("DeepSeek API timed out after 45 s. Try again.")
    except requests.RequestException as exc:
        raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc

    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip accidental markdown fences that some models still produce
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) >= 2 else raw

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"DeepSeek returned invalid JSON ({exc}). "
            f"Preview: {raw[:200]}"
        ) from exc


# ── D3.js Force-Directed Graph ────────────────────────────────────────────────

_D3_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0e1117;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    overflow: hidden;
  }
  svg { display: block; width: 100vw; height: 100vh; }
  #tooltip {
    position: fixed;
    background: #1f2937;
    color: #f9fafb;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    pointer-events: none;
    display: none;
    max-width: 260px;
    border: 1px solid #374151;
    line-height: 1.6;
    z-index: 100;
  }
</style>
</head>
<body>
<div id="tooltip"></div>
<svg id="graph">
  <defs>
    <!-- Arrow for hub→product spokes -->
    <marker id="arrow-hub" viewBox="0 -4 10 8" refX="52" refY="0"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,-4L10,0L0,4" fill="#a16207"/>
    </marker>
    <!-- Arrow for product→sector links -->
    <marker id="arrow-sector" viewBox="0 -4 10 8" refX="55" refY="0"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,-4L10,0L0,4" fill="#374151"/>
    </marker>
  </defs>
</svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const DATA = __GRAPH_JSON__;

const W = window.innerWidth;
const H = window.innerHeight;
const svg = d3.select("#graph");
const g   = svg.append("g");

// ── Zoom / pan ──────────────────────────────────────────────────────────────
svg.call(
  d3.zoom().scaleExtent([0.15, 5])
    .on("zoom", e => g.attr("transform", e.transform))
);

// ── Build nodes ─────────────────────────────────────────────────────────────
// Company hub is pinned to centre; products and sectors orbit around it.
const COMPANY_ID = DATA.company_name;

const nodes = [
  { id: COMPANY_ID, type: "company", fx: W / 2, fy: H / 2 },
  ...DATA.products.map(id      => ({ id, type: "product" })),
  ...DATA.macro_sectors.map(id => ({ id, type: "sector"  })),
];

// Hub → product spokes  +  product → sector links
const links = [
  ...DATA.products.map(p  => ({ source: COMPANY_ID, target: p,        kind: "hub"    })),
  ...DATA.links.map(l     => ({ source: l.source,   target: l.target, kind: "sector" })),
];

// ── Force simulation ────────────────────────────────────────────────────────
const sim = d3.forceSimulation(nodes)
  .force("link",      d3.forceLink(links).id(d => d.id)
                        .distance(d => d.kind === "hub" ? 200 : 220))
  .force("charge",    d3.forceManyBody().strength(-600))
  .force("collision", d3.forceCollide().radius(d => d.type === "company" ? 70 : 75));

// ── Links ───────────────────────────────────────────────────────────────────
const linkEl = g.selectAll(".link")
  .data(links)
  .join("line")
  .attr("stroke",         d => d.kind === "hub" ? "#92400e" : "#374151")
  .attr("stroke-width",   d => d.kind === "hub" ? 1.5 : 2)
  .attr("stroke-opacity", d => d.kind === "hub" ? 0.6  : 0.8)
  .attr("marker-end",     d => d.kind === "hub" ? "url(#arrow-hub)" : "url(#arrow-sector)");

// ── Drag (company hub stays draggable but re-pins after release) ─────────────
const drag = d3.drag()
  .on("start", (e, d) => {
    if (!e.active) sim.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
  })
  .on("drag",  (e, d) => { d.fx = e.x; d.fy = e.y; })
  .on("end",   (e, d) => {
    if (!e.active) sim.alphaTarget(0);
    // Re-pin company to wherever the user dropped it; free the others
    if (d.type !== "company") { d.fx = null; d.fy = null; }
  });

// ── Node groups ─────────────────────────────────────────────────────────────
const nodeEl = g.selectAll(".node")
  .data(nodes)
  .join("g")
  .attr("class", "node")
  .style("cursor", "grab")
  .call(drag);

// Company hub — large gold circle
nodeEl.filter(d => d.type === "company")
  .append("circle")
  .attr("r", 58)
  .attr("fill", "#78350f")
  .attr("stroke", "#fbbf24")
  .attr("stroke-width", 2.5);

// Product — rounded rectangle (blue)
nodeEl.filter(d => d.type === "product")
  .append("rect")
  .attr("width", 152).attr("height", 46)
  .attr("x", -76).attr("y", -23)
  .attr("rx", 9)
  .attr("fill", "#1e40af")
  .attr("stroke", "#60a5fa")
  .attr("stroke-width", 1.5);

// Sector — circle (orange-red)
nodeEl.filter(d => d.type === "sector")
  .append("circle")
  .attr("r", 50)
  .attr("fill", "#7c2d12")
  .attr("stroke", "#fb923c")
  .attr("stroke-width", 1.5);

// ── Labels ──────────────────────────────────────────────────────────────────
function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }

nodeEl.each(function(d) {
  const el = d3.select(this);

  if (d.type === "company") {
    // Company: show ticker on top line, short name below
    const sep = d.id.indexOf(" / ");
    const en  = truncate(sep >= 0 ? d.id.slice(0, sep) : d.id, 14);
    const zh  = sep >= 0 ? truncate(d.id.slice(sep + 3), 7) : "";
    el.append("text")
      .attr("text-anchor", "middle").attr("y", zh ? -8 : 2)
      .attr("dominant-baseline", "middle")
      .style("fill", "#fef3c7").style("font-size", "11px").style("font-weight", "700")
      .style("pointer-events", "none").text(DATA.ticker);
    el.append("text")
      .attr("text-anchor", "middle").attr("y", 7)
      .attr("dominant-baseline", "middle")
      .style("fill", "#fde68a").style("font-size", "10px").style("font-weight", "600")
      .style("pointer-events", "none").text(en);
    if (zh) {
      el.append("text")
        .attr("text-anchor", "middle").attr("y", 21)
        .attr("dominant-baseline", "middle")
        .style("fill", "#fcd34d").style("font-size", "9px")
        .style("pointer-events", "none").text(zh);
    }
    return;
  }

  // Products & sectors: bilingual split
  const sep = d.id.indexOf(" / ");
  const en  = truncate(sep >= 0 ? d.id.slice(0, sep)    : d.id, 18);
  const zh  = sep >= 0          ? truncate(d.id.slice(sep + 3), 8) : "";

  el.append("text")
    .attr("text-anchor", "middle")
    .attr("y", zh ? -5 : 1)
    .attr("dominant-baseline", "middle")
    .style("fill", "#f1f5f9").style("font-size", "10.5px").style("font-weight", "600")
    .style("pointer-events", "none").text(en);
  if (zh) {
    el.append("text")
      .attr("text-anchor", "middle").attr("y", 10)
      .attr("dominant-baseline", "middle")
      .style("fill", "#94a3b8").style("font-size", "9.5px")
      .style("pointer-events", "none").text(zh);
  }
});

// ── Tooltip ─────────────────────────────────────────────────────────────────
const tip = d3.select("#tooltip");
const KIND_LABEL = { company: "🏢 Company 公司", product: "📦 Product 产品", sector: "🏭 Macro Sector 行业" };

nodeEl
  .on("mouseover", (e, d) => {
    tip.style("display", "block")
       .html(`<strong>${d.id}</strong><br><em style="color:#9ca3af">${KIND_LABEL[d.type]}</em>`);
  })
  .on("mousemove", e => {
    tip.style("left", (e.clientX + 14) + "px").style("top", (e.clientY - 36) + "px");
  })
  .on("mouseleave", () => tip.style("display", "none"));

// ── Legend ──────────────────────────────────────────────────────────────────
const lg = svg.append("g").attr("transform", "translate(16,16)");

lg.append("circle").attr("cx", 7).attr("cy", 7).attr("r", 7)
  .attr("fill", "#78350f").attr("stroke", "#fbbf24").attr("stroke-width", 1.5);
lg.append("text").attr("x", 22).attr("y", 11)
  .style("fill", "#fde68a").style("font-size", "12px").text("Company 公司");

lg.append("rect").attr("x", 0).attr("y", 24).attr("width", 14).attr("height", 14).attr("rx", 3)
  .attr("fill", "#1e40af").attr("stroke", "#60a5fa").attr("stroke-width", 1);
lg.append("text").attr("x", 22).attr("y", 35)
  .style("fill", "#d1d5db").style("font-size", "12px").text("Product 产品");

lg.append("circle").attr("cx", 7).attr("cy", 56).attr("r", 7)
  .attr("fill", "#7c2d12").attr("stroke", "#fb923c").attr("stroke-width", 1);
lg.append("text").attr("x", 22).attr("y", 60)
  .style("fill", "#d1d5db").style("font-size", "12px").text("Macro Sector 行业");

lg.append("text").attr("x", 0).attr("y", 78)
  .style("fill", "#4b5563").style("font-size", "10px").text("Drag nodes · Scroll to zoom");

// ── Tick ────────────────────────────────────────────────────────────────────
sim.on("tick", () => {
  linkEl
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  nodeEl.attr("transform", d => `translate(${d.x},${d.y})`);
});
</script>
</body>
</html>"""


def render_supply_chain_graph(graph_data: dict, height: int = 560) -> None:
    """
    Render the supply chain D3.js network graph in the current Streamlit context.

    Safe to call from any page — watchlist, individual stock analysis, etc.

    Args:
        graph_data: The parsed graph dict returned by generate_supply_chain_graph()
                    or data_manager.get_supply_chain_graph().
        height:     Pixel height of the embedded iframe (default 560).
    """
    html = _D3_TEMPLATE.replace(
        "__GRAPH_JSON__",
        json.dumps(graph_data, ensure_ascii=False),
    )
    components.html(html, height=height, scrolling=False)
