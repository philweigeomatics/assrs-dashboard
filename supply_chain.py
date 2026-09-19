"""
supply_chain.py — generating a company's supply-chain graph, without Streamlit.

Carved out of supply_chain_ui.py so the API can call it. That module imports
streamlit.components.v1 at module level for the D3 renderer, which is fine in
the Streamlit app and impossible in the Cloud Run container — the prompt and
the DeepSeek call have no such dependency and now live here. supply_chain_ui
imports them back, so the Streamlit page is unchanged.

The graph shape is products to macro sectors, NOT the generic nodes/edges a
reader might assume:

    {ticker, company_name, products: [...], macro_sectors: [...],
     links: [{source, target}]}

where every source names a product and every target names a sector, both
matched by their exact string.

The prompt is per market. The schema is identical everywhere — the graph a
chart draws must not change shape by exchange — but the sourcing instruction
cannot be: telling a model to verify a US company against cninfo.com.cn is
telling it to consult a database that does not contain the company, and what
comes back is invention rather than recall. So the last rule is swapped for
the filings that actually exist in that market, and everything above it stays
byte-identical.
"""

from __future__ import annotations

import ai_client

_PROMPT = """\
You are an elite quantitative supply chain analyst specialising in {speciality}.
Your task is to map the core physical products and the downstream macroeconomic sectors \
for a given stock.

Return ONLY a raw, valid JSON object. Do not include any conversational text, \
explanations, or markdown code fences. Start your response directly with { and end with }.

Follow this exact schema:
{
  "ticker": "String — the ticker provided",
  "company_name": "String — 公司中文名称 / Official English or Pinyin name  (Chinese first, then ' / ', then English)",
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

Base rules:
1. products must be specific, tangible items or services the company manufactures.
2. macro_sectors must be broad downstream industries that consume those products.
3. links may only connect a product to a macro_sector — never product→product.
4. source and target values must be spelled exactly as they appear in their arrays.
5. Every value (not key) must contain both English and Chinese separated by ' / '.

CRITICAL RULES (strictly enforced):
1. Every macro_sector in the "macro_sectors" array MUST appear as a "target" in at \
least one link. Do not list a sector unless it is linked to at least one product.
2. Every product in the "products" array SHOULD appear as a "source" in at least one \
link. Prefer to map every product; only omit a product if it has absolutely no clear \
downstream sector connection.
3. Base the mapping on the company's ACTUAL downstream customers — if a sector appears \
in the company's segment revenue breakdown or its regulatory filings, include it. \
If not verified, exclude it.
4. {sources}
"""

#: What to consult, per market. Every value still carries English and Chinese,
#: because the app is read in Chinese whatever the listing is.
_MARKETS = {
    "CN": {
        "speciality": "Chinese A-Shares",
        "label": "Chinese A-share",
        "sources": ("For A-share companies, consult: annual report segment revenue "
                    "breakdown, investor Q&A on cninfo.com.cn, or industry "
                    "association reports as your primary source of truth."),
    },
    "US": {
        "speciality": "US-listed equities",
        "label": "US-listed",
        "sources": ("For US-listed companies, consult: the segment disclosures in "
                    "the 10-K and 10-Q, the customer-concentration note, and "
                    "investor-day materials as your primary source of truth. "
                    "Do NOT cite Chinese regulatory sources for a US filer."),
    },
    "CA": {
        "speciality": "Canadian-listed equities (TSX / TSXV)",
        "label": "Canadian-listed",
        "sources": ("For Canadian-listed companies, consult: the Annual "
                    "Information Form and MD&A filed on SEDAR+, the segmented "
                    "information note in the financial statements, and investor "
                    "presentations as your primary source of truth. "
                    "Do NOT cite Chinese regulatory sources for a Canadian filer."),
    },
}


def _prompt(market: str) -> tuple[str, str]:
    """
    (system prompt, market label) for one market, defaulting to A-shares.

    Substituted with str.replace, not str.format: the prompt embeds a JSON
    schema, so every brace in it would have to be doubled to survive format(),
    which makes the schema unreadable and one missed brace silently breaks the
    whole prompt.
    """
    spec = _MARKETS.get(market, _MARKETS["CN"])
    return (_PROMPT.replace("{speciality}", spec["speciality"])
                   .replace("{sources}", spec["sources"]), spec["label"])


def generate_supply_chain_graph(ticker: str, company_name: str,
                                market: str = "CN") -> dict:
    """
    Call DeepSeek to produce a supply chain knowledge graph for *ticker*.

    `market` is the two-letter code from markets.split(). Returns the parsed
    graph dict; raises RuntimeError with a user-friendly message on failure.
    """
    system, label = _prompt(market)
    user_msg = (
        f"Generate the supply chain knowledge graph for this {label} company:\n"
        f"Ticker: {ticker}\n"
        f"Company Name: {company_name}"
    )
    return ai_client.call_json(
        system, user_msg,
        # This is recall + formatting (products, sectors, links from what the
        # model already knows about the company), not multi-step reasoning, so
        # reasoning_effort="low" is appropriate and directly reduces how much
        # of max_tokens the thinking trace burns before the JSON is written —
        # that burn, not a too-small max_tokens, was the actual cause of the
        # "ran out" failures on obscure/thinly-documented tickers.
        # max_tokens raised as a safety margin on top of that fix, not instead
        # of it — an unfamiliar ticker can still make the model reason for a
        # while even at low effort.
        max_tokens=16000,
        temperature=0.3,
        reasoning_effort="low",
    )
