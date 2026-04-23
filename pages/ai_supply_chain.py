"""
AI Data Centre Supply Chain Map — AI数据中心供应链图谱
A comprehensive map of China A-share stocks across every layer of the AI data centre supply chain.
Displayed under Portfolio section.
"""

import streamlit as st
import data_manager
import auth_manager

auth_manager.require_login()

# ─────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Supply Chain Page Styles ─────────────────────────────── */
.sc-hero {
    background: linear-gradient(135deg, #0f1117 0%, #1a1e2e 100%);
    border: 1px solid #1e2333;
    border-radius: 12px;
    padding: 2rem 2.5rem 1.5rem;
    margin-bottom: 1.5rem;
}
.sc-hero-title {
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0 0 0.35rem;
    color: #e2e4ed;
    letter-spacing: -0.02em;
}
.sc-hero-title span { color: #4F8EF7; }
.sc-hero-sub {
    color: #7a7f96;
    font-size: 0.88rem;
    line-height: 1.6;
    margin-bottom: 1.25rem;
}
.sc-stats {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
}
.sc-stat { text-align: left; }
.sc-stat-num {
    font-size: 1.5rem;
    font-weight: 800;
    color: #e2e4ed;
    font-family: 'JetBrains Mono', monospace;
}
.sc-stat-label {
    font-size: 0.65rem;
    color: #7a7f96;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Layer Header ──────────────────────────────────────────── */
.sc-layer-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0 0.4rem;
}
.sc-layer-icon {
    font-size: 1.25rem;
    width: 36px;
    height: 36px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
}
.sc-layer-title {
    font-size: 1rem;
    font-weight: 700;
    margin: 0;
}
.sc-layer-desc {
    font-size: 0.78rem;
    color: #7a7f96;
    line-height: 1.6;
    padding: 0.5rem 0.75rem;
    border-left: 3px solid;
    margin-bottom: 0.75rem;
    background: rgba(255,255,255,0.02);
    border-radius: 0 6px 6px 0;
}
.sc-why-label {
    font-weight: 700;
    color: #b0b3c6;
}

/* ── Company Card ──────────────────────────────────────────── */
.sc-company-card {
    background: #0f1117;
    border: 1px solid #1e2333;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    transition: border-color 0.15s;
}
.sc-company-card:hover { border-color: #2d3350; }
.sc-card-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.5rem;
}
.sc-ticker-badge {
    font-family: monospace;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    border: 1px solid;
    flex-shrink: 0;
    letter-spacing: 0.04em;
}
.sc-market-badge {
    font-size: 0.58rem;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    background: #1a1e28;
    color: #7a7f96;
    border: 1px solid #1e2333;
    flex-shrink: 0;
    font-family: monospace;
}
.sc-name-en {
    font-size: 0.82rem;
    font-weight: 700;
    color: #e2e4ed;
    line-height: 1.25;
}
.sc-name-zh {
    font-size: 0.7rem;
    color: #7a7f96;
    font-family: monospace;
}
.sc-product {
    font-size: 0.73rem;
    color: #9498b0;
    line-height: 1.55;
    flex: 1;
}
.sc-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    margin-top: auto;
}
.sc-tag {
    font-size: 0.58rem;
    padding: 0.1rem 0.4rem;
    border-radius: 12px;
    background: #141720;
    color: #7a7f96;
    border: 1px solid #1e2333;
}
.sc-revenue {
    font-size: 0.62rem;
    color: #3d4259;
    font-family: monospace;
}

/* ── Flow connector ────────────────────────────────────────── */
.sc-flow-connector {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    margin: 0.25rem 0;
}
.sc-flow-line {
    height: 1px;
    flex: 1;
    max-width: 80px;
    opacity: 0.25;
}
.sc-flow-label {
    font-size: 0.62rem;
    color: #3d4259;
    font-family: monospace;
    text-align: center;
    letter-spacing: 0.04em;
    padding: 0.2rem 0.6rem;
    border: 1px dashed #1e2333;
    border-radius: 12px;
}
.sc-flow-arrow {
    font-size: 0.9rem;
    opacity: 0.3;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SUPPLY CHAIN DATA
# ─────────────────────────────────────────────────────────────────
SUPPLY_CHAIN = [
    {
        "id": "raw-materials",
        "name": "Layer 1 · Raw Materials & Critical Minerals",
        "short": "原材料 · Raw Materials",
        "color": "#8B5CF6",
        "icon": "⛏️",
        "why": "The foundation of the entire AI supply chain. These companies mine and refine copper, cobalt, rare earths, and fluorochemicals required for every chip, PCB, cable, and cooling system. China controls ~60% of global rare earth processing.",
        "flow_label": "Copper · Silicon · Fluorochemicals → Fabs & PCB makers",
        "companies": [
            {"ticker": "603993", "name": "CMOC Group", "nameZh": "洛阳钼业", "market": "SSE", "revenue": "¥180B+",
             "product": "World's largest cobalt producer; copper mining in DRC & Brazil. Cobalt essential for battery cooling and power storage in data centres.",
             "tags": ["Cobalt", "Copper"]},
            {"ticker": "600362", "name": "Jiangxi Copper", "nameZh": "江西铜业", "market": "SSE", "revenue": "¥520B+",
             "product": "China's largest copper producer. Copper wire and tubing is critical for power distribution and liquid cooling pipes in data centres.",
             "tags": ["Copper"]},
            {"ticker": "603799", "name": "Huayou Cobalt", "nameZh": "华友钴业", "market": "SSE", "revenue": "¥60B+",
             "product": "Major cobalt & nickel refiner. Key battery material supplier also used in precision electronics and heat exchangers.",
             "tags": ["Cobalt", "Nickel"]},
            {"ticker": "000807", "name": "Yunnan Aluminium", "nameZh": "云南铝业", "market": "SZSE", "revenue": "¥25B+",
             "product": "Primary aluminium production. Aluminium heat sinks and structural casings in server chassis and cooling systems.",
             "tags": ["Aluminium"]},
            {"ticker": "000408", "name": "Zangge Mining", "nameZh": "藏格矿业", "market": "SZSE", "revenue": "¥10B+",
             "product": "Lithium and potash mining. Lithium used in high-capacity UPS batteries and power storage for data centres.",
             "tags": ["Lithium"]},
            {"ticker": "600160", "name": "Juhua Group", "nameZh": "巨化股份", "market": "SSE", "revenue": "¥18B+",
             "product": "China's fluorochemical leader. Produces semiconductor-grade HFE coolants — the key fluid in immersion liquid cooling. Verified by TSMC and NVIDIA GB300 ecosystem. Single-ton price >¥1.5M.",
             "tags": ["Fluorochemicals", "Immersion Coolant"]},
        ]
    },
    {
        "id": "semiconductor",
        "name": "Layer 2 · Semiconductors, AI Chips & EDA",
        "short": "半导体 · Semiconductors",
        "color": "#EC4899",
        "icon": "🧪",
        "why": "Without foundries and AI chip designers, no training or inference is possible. China is aggressively building domestic alternatives (SMIC, Cambricon, HYGON) due to US export controls on NVIDIA H100/A100 chips.",
        "flow_label": "Wafers · Chips · NPUs → Server assembly",
        "companies": [
            {"ticker": "688981", "name": "SMIC", "nameZh": "中芯国际", "market": "STAR", "revenue": "¥56B+",
             "product": "China's largest foundry. Manufactures AI chips including Cambricon NPUs, Loongson CPUs, and domestic AI accelerators. Limited to ≤7nm due to US export controls.",
             "tags": ["Foundry", "Logic"]},
            {"ticker": "688183", "name": "Hua Hong Semiconductor", "nameZh": "华虹半导体", "market": "STAR", "revenue": "¥14B+",
             "product": "China's #2 foundry (90nm–28nm). Produces power management ICs and AI edge inferencing chips used in data centre PSUs and edge nodes.",
             "tags": ["Foundry", "Mature Node"]},
            {"ticker": "688779", "name": "Cambricon", "nameZh": "寒武纪", "market": "STAR", "revenue": "¥1.5B+",
             "product": "China's leading AI chip designer. Develops neural processing units (NPUs) for AI training and inference — the domestic NPU challenger to NVIDIA.",
             "tags": ["AI Chip", "NPU"]},
            {"ticker": "688041", "name": "HYGON (DCU)", "nameZh": "海光信息", "market": "STAR", "revenue": "¥6B+",
             "product": "AI-grade Deep Computing Units (DCUs) and x86-compatible CPUs. Targeting data centre training workloads as a domestic H100 alternative. Widely deployed where foreign GPUs are restricted.",
             "tags": ["AI Accelerator", "x86 CPU"]},
            {"ticker": "688126", "name": "JCET Group", "nameZh": "长电科技", "market": "STAR", "revenue": "¥24B+",
             "product": "China's largest semiconductor packaging & testing (OSAT). Advanced packaging (SiP, Fan-Out WLP) crucial for AI chip density and thermal performance.",
             "tags": ["OSAT", "Advanced Packaging"]},
            {"ticker": "688521", "name": "Empyrean Technology", "nameZh": "概伦电子", "market": "STAR", "revenue": "¥0.3B+",
             "product": "China's domestic EDA (Electronic Design Automation) software — strategic substitute for Synopsys/Cadence. Without EDA tools, no chip can be designed.",
             "tags": ["EDA", "Design Tools"]},
            {"ticker": "600460", "name": "Silan Microelectronics", "nameZh": "士兰微", "market": "SSE", "revenue": "¥5B+",
             "product": "Power semiconductors (MOSFETs, IGBTs) used in data centre UPS systems and power delivery units (PDUs). Critical for reliable power rail management.",
             "tags": ["Power Semi", "MOSFET"]},
        ]
    },
    {
        "id": "pcb",
        "name": "Layer 3 · PCBs, Substrates & Copper Clad Laminates",
        "short": "PCB / 电路板",
        "color": "#10B981",
        "icon": "🔌",
        "why": "AI GPUs require ultra-high-density interconnect (HDI) PCBs with trace geometries <50μm and low-loss dielectrics for high-speed GPU-to-GPU signalling. AI server PCBs approach semiconductor fabrication precision.",
        "flow_label": "HDI PCBs · IC Substrates → Server assembly & optical modules",
        "companies": [
            {"ticker": "002916", "name": "Shennan Circuits", "nameZh": "深南电路", "market": "SZSE", "revenue": "¥12B+",
             "product": "China's top HDI & IC substrate manufacturer. Mastered packaging substrates — the microscopic interface between silicon die and motherboard. Supplier to Huawei, AMD, and domestic AI chipmakers.",
             "tags": ["HDI PCB", "IC Substrate"]},
            {"ticker": "300476", "name": "Victory Giant Tech", "nameZh": "胜宏科技", "market": "ChiNext", "revenue": "¥8B+",
             "product": "High-frequency, high-speed PCBs for AI servers. One of the fastest-growing AI PCB suppliers. +597% stock return in 2025 driven by AI server demand surge.",
             "tags": ["AI PCB", "High-Speed"]},
            {"ticker": "002384", "name": "Dongshan Precision", "nameZh": "东山精密", "market": "SZSE", "revenue": "¥30B+",
             "product": "#2 global flexible PCB (FPC) manufacturer. FPCs connect GPU modules in tight AI server spaces. Also in Apple and EV supply chains.",
             "tags": ["FPC", "Flexible PCB"]},
            {"ticker": "002475", "name": "Shengyi Technology", "nameZh": "生益科技", "market": "SZSE", "revenue": "¥18B+",
             "product": "China's largest copper clad laminate (CCL) producer. CCL is the raw substrate every PCB is etched from. High-frequency CCL for AI server boards is a key growth product.",
             "tags": ["CCL", "Laminate"]},
            {"ticker": "002466", "name": "Tianshui Huatian", "nameZh": "天水华天电子", "market": "SZSE", "revenue": "¥8B+",
             "product": "Semiconductor packaging & testing for AI accelerators and memory chips. OSAT services critical for AI chip volume production at scale.",
             "tags": ["OSAT", "Packaging"]},
        ]
    },
    {
        "id": "optical",
        "name": "Layer 4 · Optical Modules & High-Speed Networking",
        "short": "光模块 · Optical Modules",
        "color": "#06B6D4",
        "icon": "💡",
        "why": "An NVIDIA GB200 NVL72 rack requires thousands of optical connections. AI clusters running LLM training demand >400Gbps per port. Traditional copper cannot scale to these speeds and distances — optical transceivers are mandatory.",
        "flow_label": "800G/1.6T Transceivers → Switch fabrics in AI racks",
        "companies": [
            {"ticker": "300308", "name": "Zhongji Innolight", "nameZh": "中际旭创", "market": "ChiNext", "revenue": "¥20B+",
             "product": "Top-tier 400G/800G/1.6T optical transceiver manufacturer. Supplies NVIDIA, Microsoft, Google, and Amazon data centres globally. Stock +410% in 2025 driven by hyperscaler demand.",
             "tags": ["800G/1.6T Transceiver", "CPO"]},
            {"ticker": "300394", "name": "Suzhou TFC Optical", "nameZh": "天孚通信", "market": "ChiNext", "revenue": "¥4B+",
             "product": "High-margin passive optical components (connectors, collimators, isolators) for AI data centre fiber networks. Low capex, high precision manufacturing moat.",
             "tags": ["Passive Optics", "Connectors"]},
            {"ticker": "688018", "name": "Accelink Technologies", "nameZh": "华工科技", "market": "STAR", "revenue": "¥10B+",
             "product": "Optical components, modules and subsystems for data communications. 100G–400G transceivers for hyperscale data centres and telecom operators.",
             "tags": ["Transceivers", "Optical Components"]},
            {"ticker": "002281", "name": "Hisense Broadband", "nameZh": "海信宽带", "market": "SZSE", "revenue": "¥6B+",
             "product": "100G–400G optical transceivers and laser chips for data centres. Active Co-Packaged Optics (CPO) development roadmap for next-gen AI racks.",
             "tags": ["Transceivers", "Laser Chip"]},
            {"ticker": "300548", "name": "Shengke Photonics", "nameZh": "仕佳光子", "market": "ChiNext", "revenue": "¥1B+",
             "product": "Photonic integrated circuits (PICs) and planar lightwave circuits (PLC) for signal splitting in data centre fiber networks. Precision micro-optical components.",
             "tags": ["PIC", "Photonics"]},
            {"ticker": "603083", "name": "Jianqiao Technology", "nameZh": "剑桥科技", "market": "SSE", "revenue": "¥3B+",
             "product": "Optical networking equipment and transceivers for data centre interconnect. Ethernet optical modules for AI cluster internal networks.",
             "tags": ["Optical Networking"]},
        ]
    },
    {
        "id": "servers",
        "name": "Layer 5 · AI Servers, Storage & ODM Assembly",
        "short": "AI服务器 · AI Servers",
        "color": "#F59E0B",
        "icon": "🖥️",
        "why": "AI training requires servers with 8–72 GPUs per node, high-bandwidth memory (HBM), NVLink interconnects, and specialised power delivery at 10–100kW per rack. ODMs integrate GPUs, CPUs, memory, cooling, and networking into deployable rack systems.",
        "flow_label": "Assembled AI server racks → Data centre operators",
        "companies": [
            {"ticker": "601138", "name": "Foxconn Ind. Internet (FII)", "nameZh": "工业富联", "market": "SSE", "revenue": "¥600B+",
             "product": "World's largest server assembler. NVIDIA GB200 NVL72 rack supplier to Microsoft, Amazon, Google. Liquid cooling leader — CSP revenue +150% in 2025. Pivoted to liquid-cooled rack manufacturing.",
             "tags": ["Server ODM", "Liquid Cooling"]},
            {"ticker": "000977", "name": "Inspur Information", "nameZh": "浪潮信息", "market": "SZSE", "revenue": "¥80B+",
             "product": "China's largest AI server manufacturer. Global liquid-cooled server leader with >50% domestic market share. Delivers cold-plate liquid cooling chassis. #1 domestic AI server ODM.",
             "tags": ["AI Server", "#1 China"]},
            {"ticker": "603019", "name": "Sugon / Dawning", "nameZh": "中科曙光", "market": "SSE", "revenue": "¥20B+",
             "product": "China's top HPC and AI server maker backed by Chinese Academy of Sciences. 65% market share in immersion liquid cooling. Deploys AI clusters for national supercomputing centres.",
             "tags": ["HPC Server", "Immersion Cooling"]},
            {"ticker": "000938", "name": "Unisplendour (H3C parent)", "nameZh": "紫光股份", "market": "SZSE", "revenue": "¥70B+",
             "product": "Parent of New H3C Group — full-stack data centre hardware: AI servers, 400G switches, storage, routers. #2 server OEM in China. Deep Huawei ecosystem integration.",
             "tags": ["Servers", "Switches", "Storage"]},
        ]
    },
    {
        "id": "cooling",
        "name": "Layer 6 · Power Supply, Cooling & Thermal Management",
        "short": "液冷 · Power & Cooling",
        "color": "#EF4444",
        "icon": "❄️",
        "why": "AI GPUs now consume 700W–2,300W per chip. At rack level, this means 100–500kW of heat that air cooling physically cannot dissipate. Liquid cooling (cold-plate, immersion, spray) is now mandatory for high-density AI infrastructure.",
        "flow_label": "CDUs · Coolant systems → Installed in DC builds",
        "companies": [
            {"ticker": "002837", "name": "InvenStar (Yingweike)", "nameZh": "英维克", "market": "SZSE", "revenue": "¥6B+",
             "product": "Global CDU technology leader. World's first liquid-ring vacuum CDU (eliminates leakage risk). Intel-certified cold plates. In NVIDIA MGX ecosystem. >30% cold-plate market share.",
             "tags": ["CDU", "Cold-Plate", "NVIDIA Cert."]},
            {"ticker": "300499", "name": "Gaolan Co.", "nameZh": "高澜股份", "market": "ChiNext", "revenue": "¥2B+",
             "product": "Core certified supplier for NVIDIA GB300 liquid cooling modules. Passed Google supplier audit. Covers cold-plate, immersion, and spray cooling. ByteDance's core cooling partner.",
             "tags": ["NVIDIA GB300", "Liquid Cooling"]},
            {"ticker": "301018", "name": "Shenling Environment", "nameZh": "申菱环境", "market": "ChiNext", "revenue": "¥2B+",
             "product": "Huawei's core data centre thermal management partner. Single-phase immersion systems compatible with NVIDIA DGX SuperPOD. Single CDU cooling capacity 200kW. 38% gross margin.",
             "tags": ["Huawei Partner", "Immersion"]},
            {"ticker": "002011", "name": "Kehua Data", "nameZh": "科华数据", "market": "SZSE", "revenue": "¥10B+",
             "product": "UPS power systems, data centre power distribution, and liquid cooling integration. Waste heat recovery improves energy utilisation by 30%. Deployed at national computing hubs.",
             "tags": ["UPS", "Power", "DC Operator"]},
            {"ticker": "601877", "name": "Zhengtai Electric", "nameZh": "正泰电器", "market": "SSE", "revenue": "¥30B+",
             "product": "Electrical switchgear, circuit breakers, and transformers for data centre power infrastructure. High-voltage distribution equipment from grid to rack.",
             "tags": ["Switchgear", "Transformer"]},
            {"ticker": "002028", "name": "Sieyuan Electric", "nameZh": "思源电气", "market": "SZSE", "revenue": "¥8B+",
             "product": "High-voltage power transformers and STATCOM reactive power compensation for data centre electrical infrastructure. Grid-level power quality management.",
             "tags": ["HV Transformer", "Power Quality"]},
            {"ticker": "301489", "name": "Siquan New Materials", "nameZh": "思泉新材", "market": "ChiNext", "revenue": "¥0.8B+",
             "product": "Thermal interface materials (TIMs) and phase-change materials for GPU heat management. Minimises chip-to-cold-plate thermal resistance — critical at 1,000W+ chip power levels.",
             "tags": ["TIM", "Thermal Materials"]},
            {"ticker": "002897", "name": "Qiangrui Technology", "nameZh": "强瑞技术", "market": "SZSE", "revenue": "¥1B+",
             "product": "Liquid cooling quick-disconnect connectors (UQDs) for modular data centre assembly. In Huawei and NVIDIA supply chains. Order growth >100% YoY.",
             "tags": ["UQD Connectors"]},
        ]
    },
    {
        "id": "infrastructure",
        "name": "Layer 7 · Data Centre Construction & Infrastructure",
        "short": "基础设施 · DC Infrastructure",
        "color": "#64748B",
        "icon": "🏗️",
        "why": "Building an AI data centre requires specialised modular construction, precision civil & electrical engineering, fire suppression, and pre-fabricated power/cooling modules. Modern approaches compress deployment from 3 years to 18 months.",
        "flow_label": "Physical DC buildings · Power grid → Handed to DC operators",
        "companies": [
            {"ticker": "600522", "name": "ZTT (Zhongtian Tech)", "nameZh": "中天科技", "market": "SSE", "revenue": "¥35B+",
             "product": "Data centre cabling: fibre-optic cables, copper power cables, and submarine cables. Complete cabling solutions for AI data centre builds — from grid connection to rack.",
             "tags": ["Fibre Cable", "Power Cable"]},
            {"ticker": "000021", "name": "Shenzhen Kaifa", "nameZh": "深科技", "market": "SZSE", "revenue": "¥20B+",
             "product": "Hard disk drive (HDD) manufacturing and storage device assembly. Storage hardware for AI training data repositories — raw data lakes for LLM pre-training.",
             "tags": ["HDD", "Storage"]},
            {"ticker": "300308", "name": "ZTE Corporation", "nameZh": "中兴通讯", "market": "SZSE", "revenue": "¥120B+",
             "product": "Telecom & enterprise data centre switching fabric, 5G base stations, and optical network gear. AI network infrastructure switch supplier.",
             "tags": ["Telecom", "DC Switching"]},
            {"ticker": "600584", "name": "Longsys (Flash Storage)", "nameZh": "朗科科技", "market": "SSE", "revenue": "¥5B+",
             "product": "NAND-based SSDs and DRAM modules for AI data centre storage tiers. Flash storage for AI inference caching and hot data lakes.",
             "tags": ["SSD", "Flash Storage"]},
        ]
    },
    {
        "id": "operators",
        "name": "Layer 8 · Data Centre Operators (IDC / AIDC)",
        "short": "IDC运营 · DC Operators",
        "color": "#3B82F6",
        "icon": "🏢",
        "why": "IDC operators are the 'AI factory' landlords — they own the physical facilities, power contracts, and network infrastructure, and sell GPU compute time or colocation rack space to cloud tenants and AI labs. National policy drives massive AIDC investment.",
        "flow_label": "Compute capacity sold as service → Cloud & AI platforms",
        "companies": [
            {"ticker": "603881", "name": "Shujugang (Data Harbor)", "nameZh": "数据港", "market": "SSE", "revenue": "¥2B+",
             "product": "Alibaba Cloud's IDC partner. Builds and operates hyperscale data centres with immersion liquid cooling. PUE as low as 1.09. Modular design enables rapid capacity expansion.",
             "tags": ["Alibaba Partner", "Liquid DC"]},
            {"ticker": "300869", "name": "Runze Technology", "nameZh": "润泽科技", "market": "ChiNext", "revenue": "¥6B+",
             "product": "ByteDance's liquid cooling data centre partner. Delivered liquid-cooled DC at PUE 1.06. Annual electricity savings >¥10M per project. National computing hub node operator.",
             "tags": ["ByteDance Partner", "AIDC"]},
            {"ticker": "002252", "name": "Sinnet Cloud", "nameZh": "中电数据", "market": "SZSE", "revenue": "¥20B+",
             "product": "State-owned IDC operator providing AI computing power to government and enterprise. Part of China's 'eastern data, western computing' national strategy.",
             "tags": ["State IDC", "Gov AI"]},
            {"ticker": "300383", "name": "Guanghuan New Network", "nameZh": "光环新网", "market": "ChiNext", "revenue": "¥3B+",
             "product": "Beijing-based colocation IDC operator. Premium connectivity via internet exchange points (IXPs). Serves fintech, AI labs, and cloud tenants.",
             "tags": ["Colocation", "Beijing IXP"]},
        ]
    },
    {
        "id": "cloud",
        "name": "Layer 9 · Cloud Platforms, AI Models & Applications",
        "short": "AI应用 · Cloud & AI",
        "color": "#059669",
        "icon": "☁️",
        "why": "The top of the stack — cloud providers sell AI infrastructure as-a-service, AI labs train and deploy LLMs, and application companies embed AI inference into products. China's domestic cloud must operate on domestic infrastructure due to data sovereignty laws.",
        "flow_label": "AI APIs · LLMs · Cloud services → End users",
        "companies": [
            {"ticker": "002230", "name": "iFlytek", "nameZh": "科大讯飞", "market": "SZSE", "revenue": "¥22B+",
             "product": "China's leading AI voice/NLP company. Operates Spark (Xinghuo) LLM deployed on domestic AI infrastructure. Major consumer of domestic AI data centre compute.",
             "tags": ["NLP AI", "Spark LLM"]},
            {"ticker": "002236", "name": "DaHua Technology", "nameZh": "大华技术", "market": "SZSE", "revenue": "¥35B+",
             "product": "AI-powered video surveillance and computer vision. Deep learning inferencing runs on domestic AI data centre infrastructure at massive scale.",
             "tags": ["Computer Vision", "AI Inference"]},
            {"ticker": "002415", "name": "Hikvision", "nameZh": "海康威视", "market": "SZSE", "revenue": "¥95B+",
             "product": "World's largest video surveillance company. AI analytics and computer vision — major AI inference workload consumer. Develops edge AI chips for real-time processing.",
             "tags": ["AI Vision", "Edge AI Chip"]},
            {"ticker": "601360", "name": "360 Security", "nameZh": "三六零", "market": "SSE", "revenue": "¥8B+",
             "product": "AI cybersecurity platforms and intelligent edge computing. AI-powered threat detection consumes significant data centre inference capacity.",
             "tags": ["AI Security", "Edge AI"]},
            {"ticker": "600100", "name": "Tsinghua Tongfang", "nameZh": "同方股份", "market": "SSE", "revenue": "¥15B+",
             "product": "AI computing power platforms, smart city systems, and cloud infrastructure for government AI applications. National 'city brain' projects.",
             "tags": ["Gov AI", "Smart City"]},
            {"ticker": "300760", "name": "Mindray Medical", "nameZh": "迈瑞医疗", "market": "ChiNext", "revenue": "¥38B+",
             "product": "AI-powered medical imaging and diagnostics. Runs deep learning inference on cloud AI infrastructure for hospital deployment across China.",
             "tags": ["Medical AI", "Imaging"]},
        ]
    }
]

# Color map for layer accents
LAYER_COLORS = {layer["id"]: layer["color"] for layer in SUPPLY_CHAIN}


# ─────────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────────
total_companies = sum(len(l["companies"]) for l in SUPPLY_CHAIN)

st.markdown(f"""
<div class="sc-hero">
  <div class="sc-hero-title">AI Data Centre <span>Supply Chain Map</span></div>
  <div class="sc-hero-title" style="font-size:1rem;font-weight:500;color:#7a7f96;margin-bottom:0.75rem">AI数据中心供应链图谱 · A股全景</div>
  <div class="sc-hero-sub">
    Every layer from raw silicon to cloud delivery — mapped with China A-share listed companies, 
    their specific role in the supply chain, and what they produce. 
    Click <strong style="color:#e2e4ed">➕ Watchlist</strong> to track or <strong style="color:#e2e4ed">🔍 Analyze</strong> to open Stock Analysis.
  </div>
  <div class="sc-stats">
    <div class="sc-stat"><div class="sc-stat-num">9</div><div class="sc-stat-label">Supply Layers</div></div>
    <div class="sc-stat"><div class="sc-stat-num">{total_companies}</div><div class="sc-stat-label">A-Share Companies</div></div>
    <div class="sc-stat"><div class="sc-stat-num">¥336B</div><div class="sc-stat-label">China DC Market 2032E</div></div>
    <div class="sc-stat"><div class="sc-stat-num">↑85%</div><div class="sc-stat-label">AI Server Demand YoY</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# FEEDBACK AREA — watchlist/analyze feedback messages
# ─────────────────────────────────────────────────────────────────
feedback_container = st.container()


# ─────────────────────────────────────────────────────────────────
# RENDER EACH LAYER
# ─────────────────────────────────────────────────────────────────
for layer_idx, layer in enumerate(SUPPLY_CHAIN):
    color = layer["color"]

    # ── Flow connector between layers ──────────────────────────
    if layer_idx > 0:
        st.markdown(f"""
        <div class="sc-flow-connector">
            <div class="sc-flow-line" style="background:linear-gradient(to right,{SUPPLY_CHAIN[layer_idx-1]['color']},{color})"></div>
            <div class="sc-flow-arrow">▼</div>
            <div class="sc-flow-label">{layer['flow_label']}</div>
            <div class="sc-flow-arrow">▼</div>
            <div class="sc-flow-line" style="background:linear-gradient(to right,{SUPPLY_CHAIN[layer_idx-1]['color']},{color})"></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Layer expander ─────────────────────────────────────────
    with st.expander(f"{layer['icon']}  {layer['name']}  —  {len(layer['companies'])} companies", expanded=(layer_idx == 0)):

        # Why this layer matters
        st.markdown(f"""
        <div class="sc-layer-desc" style="border-left-color:{color}">
            <span class="sc-why-label">Why this layer matters: </span>{layer['why']}
        </div>
        """, unsafe_allow_html=True)

        # ── Company cards: 3 per row ──────────────────────────
        companies = layer["companies"]
        rows = [companies[i:i+3] for i in range(0, len(companies), 3)]

        for row in rows:
            cols = st.columns(len(row))
            for col_idx, company in enumerate(row):
                ticker = company["ticker"]
                tags_html = "".join(f'<span class="sc-tag">{t}</span>' for t in company["tags"])

                with cols[col_idx]:
                    # Visual card (HTML)
                    st.markdown(f"""
                    <div class="sc-company-card">
                        <div class="sc-card-top">
                            <span class="sc-ticker-badge" style="color:{color};background:color-mix(in srgb,{color} 12%,#0f1117);border-color:color-mix(in srgb,{color} 28%,transparent)">{ticker}</span>
                            <span class="sc-market-badge">{company['market']}</span>
                        </div>
                        <div class="sc-name-en">{company['name']}</div>
                        <div class="sc-name-zh">{company['nameZh']}</div>
                        <div class="sc-product">{company['product']}</div>
                        <div class="sc-tags">{tags_html}</div>
                        <div class="sc-revenue">Rev: {company['revenue']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Action buttons — two side by side
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button(
                            "➕ Watchlist",
                            key=f"wl_{layer['id']}_{ticker}_{col_idx}",
                            use_container_width=True,
                            help=f"Add {company['name']} ({ticker}) to your watchlist"
                        ):
                            success, message = data_manager.add_to_watchlist(
                                ticker,
                                stock_name=company["nameZh"]  # pass Chinese name directly, avoids API call
                            )
                            if success:
                                with feedback_container:
                                    st.success(f"✅ Added **{company['name']}** ({ticker}) to watchlist", icon="✅")
                            else:
                                with feedback_container:
                                    st.warning(message, icon="⚠️")

                    with btn_col2:
                        if st.button(
                            "🔍 Analyze",
                            key=f"an_{layer['id']}_{ticker}_{col_idx}",
                            use_container_width=True,
                            help=f"Open Stock Analysis for {ticker}"
                        ):
                            st.session_state.active_ticker = ticker
                            st.session_state.ticker_input = ticker
                            st.switch_page("pages/2_Single_Stock_Analysis_个股分析.py")

            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# FOOTER NOTE
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data sourced from public company filings, CNKI, and market research. "
    "For informational purposes only — not investment advice. "
    "Revenue figures are approximate annual estimates in CNY."
)