"""
AI Supply Chain Map - AI供应链
Integrated with ASSRS V2 Database and Watchlist
"""

import streamlit as st
import pandas as pd
import data_manager
import auth_manager

# ==================== PAGE CONFIG & AUTH ====================
auth_manager.require_login()
user = auth_manager.get_current_user()

st.title("🌐 AI Data Centre Supply Chain | AI供应链")
st.markdown("Real-time valuation metrics and structural map for the China A-Share ecosystem.")
st.divider()

# ==================== 1. FULL SUPPLY CHAIN DATA ====================
SUPPLY_CHAIN = [
    {
        "id": "raw-materials",
        "name": "Raw Materials & Critical Minerals",
        "shortName": "Raw Materials",
        "icon": "⛏",
        "description": "The bedrock of the entire supply chain. Miners and refiners produce copper, cobalt, rare earths, silicon, fluorite, and boron — feedstocks for every downstream layer from chip fabs to cooling systems. Without refined minerals, nothing downstream can be made.",
        "companies": [
            {"ticker": "603993", "name": "CMOC Group", "nameZh": "洛阳钼业", "market": "SSE", "product": "World's largest cobalt producer and major copper miner (DRC & Brazil). Cobalt is critical for battery storage in data centre UPS systems. Controls ~20% of global cobalt supply."},
            {"ticker": "600362", "name": "Jiangxi Copper", "nameZh": "江西铜业", "market": "SSE", "product": "China's largest integrated copper producer. Copper rod and cathode feed into copper foil, cooling pipes, power cables, and busbars."},
            {"ticker": "603799", "name": "Huayou Cobalt", "nameZh": "华友钴业", "market": "SSE", "product": "Major cobalt & nickel refiner. Processes DRC ore into battery-grade cobalt sulphate and nickel sulphate."},
            {"ticker": "000807", "name": "Yunnan Aluminium", "nameZh": "云南铝业", "market": "SZSE", "product": "Primary aluminium smelting. Aluminium is the dominant material in server chassis, heat sink fins, cold plate bodies, and cooling manifolds."},
            {"ticker": "000408", "name": "Zangge Mining", "nameZh": "藏格矿业", "market": "SZSE", "product": "Lithium carbonate and potash from Qinghai salt lakes. Lithium feeds into high-capacity LiFePO4 UPS batteries."},
            {"ticker": "600160", "name": "Juhua Group", "nameZh": "巨化股份", "market": "SSE", "product": "China's fluorochemical champion. Produces semiconductor-grade hydrofluoroether (HFE) — the dielectric immersion cooling fluid."},
            {"ticker": "600206", "name": "GRICEM (Gallium/Germanium)", "nameZh": "有研新材", "market": "SSE", "product": "China's leading producer of gallium, germanium, and semiconductor-grade silicon. Critical chokepoint supplier."},
            {"ticker": "002467", "name": "Shenghe Resources", "nameZh": "盛和资源", "market": "SZSE", "product": "China's largest rare earth resources company. Produces neodymium-iron-boron (NdFeB) magnet precursors used in data centre cooling fans."}
        ]
    },
    {
        "id": "specialty-materials",
        "name": "E-Glass Cloth, Copper Foil & Specialty Substrate",
        "shortName": "E-Glass / Cu Foil",
        "icon": "🪡",
        "description": "Converts raw minerals into precision intermediate materials. E-glass cloth, Copper foil, and Epoxy resin bond together to form the circuit boards in every AI server.",
        "companies": [
            {"ticker": "600176", "name": "China Jushi", "nameZh": "中国巨石", "market": "SSE", "product": "Global #1 fiberglass and e-glass cloth producer. Controls ~25% of China's e-glass market."},
            {"ticker": "603256", "name": "Honghe Technology", "nameZh": "宏和科技", "market": "SSE", "product": "Global leader in high-end specialty e-glass cloth. The only A-share pure-play on ultra-thin and specialty Low-Dk/Low-CTE e-glass cloth for AI server PCBs."},
            {"ticker": "002080", "name": "Zhongcai Technology", "nameZh": "中材科技", "market": "SZSE", "product": "Parent of Taishan Glass Fiber — China's #2 e-glass yarn producer. Global #3 in ultra-low-loss (Low-Dk) e-glass cloth."},
            {"ticker": "301526", "name": "Chongqing Polycomp", "nameZh": "国际复材", "market": "ChiNext", "product": "China's #3 fiberglass producer. Produces standard and high-performance e-glass yarn and cloth."},
            {"ticker": "301511", "name": "Defetech (Defu)", "nameZh": "德福科技", "market": "ChiNext", "product": "China's premium AI-grade copper foil specialist. Produces HVLP4/HVLP5 ultra-low-profile copper foil."},
            {"ticker": "301217", "name": "Tongguan Copper Foil", "nameZh": "铜冠铜箔", "market": "ChiNext", "product": "Listed subsidiary of Tongling Nonferrous. Produces high-performance electrodeposited copper foil for PCBs."},
            {"ticker": "688363", "name": "Huazheng New Materials", "nameZh": "华正新材", "market": "STAR", "product": "Specialty epoxy resin and halogen-free CCL resin systems. High-Tg epoxy with stable dielectric properties."},
            {"ticker": "605289", "name": "Kunshan Guoli", "nameZh": "荣昌科技", "market": "SSE", "product": "Rolled annealed (RA) copper foil for flexible PCBs (FPCs). Essential for flexible circuit connections in dense AI servers."}
        ]
    },
    {
        "id": "pcb-substrate",
        "name": "PCBs, Substrates & Copper Clad Laminates (CCL)",
        "shortName": "PCB / CCL",
        "icon": "🔌",
        "description": "CCL is manufactured by pressing e-glass cloth with epoxy resin and copper foil. PCBs are then etched from CCL into circuit boards.",
        "companies": [
            {"ticker": "002916", "name": "Shennan Circuits", "nameZh": "深南电路", "market": "SZSE", "product": "China's most advanced PCB and IC substrate maker. Produces packaging substrates for AI chipmakers."},
            {"ticker": "002475", "name": "Shengyi Technology", "nameZh": "生益科技", "market": "SZSE", "product": "China's largest CCL (Copper Clad Laminate) manufacturer. Produces high-Tg, low-loss CCL for AI server motherboards."},
            {"ticker": "300476", "name": "Victory Giant", "nameZh": "胜宏科技", "market": "ChiNext", "product": "High-frequency, high-speed multi-layer PCBs for AI servers and optical network equipment."},
            {"ticker": "002384", "name": "Dongshan Precision", "nameZh": "东山精密", "market": "SZSE", "product": "Global #2 flexible PCB (FPC) manufacturer by volume. Connects GPU modules inside AI server chassis."},
            {"ticker": "002466", "name": "Tianshui Huatian", "nameZh": "天水华天电子", "market": "SZSE", "product": "Semiconductor OSAT (packaging & testing) for AI accelerator chips. Also produces substrates."},
            {"ticker": "603228", "name": "Chaohua Technology", "nameZh": "超华科技", "market": "SSE", "product": "CCL materials and high-performance copper clad laminates including halogen-free grades."},
            {"ticker": "002933", "name": "Xinlun New Materials", "nameZh": "新纶新材料", "market": "SZSE", "product": "Specialty polymer films and functional materials used in FPC and rigid-flex PCBs."}
        ]
    },
    {
        "id": "semiconductor",
        "name": "Semiconductors, AI Chips & EDA Tools",
        "shortName": "AI Chips / Semi",
        "icon": "🧠",
        "description": "Foundries fabricate the silicon chips; AI chip designers create the NPUs/GPUs; OSAT companies package them. The core of computation.",
        "companies": [
            {"ticker": "688981", "name": "SMIC", "nameZh": "中芯国际", "market": "STAR", "product": "China's largest foundry. Manufactures Cambricon NPUs, HYGON DCUs, and domestic AI accelerators."},
            {"ticker": "688183", "name": "Hua Hong Semi", "nameZh": "华虹半导体", "market": "STAR", "product": "China's #2 foundry. Produces power management ICs for data centre PSUs."},
            {"ticker": "688779", "name": "Cambricon", "nameZh": "寒武纪", "market": "STAR", "product": "China's flagship AI chip designer. Produces MLU series NPUs for AI training and inference."},
            {"ticker": "688041", "name": "HYGON Information", "nameZh": "海光信息", "market": "STAR", "product": "Produces x86-compatible CPUs and Deep Computing Units (DCUs) — primary NVIDIA H-series substitutes."},
            {"ticker": "688126", "name": "JCET Group", "nameZh": "长电科技", "market": "STAR", "product": "China's largest semiconductor OSAT. Advanced packaging including Fan-Out WLP, SiP, and flip-chip."},
            {"ticker": "688521", "name": "Empyrean Technology", "nameZh": "概伦电子", "market": "STAR", "product": "China's leading EDA (Electronic Design Automation) software company for chip design."},
            {"ticker": "600460", "name": "Silan Microelectronics", "nameZh": "士兰微", "market": "SSE", "product": "Power semiconductors (MOSFETs, IGBTs) for data centre UPS and PDUs."},
            {"ticker": "688536", "name": "Primarius Tech", "nameZh": "芯华章", "market": "STAR", "product": "EDA verification tools and hardware emulation platforms for AI chip design validation."}
        ]
    },
    {
        "id": "passive-components",
        "name": "Passive Components — MLCC, Resistors & Inductors",
        "shortName": "Passives / MLCC",
        "icon": "🔋",
        "description": "A single AI server motherboard contains thousands of MLCCs to filter power and stabilise voltage rails.",
        "companies": [
            {"ticker": "002138", "name": "Sunlord Electronics", "nameZh": "顺络电子", "market": "SZSE", "product": "China's largest inductor maker. Power inductors are mandatory on every GPU voltage regulator module (VRM)."},
            {"ticker": "603876", "name": "Fenghua Advanced", "nameZh": "风华高科", "market": "SSE", "product": "China's largest MLCC manufacturer. A GPU has 3,000–8,000 MLCCs to decouple power supply noise."},
            {"ticker": "688676", "name": "Sanhuan Group", "nameZh": "三环集团", "market": "STAR", "product": "Produces ceramic-based passive components, MLCC, and ceramic substrates for optical transceivers."},
            {"ticker": "603659", "name": "Shanghai Putian", "nameZh": "璞泰来", "market": "SSE", "product": "Advanced polymer separator films and dielectric films in high-frequency capacitors."},
            {"ticker": "301308", "name": "Ninestar", "nameZh": "纳芯微", "market": "ChiNext", "product": "Analogue and mixed-signal ICs (gate drivers, sensors) used in AI server power management."},
            {"ticker": "688286", "name": "Kejie Technology", "nameZh": "科技汇川", "market": "STAR", "product": "High-frequency power conversion ICs and magnetic components for server PSU units."}
        ]
    },
    {
        "id": "optical-networking",
        "name": "Optical Modules & High-Speed Networking",
        "shortName": "Optical / Network",
        "icon": "💡",
        "description": "Every GPU in a training cluster must communicate at terabit speeds using optical transceivers converting signals to photons.",
        "companies": [
            {"ticker": "300308", "name": "Zhongji Innolight", "nameZh": "中际旭创", "market": "ChiNext", "product": "China's #1 optical transceiver maker. Produces 800G/1.6T modules for global hyperscalers."},
            {"ticker": "300394", "name": "Suzhou TFC Optical", "nameZh": "天孚通信", "market": "ChiNext", "product": "High-margin passive optical components: precision fibre connectors, isolators, and lenses."},
            {"ticker": "688018", "name": "Accelink Tech", "nameZh": "华工科技", "market": "STAR", "product": "Full-range optical transceiver and component manufacturer. 100G–400G modules for hyperscale DCs."},
            {"ticker": "688097", "name": "Yuanjie Technology", "nameZh": "源杰科技", "market": "STAR", "product": "China's leading laser chip (EML, DFB) designer. The light-emitting engine inside every optical transceiver."},
            {"ticker": "688548", "name": "Liguo Technology", "nameZh": "利扬芯片", "market": "STAR", "product": "Optical chip testing services and silicon photonic PIC characterisation."},
            {"ticker": "300548", "name": "Shengke Photonics", "nameZh": "仕佳光子", "market": "ChiNext", "product": "Photonic integrated circuits (PICs) and planar lightwave circuits (PLCs) for WDM systems."},
            {"ticker": "002281", "name": "Hisense Broadband", "nameZh": "海信宽带", "market": "SZSE", "product": "100G–400G transceiver modules and laser components for enterprise and data centre networks."}
        ]
    },
    {
        "id": "servers-hardware",
        "name": "AI Servers, Storage & ODM/OEM Assembly",
        "shortName": "AI Servers",
        "icon": "🖥",
        "description": "Server ODMs integrate GPUs, CPUs, memory, PCBs, optical modules, and cooling into complete AI compute nodes.",
        "companies": [
            {"ticker": "601138", "name": "Foxconn (FII)", "nameZh": "工业富联", "market": "SSE", "product": "World's largest server ODM. Liquid-cooled rack assembler for hyperscalers (GB200 NVL72)."},
            {"ticker": "000977", "name": "Inspur Information", "nameZh": "浪潮信息", "market": "SZSE", "product": "China's #1 AI server manufacturer. >50% domestic AI server market share."},
            {"ticker": "603019", "name": "Sugon / Dawning", "nameZh": "中科曙光", "market": "SSE", "product": "China's HPC and AI server champion backed by Chinese Academy of Sciences."},
            {"ticker": "000938", "name": "Unisplendour / H3C", "nameZh": "紫光股份", "market": "SZSE", "product": "China's #2 server OEM. Turnkey GPU cluster offerings and switches."},
            {"ticker": "000050", "name": "Shenzhen Tianma", "nameZh": "深天马A", "market": "SZSE", "product": "Display panels for data centre KVM consoles and DCIM management dashboards."}
        ]
    },
    {
        "id": "power-cooling",
        "name": "Power Supply, Cooling & Thermal Management",
        "shortName": "Power & Cooling",
        "icon": "❄",
        "description": "Liquid cooling is now mandatory for high-density AI server racks passing 100kW+ per rack.",
        "companies": [
            {"ticker": "002837", "name": "InvenStar", "nameZh": "英维克", "market": "SZSE", "product": "Global CDU tech leader. Intel/NVIDIA certified cold plates; >30% domestic CDU market share."},
            {"ticker": "300499", "name": "Gaolan Co.", "nameZh": "高澜股份", "market": "ChiNext", "product": "NVIDIA GB300 certified liquid cooling supplier. ByteDance's core hardware partner."},
            {"ticker": "920808", "name": "Shuguang Shuchuang", "nameZh": "曙光数创", "market": "BSE", "product": "Immersion liquid cooling absolute market leader. Achieves PUE of 1.03 for 50kW racks."},
            {"ticker": "300339", "name": "Shenling Env", "nameZh": "申菱环境", "market": "ChiNext", "product": "Huawei's designated thermal partner. Single-phase immersion cooling systems."},
            {"ticker": "300990", "name": "Tongfei Shares", "nameZh": "同飞股份", "market": "ChiNext", "product": "Full-spectrum liquid cooling: CDUs, TANK immersion units, and cold-plate assemblies."},
            {"ticker": "301489", "name": "Siquan New Mat", "nameZh": "思泉新材", "market": "ChiNext", "product": "Thermal interface materials (TIMs) and phase-change pads for GPU heat management."},
            {"ticker": "301492", "name": "Chuanhuan Tech", "nameZh": "川环科技", "market": "ChiNext", "product": "PTFE high-pressure liquid cooling pipes meeting NVIDIA GB300 specifications."},
            {"ticker": "002897", "name": "Qiangrui Tech", "nameZh": "强瑞技术", "market": "SZSE", "product": "Liquid cooling quick-disconnect connectors (UQDs) for hot-swapping server blades."},
            {"ticker": "002011", "name": "Kehua Data", "nameZh": "科华数据", "market": "SZSE", "product": "Data centre UPS power systems and PDUs. Waste heat recovery tech."},
            {"ticker": "601877", "name": "Zhengtai Electric", "nameZh": "正泰电器", "market": "SSE", "product": "Electrical switchgear and circuit breakers for data centre power infrastructure."},
            {"ticker": "002028", "name": "Sieyuan Electric", "nameZh": "思源电气", "market": "SZSE", "product": "High-voltage power transformers and STATCOM reactive power compensation systems."}
        ]
    },
    {
        "id": "dc-infrastructure",
        "name": "Data Centre Construction, Cabling & Infrastructure",
        "shortName": "DC Infrastructure",
        "icon": "🏗",
        "description": "The physical container for AI compute. Facility building, cabling, grid connections, and security.",
        "companies": [
            {"ticker": "600522", "name": "ZTT", "nameZh": "中天科技", "market": "SSE", "product": "Fibre-optic trunk cables, copper power cables, and specialty shielded cables for DC builds."},
            {"ticker": "601669", "name": "PowerChina", "nameZh": "中国电建", "market": "SSE", "product": "State-owned EPC giant building the high-voltage substations for national computing hubs."},
            {"ticker": "000021", "name": "Shenzhen Kaifa", "nameZh": "深科技", "market": "SZSE", "product": "Hard disk drive (HDD) assembly. Essential for AI training data cold storage."},
            {"ticker": "600584", "name": "Lanke Tech", "nameZh": "兰科集团", "market": "SSE", "product": "NAND flash-based SSD modules accelerating active AI dataset access."},
            {"ticker": "002065", "name": "Donghua Software", "nameZh": "东华软件", "market": "SZSE", "product": "DCIM (Data Centre Infrastructure Management) software for power/cooling visibility."}
        ]
    },
    {
        "id": "dc-operators",
        "name": "Data Centre Operators — IDC & AIDC",
        "shortName": "IDC Operators",
        "icon": "🏢",
        "description": "Operators providing rack space, colocation, and managed GPU compute to cloud tenants and AI labs.",
        "companies": [
            {"ticker": "603881", "name": "Shujugang", "nameZh": "数据港", "market": "SSE", "product": "Alibaba Cloud's dedicated IDC partner. Builds immersion liquid cooling AIDC facilities."},
            {"ticker": "300869", "name": "Runze Technology", "nameZh": "润泽科技", "market": "ChiNext", "product": "ByteDance's primary liquid-cooled AIDC partner. Operates national computing hubs."},
            {"ticker": "002252", "name": "ChinaNet Online", "nameZh": "中电数据", "market": "SZSE", "product": "State-owned IDC operator providing secure AI computing power to central government."},
            {"ticker": "300383", "name": "Guanghuan Network", "nameZh": "光环新网", "market": "ChiNext", "product": "Beijing-based carrier-neutral colocation IDC for fintech and AI labs."},
            {"ticker": "300352", "name": "Zhongjunercheng", "nameZh": "中润四方", "market": "ChiNext", "product": "High-density GPU rack colocation and AI computing power rental operator."}
        ]
    },
    {
        "id": "cloud-ai-platforms",
        "name": "Cloud Platforms, LLMs & AI Applications",
        "shortName": "Cloud & Apps",
        "icon": "☁",
        "description": "The top of the stack — cloud providers, AI model developers, and enterprise applications consuming the infrastructure.",
        "companies": [
            {"ticker": "002230", "name": "iFlytek", "nameZh": "科大讯飞", "market": "SZSE", "product": "Operates Spark LLM family. Deploys AI across education and healthcare."},
            {"ticker": "002415", "name": "Hikvision", "nameZh": "海康威视", "market": "SZSE", "product": "World's largest video surveillance company. Huge consumer of AI inference infrastructure."},
            {"ticker": "002236", "name": "DaHua Technology", "nameZh": "大华技术", "market": "SZSE", "product": "Smart city AI and deep learning video surveillance models."},
            {"ticker": "300760", "name": "Mindray Medical", "nameZh": "迈瑞医疗", "market": "ChiNext", "product": "AI-powered medical imaging and diagnostic foundation models."},
            {"ticker": "601360", "name": "360 Security", "nameZh": "三六零", "market": "SSE", "product": "AI cybersecurity platform and '360 Brain' enterprise ecosystem."},
            {"ticker": "600100", "name": "Tsinghua Tongfang", "nameZh": "同方股份", "market": "SSE", "product": "AI computing platforms for government applications and smart cities."},
            {"ticker": "688271", "name": "Zhipu AI", "nameZh": "智谱AI", "market": "STAR", "product": "Top-tier developer of GLM series LLMs. Massive consumer of domestic compute."}
        ]
    }
]

# ==================== 2. DATA ENRICHMENT ====================
@st.cache_data(ttl=3600)
def get_supply_chain_metrics():
    """Extract all tickers from the supply chain array and fetch their DB metrics in bulk."""
    all_tickers = []
    for layer in SUPPLY_CHAIN:
        for comp in layer.get('companies', []):
            all_tickers.append(comp['ticker'])
            
    # Fetch from DB using your data_manager function
    try:
        db_metrics = data_manager.get_daily_basic_for_tickers(all_tickers)
        return db_metrics
    except Exception as e:
        st.error(f"Failed to load database metrics: {e}")
        return pd.DataFrame()

db_metrics = get_supply_chain_metrics()

def get_metrics_for_ticker(ticker, df):
    """Safely extract PE_TTM and Total Market Value from the fetched DB dataframe."""
    if df is None or df.empty:
        return "N/A", "N/A"
    
    # Match the ticker column created inside get_daily_basic_for_tickers
    match = df[df['ticker'] == ticker]
    
    if not match.empty:
        pe = match.iloc[0].get('pe_ttm', 'N/A')
        # get_daily_basic_for_tickers already creates total_mv_yi (亿元)
        mv_yi = match.iloc[0].get('total_mv_yi', 'N/A') 
        
        pe_str = f"{pe:.1f}" if pd.notna(pe) and isinstance(pe, (int, float)) else "N/A"
        mv_str = f"{mv_yi:.1f}亿" if pd.notna(mv_yi) and isinstance(mv_yi, (int, float)) else "N/A"
        return pe_str, mv_str
        
    return "N/A", "N/A"

# ==================== 3. UI RENDERING ====================
for layer in SUPPLY_CHAIN:
    with st.expander(f"{layer['icon']} {layer['name']}", expanded=True):
        st.caption(layer['description'])
        
        # Create a responsive 3-column grid layout for cards
        cols = st.columns(3)
        
        for idx, comp in enumerate(layer.get('companies', [])):
            ticker = comp['ticker']
            pe_val, mv_val = get_metrics_for_ticker(ticker, db_metrics)
            
            with cols[idx % 3]:
                # Native Streamlit Container (acts as a UI Card)
                with st.container(border=True):
                    
                    # Header: Clickable Stock Code + Market Label
                    head1, head2 = st.columns([1.5, 1])
                    with head1:
                        # Sets session state and routes to analysis page
                        if st.button(f"🔍 {ticker}", key=f"nav_{ticker}_{layer['id']}", use_container_width=True):
                            st.session_state.active_ticker = ticker
                            st.switch_page("pages/2_Single_Stock_Analysis_个股分析.py")
                    with head2:
                        st.markdown(f"<div style='text-align:right; color:gray; font-size:0.8rem; margin-top:8px;'>{comp['market']}</div>", unsafe_allow_html=True)
                    
                    # Titles
                    st.markdown(f"**{comp['name']}**")
                    st.caption(f"{comp.get('nameZh', '')}")
                    
                    # DB Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("PE (TTM)", pe_val)
                    m2.metric("Total MV", mv_val)
                    
                    # Product Description
                    st.markdown(
                        f"<div style='font-size:0.8rem; color:gray; height:4.5em; overflow:hidden; margin-bottom:10px; line-height: 1.4;'>"
                        f"{comp.get('product', '')}</div>", 
                        unsafe_allow_html=True
                    )
                    
                    # Watchlist Action
                    if st.button("➕ Watchlist", key=f"wl_{ticker}_{layer['id']}", type="primary", use_container_width=True):
                        try:
                            # Use existing is_in_watchlist DB query
                            if data_manager.is_in_watchlist(ticker):
                                st.warning(f"⚠️ {ticker} is already in your watchlist!")
                            else:
                                success, msg = data_manager.add_to_watchlist(ticker, stock_name=comp['nameZh'])
                                if success:
                                    st.toast(msg, icon="✅")
                                else:
                                    st.error(msg)
                        except Exception as e:
                            st.error(f"Database error: {e}")