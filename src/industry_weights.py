"""
industry_weights.py -- GICS Sector Profiles for Industry-Aware Sentiment Scoring

Each sector profile defines keyword categories with relevance weights.
Used by sentiment.py to amplify/dampen LLM conviction scores based on
whether a headline is relevant to that sector's key drivers.
"""

SECTOR_PROFILES = {
    'Technology': {
        'description': 'Semiconductors, software, cloud, AI',
        'categories': {
            'ai_compute':   {'weight': 2.0, 'description': 'AI hardware/compute', 'keywords': ['ai chip','gpu','data center','inference','accelerator','blackwell','h100','h200','gb200','nvidia','ai infrastructure','compute demand','ai demand','jensen huang']},
            'semiconductor':{'weight': 1.8, 'description': 'Chip manufacturing', 'keywords': ['semiconductor','chip','foundry','tsmc','wafer','lithography','chip shortage','chip supply','fab']},
            'cloud_software':{'weight': 1.6, 'description': 'Cloud and software', 'keywords': ['cloud','azure','aws','google cloud','saas','software','subscription','arr','platform','gcp']},
            'export_trade': {'weight': 1.7, 'description': 'Export controls and trade', 'keywords': ['export ban','export control','china ban','trade restriction','entity list','chip export','semiconductor export','tariff tech','huawei','china tariff','china trade','trade tariff','tech tariff','import duty']},
            'financial':    {'weight': 1.2, 'description': 'Earnings/financials', 'keywords': ['earnings','revenue','guidance','quarterly','eps','beat','miss','profit','margin','forecast','outlook','q1','q2','q3','q4']},
        },
        'default_weight': 0.7,
    },
    'Health Care': {
        'description': 'Pharma, biotech, medical devices',
        'categories': {
            'drug_approval':  {'weight': 2.0, 'description': 'FDA approvals', 'keywords': ['fda','approval','approved','rejected','pdufa','nda','bla','accelerated approval','breakthrough therapy','clinical hold']},
            'clinical_trial': {'weight': 1.9, 'description': 'Trial results', 'keywords': ['clinical trial','phase 1','phase 2','phase 3','trial results','primary endpoint','efficacy','safety data','placebo']},
            'pipeline':       {'weight': 1.6, 'description': 'Drug pipeline', 'keywords': ['pipeline','candidate','drug','therapy','treatment','indication','biotech','biologic','antibody','mrna']},
            'regulatory':     {'weight': 1.7, 'description': 'Healthcare regulation', 'keywords': ['medicare','medicaid','cms','drug pricing','rebate','regulation','hhs','patent expiry','generic','biosimilar']},
            'financial':      {'weight': 1.2, 'description': 'Earnings/financials', 'keywords': ['earnings','revenue','guidance','quarterly','eps','beat','miss','profit','margin','forecast']},
        },
        'default_weight': 0.6,
    },
    'Financials': {
        'description': 'Banks, insurance, investment management',
        'categories': {
            'interest_rates': {'weight': 2.0, 'description': 'Fed and rates', 'keywords': ['federal reserve','fed rate','interest rate','rate hike','rate cut','fomc','powell','monetary policy','yield curve','treasury yield']},
            'credit_quality': {'weight': 1.8, 'description': 'Loan quality', 'keywords': ['loan loss','credit loss','provision','npl','delinquency','charge-off','credit quality','net interest margin','nim']},
            'regulation':     {'weight': 1.7, 'description': 'Banking regulation', 'keywords': ['regulation','capital requirement','stress test','basel','fdic','occ','compliance','fine','settlement','enforcement']},
            'financial':      {'weight': 1.3, 'description': 'Earnings/financials', 'keywords': ['earnings','revenue','guidance','quarterly','eps','beat','miss','net income','roe','return on equity','dividend']},
        },
        'default_weight': 0.8,
    },
    'Consumer Discretionary': {
        'description': 'Retail, autos, hotels, e-commerce',
        'categories': {
            'consumer_spending': {'weight': 2.0, 'description': 'Consumer confidence/spending', 'keywords': ['consumer confidence','consumer spending','retail sales','discretionary spending','consumer sentiment','spending cut']},
            'trade_tariff':      {'weight': 1.9, 'description': 'Tariffs and trade', 'keywords': ['tariff','trade war','import tax','trade deal','china tariff','supply chain cost','input cost']},
            'ecommerce':         {'weight': 1.6, 'description': 'E-commerce trends', 'keywords': ['e-commerce','online sales','amazon','shopify','marketplace','prime','delivery','fulfillment']},
            'financial':         {'weight': 1.2, 'description': 'Earnings/financials', 'keywords': ['earnings','revenue','guidance','same-store sales','comp sales','eps','beat','miss','outlook','quarterly']},
        },
        'default_weight': 0.7,
    },
    'Communication Services': {
        'description': 'Social media, telecom, media, advertising',
        'categories': {
            'advertising':    {'weight': 2.0, 'description': 'Digital advertising market', 'keywords': ['advertising','ad revenue','digital ads','ad spend','cpm','programmatic','search ads','youtube ads','meta ads']},
            'ai_competition': {'weight': 1.9, 'description': 'AI threat to platforms', 'keywords': ['ai search','chatgpt','perplexity','openai','search market share','ai competition','llm','generative ai','search engine']},
            'antitrust':      {'weight': 1.8, 'description': 'Antitrust scrutiny', 'keywords': ['antitrust','monopoly','doj','ftc','competition','breakup','market dominance','search monopoly','consent decree']},
            'user_growth':    {'weight': 1.5, 'description': 'User engagement', 'keywords': ['dau','mau','user growth','engagement','subscribers','streaming','content','platform growth']},
            'financial':      {'weight': 1.2, 'description': 'Earnings/financials', 'keywords': ['earnings','revenue','guidance','quarterly','eps','beat','miss','profit','margin','outlook']},
        },
        'default_weight': 0.6,
    },
    'Energy': {
        'description': 'Oil, gas, pipelines, renewables',
        'categories': {
            'oil_price':   {'weight': 2.0, 'description': 'Crude oil price/supply', 'keywords': ['crude oil','oil price','wti','brent','opec','oil production','oil demand','inventory','barrel','energy price']},
            'geopolitics': {'weight': 1.8, 'description': 'Geopolitical energy risk', 'keywords': ['geopolitic','sanctions','iran','russia','middle east','pipeline','lng','energy security','conflict']},
            'renewables':  {'weight': 1.5, 'description': 'Clean energy transition', 'keywords': ['renewable','solar','wind','clean energy','transition','carbon','ira','inflation reduction act','ev','battery']},
            'financial':   {'weight': 1.2, 'description': 'Earnings/financials', 'keywords': ['earnings','revenue','capex','dividend','buyback','production','output','quarterly','eps','outlook']},
        },
        'default_weight': 0.7,
    },
    'Industrials': {
        'description': 'Defense, aerospace, machinery, transport',
        'categories': {
            'defense':      {'weight': 2.0, 'description': 'Defense contracts/spending', 'keywords': ['defense','military','dod','pentagon','contract award','weapons','nato','geopolitic','aerospace']},
            'supply_chain': {'weight': 1.8, 'description': 'Manufacturing/supply chain', 'keywords': ['supply chain','manufacturing','production','backlog','orders','capacity','lead time','raw material','freight']},
            'macro_cycle':  {'weight': 1.5, 'description': 'Economic cycle', 'keywords': ['pmi','industrial production','manufacturing index','ism','recession','economic activity','infrastructure']},
            'financial':    {'weight': 1.2, 'description': 'Earnings/financials', 'keywords': ['earnings','revenue','guidance','quarterly','eps','beat','miss','margin','outlook','order book']},
        },
        'default_weight': 0.7,
    },
    'Consumer Staples': {
        'description': 'Food, beverages, household products',
        'categories': {
            'commodity_costs': {'weight': 2.0, 'description': 'Input commodity costs', 'keywords': ['commodity','wheat','corn','sugar','coffee','input cost','raw material','food inflation','packaging cost']},
            'pricing_power':   {'weight': 1.7, 'description': 'Brand pricing power', 'keywords': ['price increase','pricing power','volume decline','trade down','private label','brand','market share']},
            'financial':       {'weight': 1.2, 'description': 'Earnings/financials', 'keywords': ['earnings','revenue','organic growth','volume','eps','beat','miss','margin','guidance']},
        },
        'default_weight': 0.8,
    },
    'Real Estate': {
        'description': 'REITs, commercial real estate',
        'categories': {
            'rates':     {'weight': 2.0, 'description': 'Interest rate sensitivity', 'keywords': ['interest rate','mortgage rate','fed rate','yield','cap rate','rate cut','rate hike','refinancing','bond yield']},
            'occupancy': {'weight': 1.7, 'description': 'Occupancy and rent trends', 'keywords': ['occupancy','vacancy','rent','lease','noi','ffo','affo','property value','tenant']},
            'financial': {'weight': 1.2, 'description': 'REIT financials', 'keywords': ['ffo','affo','dividend','distribution','earnings','revenue','guidance','quarterly']},
        },
        'default_weight': 0.8,
    },
    'Materials': {
        'description': 'Mining, chemicals, metals',
        'categories': {
            'commodity_prices': {'weight': 2.0, 'description': 'Metal and material prices', 'keywords': ['copper','gold','silver','lithium','nickel','aluminum','steel','iron ore','commodity price','metal price']},
            'china_demand':     {'weight': 1.8, 'description': 'China industrial demand', 'keywords': ['china demand','china economy','chinese manufacturing','china pmi','china stimulus','emerging markets']},
            'financial':        {'weight': 1.2, 'description': 'Earnings/financials', 'keywords': ['earnings','revenue','quarterly','production','eps','beat','miss','margin','guidance']},
        },
        'default_weight': 0.7,
    },
    'Utilities': {
        'description': 'Electric, gas, water utilities',
        'categories': {
            'ai_power_demand': {'weight': 2.0, 'description': 'AI data center power demand', 'keywords': ['data center','ai power','electricity demand','gigawatt','power demand','grid','load growth','ai infrastructure','hyperscaler']},
            'regulation':      {'weight': 1.8, 'description': 'Rate regulation', 'keywords': ['rate case','regulatory','puc','ferc','rate increase','allowed return','rate approval','utility regulation']},
            'rates':           {'weight': 1.6, 'description': 'Interest rate sensitivity', 'keywords': ['interest rate','bond yield','dividend yield','rate cut','rate hike','refinancing','cost of capital']},
            'financial':       {'weight': 1.2, 'description': 'Earnings/financials', 'keywords': ['earnings','revenue','eps','dividend','guidance','quarterly','beat','miss','capex']},
        },
        'default_weight': 0.8,
    },
}

DEFAULT_SECTOR_PROFILE = {
    'description': 'General market',
    'categories': {
        'financial': {'weight': 1.2, 'description': 'Earnings/financials', 'keywords': ['earnings','revenue','guidance','quarterly','eps','beat','miss','profit','margin','forecast','outlook']},
    },
    'default_weight': 0.8,
}

# Known ticker -> sector mapping for quick lookup (used by demo scripts and backtest cache)
TICKER_SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
    'GOOGL': 'Communication Services', 'GOOG': 'Communication Services',
    'META': 'Communication Services', 'AMZN': 'Consumer Discretionary',
    'TSLA': 'Consumer Discretionary', 'JPM': 'Financials', 'BAC': 'Financials',
    'JNJ': 'Health Care', 'UNH': 'Health Care', 'XOM': 'Energy', 'CVX': 'Energy',
    'WMT': 'Consumer Staples', 'PG': 'Consumer Staples',
}


# Wikipedia GICS names -> SECTOR_PROFILES keys (Wikipedia uses longer forms)
SECTOR_NAME_MAP = {
    'Information Technology': 'Technology',
    'Health Care':            'Health Care',
    'Financials':             'Financials',
    'Consumer Discretionary': 'Consumer Discretionary',
    'Communication Services': 'Communication Services',
    'Energy':                 'Energy',
    'Industrials':            'Industrials',
    'Consumer Staples':       'Consumer Staples',
    'Real Estate':            'Real Estate',
    'Materials':              'Materials',
    'Utilities':              'Utilities',
}


def normalize_sector(sector: str) -> str:
    """Map Wikipedia GICS sector name to SECTOR_PROFILES key."""
    return SECTOR_NAME_MAP.get(sector, sector)


def get_sector_weight(sector: str, headline_text: str) -> tuple:
    """
    Returns (weight: float, category: str, description: str)
    Highest-weight matching category for the sector, or default weight.
    Accepts both Wikipedia GICS names and short names.
    """
    sector = normalize_sector(sector or '')
    profile = SECTOR_PROFILES.get(sector, DEFAULT_SECTOR_PROFILE)
    text_lower = headline_text.lower()
    best_weight = profile.get('default_weight', 0.8)
    best_category = 'default'
    best_description = 'No category matched'
    for cat_name, cat_data in profile.get('categories', {}).items():
        for keyword in cat_data['keywords']:
            if keyword.lower() in text_lower:
                if cat_data['weight'] > best_weight:
                    best_weight = cat_data['weight']
                    best_category = cat_name
                    best_description = cat_data['description']
                break
    return best_weight, best_category, best_description


# ---------------------------------------------------------------------------
# Source Whitelist (Strategy PDF, Page 4, Section 5)
# Only headlines from these domains contribute to sentiment scoring.
# Goal: block unknown blogs and low-credibility aggregators while keeping
# the major financial news outlets that Alpaca commonly surfaces.
# ---------------------------------------------------------------------------
TRUSTED_SOURCES = {
    # Wire services (Alpaca short-name format)
    'reuters', 'apnews', 'ap',
    # Financial press
    'bloomberg', 'ft', 'wsj', 'barrons',
    'marketwatch', 'thestreet', 'investing',
    # Broadcast / digital
    'cnbc', 'foxbusiness', 'fortune', 'businessinsider',
    # Specialist financial (Benzinga is the dominant Alpaca source)
    'seekingalpha', 'benzinga', 'motleyfool', 'zacks',
    # Press release wires (official company statements)
    'prnewswire', 'businesswire', 'globenewswire',
    # Major news orgs with finance coverage
    'nytimes', 'washingtonpost', 'economist',
    'techcrunch', 'wired',
}
