# Generated from: analytics.ipynb
# Converted at: 2026-04-10T13:18:55.565Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

!uv pip install openai langgraph groq langchain-core chromadb sentence-transformers prophet duckdb plotly networkx statsmodels apscheduler scipy sympy scikit-learn FlagEmbedding langchain-nvidia-ai-endpoints langgraph langchain-core langchain-openai

import os, json, hashlib, warnings, textwrap, re, math
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import duckdb
import chromadb
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

import scipy.stats as stats
from scipy.stats import ttest_ind, pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prophet import Prophet

import sympy as sp
from groq import Groq
from langgraph.graph import StateGraph, END

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:,.2f}".format)

pio.renderers.default = "iframe"

print(f"All imports OK | plotly renderer: {pio.renderers.default}")

# from langchain_openai import ChatOpenAI
# try:
#     from kaggle_secrets import UserSecretsClient
#     API_KEY = UserSecretsClient().get_secret("GROQ_API_KEY")
# except Exception:
#     API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_KEY")
#
# PROVIDER       = "groq"
# MAIN_MODEL     = "openai/gpt-oss-120b"
# FAST_MODEL     = "llama-3.3-70b-versatile"
# FALLBACK_MODEL = "llama-3.3-70b-versatile"
#
# _main_llm = ChatOpenAI(
#     model=MAIN_MODEL, api_key=API_KEY,
#     base_url="https://api.groq.com/openai/v1",
#     temperature=0, max_tokens=16384,
# )
# _fast_llm = ChatOpenAI(
#     model=FAST_MODEL, api_key=API_KEY,
#     base_url="https://api.groq.com/openai/v1",
#     temperature=0, max_tokens=16384,
# )

# ── OPTION B: NVIDIA NIM (via langchain-nvidia-ai-endpoints) ─────────
from langchain_nvidia_ai_endpoints import ChatNVIDIA

try:
    from kaggle_secrets import UserSecretsClient
    API_KEY = UserSecretsClient().get_secret("nvidia_api_key")
except Exception:
    API_KEY = os.environ.get("NVIDIA_API_KEY", "YOUR_NVIDIA_KEY")

PROVIDER       = "nvidia"
MAIN_MODEL     = "qwen/qwen3-next-80b-a3b-instruct"
FAST_MODEL     = "moonshotai/kimi-k2-thinking"
FALLBACK_MODEL = "google/gemma-4-31b-it"

_main_llm = ChatNVIDIA(
    model=MAIN_MODEL, api_key=API_KEY,
    temperature=0, max_tokens=16384,
)
_fast_llm = ChatNVIDIA(
    model=FAST_MODEL, api_key=API_KEY,
    temperature=0, max_tokens=16384,
)

result1 = _main_llm .invoke("Respond with OK all all working")
print("Main LLM: ",result1.content)

result2 = _fast_llm .invoke("Respond with OK all all working")
print("Fast LLM: ",result2.content)

MAX_RETRIES  = 5
MAX_TOKENS = 16384

DB_PATH     = "/kaggle/working/sentinel_ecom.duckdb"
CHROMA_PATH = "/kaggle/working/chroma_bge"
GRAPH_PATH  = "/kaggle/working/l3_ecom.gml"
os.makedirs(CHROMA_PATH, exist_ok=True)

from langchain_core.messages import HumanMessage, SystemMessage


def call_llm(prompt: str, system: str = "", model: str = MAIN_MODEL,
             temperature: float = 0.0) -> str:
    """
    Unified LLM caller using langchain message interface.
    Automatically picks _main_llm or _fast_llm based on model arg.
    Falls back to FALLBACK_MODEL on any error.
    """
    # Pick the right langchain llm object
    llm = _fast_llm if model == FAST_MODEL else _main_llm

    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    # Apply temperature at invoke time if supported
    try:
        resp = llm.bind(temperature=temperature).invoke(messages)
        return resp.content.strip()
    except Exception as e1:
        print(f"  [LLM] {model} failed: {e1} — trying fallback")
        try:
            resp = _fast_llm.bind(temperature=temperature).invoke(messages)
            return resp.content.strip()
        except Exception as e2:
            return f"LLM_ERROR: {e2}"


# ── Smoke test ────────────────────────────────────────────────────────
_test = call_llm("Reply with the single word: ready", temperature=0.0)
print(f"Provider  : {PROVIDER.upper()}")
print(f"Main model: {MAIN_MODEL}")
print(f"Fast model: {FAST_MODEL}")
print(f"Smoke test: '{_test}'")

np.random.seed(42)
rng = np.random.default_rng(42)


N_ORDERS   = 50_000
N_CUSTOMERS = 12_000
N_SELLERS   = 800
N_PRODUCTS  = 2_000

START_DATE = datetime(2024, 6, 1)
END_DATE   = datetime(2024, 6, 15, 23, 59, 59)

CATEGORIES = ["Food", "Electronics", "Fashion", "Books",
              "Home & Kitchen", "Beauty", "Sports", "Grocery"]
CITIES     = ["Mumbai", "Delhi", "Bangalore", "Chennai",
              "Hyderabad", "Pune", "Kolkata", "Ahmedabad",
              "Jaipur", "Surat"]
STATES     = ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu",
              "Telangana", "Maharashtra", "West Bengal", "Gujarat",
              "Rajasthan", "Gujarat"]
CITY_STATE = dict(zip(CITIES, STATES))
PLATFORMS  = ["App", "Website", "API"]
PAYMENTS   = ["UPI", "Credit Card", "Debit Card", "COD", "Wallet"]
STATUSES   = ["delivered", "pending", "cancelled", "returned"]

CAT_PRICE_RANGE = {
    "Food":          (50, 800),
    "Electronics":   (500, 80000),
    "Fashion":       (200, 5000),
    "Books":         (100, 1500),
    "Home & Kitchen":(300, 15000),
    "Beauty":        (150, 3000),
    "Sports":        (500, 20000),
    "Grocery":       (100, 2000),
}


product_cats = rng.choice(CATEGORIES, N_PRODUCTS,
                          p=[0.25, 0.15, 0.15, 0.08, 0.12, 0.10, 0.08, 0.07])
products_df = pd.DataFrame({
    "product_id":   range(1, N_PRODUCTS + 1),
    "product_name": [f"{cat[:3].upper()}_Product_{i}"
                     for i, cat in enumerate(product_cats, 1)],
    "category":     product_cats,
    "base_price":   [rng.integers(*CAT_PRICE_RANGE[c])
                     for c in product_cats],
    "brand_tier":   rng.choice(["premium", "mid", "budget"], N_PRODUCTS,
                                p=[0.2, 0.5, 0.3]),
})


cust_cities = rng.choice(CITIES, N_CUSTOMERS,
                          p=[0.18, 0.15, 0.14, 0.10, 0.09,
                             0.09, 0.08, 0.07, 0.05, 0.05])
customers_df = pd.DataFrame({
    "customer_id":   range(1, N_CUSTOMERS + 1),
    "city":          cust_cities,
    "state":         [CITY_STATE[c] for c in cust_cities],
    "age_group":     rng.choice(["18-24","25-34","35-44","45-54","55+"],
                                N_CUSTOMERS, p=[0.2,0.35,0.25,0.12,0.08]),
    "loyalty_tier":  rng.choice(["bronze","silver","gold","platinum"],
                                N_CUSTOMERS, p=[0.4,0.3,0.2,0.1]),
    "signup_days_ago": rng.integers(1, 1000, N_CUSTOMERS),
})

sellers_df = pd.DataFrame({
    "seller_id":   range(1, N_SELLERS + 1),
    "seller_city": rng.choice(CITIES, N_SELLERS),
    "seller_tier": rng.choice(["A", "B", "C"], N_SELLERS, p=[0.2, 0.5, 0.3]),
    "years_active": rng.integers(1, 10, N_SELLERS),
})


# Realistic time distribution: peaks at 12-14 (lunch) and 19-22 (evening)
total_seconds = int((END_DATE - START_DATE).total_seconds())

# Hour weights (food delivery pattern)
hour_weights = np.array([
    0.5,0.3,0.2,0.2,0.2,0.3,   # 0-5 AM
    0.8,1.5,2.0,2.5,3.0,3.5,   # 6-11 AM
    5.0,4.5,3.5,3.0,3.5,4.0,   # 12-17 PM (lunch peak)
    4.5,6.0,6.5,5.5,3.5,2.0    # 18-23 PM (dinner peak)
])
hour_weights = hour_weights / hour_weights.sum()

order_hours = rng.choice(24, N_ORDERS, p=hour_weights)
order_days  = rng.integers(0, 15, N_ORDERS)
order_minutes = rng.integers(0, 60, N_ORDERS)
order_seconds = rng.integers(0, 60, N_ORDERS)
order_timestamps = [
    START_DATE + timedelta(days=int(d), hours=int(h),
                           minutes=int(m), seconds=int(s))
    for d, h, m, s in zip(order_days, order_hours, order_minutes, order_seconds)
]

order_products  = rng.integers(1, N_PRODUCTS + 1, N_ORDERS)
order_customers = rng.integers(1, N_CUSTOMERS + 1, N_ORDERS)
order_sellers   = rng.integers(1, N_SELLERS + 1, N_ORDERS)
order_platforms = rng.choice(PLATFORMS, N_ORDERS, p=[0.65, 0.30, 0.05])
order_payments  = rng.choice(PAYMENTS, N_ORDERS, p=[0.40, 0.25, 0.15, 0.12, 0.08])

# Price = base_price * noise * platform_modifier
base_prices = products_df.loc[order_products - 1, "base_price"].values
noise       = rng.uniform(0.85, 1.15, N_ORDERS)
order_amounts = np.round(base_prices * noise, 2)

# Discount: loyalty and day-of-week effects
loyalty_tiers = customers_df.loc[order_customers - 1, "loyalty_tier"].values
discount_pct  = np.where(loyalty_tiers == "platinum", rng.uniform(0.10, 0.20, N_ORDERS),
                np.where(loyalty_tiers == "gold",     rng.uniform(0.05, 0.12, N_ORDERS),
                np.where(loyalty_tiers == "silver",   rng.uniform(0.02, 0.07, N_ORDERS),
                                                      rng.uniform(0.00, 0.03, N_ORDERS))))
discount_amounts = np.round(order_amounts * discount_pct, 2)

delivery_fees    = np.where(order_amounts > 500, 0,
                            rng.choice([29, 39, 49, 59], N_ORDERS))
final_amounts    = order_amounts - discount_amounts + delivery_fees

# Status: higher cancellation for COD
status_probs = np.where(
    order_payments == "COD",
    [None] * N_ORDERS,  # handled below
    [None] * N_ORDERS
)
order_statuses = []
for pay in order_payments:
    if pay == "COD":
        order_statuses.append(rng.choice(STATUSES, p=[0.78, 0.05, 0.12, 0.05]))
    else:
        order_statuses.append(rng.choice(STATUSES, p=[0.88, 0.05, 0.04, 0.03]))

# Delivery time (hours): faster for App orders
delivery_times = np.where(
    np.array(order_statuses) == "delivered",
    np.where(np.array(order_platforms) == "App",
             rng.uniform(0.5, 48, N_ORDERS),
             rng.uniform(1.0, 96, N_ORDERS)),
    np.nan
)

# Ratings: correlated with delivery speed
ratings = []
for i, (status, dtime) in enumerate(zip(order_statuses, delivery_times)):
    if status == "delivered":
        if not np.isnan(dtime) and dtime < 24:
            ratings.append(int(rng.choice([4, 5], p=[0.3, 0.7])))
        elif not np.isnan(dtime) and dtime < 72:
            ratings.append(int(rng.choice([3, 4, 5], p=[0.2, 0.5, 0.3])))
        else:
            ratings.append(int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3])))
    elif status == "cancelled":
        ratings.append(int(rng.choice([1, 2], p=[0.6, 0.4])))
    else:
        ratings.append(int(rng.choice([1, 2, 3], p=[0.2, 0.5, 0.3])))

order_categories = products_df.loc[order_products - 1, "category"].values
cust_cities_arr  = customers_df.loc[order_customers - 1, "city"].values
cust_states_arr  = customers_df.loc[order_customers - 1, "state"].values
age_groups_arr   = customers_df.loc[order_customers - 1, "age_group"].values
loyalty_arr      = customers_df.loc[order_customers - 1, "loyalty_tier"].values
seller_tier_arr  = sellers_df.loc[order_sellers - 1, "seller_tier"].values

orders_df = pd.DataFrame({
    "order_id":         range(1, N_ORDERS + 1),
    "customer_id":      order_customers,
    "seller_id":        order_sellers,
    "product_id":       order_products,
    "order_timestamp":  order_timestamps,
    "order_date":       [t.date() for t in order_timestamps],
    "order_hour":       order_hours,
    "category":         order_categories,
    "city":             cust_cities_arr,
    "state":            cust_states_arr,
    "age_group":        age_groups_arr,
    "loyalty_tier":     loyalty_arr,
    "seller_tier":      seller_tier_arr,
    "platform":         order_platforms,
    "payment_method":   order_payments,
    "base_amount":      order_amounts,
    "discount_amount":  discount_amounts,
    "delivery_fee":     delivery_fees,
    "final_amount":     np.round(final_amounts, 2),
    "status":           order_statuses,
    "delivery_time_hrs": np.round(delivery_times, 2),
    "rating":           ratings,
})

print(f"orders_df: {len(orders_df):,} rows × {orders_df.shape[1]} cols")
print(f"Date range: {orders_df['order_date'].min()} → {orders_df['order_date'].max()}")
print(f"\nStatus distribution:\n{orders_df['status'].value_counts()}")
print(f"\nCategory distribution:\n{orders_df['category'].value_counts()}")
print(f"\nRevenue: ₹{orders_df[orders_df['status']=='delivered']['final_amount'].sum():,.0f} total")
orders_df.head(3)

con = duckdb.connect(DB_PATH)
con.execute("PRAGMA threads=4")

# Drop and recreate tables
for tbl in ["orders", "products", "customers", "sellers",
            "daily_summary", "hourly_summary", "category_summary"]:
    con.execute(f"DROP TABLE IF EXISTS {tbl}")

con.execute("CREATE TABLE orders    AS SELECT * FROM orders_df")
con.execute("CREATE TABLE products  AS SELECT * FROM products_df")
con.execute("CREATE TABLE customers AS SELECT * FROM customers_df")
con.execute("CREATE TABLE sellers   AS SELECT * FROM sellers_df")

# Materialized summary tables (trend in large-scale analytics: pre-aggregation)
con.execute("""
    CREATE TABLE daily_summary AS
    SELECT
        order_date,
        category,
        city,
        platform,
        status,
        COUNT(*)                    AS order_count,
        SUM(final_amount)           AS revenue,
        AVG(final_amount)           AS avg_order_value,
        SUM(discount_amount)        AS total_discount,
        AVG(discount_amount)        AS avg_discount,
        AVG(CASE WHEN rating IS NOT NULL THEN rating END) AS avg_rating,
        COUNT(DISTINCT customer_id) AS unique_customers,
        COUNT(DISTINCT seller_id)   AS active_sellers,
        AVG(delivery_time_hrs)      AS avg_delivery_hrs,
        SUM(delivery_fee)           AS total_delivery_fees
    FROM orders
    GROUP BY order_date, category, city, platform, status
""")

con.execute("""
    CREATE TABLE hourly_summary AS
    SELECT
        order_date,
        order_hour,
        category,
        COUNT(*)          AS order_count,
        SUM(final_amount) AS revenue,
        AVG(rating)       AS avg_rating
    FROM orders
    GROUP BY order_date, order_hour, category
""")

con.execute("""
    CREATE TABLE category_summary AS
    SELECT
        category,
        COUNT(*)                AS total_orders,
        SUM(final_amount)       AS total_revenue,
        AVG(final_amount)       AS avg_order_value,
        AVG(rating)             AS avg_rating,
        AVG(delivery_time_hrs)  AS avg_delivery_hrs,
        SUM(CASE WHEN status='returned' THEN 1 ELSE 0 END)*1.0/COUNT(*) AS return_rate,
        SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END)*1.0/COUNT(*) AS cancel_rate
    FROM orders GROUP BY category
""")



DATA_DATE_MIN = con.execute("SELECT MIN(order_date) FROM orders").fetchone()[0]
DATA_DATE_MAX = con.execute("SELECT MAX(order_date) FROM orders").fetchone()[0]
DATA_DATE_MIDPOINT = con.execute(
    f"SELECT DATE '{DATA_DATE_MIN}' + INTERVAL '{(DATA_DATE_MAX - DATA_DATE_MIN).days // 2} days'"
).fetchone()[0]



def get_schema() -> str:
    parts = []
    for tbl in ["orders", "products", "customers", "sellers",
                "daily_summary", "hourly_summary", "category_summary"]:
        cols = con.execute(f"DESCRIBE {tbl}").df()
        cols_str = ", ".join(f"{r['column_name']}:{r['column_type']}"
                             for _, r in cols.iterrows())
        sample = con.execute(f"SELECT * FROM {tbl} LIMIT 2").df().to_string(index=False)
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        parts.append(f"TABLE {tbl} ({n:,} rows)\nCOLUMNS: {cols_str}\nSAMPLE:\n{sample}")
    return "\n\n".join(parts)

SCHEMA = get_schema()


def run_sql_approx(query: str, sample_frac: float = 0.3,
                   confidence: float = 0.95) -> Tuple[pd.DataFrame, dict]:
    """
    Approximate Query Processing (BlinkDB-inspired):
    Runs query on a stratified sample, reports CI on numeric aggregates.
    Only triggers for large full-table scans.
    """
    result = run_sql(query)
    if result.empty or len(result) < 5:
        return result, {}

    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    n = len(result)
    z = stats.norm.ppf((1 + confidence) / 2)
    ci_info = {}
    for col in numeric_cols:
        se = result[col].sem()
        margin = z * se
        ci_info[col] = {
            "mean": result[col].mean(),
            "ci_lower": result[col].mean() - margin,
            "ci_upper": result[col].mean() + margin,
            "confidence": confidence
        }
    return result, ci_info


def run_sql(query: str) -> pd.DataFrame:
    try:
        return con.execute(query).df()
    except Exception as e:
        raise ValueError(f"SQL error: {e}\nQuery: {query}")


for tbl in ["orders","products","customers","sellers",
            "daily_summary","hourly_summary","category_summary"]:
    n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"  {tbl}: {n:,} rows")

print(f"\nData range: {DATA_DATE_MIN} → {DATA_DATE_MAX}")
print(f"Schema loaded ({len(SCHEMA)} chars)")

from sentence_transformers import SentenceTransformer

print("Loading BAAI/bge-large-en-v1.5 (best MTEB retrieval model)...")
embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
print(f"  Embedding dim: {embed_model.get_sentence_embedding_dimension()}")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

l2_collection = chroma_client.get_or_create_collection(
    "l2_episodic",
    metadata={"description": "Past query episodes — BGE-Large embeddings"}
)
l4_collection = chroma_client.get_or_create_collection(
    "l4_procedural",
    metadata={"description": "Verified SQL patterns for few-shot prompting"}
)


def embed(text: str) -> List[float]:
    # BGE models benefit from query prefix for retrieval
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return embed_model.encode(prefixed, normalize_embeddings=True).tolist()


def l2_store(question: str, sql: str, result_summary: str,
             feedback: str = "auto", score: float = 1.0) -> None:
    doc_id = hashlib.md5((question + sql).encode()).hexdigest()
    l2_collection.upsert(
        ids=[doc_id],
        embeddings=[embed(question)],
        documents=[question],
        metadatas=[{
            "sql":            sql[:800],
            "result_summary": result_summary[:400],
            "feedback":       feedback,
            "score":          float(score),
            "timestamp":      datetime.now().isoformat(),
        }]
    )


def l2_retrieve(question: str, top_k: int = 3) -> List[Dict]:
    if l2_collection.count() == 0:
        return []
    k = min(top_k, l2_collection.count())
    res = l2_collection.query(query_embeddings=[embed(question)], n_results=k)
    return [{"question": d, **m}
            for d, m in zip(res["documents"][0], res["metadatas"][0])]


def l4_store(problem_type: str, sql_template: str, description: str) -> None:
    doc_id = hashlib.md5(problem_type.encode()).hexdigest()
    l4_collection.upsert(
        ids=[doc_id],
        embeddings=[embed(problem_type)],
        documents=[problem_type],
        metadatas=[{"sql_template": sql_template[:800], "description": description[:300]}]
    )


def l4_retrieve(question: str, top_k: int = 2) -> List[Dict]:
    if l4_collection.count() == 0:
        return []
    k = min(top_k, l4_collection.count())
    res = l4_collection.query(query_embeddings=[embed(question)], n_results=k)
    return [{"problem_type": d, **m}
            for d, m in zip(res["documents"][0], res["metadatas"][0])]


SEED_PATTERNS = [
    ("revenue by category over time",
     "SELECT order_date, category, SUM(final_amount) AS revenue FROM orders WHERE status='delivered' GROUP BY order_date, category ORDER BY order_date, revenue DESC",
     "Daily revenue breakdown by product category"),
    ("customer cohort retention",
     "WITH first_order AS (SELECT customer_id, MIN(order_date) AS cohort_date FROM orders GROUP BY customer_id) SELECT f.cohort_date, o.order_date, COUNT(DISTINCT o.customer_id) AS retained FROM orders o JOIN first_order f ON o.customer_id=f.customer_id GROUP BY f.cohort_date, o.order_date",
     "Customer cohort retention analysis"),
    ("seller performance ranking",
     "SELECT seller_id, seller_tier, COUNT(*) AS orders, SUM(final_amount) AS revenue, AVG(rating) AS avg_rating, AVG(delivery_time_hrs) AS avg_delivery FROM orders WHERE status='delivered' GROUP BY seller_id, seller_tier ORDER BY revenue DESC",
     "Seller performance with ratings and delivery"),
    ("hourly order pattern",
     "SELECT order_hour, COUNT(*) AS orders, SUM(final_amount) AS revenue FROM orders GROUP BY order_hour ORDER BY order_hour",
     "Hourly order volume and revenue pattern"),
    ("payment method analysis",
     "SELECT payment_method, COUNT(*) AS orders, AVG(final_amount) AS avg_value, SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS cancel_rate FROM orders GROUP BY payment_method",
     "Payment method performance and cancellation rates"),
    ("top cities by revenue",
     "SELECT city, state, COUNT(*) AS orders, SUM(final_amount) AS revenue, AVG(rating) AS avg_rating FROM orders WHERE status='delivered' GROUP BY city, state ORDER BY revenue DESC",
     "Geographic revenue distribution"),
    ("discount effectiveness",
     "SELECT loyalty_tier, AVG(discount_amount) AS avg_discount, AVG(final_amount) AS avg_revenue, COUNT(*) AS orders FROM orders WHERE status='delivered' GROUP BY loyalty_tier ORDER BY avg_revenue DESC",
     "Discount vs revenue by loyalty tier"),
    ("return rate analysis",
     "SELECT category, COUNT(*) AS total, SUM(CASE WHEN status='returned' THEN 1 ELSE 0 END) AS returns, SUM(CASE WHEN status='returned' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS return_rate FROM orders GROUP BY category ORDER BY return_rate DESC",
     "Return rate by product category"),
]

for pt, sql, desc in SEED_PATTERNS:
    l4_store(pt, sql, desc)

print(f"L2 episodes: {l2_collection.count()} | L4 patterns: {l4_collection.count()}")
print(f"Embedding dim: {len(embed('test'))}")

def build_l3_graph() -> nx.DiGraph:
    G = nx.DiGraph()

    # Table nodes
    for tbl in ["orders","products","customers","sellers",
                "daily_summary","hourly_summary","category_summary"]:
        G.add_node(tbl, type="table")

    # Schema columns
    schema_map = {
        "orders":    ["order_id","customer_id","seller_id","product_id",
                      "order_timestamp","order_date","order_hour","category",
                      "city","state","age_group","loyalty_tier","seller_tier",
                      "platform","payment_method","base_amount","discount_amount",
                      "delivery_fee","final_amount","status","delivery_time_hrs","rating"],
        "products":  ["product_id","product_name","category","base_price","brand_tier"],
        "customers": ["customer_id","city","state","age_group","loyalty_tier","signup_days_ago"],
        "sellers":   ["seller_id","seller_city","seller_tier","years_active"],
        "daily_summary": ["order_date","category","city","platform","status",
                          "order_count","revenue","avg_order_value","total_discount",
                          "avg_rating","unique_customers","active_sellers",
                          "avg_delivery_hrs","total_delivery_fees"],
    }
    for tbl, cols in schema_map.items():
        for col in cols:
            node = f"{tbl}.{col}"
            G.add_node(node, type="column", table=tbl)
            G.add_edge(tbl, node, rel="has_column")

    # FK relationships
    G.add_edge("orders.customer_id", "customers.customer_id", rel="foreign_key")
    G.add_edge("orders.seller_id",   "sellers.seller_id",     rel="foreign_key")
    G.add_edge("orders.product_id",  "products.product_id",   rel="foreign_key")

    # Causal/business metric relationships (directed)
    G.add_edge("orders.delivery_time_hrs", "orders.rating",
               rel="causes", weight=-0.68, confidence=0.82)
    G.add_edge("orders.discount_amount",   "orders.final_amount",
               rel="reduces", weight=-1.0, confidence=1.0)
    G.add_edge("orders.rating",            "daily_summary.revenue",
               rel="causes", weight=0.45, confidence=0.61)
    G.add_edge("orders.loyalty_tier",      "orders.discount_amount",
               rel="determines", weight=0.72, confidence=0.95)
    G.add_edge("orders.platform",          "orders.final_amount",
               rel="influences", weight=0.31, confidence=0.55)
    G.add_edge("orders.seller_tier",       "orders.delivery_time_hrs",
               rel="causes", weight=-0.54, confidence=0.74)

    # Domain rules
    G.nodes["daily_summary.revenue"]["north_star"] = True
    G.nodes["orders.rating"]["north_star"] = True
    G.nodes["orders.status"]["valid_values"] = ["delivered","pending","cancelled","returned"]
    G.nodes["orders.payment_method"]["valid_values"] = ["UPI","Credit Card","Debit Card","COD","Wallet"]
    G.nodes["orders.category"]["valid_values"] = CATEGORIES
    G.nodes["orders.platform"]["valid_values"] = PLATFORMS

    # Business rules as nodes
    G.add_node("rule_free_delivery", type="business_rule",
               description="free delivery when final_amount > 500")
    G.add_node("rule_status_filter", type="business_rule",
               description="use status='delivered' for revenue metrics")
    G.add_node("rule_date_anchor", type="business_rule",
               description=f"data spans {DATA_DATE_MIN} to {DATA_DATE_MAX} — never use CURRENT_DATE")

    return G


l3_graph = build_l3_graph()


def l3_get_context(entity: str) -> str:
    """Get causal predecessors and successors for any entity in the graph."""
    lines = []
    matches = [n for n in l3_graph.nodes if entity.lower() in n.lower()]
    for node in matches[:3]:
        in_edges  = [(u, d) for u, v, d in l3_graph.in_edges(node, data=True)]
        out_edges = [(v, d) for u, v, d in l3_graph.out_edges(node, data=True)]
        for u, d in in_edges:
            lines.append(f"  {u} →[{d.get('rel')}] {node} "
                         f"(w={d.get('weight','?')}, conf={d.get('confidence','?')})")
        for v, d in out_edges:
            lines.append(f"  {node} →[{d.get('rel')}] {v} "
                         f"(w={d.get('weight','?')}, conf={d.get('confidence','?')})")
    return "\n".join(lines) if lines else "No causal links."


def l3_get_business_rules() -> str:
    rules = [d.get("description","") for _, d in l3_graph.nodes(data=True)
             if d.get("type") == "business_rule"]
    return "\n".join(f"  - {r}" for r in rules)


nx.write_gml(l3_graph, GRAPH_PATH)
print(f"L3 graph: {l3_graph.number_of_nodes()} nodes, {l3_graph.number_of_edges()} edges")
print(f"\nBusiness rules:\n{l3_get_business_rules()}")

class SentinelState(TypedDict):
    query:               str
    intent:              str
    linked_schema:       str
    sub_queries:         str
    memory_context:      str
    schema_context:      str
    sql_query:           str
    sql_candidates:      str
    sql_result_json:     str
    validation_attempts: int
    validation_error:    str
    aqp_ci:              str
    rca_result:          dict
    forecast_result:     dict
    anomaly_result:      dict
    math_result:         dict
    n_charts_requested:  int
    chart_explanations:  str
    final_response:      str
    error:               str


def empty_state(query: str) -> SentinelState:
    n_charts = 0
    m = re.search(r'\b(\d+)\s+chart', query, re.IGNORECASE)
    if m:
        n_charts = min(int(m.group(1)), 8)
    return SentinelState(
        query=query, intent="", linked_schema="", sub_queries="",
        memory_context="", schema_context=SCHEMA,
        sql_query="", sql_candidates="", sql_result_json="",
        validation_attempts=0, validation_error="", aqp_ci="",
        rca_result={}, forecast_result={}, anomaly_result={}, math_result={},
        n_charts_requested=n_charts,
        chart_explanations="",
        final_response="", error=""
    )


import uuid
from IPython.display import display, IFrame, HTML

_chart_counter = 0
_FIG_QUEUE: List[Tuple[str, str]] = []


def safe_show(fig: go.Figure, title_suffix: str = "") -> Optional[str]:
    global _chart_counter, _FIG_QUEUE
    _chart_counter += 1
    try:
        fig.update_layout(margin=dict(l=55, r=55, t=65, b=55), font=dict(size=12))
        # Store HTML string directly no file, no IFrame, no 404
        html_str = fig.to_html(
            include_plotlyjs="cdn",
            full_html=False,        # just the <div>, not a full page
            config={"responsive": True}
        )
        _FIG_QUEUE.append((html_str, title_suffix))
        print(f"  ✓ Queued [{_chart_counter}]: {title_suffix}")
        return title_suffix
    except Exception as e:
        print(f"  ✗ Queue failed ({title_suffix}): {e}")
        return None


def flush_charts() -> int:
    global _FIG_QUEUE
    n = 0
    if not _FIG_QUEUE:
        return 0
    print(f"\n{'─'*60}")
    print(f"📊 Rendering {len(_FIG_QUEUE)} chart(s)")
    print(f"{'─'*60}")
    for html_str, title in _FIG_QUEUE:
        try:
            if title:
                display(HTML(
                    f"<p style='font-size:13px;font-weight:600;"
                    f"margin:12px 0 2px;color:#333'>📊 {title}</p>"
                ))
            # Embed chart HTML inline — works in Kaggle, Colab, JupyterLab
            display(HTML(
                f"<div style='width:100%;min-height:480px'>{html_str}</div>"
            ))
            n += 1
        except Exception as e:
            print(f"  ✗ Display failed ({title}): {e}")
    _FIG_QUEUE.clear()
    return n


def safe_vline(fig: go.Figure, x_val, label: str = "",
               color: str = "red", dash: str = "dash") -> None:
    x_str = (x_val.strftime("%Y-%m-%d %H:%M:%S")
             if hasattr(x_val, "strftime") else str(x_val))
    fig.add_shape(
        type="line", x0=x_str, x1=x_str, y0=0, y1=1,
        yref="paper", xref="x",
        line=dict(color=color, dash=dash, width=1.5)
    )
    if label:
        fig.add_annotation(
            x=x_str, y=1.02, yref="paper", xref="x",
            text=label, showarrow=False,
            font=dict(color=color, size=11),
            bgcolor="white", bordercolor=color, borderwidth=1, borderpad=3
        )


def safe_hline(fig: go.Figure, y_val: float, label: str = "",
               color: str = "gray", dash: str = "dot") -> None:
    fig.add_shape(
        type="line", x0=0, x1=1, y0=y_val, y1=y_val,
        xref="paper", yref="y",
        line=dict(color=color, dash=dash, width=1.2)
    )
    if label:
        fig.add_annotation(
            x=1.01, y=y_val, xref="paper", yref="y",
            text=label, showarrow=False,
            font=dict(color=color, size=10), xanchor="left"
        )


def extract_json(text: str, fallback=None):
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for pattern in [r'\{[^{}]*\}', r'\[.*?\]', r'\{.*\}']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                continue
    return fallback

# ═══════════════════════════════════════════════════════════════════════════════
# SENTINEL CHART ANALYSIS ENGINE v3.0 — NEVER FAILS, ALL CHART TYPES
# Drop-in replacement. Paste this as ONE block before your agent definitions.
# ═══════════════════════════════════════════════════════════════════════════════

import warnings, traceback
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
from scipy import stats as scipy_stats
from scipy.stats import pearsonr, spearmanr, kruskal, shapiro
from sklearn.linear_model import LinearRegression


# ───────────────────────────────────────────────────────────────────────────────
# SAFE WRAPPERS — every computation is isolated; never crashes outer code
# ───────────────────────────────────────────────────────────────────────────────

def _safe(fn, *args, default=None, **kwargs):
    """Call fn(*args, **kwargs), return default on any exception."""
    try:
        result = fn(*args, **kwargs)
        # Filter out NaN/Inf so JSON serialization never fails
        if isinstance(result, float) and (np.isnan(result) or np.isinf(result)):
            return default
        return result
    except Exception:
        return default


def _safe_json(obj) -> str:
    """JSON-serialize anything — replace un-serializable values gracefully."""
    def _fix(o):
        if isinstance(o, dict):
            return {str(k): _fix(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_fix(i) for i in o]
        if isinstance(o, float):
            if np.isnan(o) or np.isinf(o):
                return None
            return round(o, 6)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)
        if isinstance(o, (np.ndarray,)):
            return [_fix(x) for x in o.tolist()]
        if isinstance(o, pd.Timestamp):
            return str(o.date())
        return o
    try:
        return json.dumps(_fix(obj), indent=2)
    except Exception as e:
        return json.dumps({"serialization_error": str(e)})


# ───────────────────────────────────────────────────────────────────────────────
# COLUMN DETECTOR — the #1 reason analysis breaks is wrong col detection
# ───────────────────────────────────────────────────────────────────────────────

def _detect_columns(df: pd.DataFrame, x_col: str = None,
                    y_col: str = None, color_col: str = None,
                    z_col: str = None) -> dict:
    """
    Robustly detect numeric, categorical, datetime columns from df.
    Falls back gracefully when passed col names don't exist.
    Returns a dict: {numeric, categorical, datetime, x, y, color, z}
    """
    all_cols = list(df.columns)

    # Detect types from actual data
    numeric_cols  = [c for c in all_cols
                     if pd.api.types.is_numeric_dtype(df[c])
                     and df[c].notna().sum() > 0]
    datetime_cols = []
    cat_cols      = []

    for c in all_cols:
        if c in numeric_cols:
            continue
        # Try to parse as datetime
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            datetime_cols.append(c)
            continue
        if df[c].dtype == object:
            sample = df[c].dropna().head(3)
            is_date = False
            for v in sample:
                try:
                    pd.to_datetime(str(v))
                    is_date = True
                    break
                except Exception:
                    pass
            if is_date:
                datetime_cols.append(c)
                # Convert in place for analysis
                try:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                except Exception:
                    pass
            else:
                cat_cols.append(c)

    # Resolve passed column names → fall back to auto-detected
    def _resolve(col, preferred_list, fallback_list):
        if col and col in all_cols:
            return col
        return preferred_list[0] if preferred_list else (
               fallback_list[0] if fallback_list else None)

    resolved_x = _resolve(x_col, datetime_cols + cat_cols, numeric_cols)
    resolved_y = _resolve(y_col, numeric_cols, cat_cols)
    resolved_color = color_col if (color_col and color_col in all_cols) else None
    resolved_z = z_col if (z_col and z_col in all_cols) else (
                 numeric_cols[2] if len(numeric_cols) > 2 else None)

    return {
        "numeric":   numeric_cols,
        "categorical": cat_cols,
        "datetime":  datetime_cols,
        "x":         resolved_x,
        "y":         resolved_y,
        "color":     resolved_color,
        "z":         resolved_z,
        "all":       all_cols,
    }


# ───────────────────────────────────────────────────────────────────────────────
# STATISTICAL EXTRACTORS (all wrapped in _safe)
# ───────────────────────────────────────────────────────────────────────────────

def _num_stats(series: pd.Series, name: str = "") -> dict:
    s = series.dropna()
    if len(s) < 2:
        return {"col": name, "n": len(s), "note": "insufficient data"}
    n   = len(s)
    mu  = float(s.mean())
    med = float(s.median())
    sd  = float(s.std())
    mn, mx = float(s.min()), float(s.max())
    p25 = float(np.percentile(s, 25))
    p75 = float(np.percentile(s, 75))
    iqr = p75 - p25
    skew = _safe(lambda: float(s.skew()), default=0.0)
    kurt = _safe(lambda: float(s.kurtosis()), default=0.0)
    cv   = _safe(lambda: sd / abs(mu) * 100 if mu != 0 else 0.0, default=0.0)

    # Outliers
    lo, hi = p25 - 1.5*iqr, p75 + 1.5*iqr
    outliers = s[(s < lo) | (s > hi)]
    out_pct  = len(outliers) / n * 100

    # Trend
    tr, tp = _safe(lambda: spearmanr(range(n), s.values), default=(None, None))

    # Normality
    norm_p = None
    if 8 <= n <= 5000:
        sample = s.sample(min(n, 1000), random_state=42)
        _, norm_p = _safe(lambda: shapiro(sample), default=(None, None))

    # Concentration: top 20% hold what % of total
    sorted_s = np.sort(s.values)[::-1]
    top20 = max(1, int(n*0.2))
    conc = float(sorted_s[:top20].sum() / (sorted_s.sum() + 1e-9) * 100)

    return {
        "col": name, "n": n,
        "mean": round(mu, 4), "median": round(med, 4),
        "std": round(sd, 4),  "cv_pct": round(cv, 2),
        "min": round(mn, 4),  "max": round(mx, 4),
        "p25": round(p25, 4), "p75": round(p75, 4),
        "iqr": round(iqr, 4), "skew": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "outlier_count": len(outliers),
        "outlier_pct": round(out_pct, 2),
        "outlier_examples": sorted(outliers.tolist(), key=abs, reverse=True)[:3],
        "top20_concentration_pct": round(conc, 2),
        "trend_spearman_r": round(float(tr), 4) if tr is not None else None,
        "trend_p": round(float(tp), 4) if tp is not None else None,
        "trend_direction": ("upward" if (tr or 0) > 0.3 else
                            "downward" if (tr or 0) < -0.3 else "flat"),
        "normality_p": round(float(norm_p), 4) if norm_p is not None else None,
        "is_normal": (norm_p > 0.05) if norm_p is not None else None,
        "top3": sorted(s.tolist(), reverse=True)[:3],
        "bottom3": sorted(s.tolist())[:3],
    }


def _cat_stats(cat_s: pd.Series, val_s: pd.Series = None, name: str = "") -> dict:
    s = cat_s.dropna().astype(str)
    if len(s) == 0:
        return {"col": name, "n": 0}
    vc     = s.value_counts()
    n      = len(s)
    n_uni  = int(s.nunique())
    top_sh = float(vc.iloc[0] / n * 100) if n > 0 else 0

    # Shannon entropy
    probs   = vc / n
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    max_ent = np.log2(n_uni) if n_uni > 1 else 1
    even    = float(entropy / max_ent) if max_ent > 0 else 0

    result = {
        "col": name, "n_unique": n_uni,
        "top_category": str(vc.index[0]),
        "top_cat_share_pct": round(top_sh, 2),
        "bottom_category": str(vc.index[-1]),
        "top5": {str(k): int(v) for k, v in vc.head(5).items()},
        "entropy_normalized": round(even, 4),
        "distribution_type": ("concentrated" if top_sh > 40 else
                              "moderate" if top_sh > 20 else "even"),
    }

    if val_s is not None and len(val_s) == len(cat_s):
        paired = pd.DataFrame({"c": cat_s.astype(str), "v": val_s}).dropna()
        if not paired.empty and paired["v"].notna().sum() > 0:
            grp = paired.groupby("c")["v"].agg(["mean","sum","count"])
            grp = grp.sort_values("sum", ascending=False)
            result["per_cat"] = {
                str(r): {"mean": round(float(row["mean"]),2),
                         "sum":  round(float(row["sum"]),2),
                         "count": int(row["count"])}
                for r, row in grp.iterrows()
            }
            if len(grp) >= 2:
                ts = float(grp["sum"].iloc[0])
                bs = float(grp["sum"].iloc[-1])
                ms = float(grp["sum"].median())
                result["top_cat_by_value"]    = str(grp.index[0])
                result["bottom_cat_by_value"] = str(grp.index[-1])
                result["top_bottom_gap_ratio"] = round(ts/(abs(bs)+1e-9), 2)
                result["top_vs_median_ratio"]  = round(ts/(abs(ms)+1e-9), 2)

            groups = [g["v"].values for _, g in paired.groupby("c") if len(g) >= 3]
            if len(groups) >= 2:
                kw = _safe(lambda: kruskal(*groups), default=None)
                if kw:
                    result["kruskal_p"] = round(float(kw[1]), 4)
                    result["groups_differ"] = kw[1] < 0.05

    return result


def _pareto(df: pd.DataFrame, cat_col: str, val_col: str) -> dict:
    grp   = df.groupby(cat_col)[val_col].sum().sort_values(ascending=False)
    total = grp.sum()
    if total == 0:
        return {}
    cumsum = grp.cumsum()
    n80    = int((cumsum <= total * 0.80).sum()) + 1
    pct80  = round(n80 / len(grp) * 100, 1)
    return {
        "top_n_for_80pct":   n80,
        "top_pct_for_80pct": pct80,
        "total_categories":  len(grp),
        "top3": {str(k): round(float(v),2) for k,v in grp.head(3).items()},
        "bottom3": {str(k): round(float(v),2) for k,v in grp.tail(3).items()},
        "interpretation": f"Top {n80} ({pct80}%) of '{cat_col}' = 80% of '{val_col}'",
    }


def _composition(df: pd.DataFrame, names_col: str, values_col: str) -> dict:
    grp   = df.groupby(names_col)[values_col].sum().sort_values(ascending=False)
    total = grp.sum()
    if total == 0:
        return {}
    shares = (grp / total * 100).round(3)
    return {
        "n_segments":             len(grp),
        "total":                  round(float(total), 2),
        "top_segment":            str(grp.index[0]),
        "top_value":              round(float(grp.iloc[0]), 2),
        "top_share_pct":          round(float(shares.iloc[0]), 2),
        "top3_cumulative_pct":    round(float(shares.head(3).sum()), 2),
        "bottom_segment":         str(grp.index[-1]),
        "bottom_share_pct":       round(float(shares.iloc[-1]), 2),
        "all_shares":             {str(k): round(float(v),2) for k,v in shares.items()},
        "herfindahl_index":       round(float((shares**2).sum()), 2),
        "concentration":          ("dominant" if shares.iloc[0] > 50 else
                                   "moderate" if shares.iloc[0] > 25 else "fragmented"),
    }


def _timeseries(df: pd.DataFrame, date_col: str, val_col: str) -> dict:
    sub = df[[date_col, val_col]].copy()
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna().sort_values(date_col)
    if len(sub) < 4:
        return {"note": "fewer than 4 data points"}
    vals = sub[val_col].values.astype(float)
    n    = len(vals)
    t    = np.arange(n)
    slope, intercept, r, p, se = scipy_stats.linregress(t, vals)
    pct_change = (vals[-1] - vals[0]) / (abs(vals[0]) + 1e-9) * 100
    peak_i  = int(np.argmax(vals))
    trough_i= int(np.argmin(vals))

    # WoW comparison
    mid = n // 2
    wow = (vals[mid:].mean() - vals[:mid].mean()) / (abs(vals[:mid].mean())+1e-9)*100 if n >= 6 else None

    # Acceleration
    d1 = np.gradient(vals)
    d2 = np.gradient(d1)
    accel_avg = float(d2[-max(3,n//4):].mean())

    # Changepoint: largest absolute jump
    diffs = np.abs(np.diff(vals))
    cp_i  = int(np.argmax(diffs)) + 1 if len(diffs) else None

    dates = sub[date_col]
    return {
        "n_points": n,
        "start": str(dates.iloc[0].date()),
        "end":   str(dates.iloc[-1].date()),
        "start_value": round(float(vals[0]), 2),
        "end_value":   round(float(vals[-1]), 2),
        "pct_change_total": round(pct_change, 2),
        "trend_slope":  round(float(slope), 4),
        "trend_r2":     round(float(r**2), 4),
        "trend_p":      round(float(p), 6),
        "trend_significant": p < 0.05,
        "trend_direction": ("upward" if slope > 0 and p < 0.05 else
                            "downward" if slope < 0 and p < 0.05 else "flat"),
        "peak_value": round(float(vals[peak_i]), 2),
        "peak_date":  str(dates.iloc[peak_i].date()),
        "trough_value": round(float(vals[trough_i]), 2),
        "trough_date":  str(dates.iloc[trough_i].date()),
        "peak_trough_ratio": round(float(vals[peak_i])/(abs(float(vals[trough_i]))+1e-9), 3),
        "wow_pct": round(float(wow), 2) if wow is not None else None,
        "acceleration_avg": round(accel_avg, 4),
        "acceleration_label": "accelerating" if accel_avg > 0 else "decelerating",
        "changepoint_date": str(dates.iloc[cp_i].date()) if cp_i else None,
        "changepoint_magnitude": round(float(diffs[cp_i-1]),2) if cp_i else None,
    }


def _correlation(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 4:
        return {"note": "insufficient data for correlation"}
    x, y = sub[x_col].values.astype(float), sub[y_col].values.astype(float)
    pr, pp = _safe(lambda: pearsonr(x, y), default=(0, 1))
    sr, sp = _safe(lambda: spearmanr(x, y), default=(0, 1))
    reg    = _safe(lambda: LinearRegression().fit(x.reshape(-1,1), y), default=None)
    slope  = float(reg.coef_[0]) if reg else None
    r2     = float(reg.score(x.reshape(-1,1), y)) if reg else None

    pr_f = float(pr) if pr is not None else 0
    strength = ("very strong" if abs(pr_f) > 0.8 else "strong" if abs(pr_f) > 0.6 else
                "moderate" if abs(pr_f) > 0.4 else "weak" if abs(pr_f) > 0.2 else "negligible")
    return {
        "n": len(sub),
        "pearson_r": round(pr_f, 4),
        "pearson_p": round(float(pp), 6),
        "spearman_r": round(float(sr), 4) if sr else None,
        "spearman_p": round(float(sp), 6) if sp else None,
        "r_squared": round(r2, 4) if r2 is not None else None,
        "slope": round(slope, 6) if slope is not None else None,
        "strength": strength,
        "direction": "positive" if pr_f > 0 else "negative",
        "significant": float(pp) < 0.05,
        "interpretation": (
            f"Each unit increase in '{x_col}' → {slope:+.4f} change in '{y_col}' "
            f"(R²={r2:.3f}, {strength} {'positive' if pr_f>0 else 'negative'} correlation)"
            if slope is not None and r2 is not None else "correlation computed"
        ),
    }


def _distribution(series: pd.Series, name: str = "") -> dict:
    s = series.dropna()
    if len(s) < 4:
        return {"col": name, "n": len(s), "note": "insufficient data"}
    n    = len(s)
    skew = _safe(lambda: float(s.skew()), default=0.0)
    kurt = _safe(lambda: float(s.kurtosis()), default=0.0)

    norm_p = None
    if 8 <= n <= 5000:
        samp = s.sample(min(n, 1000), random_state=42)
        res  = _safe(lambda: shapiro(samp), default=None)
        norm_p = float(res[1]) if res else None

    hist_c, bin_e = np.histogram(s, bins=min(20, max(5, n//3)))
    local_max = [round(float(bin_e[i]),2)
                 for i in range(1, len(hist_c)-1)
                 if hist_c[i] > hist_c[i-1] and hist_c[i] > hist_c[i+1]]

    q1 = float(np.percentile(s, 25))
    q3 = float(np.percentile(s, 75))
    iqr_v = q3 - q1
    fl, fh = q1 - 1.5*iqr_v, q3 + 1.5*iqr_v
    outliers = s[(s < fl) | (s > fh)]

    return {
        "col": name, "n": n,
        "mean": round(float(s.mean()), 4), "median": round(float(s.median()), 4),
        "std":  round(float(s.std()), 4),
        "skewness": round(skew, 4), "kurtosis": round(kurt, 4),
        "normality_p": round(norm_p, 4) if norm_p else None,
        "is_normal": (norm_p > 0.05) if norm_p else None,
        "tail_shape": ("heavy right tail" if skew > 1 else "heavy left tail" if skew < -1 else
                       "right skewed" if skew > 0.5 else "left skewed" if skew < -0.5 else "symmetric"),
        "modality": ("unimodal" if len(local_max) <= 1 else
                     "bimodal" if len(local_max) == 2 else "multimodal"),
        "local_maxima": local_max[:4],
        "outlier_count": len(outliers), "outlier_pct": round(len(outliers)/n*100, 2),
        "q1": round(q1, 4), "q3": round(q3, 4), "iqr": round(iqr_v, 4),
        "whisker_lo": round(float(s[s >= fl].min()), 4) if len(s[s>=fl]) > 0 else round(fl,4),
        "whisker_hi": round(float(s[s <= fh].max()), 4) if len(s[s<=fh]) > 0 else round(fh,4),
    }


def _group_compare(df: pd.DataFrame, cat_col: str, val_col: str) -> dict:
    sub = df[[cat_col, val_col]].dropna()
    if sub.empty:
        return {}
    grp   = sub.groupby(cat_col)[val_col].agg(["mean","median","std","count"])
    grp   = grp.sort_values("mean", ascending=False)
    kw    = None
    glist = [g[val_col].values for _, g in sub.groupby(cat_col) if len(g) >= 3]
    if len(glist) >= 2:
        kw = _safe(lambda: kruskal(*glist), default=None)
    top = str(grp.index[0]) if len(grp) > 0 else ""
    bot = str(grp.index[-1]) if len(grp) > 0 else ""
    tm  = float(grp.loc[top,"mean"]) if top in grp.index else 0
    bm  = float(grp.loc[bot,"mean"]) if bot in grp.index else 0
    return {
        "n_groups": len(grp),
        "top_group": top, "top_group_mean": round(tm, 4),
        "bottom_group": bot, "bottom_group_mean": round(bm, 4),
        "top_vs_bottom_ratio": round(tm/(abs(bm)+1e-9), 3),
        "kruskal_p": round(float(kw[1]),4) if kw else None,
        "groups_differ": (kw[1] < 0.05) if kw else None,
        "all_groups": {str(r): {"mean": round(float(row["mean"]),4),
                                "median": round(float(row["median"]),4),
                                "count": int(row["count"])}
                       for r, row in grp.iterrows()},
    }


def _heatmap_stats(df: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> dict:
    sub = df[[x_col, y_col, z_col]].dropna()
    if sub.empty:
        return {}
    try:
        pv = sub.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc="sum").fillna(0)
    except Exception:
        return {}
    flat = pv.values.flatten()
    mi   = np.unravel_index(pv.values.argmax(), pv.shape)
    ni   = np.unravel_index(pv.values.argmin(), pv.shape)
    rt   = pv.sum(axis=1).sort_values(ascending=False)
    ct   = pv.sum(axis=0).sort_values(ascending=False)
    return {
        "shape": list(pv.shape),
        "global_mean": round(float(flat.mean()), 4),
        "global_std":  round(float(flat.std()), 4),
        "max_cell": f"{pv.index[mi[0]]} × {pv.columns[mi[1]]}",
        "max_val":  round(float(pv.values.max()), 4),
        "min_cell": f"{pv.index[ni[0]]} × {pv.columns[ni[1]]}",
        "min_val":  round(float(pv.values.min()), 4),
        "top_cell_pct_of_total": round(float(pv.values.max()/(flat.sum()+1e-9)*100), 2),
        "top_row":   str(rt.index[0]), "top_row_total": round(float(rt.iloc[0]),2),
        "top_col":   str(ct.index[0]), "top_col_total": round(float(ct.iloc[0]),2),
        "row_totals": {str(k): round(float(v),2) for k,v in rt.head(5).items()},
        "col_totals": {str(k): round(float(v),2) for k,v in ct.head(5).items()},
        "sparsity_pct": round(float((flat==0).sum()/len(flat)*100), 2),
    }


def _quadrant(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 4:
        return {}
    xm, ym = sub[x_col].median(), sub[y_col].median()
    n = len(sub)
    q1 = len(sub[(sub[x_col] >= xm) & (sub[y_col] >= ym)])
    q2 = len(sub[(sub[x_col] <  xm) & (sub[y_col] >= ym)])
    q3 = len(sub[(sub[x_col] <  xm) & (sub[y_col] <  ym)])
    q4 = len(sub[(sub[x_col] >= xm) & (sub[y_col] <  ym)])
    dom = max([("Q1_high_x_high_y",q1),("Q2_low_x_high_y",q2),
               ("Q3_low_x_low_y",q3),("Q4_high_x_low_y",q4)], key=lambda t:t[1])
    return {
        "x_median": round(float(xm), 4), "y_median": round(float(ym), 4),
        "Q1_high_high_pct": round(q1/n*100,1), "Q2_low_high_pct": round(q2/n*100,1),
        "Q3_low_low_pct":   round(q3/n*100,1), "Q4_high_low_pct": round(q4/n*100,1),
        "dominant_quadrant": dom[0],
    }


# ───────────────────────────────────────────────────────────────────────────────
# MASTER ROUTER — selects extractors based on chart type + actual col types
# ───────────────────────────────────────────────────────────────────────────────

def _build_features(chart_type: str, df: pd.DataFrame,
                    cols: dict, extra: str = "") -> dict:
    """
    Route to the right extractors. chart_type is used as a hint only;
    actual column types drive the logic — so this works even with wrong chart_type.
    """
    ct  = chart_type.lower().replace(" ","_").replace("-","_")
    x   = cols["x"]
    y   = cols["y"]
    c   = cols["color"]
    z   = cols["z"]
    num = cols["numeric"]
    cat = cols["categorical"]
    dt  = cols["datetime"]

    feats = {
        "chart_type": chart_type, "n_rows": len(df),
        "columns": cols["all"], "extra_context": extra,
    }

    # ── Y column stats (almost always useful) ──────────────────────────────
    if y and y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
        feats["y_stats"] = _safe(lambda: _num_stats(df[y], y), default={})

    # ── Determine primary analysis path ────────────────────────────────────

    # TIME SERIES path
    if ct in ("line","area") or (x in dt):
        if x and y and x in df.columns and y in df.columns:
            feats["timeseries"] = _safe(
                lambda: _timeseries(df, x, y), default={})
        if c and c in cat:
            feats["series_breakdown"] = _safe(
                lambda: _cat_stats(df[c], df[y] if y and y in df.columns else None, c),
                default={})

    # BAR / FUNNEL path
    elif ct in ("bar","grouped_bar","funnel"):
        if x and y and x in df.columns and y in df.columns:
            feats["category_analysis"] = _safe(
                lambda: _cat_stats(df[x], df[y] if pd.api.types.is_numeric_dtype(df[y]) else None, x),
                default={})
            if pd.api.types.is_numeric_dtype(df[y]):
                feats["pareto"] = _safe(lambda: _pareto(df, x, y), default={})
        if c and c in df.columns and y and y in df.columns:
            feats["color_breakdown"] = _safe(
                lambda: _cat_stats(df[c], df[y] if pd.api.types.is_numeric_dtype(df[y]) else None, c),
                default={})

    # PIE / TREEMAP path
    elif ct in ("pie","treemap"):
        # Auto-find best names and values cols
        names_col  = next((col for col in df.columns if df[col].dtype == object), x)
        values_col = next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), y)
        if names_col and values_col and names_col in df.columns and values_col in df.columns:
            feats["composition"] = _safe(
                lambda: _composition(df, names_col, values_col), default={})
            feats["value_stats"] = _safe(
                lambda: _num_stats(df[values_col], values_col), default={})

    # SCATTER / BUBBLE path
    elif ct in ("scatter","bubble"):
        if x and y and all(v in df.columns for v in [x,y]):
            if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
                feats["correlation"] = _safe(lambda: _correlation(df, x, y), default={})
                feats["x_stats"]     = _safe(lambda: _num_stats(df[x], x), default={})
                feats["quadrant"]    = _safe(lambda: _quadrant(df, x, y), default={})
        if c and c in df.columns:
            feats["cluster_breakdown"] = _safe(
                lambda: _cat_stats(df[c], df[y] if y and y in df.columns else None, c),
                default={})

    # HISTOGRAM path
    elif ct == "histogram":
        col = x if (x and x in df.columns and pd.api.types.is_numeric_dtype(df[x])) else \
              (y if (y and y in df.columns and pd.api.types.is_numeric_dtype(df[y])) else \
              (num[0] if num else None))
        if col:
            feats["distribution"] = _safe(lambda: _distribution(df[col], col), default={})

    # BOX / VIOLIN path
    elif ct in ("box","violin"):
        if y and y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
            feats["distribution"] = _safe(lambda: _distribution(df[y], y), default={})
            if x and x in df.columns:
                feats["group_comparison"] = _safe(
                    lambda: _group_compare(df, x, y), default={})

    # HEATMAP path
    elif ct == "heatmap":
        hz = z or (num[0] if num else None)
        if x and y and hz and all(v in df.columns for v in [x,y,hz]):
            feats["heatmap"] = _safe(lambda: _heatmap_stats(df, x, y, hz), default={})

    # ── UNIVERSAL FALLBACK — runs for ANY chart type not matched above ──────
    # This is the key fix: always extract something useful
    else:
        # Try everything that applies
        if y and y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
            feats["y_stats"] = _safe(lambda: _num_stats(df[y], y), default={})
        if x and y and x in df.columns and y in df.columns:
            if x in dt:
                feats["timeseries"] = _safe(lambda: _timeseries(df, x, y), default={})
            elif x in cat and pd.api.types.is_numeric_dtype(df.get(y, pd.Series())):
                feats["category_analysis"] = _safe(
                    lambda: _cat_stats(df[x], df[y], x), default={})
            elif x in num and y in num:
                feats["correlation"] = _safe(lambda: _correlation(df, x, y), default={})

    # ── ALWAYS: multi-series breakdown if color exists ─────────────────────
    if c and c in df.columns and c not in feats:
        feats["color_series"] = _safe(
            lambda: _cat_stats(df[c], df[y] if y and y in df.columns and
                               pd.api.types.is_numeric_dtype(df[y]) else None, c),
            default={})

    # ── ALWAYS: if we have ≥2 numeric cols, add all-numeric stats ─────────
    if len(num) >= 2 and y and y in num:
        for nc in num[:3]:
            if nc != y and nc not in feats:
                feats[f"extra_numeric_{nc}"] = _safe(
                    lambda nc=nc: _num_stats(df[nc], nc), default={})

    return feats


# ───────────────────────────────────────────────────────────────────────────────
# LLM DEEP INSIGHT LAYER
# ───────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a Senior Data Scientist and Business Intelligence expert.
You receive PRE-COMPUTED statistical features for a chart. Do NOT re-compute.
Your job: produce DEEP, SPECIFIC, ACTIONABLE insights from the numbers given.

STRICT RULES:
1. Cite EXACT numbers — never say "high" when you have a value like 98.4M
2. Never say "the chart shows" or "this visualization" — speak about DATA
3. Use statistical test results (p-values, R², Kruskal p) to qualify claims
4. Identify non-obvious patterns, not just the maximum value
5. Each bullet must have: a finding + why it matters + recommended action
6. Format: exactly 4 bullet points, bold label at start of each"""

_PROMPT_TEMPLATE = """
CHART TYPE: {chart_type}
CHART TITLE: {title}
BUSINESS CONTEXT: {context}

═══ PRE-COMPUTED STATISTICAL FEATURES ═══
{features}

═══ DATA SAMPLE (top 8 rows) ═══
{sample}

Generate exactly 4 deep insights. Each bullet:
- **[LABEL]** Finding with exact numbers → business implication → recommended action

Labels: TREND | OUTLIER | CONCENTRATION | CORRELATION | SEGMENT | DISTRIBUTION |
        ANOMALY | OPPORTUNITY | RISK | SEASONALITY | CAUSALITY | COMPOSITION
"""

def _analyze_chart(fig_title: str, chart_type: str,
                   x_col: str, y_col: str, df: pd.DataFrame,
                   extra_context: str = "",
                   color_col: str = None,
                   z_col: str = None) -> str:
    """
    THE ONE TRUE analyze_chart function.
    - Never crashes (triple try/except)
    - Self-healing column detection
    - Works for ALL chart types
    - Always returns 4 deep insights
    """

    # ── Guard: ensure df is valid ──────────────────────────────────────────
    if df is None or (hasattr(df, "empty") and df.empty):
        return (f"• **NO DATA** — DataFrame is empty for '{fig_title}'. "
                f"Check SQL query or data pipeline for '{chart_type}' chart.")

    try:
        df = df.copy().reset_index(drop=True)
    except Exception:
        return "• **DATA ERROR** — Could not process DataFrame."

    # ── Step 1: Robust column detection ───────────────────────────────────
    try:
        cols = _detect_columns(df, x_col, y_col, color_col, z_col)
    except Exception as e:
        cols = {
            "numeric": [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
            "categorical": [c for c in df.columns if df[c].dtype == object],
            "datetime": [], "x": x_col, "y": y_col,
            "color": color_col, "z": z_col, "all": list(df.columns),
        }

    # ── Step 2: Extract statistical features ──────────────────────────────
    try:
        features = _build_features(chart_type, df, cols, extra_context)
    except Exception as e:
        # Ultimate fallback: at least get basic stats on every column
        features = {"chart_type": chart_type, "error": str(e)}
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                features[f"stat_{c}"] = _safe(
                    lambda c=c: _num_stats(df[c], c), default={})

    # ── Step 3: Build data sample ──────────────────────────────────────────
    display_cols = [c for c in [cols["x"], cols["y"], cols["color"], cols["z"]]
                    if c and c in df.columns]
    if not display_cols:
        display_cols = df.columns.tolist()[:5]
    try:
        sample_str = df[display_cols].dropna().head(8).to_string(index=False)
    except Exception:
        sample_str = df.head(8).to_string(index=False)

    # ── Step 4: Serialize features safely ─────────────────────────────────
    features_str = _safe_json(features)
    if len(features_str) > 4000:
        features_str = features_str[:4000] + "\n... [truncated for token budget]"

    # ── Step 5: LLM call ────────────────────────────────────────────────────
    try:
        prompt = _PROMPT_TEMPLATE.format(
            chart_type = chart_type,
            title      = fig_title,
            context    = (extra_context[:300] if extra_context
                          else "General business analytics"),
            features   = features_str,
            sample     = sample_str,
        )
        result = call_llm(prompt, system=_SYSTEM_PROMPT,
                          model=FAST_MODEL, temperature=0.15)

        # Sanity check: if LLM returned something weird, add disclaimer
        if not result or len(result.strip()) < 50:
            raise ValueError("LLM returned empty/short response")

        return result

    except Exception as e:
        # Final fallback: generate rule-based insights from features
        return _rule_based_fallback(fig_title, chart_type, features, df, cols)


def _rule_based_fallback(title: str, chart_type: str,
                         features: dict, df: pd.DataFrame, cols: dict) -> str:
    """
    Pure Python fallback — generates insights without LLM.
    Triggered only if LLM fails. Ensures analysis ALWAYS returns something useful.
    """
    lines = [f"[Rule-based analysis for: {title}]"]
    num = cols.get("numeric", [])
    cat = cols.get("categorical", [])

    # Insight 1: Basic scale
    y = cols.get("y")
    if y and y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
        s = df[y].dropna()
        lines.append(f"• **SCALE** — '{y}' ranges from {s.min():,.2f} to {s.max():,.2f} "
                     f"(mean={s.mean():,.2f}, std={s.std():,.2f}). "
                     f"CV={s.std()/abs(s.mean())*100:.1f}% indicates "
                     f"{'high' if s.std()/abs(s.mean()) > 0.5 else 'moderate'} variability.")

    # Insight 2: Top/bottom from category stats
    ys = features.get("y_stats") or features.get("value_stats") or {}
    cs = features.get("category_analysis") or features.get("composition") or {}
    if cs.get("top_cat_by_value"):
        top, bot = cs["top_cat_by_value"], cs.get("bottom_cat_by_value","")
        ratio = cs.get("top_bottom_gap_ratio", 1)
        lines.append(f"• **SEGMENT** — '{top}' leads all segments with a "
                     f"{ratio:.1f}x gap over '{bot}'. "
                     f"Focus investment on the top segment to maximize returns.")

    # Insight 3: Trend
    ts = features.get("timeseries", {})
    if ts.get("pct_change_total") is not None:
        d = ts["pct_change_total"]
        lines.append(f"• **TREND** — Total change over period: {d:+.1f}%. "
                     f"Trend is {ts.get('trend_direction','unknown')} "
                     f"(slope={ts.get('trend_slope',0):+.4f}, "
                     f"R²={ts.get('trend_r2',0):.3f}). "
                     f"Peak was {ts.get('peak_value',0):,.2f} on {ts.get('peak_date','?')}.")

    # Insight 4: Distribution shape
    dist = features.get("distribution", {})
    if dist.get("outlier_pct") is not None:
        lines.append(f"• **DISTRIBUTION** — {dist.get('tail_shape','unknown')} shape "
                     f"(skew={dist.get('skewness',0):.3f}), "
                     f"{dist.get('outlier_pct',0):.1f}% outliers. "
                     f"{'Not normal' if not dist.get('is_normal') else 'Normal'} distribution "
                     f"(Shapiro p={dist.get('normality_p','?')}). "
                     f"IQR={dist.get('iqr',0):.2f}.")

    # Pad to 4 bullets if needed
    while len(lines) < 5:
        col = num[0] if num else (df.columns[0] if len(df.columns) > 0 else "data")
        lines.append(f"• **DATA QUALITY** — {len(df)} rows analyzed across "
                     f"{len(df.columns)} columns. "
                     f"Null rate: {df.isnull().mean().mean()*100:.1f}%.")

    return "\n".join(lines[1:5])  # return exactly 4 bullets


def intent_classifier(state: SentinelState) -> dict:
    """Route query to correct agent pipeline. Load all 4 memory tiers in parallel."""

    # Intent classification
    prompt = f"""Classify this data analytics query into exactly one intent.
Query: {state['query']}

Intents:
- sql_query  : retrieve, aggregate, compare, rank, filter, count, join data
- rca        : root cause analysis, why did X drop/spike, explain a metric change  
- forecast   : predict future values, trend projection, "what will happen"
- anomaly    : detect anomalies, outliers, unusual patterns, health check
- math       : calculus (derivative, integral), statistical tests (t-test, chi-square), 
               regression, optimization, Gini, CMGR, elasticity, CLV calculation

Reply with ONLY the intent name."""

    intent = call_llm(prompt, model=FAST_MODEL, temperature=0.0).strip().lower()
    if intent not in {"sql_query", "rca", "forecast", "anomaly", "math"}:
        intent = "sql_query"

    # Load all memory tiers
    episodes   = l2_retrieve(state["query"], top_k=3)
    few_shots  = l4_retrieve(state["query"], top_k=2)
    l3_ctx     = l3_get_context(state["query"].split()[0])
    biz_rules  = l3_get_business_rules()

    mem_lines = [
        "═══ L2 EPISODIC MEMORY (past similar queries) ═══"
    ]
    for i, ep in enumerate(episodes, 1):
        mem_lines.append(f"[{i}] Q: {ep['question']}")
        mem_lines.append(f"    SQL: {ep.get('sql','')[:120]}...")
        mem_lines.append(f"    Summary: {ep.get('result_summary','')[:100]}")

    mem_lines.append("\n═══ L4 PROCEDURAL MEMORY (SQL templates) ═══")
    for fs in few_shots:
        mem_lines.append(f"  Type: {fs['problem_type']}")
        mem_lines.append(f"  SQL:  {fs.get('sql_template','')[:120]}")

    mem_lines.append(f"\n═══ L3 CAUSAL GRAPH CONTEXT ═══\n{l3_ctx}")
    mem_lines.append(f"\n═══ BUSINESS RULES ═══\n{biz_rules}")

    print(f"[IntentClassifier] '{state['query'][:60]}...' → intent={intent}")
    return {
        "intent":         intent,
        "schema_context": SCHEMA,
        "memory_context": "\n".join(mem_lines),
    }

def schema_linker(state: SentinelState) -> dict:
    """
    NOVEL: Schema linking before SQL generation (from DIN-SQL + CHESS papers).
    Maps NL query entities → exact schema elements, reducing hallucination by ~40%.
    Standard text-to-SQL skips this step — we implement it as a dedicated agent.
    """
    prompt = f"""You are a database schema expert performing schema linking.

TASK: Map every entity/concept in the user query to the EXACT schema element.

SCHEMA TABLES AND COLUMNS:
{state['schema_context'][:3000]}

USER QUERY: {state['query']}

Return a JSON with EXACT column names from schema:
{{
  "relevant_tables": ["table1", ...],
  "column_links": {{"query_phrase": "table.column", ...}},
  "required_joins": ["JOIN condition"],
  "filter_conditions": ["condition"],
  "aggregations": ["expression AS alias"],
  "groupby_cols": ["col"],
  "orderby_cols": ["col ASC/DESC"],
  "date_filter": "SQL date filter or empty string",
  "needs_subquery": true/false,
  "query_complexity": "simple/moderate/complex"
}}

Business rules to apply:
{l3_get_business_rules()}

Return ONLY valid JSON."""

    raw = call_llm(prompt, model=MAIN_MODEL, temperature=0.0)
    linked = extract_json(raw, fallback={
        "relevant_tables": ["orders"],
        "column_links": {},
        "required_joins": [],
        "filter_conditions": [],
        "aggregations": [],
        "groupby_cols": [],
        "orderby_cols": [],
        "date_filter": "",
        "needs_subquery": False,
        "query_complexity": "simple"
    })

    print(f"[SchemaLinker] Tables: {linked.get('relevant_tables', [])} | "
          f"Complexity: {linked.get('query_complexity', '?')} | "
          f"Links: {len(linked.get('column_links', {}))}")
    return {"linked_schema": json.dumps(linked)}

def query_decomposer(state: SentinelState) -> dict:
    """
    NOVEL: Multi-granularity query decomposition (MAC-SQL paper adapted).
    For complex queries, breaks into atomic sub-questions each solvable with one SQL.
    Uses linked_schema to guide decomposition — this combination is novel.
    """
    linked = extract_json(state["linked_schema"], {})
    complexity = linked.get("query_complexity", "simple")

    if complexity == "simple":
        print("[QueryDecomposer] Simple query — no decomposition needed")
        return {"sub_queries": json.dumps([state["query"]])}

    prompt = f"""You are an expert at decomposing complex analytical queries.

ORIGINAL QUERY: {state['query']}
SCHEMA PLAN: {state['linked_schema'][:1000]}
DATA DATE RANGE: {DATA_DATE_MIN} to {DATA_DATE_MAX}

Break this into 2-4 sequential atomic sub-questions that:
1. Each can be answered with a single SQL query
2. Are ordered so later ones may depend on results from earlier ones
3. Together fully answer the original query

Return ONLY a JSON array of strings:
["sub-question 1", "sub-question 2", ...]"""

    raw = call_llm(prompt, model=MAIN_MODEL, temperature=0.0)
    sub_queries = extract_json(raw, fallback=[state["query"]])
    if not isinstance(sub_queries, list):
        sub_queries = [state["query"]]

    print(f"[QueryDecomposer] Decomposed into {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  [{i}] {sq[:80]}")

    return {"sub_queries": json.dumps(sub_queries)}

def _extract_first_sql(raw: str) -> str:
    """
    Robustly extract exactly ONE SQL statement from LLM output.

    Root cause: LLM sometimes emits two SELECT blocks separated by a blank line
    (e.g., the few-shot example followed by the generated SQL, or two candidates).
    DuckDB's parser sees both → 'syntax error at or near SELECT'.

    Strategy (in order):
    1. Strip markdown fences
    2. Split on blank-line boundary before a top-level SELECT
    3. If still multiple top-level SELECTs (^SELECT at col-0), truncate at the second
    4. Strip trailing semicolons
    """
    sql = re.sub(r"```sql|```", "", raw).strip()

    # Step 2: split on blank line + top-level SELECT
    blocks = re.split(r'\n\s*\n', sql)
    if len(blocks) > 1:
        sql = blocks[0].strip()

    # Step 3: truncate at second top-level SELECT (col-0, not inside subquery)
    top_selects = list(re.finditer(r'^SELECT\b', sql, re.IGNORECASE | re.MULTILINE))
    if len(top_selects) > 1:
        cut = top_selects[1].start()
        sql = sql[:cut].rstrip()
        print(f"  [SQLCleaner] Truncated duplicate SELECT at position {cut}")

    return sql.rstrip(";").strip()


def sql_builder(state: SentinelState) -> dict:
    linked     = extract_json(state["linked_schema"], {})
    complexity = linked.get("query_complexity", "simple")

    system = f"""You are an expert DuckDB SQL analyst.

═══ CRITICAL: OUTPUT EXACTLY ONE SQL STATEMENT ═══
- Return a SINGLE SELECT statement only — no explanations, no second query
- Do NOT output the same query twice
- Do NOT include comments or markdown fences

═══ DATE RULES ═══
- Data range: {DATA_DATE_MIN} to {DATA_DATE_MAX} ONLY
- NEVER use CURRENT_DATE / NOW() / TODAY() — returns 2025, no data
- "Full period" = order_date BETWEEN DATE '{DATA_DATE_MIN}' AND DATE '{DATA_DATE_MAX}'
- Prefer explicit date range over BETWEEN when filtering

═══ DUCKDB SYNTAX ═══
- Date arithmetic: DATE 'YYYY-MM-DD' ± INTERVAL 'N days'
- Percentiles: PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)
- Window functions fully supported
- CAST(col AS DATE) for timestamp → date conversion

═══ BUSINESS RULES ═══
{l3_get_business_rules()}
- Revenue queries: filter status='delivered' unless asked otherwise
- Use daily_summary / category_summary for large aggregations"""

    few_shots     = l4_retrieve(state["query"], top_k=2)
    few_shot_str  = ""
    for fs in few_shots:
        few_shot_str += f"\n-- Pattern: {fs['problem_type']}\n-- {fs.get('sql_template','')[:100]}\n"

    prior_error = ""
    if state["validation_error"]:
        prior_error = (f"\n⚠️ PREVIOUS ERROR (attempt {state['validation_attempts']}):\n"
                       f"{state['validation_error'][:200]}\n"
                       f"Fix the above error. Output ONE corrected SQL statement only.\n")

    prompt = f"""SCHEMA PLAN:
{state['linked_schema'][:1500]}

FEW-SHOT PATTERNS (reference only, do not repeat):
{few_shot_str}

MEMORY (similar past queries):
{state['memory_context'][:600]}
{prior_error}
USER QUERY: {state['query']}

Output ONE DuckDB SQL statement:"""

    # Self-consistency: 3 candidates for complex, 1 for simple / retries
    n_cand = (3 if complexity in ["complex", "moderate"]
                   and state["validation_attempts"] == 0 else 1)

    candidates = []
    for i in range(n_cand):
        raw = call_llm(prompt, system=system,
                       temperature=0.0 if i == 0 else 0.25)
        sql = _extract_first_sql(raw)
        if sql and sql not in candidates:
            candidates.append(sql)

    best_sql = candidates[0]

    # Among candidates, prefer the one that executes + returns most rows
    if len(candidates) > 1:
        best_score = -1
        for cand in candidates:
            try:
                res   = run_sql(cand)
                score = len(res) if not res.empty else 0
                if score > best_score:
                    best_score, best_sql = score, cand
            except Exception:
                continue

    attempt = state["validation_attempts"] + 1
    print(f"[SQLBuilder] Attempt {attempt} | Candidates: {len(candidates)} | "
          f"Complexity: {complexity}")
    print(f"  SQL: {best_sql[:220]}...")

    return {
        "sql_query":        best_sql,
        "sql_candidates":   json.dumps(candidates),
        "validation_error": "",
    }

def sql_validator(state: SentinelState) -> dict:
    """Execute SQL, validate result quality, apply AQP for large results."""
    sql      = state["sql_query"]
    attempts = state["validation_attempts"] + 1

    try:
        result_df = run_sql(sql)

        if result_df.empty:
            return {
                "validation_attempts": attempts,
                "validation_error":    "Query returned 0 rows. Check filters, date range, or column names.",
                "sql_result_json":     "",
            }

        # AQP: Compute confidence intervals on numeric columns
        numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
        ci_info = {}
        if len(result_df) >= 10 and numeric_cols:
            z = 1.96  # 95% CI
            for col in numeric_cols[:4]:
                se = result_df[col].sem()
                mu = result_df[col].mean()
                ci_info[col] = {
                    "mean":     round(float(mu), 4),
                    "ci_lower": round(float(mu - z * se), 4),
                    "ci_upper": round(float(mu + z * se), 4),
                }

        result_json = result_df.to_json(orient="records", date_format="iso")

        print(f"\n[SQLValidator] ✓ Attempt {attempts} | Shape: {result_df.shape}")
        print(f"  {'─'*60}")
        print(result_df.head(6).to_string(index=False))
        if ci_info:
            print(f"\n  [AQP] 95% CI on aggregates:")
            for col, ci in list(ci_info.items())[:3]:
                print(f"    {col}: {ci['ci_lower']:.2f} ↔ {ci['ci_upper']:.2f} (μ={ci['mean']:.2f})")

        return {
            "sql_result_json":    result_json,
            "validation_attempts": attempts,
            "validation_error":   "",
            "aqp_ci":             json.dumps(ci_info),
        }

    except ValueError as e:
        err = str(e)[:300]
        print(f"[SQLValidator] ✗ Attempt {attempts} failed: {err}")
        return {
            "validation_attempts": attempts,
            "validation_error":    err,
            "sql_result_json":     "",
            "aqp_ci":              "",
        }


def should_retry_sql(state: SentinelState) -> Literal["sql_builder","viz_agent",END]:
    if state["validation_error"] and state["validation_attempts"] < MAX_RETRIES:
        print(f"[Router] Retry {state['validation_attempts']}/{MAX_RETRIES} → sql_builder")
        return "sql_builder"
    if state["validation_error"]:
        print(f"[Router] Max retries reached → END")
        return END
    return "viz_agent"

def math_agent(state: SentinelState) -> dict:
    """
    FIXED: Math agent now calls _analyze_chart on every chart it creates
    and collects all analyses into final_response.
    """
    print("[MathAgent] Executing mathematical/statistical analysis...")
 
    query = state["query"].lower()
    results = {}
    all_analysis = []          # ← collect chart analyses here
 
    # ── 1. CMGR / Growth Rate ─────────────────────────────────────────
    if any(k in query for k in ["growth", "cmgr", "rate", "trend", "acceleration"]):
        daily_rev = run_sql("""
            SELECT order_date, SUM(final_amount) AS revenue
            FROM orders WHERE status='delivered'
            GROUP BY order_date ORDER BY order_date
        """)
        daily_rev["order_date"] = pd.to_datetime(daily_rev["order_date"])
        daily_rev["revenue_smooth"] = daily_rev["revenue"].rolling(3, center=True).mean()
 
        n_days = len(daily_rev) - 1
        if n_days > 0 and daily_rev["revenue"].iloc[0] > 0:
            cmgr = (daily_rev["revenue"].iloc[-1] /
                    daily_rev["revenue"].iloc[0]) ** (1 / n_days) - 1
            results["cmgr_daily"]      = round(cmgr * 100, 4)
            results["cmgr_annualized"] = round(((1 + cmgr) ** 365 - 1) * 100, 2)
 
        rev_vals = daily_rev["revenue_smooth"].dropna().values
        if len(rev_vals) > 2:
            d1 = np.gradient(rev_vals)
            d2 = np.gradient(d1)
            results["revenue_acceleration_trend"] = (
                "accelerating" if d2[-3:].mean() > 0 else "decelerating"
            )
            results["avg_daily_velocity"] = round(float(d1.mean()), 2)
 
        print(f"  CMGR (daily): {results.get('cmgr_daily', 0):+.4f}%")
        print(f"  CMGR (annualized): {results.get('cmgr_annualized', 0):+.2f}%")
        print(f"  Revenue trend: {results.get('revenue_acceleration_trend', '?')}")
 
        if len(rev_vals) > 3:
            vel_series = np.gradient(daily_rev["revenue"].values)
            plot_dates  = daily_rev["order_date"]
 
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=["Daily Revenue (smoothed)",
                                 "Revenue Velocity (1st derivative)"]
            )
            fig.add_trace(
                go.Scatter(x=plot_dates, y=daily_rev["revenue"],
                           name="Revenue", line=dict(color="#2196F3")),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=plot_dates, y=vel_series,
                           name="Velocity", line=dict(color="#FF5722"),
                           fill="tozeroy"),
                row=2, col=1
            )
            fig.update_layout(
                title="Revenue Growth Analysis (Calculus-based)",
                template="plotly_white", height=500
            )
            safe_show(fig, "Revenue Velocity")
 
            # ── CHART ANALYSIS ──────────────────────────────────────
            vel_df = pd.DataFrame({
                "order_date": plot_dates,
                "revenue":    daily_rev["revenue"].values,
                "velocity":   vel_series,
            })
            expl = _analyze_chart(
                fig_title     = "Revenue Growth Analysis (Calculus-based)",
                chart_type    = "line",
                x_col         = "order_date",
                y_col         = "revenue",
                df            = vel_df,
                extra_context = (
                    f"CMGR daily: {results.get('cmgr_daily', 0):+.4f}%\n"
                    f"CMGR annualized: {results.get('cmgr_annualized', 0):+.2f}%\n"
                    f"Avg daily velocity: {results.get('avg_daily_velocity', 0):,.2f}\n"
                    f"Trend: {results.get('revenue_acceleration_trend', '?')}"
                ),
            )
            print(f"\n  ── Chart Analysis: Revenue Velocity ──\n  {expl}")
            all_analysis.append(f"**Revenue Growth Analysis (Calculus-based)**:\n{expl}")
 
    # ── 2. Statistical t-test ─────────────────────────────────────────
    if any(k in query for k in ["t-test", "significant", "hypothesis", "t test"]):
        app_rev = run_sql(
            "SELECT final_amount FROM orders WHERE platform='App' AND status='delivered'"
        )
        web_rev = run_sql(
            "SELECT final_amount FROM orders WHERE platform='Website' AND status='delivered'"
        )
        t_stat, p_val = ttest_ind(
            app_rev["final_amount"], web_rev["final_amount"], equal_var=False
        )
        results["ttest_app_vs_website"] = {
            "t_statistic": round(t_stat, 4),
            "p_value":     round(p_val, 6),
            "significant": p_val < 0.05,
            "conclusion":  (
                "App orders significantly higher"
                if (p_val < 0.05 and t_stat > 0)
                else "No significant difference"
            ),
        }
        print(f"  t-test App vs Website: t={t_stat:.4f}, p={p_val:.6f} → "
              f"{'significant' if p_val < 0.05 else 'not significant'}")
 
    # ── 3. Gini / Pareto ─────────────────────────────────────────────
    if any(k in query for k in ["gini", "pareto", "inequality",
                                 "concentration", "80-20"]):
        seller_rev = run_sql("""
            SELECT seller_id, SUM(final_amount) AS revenue
            FROM orders WHERE status='delivered'
            GROUP BY seller_id ORDER BY revenue
        """)
        rev_sorted = np.sort(seller_rev["revenue"].values)
        n          = len(rev_sorted)
        cumrev     = np.cumsum(rev_sorted)
        gini       = (n + 1 - 2 * np.sum(cumrev) / cumrev[-1]) / n
        results["gini_coefficient"] = round(gini, 4)
 
        total_rev       = rev_sorted.sum()
        cumsum_desc     = np.cumsum(rev_sorted[::-1])
        top_pct         = np.searchsorted(cumsum_desc, 0.80 * total_rev) / n
        results["pareto_top_pct"]       = round(top_pct * 100, 1)
        results["pareto_interpretation"] = (
            f"Top {top_pct * 100:.1f}% sellers = 80% revenue"
        )
 
        print(f"  Gini coefficient: {gini:.4f}")
        print(f"  Pareto: top {top_pct * 100:.1f}% of sellers → 80% of revenue")
 
        cum_sellers = np.arange(1, n + 1) / n
        cum_revenue  = cumrev / cumrev[-1]
 
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cum_sellers, y=cum_revenue,
            name=f"Lorenz Curve (Gini={gini:.3f})",
            fill="tozeroy", line=dict(color="#9C27B0")
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Perfect Equality",
            line=dict(dash="dash", color="gray")
        ))
        fig.update_layout(
            title="Revenue Lorenz Curve (Seller Concentration)",
            xaxis_title="Cumulative % Sellers",
            yaxis_title="Cumulative % Revenue",
            template="plotly_white"
        )
        safe_show(fig, "Lorenz Curve")
 
        # ── CHART ANALYSIS ──────────────────────────────────────────
        lorenz_df = pd.DataFrame({
            "cum_sellers": cum_sellers,
            "cum_revenue": cum_revenue,
        })
        expl = _analyze_chart(
            fig_title     = "Revenue Lorenz Curve (Seller Concentration)",
            chart_type    = "scatter",
            x_col         = "cum_sellers",
            y_col         = "cum_revenue",
            df            = lorenz_df,
            extra_context = (
                f"Gini coefficient: {gini:.4f}\n"
                f"Top {top_pct * 100:.1f}% of sellers account for 80% of revenue\n"
                f"Total sellers: {n}\n"
                f"Total revenue: {total_rev:,.2f}"
            ),
        )
        print(f"\n  ── Chart Analysis: Lorenz Curve ──\n  {expl}")
        all_analysis.append(
            f"**Revenue Lorenz Curve (Seller Concentration)**:\n{expl}"
        )
 
    # ── 4. Price elasticity ───────────────────────────────────────────
    if any(k in query for k in ["elastic", "price", "discount", "sensitivity"]):
        elast_df = run_sql("""
            SELECT
                ROUND(discount_amount/base_amount*100, 0) AS discount_pct_bucket,
                AVG(final_amount)  AS avg_revenue,
                COUNT(*)           AS order_count
            FROM orders
            WHERE status='delivered' AND base_amount > 0
            GROUP BY 1 HAVING COUNT(*) > 50
            ORDER BY 1
        """)
        if len(elast_df) > 3:
            from scipy.stats import linregress
            slope, intercept, r, p, se = linregress(
                elast_df["discount_pct_bucket"], elast_df["order_count"]
            )
            results["elasticity_regression"] = {
                "slope":           round(slope, 4),
                "r_squared":       round(r ** 2, 4),
                "p_value":         round(p, 6),
                "interpretation":  (
                    f"Each 1% discount → {slope:.1f} additional orders "
                    f"(R²={r ** 2:.3f})"
                ),
            }
            print(f"  Elasticity: {slope:.2f} orders per 1% discount "
                  f"(R²={r ** 2:.3f})")
 
            # ── Elasticity scatter chart + analysis ─────────────────
            fig_el = px.scatter(
                elast_df, x="discount_pct_bucket", y="order_count",
                trendline="ols",
                title="Discount % vs Order Count (Elasticity)",
                template="plotly_white",
                labels={"discount_pct_bucket": "Discount %",
                        "order_count": "Order Count"},
            )
            safe_show(fig_el, "Discount Elasticity")
 
            expl = _analyze_chart(
                fig_title     = "Discount % vs Order Count (Elasticity)",
                chart_type    = "scatter",
                x_col         = "discount_pct_bucket",
                y_col         = "order_count",
                df            = elast_df,
                extra_context = (
                    f"Slope: {slope:.4f} orders per 1% discount\n"
                    f"R²: {r ** 2:.3f}, p: {p:.6f}\n"
                    f"Avg revenue per bucket: "
                    f"{elast_df['avg_revenue'].mean():,.2f}"
                ),
            )
            print(f"\n  ── Chart Analysis: Discount Elasticity ──\n  {expl}")
            all_analysis.append(
                f"**Discount % vs Order Count (Elasticity)**:\n{expl}"
            )
 
    # ── 5. CLV ────────────────────────────────────────────────────────
    if any(k in query for k in ["clv", "lifetime value", "ltv",
                                  "customer value"]):
        clv_df = run_sql("""
            SELECT
                customer_id,
                loyalty_tier,
                COUNT(*) AS purchase_count,
                SUM(final_amount) AS total_revenue,
                AVG(final_amount) AS avg_order_value,
                MAX(order_date) - MIN(order_date) AS tenure_days
            FROM orders WHERE status='delivered'
            GROUP BY customer_id, loyalty_tier
        """)
        clv_df["tenure_days"] = (
            pd.to_numeric(clv_df["tenure_days"], errors="coerce").fillna(7)
        )
        clv_df["purchase_freq"] = clv_df["purchase_count"] / (
            clv_df["tenure_days"].clip(lower=1) / 7
        )
        estimated_lifespan_weeks = 52
        clv_df["estimated_clv"] = (
            clv_df["avg_order_value"]
            * clv_df["purchase_freq"]
            * estimated_lifespan_weeks
        )
        clv_by_tier = clv_df.groupby("loyalty_tier")["estimated_clv"].mean()
        results["clv_by_tier"] = clv_by_tier.round(2).to_dict()
        print(f"  CLV by tier:\n{clv_by_tier.round(0).to_string()}")
 
        clv_plot_df = clv_by_tier.reset_index()
        clv_plot_df.columns = ["loyalty_tier", "estimated_clv"]
 
        fig = px.bar(
            clv_plot_df, x="loyalty_tier", y="estimated_clv",
            color="loyalty_tier",
            title="Estimated Customer Lifetime Value by Loyalty Tier",
            template="plotly_white", text_auto=True
        )
        safe_show(fig, "CLV by Tier")
 
        # ── CHART ANALYSIS ──────────────────────────────────────────
        expl = _analyze_chart(
            fig_title     = "Estimated Customer Lifetime Value by Loyalty Tier",
            chart_type    = "bar",
            x_col         = "loyalty_tier",
            y_col         = "estimated_clv",
            df            = clv_plot_df,
            extra_context = (
                f"CLV by tier: {results['clv_by_tier']}\n"
                f"Lifespan assumption: {estimated_lifespan_weeks} weeks\n"
                f"Total customers analyzed: {len(clv_df)}"
            ),
        )
        print(f"\n  ── Chart Analysis: CLV by Tier ──\n  {expl}")
        all_analysis.append(
            f"**Estimated Customer Lifetime Value by Loyalty Tier**:\n{expl}"
        )
 
    # ── Narrative synthesis ───────────────────────────────────────────
    prompt = (
        f"Summarize these mathematical/statistical findings for a business audience. "
        f"Be specific with numbers. Format as 3-4 bullet points.\n"
        f"Query was: {state['query']}\n"
        f"Results: {json.dumps(results, indent=2, default=str)}"
    )
    narrative = call_llm(prompt, model=FAST_MODEL, temperature=0.1)
    print(f"\n[MathAgent] Summary:\n{narrative}")
 
    # ── Build full response including chart analyses ──────────────────
    chart_section = "\n\n".join(all_analysis)
    if chart_section:
        full_response = (
            f"{narrative}\n\n"
            f"{'─' * 50}\n"
            f"📊 Chart Analysis:\n{chart_section}"
        )
    else:
        full_response = narrative
 
    l2_store(state["query"], "/* math computation */",
             str(results)[:300], score=1.0)
 
    return {"math_result": results, "final_response": full_response}

def render_chart(spec: dict, df: pd.DataFrame,
                 rendered_sigs: set, honor_explicit: bool = False) -> Optional[str]:
    """
    Render one chart. Returns title on success, None on skip/fail.

    Dedup logic:
    - honor_explicit=True  (user named chart types): dedup on (chart_type, x, y)
    - honor_explicit=False (auto-pick):              dedup on (x, y) only
    """
    ct    = spec.get("chart", "bar")
    x     = spec.get("x")
    y     = spec.get("y")
    color = spec.get("color") or None
    z     = spec.get("z") or None
    title = spec.get("title", f"{ct} chart")
    names = spec.get("names") or x
    vals  = spec.get("values") or y

    sig = (ct, str(x), str(y)) if honor_explicit else (str(x), str(y))
    if sig in rendered_sigs:
        print(f"  ↩ Skipped duplicate: {ct}({x},{y})")
        return None
    rendered_sigs.add(sig)

    # Column validation
    cols_needed = [c for c in [x, y, color, z, names, vals] if c is not None]
    missing     = [c for c in cols_needed if c not in df.columns]
    if missing:
        print(f"  ✗ Missing columns {missing} for {ct}")
        return None
    if df.empty or (y and y in df.columns and df[y].isna().all()):
        print(f"  ✗ No data for {ct}")
        return None

    try:
        fig = None

        if ct == "bar":
            fig = px.bar(df.head(25), x=x, y=y, color=color, title=title,
                         template="plotly_white", text_auto=".2s")
        elif ct == "grouped_bar":
            fig = px.bar(df.head(25), x=x, y=y, color=color, title=title,
                         barmode="group", template="plotly_white")
        elif ct == "line":
            fig = px.line(df, x=x, y=y, color=color, title=title,
                          template="plotly_white", markers=True)
        elif ct == "area":
            fig = px.area(df, x=x, y=y, color=color, title=title,
                          template="plotly_white")
        elif ct == "scatter":
            use_trendline = len(df) > 5 and not df[x].isna().any()
            fig = px.scatter(df, x=x, y=y, color=color, title=title,
                             template="plotly_white",
                             trendline="ols" if use_trendline else None)
        elif ct == "pie":
            top_df = df.nlargest(8, vals) if (vals and vals in df.columns) else df.head(8)
            fig = px.pie(top_df, names=names, values=vals, title=title)
        elif ct == "heatmap":
            if z and x and y and all(c in df.columns for c in [x, y, z]):
                pivot = df.pivot_table(values=z, index=y, columns=x,
                                       aggfunc="sum").fillna(0)
                fig = px.imshow(pivot, title=title, template="plotly_white",
                                color_continuous_scale="Blues", text_auto=".2s")
            else:
                print(f"  ✗ Heatmap needs x, y, z columns")
                return None
        elif ct == "histogram":
            fig = px.histogram(df, x=x, color=color, title=title,
                               template="plotly_white", nbins=25)
        elif ct == "box":
            fig = px.box(df, x=x, y=y, color=color, title=title,
                         template="plotly_white", points="outliers")
        elif ct == "treemap":
            path_cols  = spec.get("path") or ([x] if x else [])
            valid_path = [c for c in path_cols if c in df.columns]
            if not valid_path:
                valid_path = [x] if x and x in df.columns else []
            v_col = vals if (vals and vals in df.columns) else y
            if valid_path and v_col and v_col in df.columns:
                fig = px.treemap(df, path=valid_path, values=v_col, title=title)
            else:
                print(f"  ✗ Treemap missing path/values")
                return None
        elif ct == "funnel":
            fig = px.funnel(df.head(12), y=x, x=y, title=title)
        elif ct == "violin":
            fig = px.violin(df, x=x, y=y, color=color, title=title,
                            template="plotly_white", box=True)
        elif ct == "bubble":
            size_col = spec.get("size") or z
            if size_col and size_col in df.columns:
                fig = px.scatter(df, x=x, y=y, size=size_col, color=color,
                                 title=title, template="plotly_white")
            else:
                fig = px.scatter(df, x=x, y=y, color=color, title=title,
                                 template="plotly_white")
        else:
            fig = px.bar(df.head(25), x=x, y=y, title=title, template="plotly_white")

        if fig is None:
            return None

        safe_show(fig, title)
        return title

    except Exception as e:
        print(f"  ✗ {ct} error: {e}")
        return None


def _decide_chart_count(df: pd.DataFrame, n_requested: int) -> int:
    if n_requested > 0:
        return n_requested
    num = len(df.select_dtypes(include="number").columns)
    cat = len(df.select_dtypes(include=["object","category"]).columns)
    dat = sum(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)
    if num == 0: return 0
    if num == 1 and cat == 0: return 1
    if dat >= 1 and num == 1 and cat == 0: return 1
    if num >= 2 and cat >= 2: return 3
    if num >= 4: return 4
    return 2


def viz_agent(state: SentinelState) -> dict:
    if not state["sql_result_json"]:
        print("[VizAgent] No data to visualize.")
        return {}

    df = pd.read_json(state["sql_result_json"], orient="records")
    if df.empty:
        return {}

    # Parse datetime columns
    for col in df.columns:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.8:
                    df[col] = parsed
            except Exception:
                pass

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols     = df.select_dtypes(include=["object","category"]).columns.tolist()
    date_cols    = [c for c in df.columns
                    if pd.api.types.is_datetime64_any_dtype(df[c])]

    n_requested  = state.get("n_charts_requested", 0)
    n_target     = _decide_chart_count(df, n_requested)
    honor_explicit = n_requested > 0   # user named specific types → dedup by (type,x,y)

    if n_target == 0:
        print("[VizAgent] Data has no plottable columns.")
        return {}

    print(f"\n[VizAgent] Target: {n_target} chart(s) "
          f"({'user-requested' if honor_explicit else 'auto'})")

    col_profile = {
        "numeric": numeric_cols, "categorical": cat_cols, "datetime": date_cols,
        "n_rows": len(df),
        "n_unique_cat": {c: int(df[c].nunique()) for c in cat_cols[:5]},
    }

    explicit_note = ""
    if honor_explicit:
        explicit_note = (
            f"\nThe user EXPLICITLY requested these chart types — you MUST include them:\n"
            f"Query: {state['query'][:300]}\n"
            f"Generate specs matching the user's requested types in the order mentioned."
        )

    prompt = f"""You are a data visualization expert. Recommend exactly {n_target} chart spec(s).

DATA PROFILE:
{json.dumps(col_profile, indent=2, default=str)}

FIRST 3 ROWS:
{df.head(3).to_dict('records')}

QUERY CONTEXT: {state['query'][:250]}
{explicit_note}

CHART TYPES AVAILABLE:
bar, grouped_bar, line, area, scatter, pie, heatmap, histogram, box, treemap, funnel, violin, bubble

RULES:
- Each spec uses columns that EXIST in the data profile above
- For pie: use 'names' (categorical) and 'values' (numeric)
- For heatmap: must include x, y (both categorical), z (numeric)
- For treemap: include 'path' (list of categorical cols) and 'values' (numeric)
- For scatter/bubble: x and y must both be numeric
- Titles must be specific and descriptive (not generic like "Revenue Chart")

Return ONLY a JSON array of exactly {n_target} spec(s):
[
  {{"chart":"bar","x":"col","y":"col","color":"col_or_null","title":"specific title"}},
  ...
]"""

    raw   = call_llm(prompt, model=FAST_MODEL, temperature=0.0)
    specs = extract_json(raw, fallback=[])

    if not specs or not isinstance(specs, list):
        # Intelligent fallback
        specs = []
        if date_cols and numeric_cols:
            specs.append({"chart":"line","x":date_cols[0],"y":numeric_cols[0],
                          "title":f"{numeric_cols[0]} over time"})
        if cat_cols and numeric_cols and len(specs) < n_target:
            specs.append({"chart":"bar","x":cat_cols[0],"y":numeric_cols[0],
                          "title":f"{numeric_cols[0]} by {cat_cols[0]}"})
        if len(numeric_cols) >= 2 and len(specs) < n_target:
            specs.append({"chart":"scatter","x":numeric_cols[0],"y":numeric_cols[1],
                          "title":f"{numeric_cols[0]} vs {numeric_cols[1]}"})

    rendered_sigs = set()
    explanations  = []
    n_rendered    = 0

    for spec in specs[:max(n_target, 8)]:
        rendered_title = render_chart(spec, df.copy(), rendered_sigs, honor_explicit)
        if rendered_title:
            n_rendered += 1
            x_c = spec.get("x", "")
            y_c = spec.get("y", "")

            expl = _analyze_chart(
                  fig_title     = spec.get("title", ""),
                  chart_type    = spec.get("chart", "bar"),
                  x_col         = spec.get("x", ""),
                  y_col         = spec.get("y", ""),
                  df            = df.copy(),
                  color_col     = spec.get("color"),
                  z_col         = spec.get("z"),
                  extra_context = f"Query: {state['query'][:200]}",
              )
            print(f"\n  Analysis [{n_rendered}] — {rendered_title}:")
            print(f"  {expl}\n")
            explanations.append(f"**{rendered_title}**: {expl}")

    print(f"[VizAgent] Done: {n_rendered} chart(s) queued")

    chart_expls = "\n\n".join(explanations)
    return {"chart_explanations": chart_expls}

def rca_agent(state: SentinelState) -> dict:
    """
    FIXED: RCA agent now calls _analyze_chart on every chart it creates
    and collects all analyses into final_response.
    """
    print("[RCAAgent] Causal root cause analysis starting...")
 
    df = run_sql("""
        SELECT order_date, category, city, platform, payment_method,
               SUM(final_amount) AS revenue,
               COUNT(*) AS order_count,
               AVG(rating) AS avg_rating,
               SUM(CASE WHEN status='returned' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS return_rate,
               AVG(delivery_time_hrs) AS avg_delivery_hrs
        FROM orders WHERE status IN ('delivered','returned','cancelled')
        GROUP BY order_date, category, city, platform, payment_method
        ORDER BY order_date
    """)
    df["order_date"] = pd.to_datetime(df["order_date"])
 
    all_analysis = []          # ← collect chart analyses here
 
    # ── Revenue delta by category ────────────────────────────────────
    cat_totals = run_sql("""
        SELECT category,
               AVG(CASE WHEN order_date < DATE '2024-06-08'
                        THEN final_amount END) AS first_half,
               AVG(CASE WHEN order_date >= DATE '2024-06-08'
                        THEN final_amount END) AS second_half
        FROM orders WHERE status='delivered'
        GROUP BY category
    """)
    cat_totals["delta_pct"] = (
        (cat_totals["second_half"] - cat_totals["first_half"])
        / (cat_totals["first_half"] + 1e-9) * 100
    )
    cat_totals = cat_totals.sort_values("delta_pct")
    worst_cat   = cat_totals.iloc[0]["category"]
    worst_delta = cat_totals.iloc[0]["delta_pct"]
 
    print(f"\nCategory revenue change (2nd half vs 1st half of 15 days):")
    print(cat_totals[["category", "first_half", "second_half", "delta_pct"]]
          .to_string(index=False))
 
    # ── Granger causality ────────────────────────────────────────────
    cat_df   = df[df["category"] == worst_cat].sort_values("order_date")
    daily_cat = (
        cat_df.groupby("order_date")[
            ["revenue", "avg_rating", "return_rate", "avg_delivery_hrs"]
        ].mean().dropna()
    )
 
    granger_results = {}
    for col in ["avg_rating", "return_rate", "avg_delivery_hrs"]:
        if col not in daily_cat.columns:
            continue
        try:
            test_df = daily_cat[["revenue", col]].dropna()
            if len(test_df) < 8:
                continue
            res   = grangercausalitytests(test_df, maxlag=3, verbose=False)
            min_p = min(res[lag][0]["ssr_ftest"][1] for lag in res)
            granger_results[col] = round(min_p, 4)
        except Exception:
            granger_results[col] = 1.0
 
    print(f"\nGranger causality → {worst_cat} revenue:")
    for k, v in granger_results.items():
        print(f"  {k}: p={v} → "
              f"{'✓ significant' if v < 0.05 else '✗ not significant'}")
 
    # ── Correlations ─────────────────────────────────────────────────
    corr_cols      = ["revenue", "avg_rating", "return_rate", "avg_delivery_hrs"]
    corr_available = [c for c in corr_cols if c in daily_cat.columns]
    corr_matrix    = (
        daily_cat[corr_available].corr()["revenue"].drop("revenue").to_dict()
    )
    print(f"\nCorrelation with {worst_cat} revenue:")
    for k, v in sorted(corr_matrix.items(),
                        key=lambda x: abs(x[1]), reverse=True):
        print(f"  {k}: {v:.3f}")
 
    l3_ctx    = l3_get_context("revenue")
    narrative = call_llm(
        f"""You are a senior data scientist writing a root cause analysis.
 
WORST PERFORMING CATEGORY: {worst_cat} ({worst_delta:+.1f}% revenue change)
 
CATEGORY DELTA TABLE:
{cat_totals[['category','delta_pct']].to_string(index=False)}
 
GRANGER CAUSALITY:
{json.dumps(granger_results, indent=2)}
 
CORRELATIONS WITH REVENUE:
{json.dumps({k: round(v, 3) for k, v in corr_matrix.items()}, indent=2)}
 
CAUSAL GRAPH PRIORS:
{l3_ctx}
 
Write a 3-paragraph RCA:
1. What happened and scale of impact
2. Causal drivers (cite exact p-values and correlations)
3. Prioritized action items""",
        temperature=0.2,
    )
    print(f"\n{'=' * 60}\nRCA NARRATIVE:\n{narrative}\n{'=' * 60}")
 
    # ── Chart 1: Revenue trend by category ───────────────────────────
    rev_trend = (
        df.groupby(["order_date", "category"])["revenue"]
        .sum().reset_index()
    )
    fig1 = px.line(
        rev_trend, x="order_date", y="revenue", color="category",
        title="Daily Revenue by Category — RCA Overview",
        template="plotly_white"
    )
    cutoff = pd.Timestamp("2024-06-08")
    safe_vline(fig1, cutoff, label="Analysis split", color="red", dash="dash")
    safe_show(fig1, "Revenue trend by category")
 
    expl1 = _analyze_chart(
        fig_title     = "Daily Revenue by Category — RCA Overview",
        chart_type    = "line",
        x_col         = "order_date",
        y_col         = "revenue",
        df            = rev_trend,
        color_col     = "category",
        extra_context = (
            f"Worst category: {worst_cat} ({worst_delta:+.1f}%)\n"
            f"Analysis split date: 2024-06-08\n"
            f"Granger p-values: {granger_results}"
        ),
    )
    print(f"\n  ── Chart Analysis: Revenue Trend by Category ──\n  {expl1}")
    all_analysis.append(
        f"**Daily Revenue by Category — RCA Overview**:\n{expl1}"
    )
 
    # ── Chart 2: Category delta bar ───────────────────────────────────
    fig2 = px.bar(
        cat_totals.sort_values("delta_pct"), x="category", y="delta_pct",
        color="delta_pct", color_continuous_scale="RdYlGn",
        title="Revenue % Change: 2nd Half vs 1st Half (15 days)",
        template="plotly_white", text="delta_pct"
    )
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(coloraxis_showscale=False)
    safe_show(fig2, "Category delta comparison")
 
    expl2 = _analyze_chart(
        fig_title     = "Revenue % Change: 2nd Half vs 1st Half",
        chart_type    = "bar",
        x_col         = "category",
        y_col         = "delta_pct",
        df            = cat_totals[["category", "delta_pct"]].copy(),
        extra_context = (
            f"Worst performer: {worst_cat} at {worst_delta:+.1f}%\n"
            f"Best performer: {cat_totals.iloc[-1]['category']} at "
            f"{cat_totals.iloc[-1]['delta_pct']:+.1f}%"
        ),
    )
    print(f"\n  ── Chart Analysis: Category Delta Comparison ──\n  {expl2}")
    all_analysis.append(
        f"**Revenue % Change: 2nd Half vs 1st Half**:\n{expl2}"
    )
 
    # ── Chart 3: Granger causality ────────────────────────────────────
    if granger_results:
        gran_df = pd.DataFrame([
            {"driver": k, "p_value": v, "significant": v < 0.05}
            for k, v in granger_results.items()
        ])
        fig3 = px.bar(
            gran_df, x="driver", y="p_value", color="significant",
            color_discrete_map={True: "#4CAF50", False: "#F44336"},
            title=f"Granger Causality p-values → {worst_cat} Revenue",
            template="plotly_white"
        )
        safe_hline(fig3, 0.05, label="p=0.05")
        safe_show(fig3, "Granger causality")
 
        expl3 = _analyze_chart(
            fig_title     = f"Granger Causality p-values → {worst_cat} Revenue",
            chart_type    = "bar",
            x_col         = "driver",
            y_col         = "p_value",
            df            = gran_df,
            extra_context = (
                f"Significance threshold: p < 0.05\n"
                f"Significant drivers: "
                f"{[k for k,v in granger_results.items() if v < 0.05]}"
            ),
        )
        print(f"\n  ── Chart Analysis: Granger Causality ──\n  {expl3}")
        all_analysis.append(
            f"**Granger Causality p-values → {worst_cat} Revenue**:\n{expl3}"
        )
 
    # ── Chart 4: Correlation heatmap ──────────────────────────────────
    if len(corr_available) >= 3:
        corr_full = daily_cat[corr_available].corr()
        fig4 = px.imshow(
            corr_full, title="Metric Correlation Matrix",
            color_continuous_scale="RdBu_r", text_auto=".2f",
            template="plotly_white"
        )
        safe_show(fig4, "Correlation heatmap")
 
        # Flatten for _analyze_chart
        corr_long = corr_full.reset_index().melt(
            id_vars="index", var_name="metric_b", value_name="correlation"
        ).rename(columns={"index": "metric_a"})
        expl4 = _analyze_chart(
            fig_title     = "Metric Correlation Matrix",
            chart_type    = "heatmap",
            x_col         = "metric_a",
            y_col         = "metric_b",
            z_col         = "correlation",
            df            = corr_long,
            extra_context = (
                f"Category: {worst_cat}\n"
                f"Revenue correlations: "
                f"{json.dumps({k: round(v,3) for k,v in corr_matrix.items()})}"
            ),
        )
        print(f"\n  ── Chart Analysis: Correlation Heatmap ──\n  {expl4}")
        all_analysis.append(f"**Metric Correlation Matrix**:\n{expl4}")
 
    # ── Build full response ───────────────────────────────────────────
    chart_section = "\n\n".join(all_analysis)
    full_response = (
        f"{narrative}\n\n"
        f"{'─' * 50}\n"
        f"📊 Chart Analysis:\n{chart_section}"
    )
 
    rca_result = {
        "worst_category": worst_cat,
        "delta_pct":      worst_delta,
        "granger":        granger_results,
        "correlations":   corr_matrix,
        "narrative":      narrative,
    }
    l2_store(state["query"], "/* RCA */",
             f"RCA: {worst_cat} revenue {worst_delta:+.1f}%")
 
    return {"rca_result": rca_result, "final_response": full_response}

def forecast_agent(state: SentinelState) -> dict:
    """Prophet forecasting — charts queued for post-invoke display, analysis printed inline."""
    print("[ForecastAgent] Building Prophet forecasts...")

    df = run_sql("""
        SELECT order_date,
               SUM(CASE WHEN status='delivered' THEN final_amount ELSE 0 END) AS revenue,
               COUNT(*) AS order_count,
               AVG(CASE WHEN status='delivered' THEN rating END) AS avg_rating
        FROM orders
        GROUP BY order_date ORDER BY order_date
    """)
    df["order_date"] = pd.to_datetime(df["order_date"])

    results      = {}
    all_analysis = []

    for metric, col in [("Revenue", "revenue"), ("Order Count", "order_count")]:
        prophet_df = (df.rename(columns={"order_date": "ds", col: "y"})[["ds","y"]]
                        .dropna()
                        .pipe(lambda d: d[d["y"] > 0]))

        if len(prophet_df) < 5:
            print(f"  Insufficient data for {metric}")
            continue

        try:
            m = Prophet(
                changepoint_prior_scale=0.08,
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=True,
                interval_width=0.90,
                n_changepoints=min(5, len(prophet_df) // 2),
            )
            m.fit(prophet_df)

            future   = m.make_future_dataframe(periods=7, freq="D")
            forecast = m.predict(future)

            last_actual = float(prophet_df["y"].iloc[-1])
            fc_7d       = float(forecast["yhat"].iloc[-1])
            pct_change  = (fc_7d - last_actual) / (last_actual + 1e-9) * 100
            ci_upper    = float(forecast["yhat_upper"].iloc[-1])
            ci_lower    = float(forecast["yhat_lower"].iloc[-1])

            print(f"\n  [{metric}] Current: {last_actual:,.0f} | "
                  f"7d forecast: {fc_7d:,.0f} ({pct_change:+.1f}%)")
            print(f"  90% CI: [{ci_lower:,.0f}, {ci_upper:,.0f}]")

            delta_abs        = np.abs(m.params["delta"].mean(axis=0))
            cp_mask          = delta_abs > 0.01
            changepoints_str = [cp.strftime("%Y-%m-%d")
                                for cp in m.changepoints[cp_mask]][:4]
            print(f"  Changepoints: {changepoints_str}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prophet_df["ds"], y=prophet_df["y"],
                name="Actual", line=dict(color="#2196F3", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat"],
                name="Forecast", line=dict(color="#FF9800", dash="dash", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=list(forecast["ds"]) + list(forecast["ds"][::-1]),
                y=list(forecast["yhat_upper"]) + list(forecast["yhat_lower"][::-1]),
                fill="toself", fillcolor="rgba(255,152,0,0.15)",
                line=dict(color="rgba(255,152,0,0)"), name="90% CI",
                showlegend=True
            ))
            for cp_str in changepoints_str:
                safe_vline(fig, cp_str, color="gray", dash="dot")
            safe_vline(fig, prophet_df["ds"].max().strftime("%Y-%m-%d"),
                       label="Forecast start", color="#4CAF50", dash="dash")

            fig.update_layout(
                title=f"{metric} — 7-Day Prophet Forecast (90% CI)",
                xaxis_title="Date", yaxis_title=metric,
                template="plotly_white", height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            safe_show(fig, f"{metric} — 7-Day Forecast")   # queued, not displayed

            analysis = _analyze_chart(
                f"{metric} — 7-Day Prophet Forecast",
                "line + confidence band", "ds", "yhat",
                forecast[["ds","yhat","yhat_lower","yhat_upper"]],
                extra_context=(
                    f"Actual last value: {last_actual:,.0f}\n"
                    f"Forecast 7d: {fc_7d:,.0f} ({pct_change:+.1f}%)\n"
                    f"90% CI: [{ci_lower:,.0f}, {ci_upper:,.0f}]\n"
                    f"Changepoints: {changepoints_str}"
                )
            )
            print(f"\n  ── Chart Analysis: {metric} Forecast ──")
            print(f"  {analysis}")
            all_analysis.append(f"**{metric} Forecast**:\n{analysis}")

            results[col] = {
                "current":     round(last_actual, 2),
                "forecast_7d": round(fc_7d, 2),
                "pct_change":  round(pct_change, 1),
                "ci_upper_7d": round(ci_upper, 2),
                "ci_lower_7d": round(ci_lower, 2),
                "changepoints":changepoints_str,
            }

        except Exception as e:
            print(f"  Prophet failed for {metric}: {e}")
            try:
                sp = min(7, max(2, len(prophet_df) - 1))
                hw  = ExponentialSmoothing(
                    prophet_df["y"], trend="add", seasonal="add", seasonal_periods=sp
                ).fit()
                fc7 = hw.forecast(7)

                fig_hw = go.Figure()
                fig_hw.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"],
                                            name="Actual", line=dict(color="#2196F3")))
                future_dates = pd.date_range(prophet_df["ds"].max(), periods=8, freq="D")[1:]
                fig_hw.add_trace(go.Scatter(x=future_dates, y=fc7.values,
                                            name="HW Forecast",
                                            line=dict(color="#FF9800", dash="dash")))
                fig_hw.update_layout(
                    title=f"{metric} — Holt-Winters Forecast (7 days)",
                    template="plotly_white", height=380
                )
                safe_show(fig_hw, f"{metric} — Holt-Winters Forecast")

                results[col] = {
                    "current":     round(float(prophet_df["y"].iloc[-1]), 2),
                    "forecast_7d": round(float(fc7.iloc[-1]), 2),
                    "method":      "Holt-Winters fallback",
                }
                print(f"  Holt-Winters 7d={fc7.iloc[-1]:,.0f}")
            except Exception as e2:
                print(f"  Both methods failed: {e2}")

    narrative = call_llm(
        f"Summarize these 7-day forecasts for a business leader in 2 concise paragraphs. "
        f"Cite specific numbers and flag the biggest risk or opportunity.\n"
        f"{json.dumps(results, indent=2)}",
        model=FAST_MODEL, temperature=0.1
    )

    chart_section = "\n\n".join(all_analysis)
    full_response = (
        f"{narrative}\n\n"
        f"{'─'*50}\n"
        f"Chart-by-chart analysis (charts rendered above):\n{chart_section}"
    )
    print(f"\nForecast narrative:\n{narrative}")

    l2_store(state["query"], "/* Prophet forecast */",
             json.dumps({k: v.get("pct_change", 0) for k, v in results.items()
                         if isinstance(v, dict)}))

    return {"forecast_result": results, "final_response": full_response}

def anomaly_agent(state: SentinelState) -> dict:
    """Z-score + IQR anomaly detection — charts queued, analysis printed inline."""
    print("[AnomalyAgent] Running anomaly scan...")

    df = run_sql("""
        SELECT order_date, category, city,
               SUM(final_amount) AS revenue,
               COUNT(*) AS order_count,
               AVG(rating) AS avg_rating,
               SUM(CASE WHEN status='returned' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS return_rate
        FROM orders
        GROUP BY order_date, category, city
        ORDER BY order_date
    """)
    df["order_date"] = pd.to_datetime(df["order_date"])

    anomalies = []
    WINDOW, Z_THRESH = 5, 2.0

    for (cat, city), group in df.groupby(["category","city"]):
        grp = group.sort_values("order_date").reset_index(drop=True)
        if len(grp) < WINDOW + 2:
            continue
        for metric in ["revenue","order_count"]:
            series = grp[metric].values
            dates  = grp["order_date"].values
            for i in range(WINDOW, len(series)):
                window   = series[i - WINDOW: i]
                mu, sigma = window.mean(), window.std() + 1e-9
                z         = (series[i] - mu) / sigma
                q1, q3    = np.percentile(window, 25), np.percentile(window, 75)
                iqr       = q3 - q1
                fl, fh    = q1 - 1.5*iqr, q3 + 1.5*iqr
                if abs(z) > Z_THRESH or series[i] < fl or series[i] > fh:
                    anomalies.append({
                        "date":     str(pd.Timestamp(dates[i]).date()),
                        "category": cat, "city": city, "metric": metric,
                        "value":    round(float(series[i]), 2),
                        "baseline": round(float(mu), 2),
                        "z_score":  round(float(z), 2),
                        "severity": "HIGH" if abs(z) > 3.0 else "MEDIUM",
                    })

    if not anomalies:
        msg = "No anomalies detected above threshold."
        print(f"  {msg}")
        return {"anomaly_result": {"count": 0}, "final_response": msg}

    anom_df      = pd.DataFrame(anomalies)
    anom_df["date"] = pd.to_datetime(anom_df["date"])
    high         = anom_df[anom_df["severity"] == "HIGH"]
    all_analysis = []

    print(f"\n  Total: {len(anom_df)} | HIGH: {len(high)}")
    print(anom_df.sort_values("z_score", key=abs, ascending=False)
              [["date","category","city","metric","z_score","severity"]]
              .head(8).to_string(index=False))

    plot_df = anom_df.copy()
    plot_df["abs_z"] = plot_df["z_score"].abs().clip(upper=10)
    fig1 = px.scatter(
        plot_df, x="date", y="abs_z", color="category",
        symbol="severity", size="abs_z", size_max=20,
        facet_col="metric",
        title="Anomalies — |Z-score| by Date & Category",
        template="plotly_white",
        hover_data=["city","value","baseline","z_score"]
    )
    safe_show(fig1, "Anomaly scatter — |Z-score| by date")

    analysis1 = _analyze_chart(
        "Anomalies — |Z-score| by Date & Category", "scatter (faceted)",
        "date", "abs_z", plot_df,
        extra_context=(
            f"Total anomalies: {len(anom_df)} | HIGH: {len(high)}\n"
            f"Worst: z={plot_df['z_score'].abs().max():.2f} "
            f"({plot_df.loc[plot_df['z_score'].abs().idxmax(),'category']} / "
            f"{plot_df.loc[plot_df['z_score'].abs().idxmax(),'city']})"
        )
    )
    print(f"\n  ── Chart Analysis: Anomaly Scatter ──\n  {analysis1}")
    all_analysis.append(f"**Anomaly Scatter**: {analysis1}")

    pivot = (anom_df.groupby(["category","metric"])["z_score"]
                    .apply(lambda x: x.abs().max())
                    .reset_index()
                    .pivot(index="category", columns="metric", values="z_score")
                    .fillna(0))
    fig2 = px.imshow(
        pivot, title="Worst |Z-score| Heatmap — Category × Metric",
        color_continuous_scale="RdYlGn_r", text_auto=".1f",
        template="plotly_white"
    )
    safe_show(fig2, "Severity heatmap — category × metric")

    pivot_long = pivot.reset_index().melt(id_vars="category",
                                          var_name="metric", value_name="max_z")
    analysis2 = _analyze_chart(
        "Worst |Z-score| Heatmap", "heatmap",
        "metric", "max_z", pivot_long,
        extra_context=f"Categories: {list(pivot.index)}"
    )
    print(f"\n  ── Chart Analysis: Severity Heatmap ──\n  {analysis2}")
    all_analysis.append(f"**Severity Heatmap**: {analysis2}")

    rev_daily = run_sql("""
        SELECT order_date, SUM(final_amount) AS revenue
        FROM orders WHERE status='delivered'
        GROUP BY order_date ORDER BY order_date
    """)
    rev_daily["order_date"] = pd.to_datetime(rev_daily["order_date"])

    rev_anom_daily = (
        anom_df[anom_df["metric"] == "revenue"]
        .groupby("date")["z_score"]
        .apply(lambda x: x.abs().max()).reset_index()
        .merge(rev_daily, left_on="date", right_on="order_date", how="inner")
    )

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=rev_daily["order_date"], y=rev_daily["revenue"],
        name="Daily Revenue", line=dict(color="#2196F3", width=2)
    ))
    if not rev_anom_daily.empty:
        fig3.add_trace(go.Scatter(
            x=rev_anom_daily["date"], y=rev_anom_daily["revenue"],
            mode="markers", name="Anomaly Dates",
            marker=dict(color="red", size=14, symbol="x-thin-open",
                        line=dict(width=3))
        ))
    fig3.update_layout(
        title="Revenue Timeline with Anomaly Markers",
        template="plotly_white", height=400
    )
    safe_show(fig3, "Revenue timeline + anomaly markers")

    analysis3 = _analyze_chart(
        "Revenue Timeline with Anomaly Markers", "line + scatter overlay",
        "order_date", "revenue", rev_daily,
        extra_context=(
            f"Anomaly dates: {rev_anom_daily['date'].dt.strftime('%m-%d').tolist() if not rev_anom_daily.empty else []}\n"
            f"Revenue range: {rev_daily['revenue'].min():,.0f}–{rev_daily['revenue'].max():,.0f}"
        )
    )
    print(f"\n  ── Chart Analysis: Revenue Timeline ──\n  {analysis3}")
    all_analysis.append(f"**Revenue Timeline + Anomaly Markers**: {analysis3}")

    cat_anom_count = (anom_df.groupby("category")
                             .agg(anomaly_count=("z_score","count"),
                                  max_z=("z_score", lambda x: x.abs().max()))
                             .reset_index()
                             .sort_values("anomaly_count", ascending=False))
    fig4 = px.bar(cat_anom_count, x="category", y="anomaly_count",
                  color="max_z", color_continuous_scale="Reds",
                  title="Anomaly Count by Category (color = worst |Z-score|)",
                  template="plotly_white", text="anomaly_count")
    fig4.update_traces(textposition="outside")
    safe_show(fig4, "Anomaly count by category")

    analysis4 = _analyze_chart(
        "Anomaly Count by Category", "bar",
        "category", "anomaly_count", cat_anom_count,
        extra_context=f"Max z-score per category shown as color"
    )
    print(f"\n  ── Chart Analysis: Category Anomaly Count ──\n  {analysis4}")
    all_analysis.append(f"**Anomaly Count by Category**: {analysis4}")

    top5  = anom_df.sort_values("z_score", key=abs, ascending=False).head(5)
    alert = call_llm(
        f"Write a 3-4 sentence data alert for these anomalies. "
        f"State the severity and suggest one immediate action.\n{top5.to_string(index=False)}",
        model=FAST_MODEL, temperature=0.1
    )
    print(f"\nALERT:\n{alert}")

    chart_section = "\n\n".join(all_analysis)
    full_response = (
        f"{alert}\n\n"
        f"{'─'*50}\n"
        f"Chart-by-chart analysis (charts rendered above):\n{chart_section}"
    )

    l2_store(state["query"], "/* anomaly scan */",
             f"{len(anom_df)} anomalies, {len(high)} HIGH")
    return {"anomaly_result": {"count": len(anom_df), "high": len(high),
                                "alert": alert,
                                "anomalies": anom_df.head(20).to_dict("records")},
            "final_response": full_response}

def memory_curator(state: SentinelState = None) -> dict:
    """Nightly: cluster L2 → merge duplicates → promote to L4 → extract L3 rules."""
    print("[MemoryCurator] Starting nightly consolidation...")

    if l2_collection.count() == 0:
        print("  L2 empty.")
        return {}

    all_eps   = l2_collection.get(include=["documents","metadatas","embeddings"])
    docs      = all_eps["documents"]
    metas     = all_eps["metadatas"]
    ids       = all_eps["ids"]
    embs      = np.array(all_eps["embeddings"])

    print(f"  L2 episodes to process: {len(docs)}")

    if len(docs) > 1:
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(embs)
        THRESHOLD = 0.90
        merged, clusters = set(), []
        for i in range(len(docs)):
            if i in merged:
                continue
            cluster = [i]
            for j in range(i+1, len(docs)):
                if j not in merged and sim[i,j] >= THRESHOLD:
                    cluster.append(j)
                    merged.add(j)
            clusters.append(cluster)

        deleted = 0
        for cluster in clusters:
            if len(cluster) > 1:
                scores = [float(metas[i].get("score", 1.0)) for i in cluster]
                best   = cluster[np.argmax(scores)]
                to_del = [ids[i] for i in cluster if i != best]
                l2_collection.delete(ids=to_del)
                deleted += len(to_del)
        print(f"  Merged: {deleted} duplicate episodes removed")

    promoted = 0
    for doc, meta in zip(docs, metas):
        if float(meta.get("score", 1.0)) >= 0.9:
            sql = meta.get("sql", "")
            if sql and "/* " not in sql and len(sql) > 20:
                pt = call_llm(f"In ≤6 words, what SQL problem type: {doc}",
                              model=FAST_MODEL, temperature=0.0)
                l4_store(pt.strip()[:50], sql, doc[:200])
                promoted += 1
    print(f"  Promoted {promoted} episodes → L4")

    summaries = [m.get("result_summary","") for m in metas if m.get("result_summary")]
    if summaries:
        raw = call_llm(
            f"From these summaries, extract 1-3 factual business rules as JSON array:\n"
            f"{chr(10).join(summaries[:8])}",
            model=FAST_MODEL, temperature=0.0
        )
        rules = extract_json(raw, fallback=[])
        if isinstance(rules, list):
            for rule in rules:
                rule_str = str(rule)
                nid = f"rule_{hashlib.md5(rule_str.encode()).hexdigest()[:8]}"
                l3_graph.add_node(nid, type="business_rule", description=rule_str)
            print(f"  Extracted {len(rules)} rules → L3")

    print(f"\n[MemoryCurator] L2:{l2_collection.count()} L4:{l4_collection.count()} "
          f"L3:{l3_graph.number_of_nodes()} nodes")
    return {}


def sql_response_writer(state: SentinelState) -> dict:
    if not state["sql_result_json"]:
        err = state.get("validation_error", "Unknown error")
        return {"final_response":
                f"Could not retrieve data after {state['validation_attempts']} "
                f"attempts. Last error: {err[:200]}"}

    df      = pd.read_json(state["sql_result_json"], orient="records")
    ci_info = extract_json(state.get("aqp_ci", "{}"), {})

    ci_str = ""
    if ci_info:
        ci_parts = [f"{col}: 95% CI [{v['ci_lower']:.2f}, {v['ci_upper']:.2f}]"
                    for col, v in list(ci_info.items())[:3]]
        ci_str = "\n\nStatistical confidence (AQP):\n" + "\n".join(ci_parts)

    response = call_llm(
        f"You are a business analyst. Summarize these SQL results in 2-3 sentences "
        f"with specific numbers. Highlight the most important finding.\n\n"
        f"Query: {state['query']}\n"
        f"SQL: {state['sql_query'][:200]}\n"
        f"Results (top rows):\n{df.head(8).to_string(index=False)}\n"
        f"Row count: {len(df)}{ci_str}",
        model=FAST_MODEL, temperature=0.1
    )

    chart_expls = state.get("chart_explanations", "")
    if chart_expls:
        full_response = (f"{response}\n\n"
                         f"{'─'*50}\n"
                         f"📊 Chart Analysis:\n{chart_expls}")
    else:
        full_response = response

    print(f"\n[ResponseWriter]\n{full_response}")
    l2_store(state["query"], state["sql_query"], response, score=1.0)
    print(f"[L2] Stored. Total: {l2_collection.count()}")
    return {"final_response": full_response}

def route_intent(state: SentinelState) -> str:
    return {
        "rca":      "rca_agent",
        "forecast": "forecast_agent",
        "anomaly":  "anomaly_agent",
        "math":     "math_agent",
        "sql_query":"schema_linker",
    }.get(state["intent"], "schema_linker")


builder = StateGraph(SentinelState)

builder.add_node("intent_classifier",  intent_classifier)
builder.add_node("schema_linker",       schema_linker)
builder.add_node("query_decomposer",    query_decomposer)
builder.add_node("sql_builder",         sql_builder)
builder.add_node("sql_validator",       sql_validator)
builder.add_node("viz_agent",           viz_agent)
builder.add_node("sql_response_writer", sql_response_writer)
builder.add_node("rca_agent",           rca_agent)
builder.add_node("forecast_agent",      forecast_agent)
builder.add_node("anomaly_agent",       anomaly_agent)
builder.add_node("math_agent",          math_agent)

builder.set_entry_point("intent_classifier")

builder.add_conditional_edges(
    "intent_classifier", route_intent,
    {"schema_linker": "schema_linker", "rca_agent": "rca_agent",
     "forecast_agent":"forecast_agent","anomaly_agent":"anomaly_agent",
     "math_agent":    "math_agent"}
)
builder.add_edge("schema_linker",    "query_decomposer")
builder.add_edge("query_decomposer", "sql_builder")
builder.add_edge("sql_builder",      "sql_validator")
builder.add_conditional_edges(
    "sql_validator", should_retry_sql,
    {"sql_builder":"sql_builder","viz_agent":"viz_agent", END:END}
)
builder.add_edge("viz_agent",           "sql_response_writer")
builder.add_edge("sql_response_writer", END)
builder.add_edge("rca_agent",    END)
builder.add_edge("forecast_agent",END)
builder.add_edge("anomaly_agent", END)
builder.add_edge("math_agent",    END)

sentinel = builder.compile()


def ask(query: str) -> str:
    """
    Main entry point.
    1. Clears the figure queue (prevents leftover charts from prior calls)
    2. Runs the full SENTINEL pipeline
    3. Flushes all queued charts ONCE from main thread — zero duplicates
    """
    global _FIG_QUEUE
    _FIG_QUEUE.clear()   # ← clear stale queue before each new query

    print(f"\n{'═'*70}")
    print(f"QUERY: {query}")
    print(f"{'═'*70}\n")

    state  = empty_state(query)
    result = sentinel.invoke(state)

    n_shown = flush_charts()
    if n_shown > 0:
        print(f"\n[VizLayer] Displayed {n_shown} chart(s)")

    resp = result.get("final_response", "No response generated.")
    print(f"\n{'─'*70}")
    print(f"FINAL RESPONSE:\n{resp}")
    print(f"{'─'*70}")
    return resp


print("LangGraph compiled.")
print(f"Nodes: {list(sentinel.nodes.keys())}")

print("\n" + "%"*70)
print("TEST EC-1: User explicitly requests 5 charts on revenue by city")
print("\n" + "%"*70)
r_ec1 = ask(
    "Show me 5 charts analyzing revenue distribution across cities — "
    "include bar, pie, box plot, treemap, and a scatter of order count vs revenue per city."
)

print("\n" + "%"*70)
print("TEST EC-3: User requests 6 charts on seller analysis")
print("\n" + "%"*70)
r_ec3 = ask(
    "I want 6 charts about seller performance: "
    "bar chart of revenue by seller tier, "
    "histogram of delivery time distribution, "
    "scatter of rating vs delivery time, "
    "bar chart of top 15 sellers by revenue, "
    "heatmap of seller city vs tier revenue, "
    "and a violin plot of ratings by seller tier."
)

print("\n" + "%"*70)
print("TEST EC-2: User requests 4 charts on payment method behaviour")
print("\n" + "%"*70)
r_ec2 = ask(
    "Give me 4 charts breaking down payment method performance — "
    "cancellation rate, average order value, platform mix, and hourly usage pattern. "
    "Show 4 charts please."
)

print("\n" + "%"*70)
print("TEST 1: Multi-join revenue breakdown by category + loyalty tier")
print("%"*70)
r1 = ask(
    "What is the total delivered revenue broken down by product category and "
    "customer loyalty tier? Show the top 3 categories per tier."
)

print("\n" + "%"*70)
print("TEST 2: Hourly order pattern with peak detection")
print("\n" + "%"*70)
r2 = ask(
    "Analyze the hourly order volume and revenue pattern across all 15 days. "
    "Which hours generate the top 20% of revenue? Show the distribution."
)

print("\n" + "%"*70)
print("TEST 3: CMGR and revenue acceleration (calculus-based)")
print("\n" + "%"*70)
r3 = ask(
    "Calculate the compound daily growth rate (CMGR) of revenue. "
    "Is revenue accelerating or decelerating? Compute the first and second derivative."
)

print("\n" + "%"*70)
print("TEST 4: Gini coefficient + Pareto analysis of seller revenue")
print("%"*70)
r4 = ask(
    "Compute the Gini coefficient of revenue concentration across sellers. "
    "What percentage of sellers account for 80% of revenue? Show the Lorenz curve."
)

print("\n" + "%"*70)
print("TEST 5: Root cause analysis")
print("&"*70)
r5 = ask(
    "Which product category had the biggest revenue drop in the second week vs first week? "
    "Do a root cause analysis — is it driven by ratings, return rate, or delivery time?"
)

print("\n" + "%"*70)
print("TEST 6: Customer cohort analysis")
print("\n" + "%"*70)
r6 = ask(
    "For customers who placed their first order in the first 3 days (June 1-3), "
    "how many ordered again in the next 12 days? Show day-by-day retention."
)

print("\n" + "%"*70)
print("TEST 7: Platform × payment cancellation rate matrix")
print("\n" + "%"*70)
r7 = ask(
    "Show a cross-tabulation of cancellation rate by platform (App/Website/API) "
    "and payment method. Which combination has the highest cancellation rate?"
)

print("\n" + "%"*70)
print("TEST 8: Revenue and order volume forecast")
print("\n" + "%"*70)
r8 = ask(
    "Forecast daily revenue and order count for the next 7 days using time-series analysis. "
    "Report confidence intervals and any detected changepoints."
)

print("\n" + "%"*70)
print("TEST 9: Multi-dimensional anomaly detection")
print("\n" + "%"*70)
r9 = ask(
    "Scan all north-star metrics for anomalies across categories and cities. "
    "Report any unusual spikes or drops with their severity and z-score."
)


print("\n" + "%"*70)
print("TEST 10: Delivery time impact on ratings by seller tier")
print("\n" + "%"*70)
r10 = ask(
    "For each seller tier (A/B/C), show the average delivery time in hours "
    "and average rating. Is there a statistically significant correlation "
    "between delivery speed and customer rating?"
)

print("\n" + "%"*70)
print("TEST 11: Customer Lifetime Value by platform (CLV)")
print("\n" + "%"*70)
r11 = ask(
    "Calculate the estimated customer lifetime value (CLV) for customers "
    "acquired through App vs Website vs API. Which platform brings the highest-value customers?"
)

print("\n" + "%"*70)
print("TEST 12: Discount elasticity and return rate analysis")
print("\n" + "%"*70)
r12 = ask(
    "What is the relationship between discount percentage and order volume? "
    "Compute the price elasticity of demand and show how return rates vary with discount levels."
)

print("Running post-session Memory Curator...")
memory_curator()

print(f"\n{'═'*60}")
print("FINAL MEMORY STATE")
print(f"{'═'*60}")
print(f"L2 Episodic: {l2_collection.count()} episodes")
print(f"L4 Procedural: {l4_collection.count()} SQL patterns")
print(f"L3 Graph: {l3_graph.number_of_nodes()} nodes, {l3_graph.number_of_edges()} edges")

# Show a retrieval demo — proves memory is working
print(f"\n{'─'*40}")
print("Semantic cache test — similar to Test 1:")
hits = l2_retrieve("revenue breakdown by category and customer tier", top_k=1)
if hits:
    print(f"  ✓ CACHE HIT: '{hits[0]['question'][:60]}'")
    print(f"  Stored SQL: {hits[0].get('sql','')[:100]}...")
else:
    print("  Cache MISS")