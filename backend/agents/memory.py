"""
SENTINEL Memory System — Multi-Domain Adaptive L2/L4 Memory (Supabase Edition)

Architecture:
  L2 — Episodic cache: past queries tagged with dataset_type to prevent
        cross-dataset contamination (e-commerce episodes never surface for
        real estate datasets).  Stored in Supabase pgvector.

  L4 — Procedural memory: domain-specific SQL patterns, seeded dynamically
        AFTER dataset upload. The seeding function detects the domain
        (e-commerce, real estate, financial, HR, SaaS, generic) from the
        actual column names and injects accurate few-shot patterns that
        reference the REAL table and column names — no more hardcoded 'orders'.
        Stored in Supabase pgvector.

  Domain detection → seed_for_dataset_type(con) is called by namespace.py
  after every dataset upload and connection swap.

  Backend: Supabase PostgreSQL with pgvector extension (replaces ChromaDB).
  Embeddings: BAAI/bge-large-en-v1.5 (local, 1024-dimensional).
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Load .env before reading Supabase credentials
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except Exception:
    pass

# Wrap supabase import — if package is missing, all functions still get defined
try:
    from supabase import create_client, Client
    _HAS_SUPABASE = True
except ImportError:
    _HAS_SUPABASE = False
    Client = None  # type stub
    print("  [Memory] WARNING: supabase package not installed! pip install supabase>=2.0.0")

from sentence_transformers import SentenceTransformer

print("Loading BAAI/bge-large-en-v1.5 (best MTEB retrieval model)...")
embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
print(f"  Embedding dim: {embed_model.get_sentence_embedding_dimension()}")

# ── Supabase client (replaces ChromaDB PersistentClient) ─────────────────────
# Try namespace-injected values first (set by namespace.py line 412-414),
# then fall back to os.environ (loaded by auth.py's load_dotenv).
_SUPABASE_URL = (
    globals().get("SUPABASE_URL")
    or locals().get("SUPABASE_URL")
    or os.environ.get("SUPABASE_URL", "")
)
_SUPABASE_KEY = (
    globals().get("SUPABASE_KEY")
    or locals().get("SUPABASE_KEY")
    or os.environ.get("SUPABASE_KEY", "")
)

supabase: Optional["Client"] = None

if not _HAS_SUPABASE:
    print("  [Memory] Memory system will run in no-op mode (supabase package missing).")
elif not _SUPABASE_URL or not _SUPABASE_KEY:
    print("  [Memory] WARNING: SUPABASE_URL or SUPABASE_KEY not set!")
    print("  [Memory] Memory system will run in no-op mode.")
else:
    try:
        supabase = create_client(_SUPABASE_URL, _SUPABASE_KEY)
        print(f"  [Memory] Supabase client connected to {_SUPABASE_URL[:40]}...")
    except Exception as _exc:
        print(f"  [Memory] WARNING: Supabase init failed: {_exc}")
        print(f"  [Memory] Memory system will run in no-op mode.")

# Track the currently active dataset type so l2_retrieve can filter
_CURRENT_DATASET_TYPE: str = "generic"


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed(text: str) -> List[float]:
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return embed_model.encode(prefixed, normalize_embeddings=True).tolist()


# ── L2 — Episodic cache ────────────────────────────────────────────────────────
def l2_store(question: str, sql: str, result_summary: str,
             feedback: str = "auto", score: float = 1.0,
             dataset_type: Optional[str] = None) -> None:
    """Store a query episode, tagged with its dataset domain."""
    if not supabase:
        return
    doc_id = hashlib.md5((question + sql).encode()).hexdigest()
    dtype  = dataset_type or _CURRENT_DATASET_TYPE
    embedding = embed(question)
    supabase.table("l2_episodic").upsert({
        "id":             doc_id,
        "question":       question,
        "embedding":      embedding,
        "sql_text":       sql[:800],
        "result_summary": result_summary[:400],
        "feedback":       feedback,
        "score":          float(score),
        "dataset_type":   dtype,
        "created_at":     datetime.now().isoformat(),
    }).execute()


def l2_retrieve(question: str, top_k: int = 3) -> List[Dict]:
    """
    Retrieve similar past queries, FILTERED by the current dataset type.
    Episodes from a different domain are excluded to prevent cross-dataset
    SQL contamination (e.g., orders SQL surfacing for a real estate query).
    """
    if not supabase:
        return []
    try:
        count_resp = supabase.table("l2_episodic").select("id", count="exact").execute()
        if count_resp.count == 0:
            return []
    except Exception:
        return []
    try:
        query_emb = embed(question)
        k = min(top_k * 4, 50)  # Fetch extra for domain filtering
        current = _CURRENT_DATASET_TYPE

        res = supabase.rpc("match_l2_episodes", {
            "query_embedding": query_emb,
            "match_count": k,
            "filter_dataset": current,
        }).execute()

        episodes = []
        for row in (res.data or []):
            episodes.append({
                "question":       row.get("question", ""),
                "sql":            row.get("sql_text", ""),
                "result_summary": row.get("result_summary", ""),
                "feedback":       row.get("feedback", "auto"),
                "score":          float(row.get("score", 1.0)),
                "dataset_type":   row.get("dataset_type", "generic"),
            })

        return episodes[:top_k]
    except Exception as e:
        print(f"  [Memory] L2 retrieve error: {e}")
        return []


# ── L4 — Procedural (pattern) memory ─────────────────────────────────────────
def l4_store(problem_type: str, sql_template: str, description: str,
             dataset_type: str = "generic") -> None:
    if not supabase:
        return
    doc_id = hashlib.md5(problem_type.encode()).hexdigest()
    embedding = embed(problem_type)
    supabase.table("l4_procedural").upsert({
        "id":           doc_id,
        "problem_type": problem_type,
        "embedding":    embedding,
        "sql_template": sql_template[:800],
        "description":  description[:300],
        "dataset_type": dataset_type,
    }).execute()


def l4_retrieve(question: str, top_k: int = 2) -> List[Dict]:
    """Retrieve SQL patterns, filtered by current dataset domain."""
    if not supabase:
        return []
    try:
        count_resp = supabase.table("l4_procedural").select("id", count="exact").execute()
        if count_resp.count == 0:
            return []
    except Exception:
        return []
    try:
        query_emb = embed(question)
        k = min(top_k * 4, 50)
        current = _CURRENT_DATASET_TYPE

        res = supabase.rpc("match_l4_patterns", {
            "query_embedding": query_emb,
            "match_count": k,
            "filter_dataset": current,
        }).execute()

        patterns = []
        for row in (res.data or []):
            patterns.append({
                "problem_type": row.get("problem_type", ""),
                "sql_template": row.get("sql_template", ""),
                "description":  row.get("description", ""),
                "dataset_type": row.get("dataset_type", "generic"),
            })

        return patterns[:top_k]
    except Exception as e:
        print(f"  [Memory] L4 retrieve error: {e}")
        return []


# ── Supabase helper functions (used by curator / namespace) ──────────────────
def l2_count() -> int:
    """Return the total number of L2 episodes."""
    if not supabase:
        return 0
    try:
        resp = supabase.table("l2_episodic").select("id", count="exact").execute()
        return resp.count or 0
    except Exception:
        return 0


def l4_count() -> int:
    """Return the total number of L4 patterns."""
    if not supabase:
        return 0
    try:
        resp = supabase.table("l4_procedural").select("id", count="exact").execute()
        return resp.count or 0
    except Exception:
        return 0


def l2_get_all() -> Dict:
    """Get all L2 episodes (for curator dedup)."""
    _empty = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    if not supabase:
        return _empty
    try:
        resp = supabase.table("l2_episodic").select("*").execute()
        return {
            "ids": [r["id"] for r in resp.data],
            "documents": [r["question"] for r in resp.data],
            "metadatas": [r for r in resp.data],
            "embeddings": [r.get("embedding") for r in resp.data],
        }
    except Exception:
        return _empty


def l2_delete(ids: List[str]) -> None:
    """Delete L2 episodes by ID list."""
    if not supabase or not ids:
        return
    supabase.table("l2_episodic").delete().in_("id", ids).execute()


def l4_get_all() -> Dict:
    """Get all L4 patterns."""
    _empty = {"ids": [], "documents": [], "metadatas": []}
    if not supabase:
        return _empty
    try:
        resp = supabase.table("l4_procedural").select("*").execute()
        return {
            "ids": [r["id"] for r in resp.data],
            "documents": [r["problem_type"] for r in resp.data],
            "metadatas": [r for r in resp.data],
        }
    except Exception:
        return _empty


def l4_delete(ids: List[str]) -> None:
    """Delete L4 patterns by ID list."""
    if not supabase or not ids:
        return
    supabase.table("l4_procedural").delete().in_("id", ids).execute()


# ── Domain detection ───────────────────────────────────────────────────────────
_DOMAIN_SIGNALS: Dict[str, List[str]] = {
    "ecommerce": [
        "order_date", "order_id", "final_amount", "base_amount", "discount_amount",
        "customer_id", "seller_id", "product_id", "delivery_time", "status",
        "payment_method", "cart", "checkout", "sku", "return_rate", "loyalty_tier",
        "platform", "order_hour", "order_count", "refund",
    ],
    "real_estate": [
        "sqft", "sq_ft", "square_feet", "bedrooms", "beds", "bathrooms", "baths",
        "year_built", "lot_size", "garage", "zoning", "listing_price", "sale_price",
        "price_per_sqft", "neighborhood", "zip_code", "parcel", "mls", "hoa",
        "property_type", "acres", "stories", "pool",
    ],
    "financial": [
        "ticker", "symbol", "open", "close", "high", "low", "volume",
        "market_cap", "dividend", "returns", "portfolio", "asset", "equity",
        "bond", "yield", "beta", "alpha", "sharpe", "drawdown", "position",
        "trade_date", "sector", "earnings",
    ],
    "hr": [
        "employee_id", "hire_date", "termination", "department", "salary",
        "compensation", "headcount", "attrition", "job_title", "seniority",
        "performance_rating", "band", "grade", "manager_id", "payroll",
        "bonus", "overtime", "absence", "leave",
    ],
    "healthcare": [
        "patient_id", "diagnosis", "icd", "procedure", "medication",
        "admission", "discharge", "insurance", "claim", "provider",
        "treatment", "condition", "lab", "prescription", "visit",
    ],
    "saas": [
        "user_id", "session", "event", "feature", "plan", "subscription",
        "churn", "mrr", "arr", "conversion", "funnel", "onboarding",
        "page_view", "click", "trial", "upgrade", "downgrade", "ltv",
    ],
}


def detect_dataset_type(con) -> str:
    """
    Detect the dataset domain by scanning column names against known signal sets.
    Returns one of: ecommerce | real_estate | financial | hr | healthcare | saas | generic
    """
    import pandas as pd
    try:
        tables_df = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).df()
        tables = tables_df.iloc[:, 0].tolist()
    except Exception:
        try:
            tables_df = con.execute("SHOW TABLES").df()
            tables = tables_df.iloc[:, 0].tolist()
        except Exception:
            return "generic"

    # Collect all lowercase column names across all non-utility tables
    all_cols: List[str] = []
    all_table_names: List[str] = []
    for tbl in tables:
        if any(s in tbl.lower() for s in ("modified", "tmp", "temp")):
            continue
        all_table_names.append(tbl.lower())
        try:
            desc = con.execute(f"DESCRIBE \"{tbl}\"").df()
            all_cols.extend(c.lower() for c in desc.iloc[:, 0].tolist())
        except Exception:
            pass

    if not all_cols:
        return "generic"

    # Score each domain by counting matching signals
    scores: Dict[str, int] = {}
    for domain, signals in _DOMAIN_SIGNALS.items():
        score = 0
        for signal in signals:
            if any(signal in col for col in all_cols):
                score += 1
        # Bonus: table name matches
        domain_table_hints = {
            "ecommerce":   ["order", "sale", "transaction", "cart", "payment"],
            "real_estate": ["listing", "property", "house", "realt", "home"],
            "financial":   ["stock", "trade", "portfolio", "price", "market"],
            "hr":          ["employee", "staff", "headcount", "payroll", "people"],
            "healthcare":  ["patient", "claim", "medical", "health", "clinical"],
            "saas":        ["event", "session", "user", "subscription", "funnel"],
        }
        for hint in domain_table_hints.get(domain, []):
            if any(hint in t for t in all_table_names):
                score += 3  # Strong table-name signal

        scores[domain] = score

    best_domain = max(scores, key=lambda d: scores[d])
    best_score  = scores[best_domain]

    print(f"  [Memory] Domain detection scores: "
          + " | ".join(f"{d}={s}" for d, s in sorted(scores.items(), key=lambda x: -x[1])))
    print(f"  [Memory] Detected domain: {best_domain} (score={best_score})")

    # Only commit if score is meaningful; otherwise fall back to generic
    return best_domain if best_score >= 3 else "generic"


# ── Pattern libraries — parameterized by real column names ────────────────────

def _ecommerce_patterns(table: str, schema: dict) -> List[tuple]:
    """
    E-commerce SQL patterns — fully parameterized with real column names.
    Pre-computes both quoted (for SELECT) and bare (for JOIN aliases) names
    so no backslash escapes are needed inside f-string expressions.
    """
    # Quoted column references for use in SELECT/WHERE
    T  = '"' + table + '"'
    D  = ('"' + schema["date"]        + '"') if schema.get("date")         else "order_date"
    A  = ('"' + schema["amount"]      + '"') if schema.get("amount")       else "final_amount"
    C  = ('"' + schema["category"]    + '"') if schema.get("category")     else "category"
    CU = ('"' + schema["customer"]    + '"') if schema.get("customer")     else "customer_id"
    SE = ('"' + schema["seller"]      + '"') if schema.get("seller")       else "seller_id"
    ST = ('"' + schema["status"]      + '"') if schema.get("status")       else "status"
    PL = ('"' + schema["platform"]    + '"') if schema.get("platform")     else "platform"
    DI = ('"' + schema["discount"]    + '"') if schema.get("discount")     else "discount_amount"
    BA = ('"' + schema["base_amount"] + '"') if schema.get("base_amount")  else "base_amount"
    RA = ('"' + schema["rating"]      + '"') if schema.get("rating")       else "rating"

    # Bare (unquoted) names for table aliases like  o.order_date
    D_bare  = schema.get("date",     "order_date")
    CU_bare = schema.get("customer", "customer_id")

    delivered_filter = ("WHERE " + ST + " = 'delivered'") if schema.get("status") else ""
    delivered_and    = ("AND "   + ST + " = 'delivered'") if schema.get("status") else ""

    return [
        (
            "revenue by category over time",
            ("SELECT " + D + ", " + C + ", SUM(" + A + ") AS revenue "
             "FROM " + T + " " + delivered_filter + " "
             "GROUP BY " + D + ", " + C + " ORDER BY " + D + ", revenue DESC"),
            "Daily revenue by " + schema.get("category", "category"),
        ),
        (
            "customer cohort retention analysis",
            ("WITH first_txn AS ("
             "  SELECT " + CU + ", MIN(" + D + ") AS cohort_date "
             "  FROM " + T + " GROUP BY " + CU + ") "
             "SELECT f.cohort_date, o." + D_bare + ", "
             "COUNT(DISTINCT o." + CU_bare + ") AS retained "
             "FROM " + T + " o JOIN first_txn f ON o." + CU_bare + "=f." + CU_bare + " "
             "GROUP BY f.cohort_date, o." + D_bare),
            "Customer cohort retention over time",
        ),
        (
            "seller performance ranking",
            ("SELECT " + SE + ", COUNT(*) AS transactions, SUM(" + A + ") AS revenue, "
             "AVG(" + RA + ") AS avg_rating "
             "FROM " + T + " " + delivered_filter + " "
             "GROUP BY " + SE + " ORDER BY revenue DESC LIMIT 20"),
            "Top sellers by revenue and rating",
        ),
        (
            "platform or channel comparison",
            ("SELECT " + PL + ", COUNT(*) AS transactions, "
             "SUM(" + A + ") AS revenue, AVG(" + A + ") AS avg_value "
             "FROM " + T + " GROUP BY " + PL + " ORDER BY revenue DESC"),
            "Revenue comparison across " + schema.get("platform", "platforms"),
        ),
        (
            "discount effectiveness analysis",
            ("SELECT ROUND(CAST(" + DI + " AS DOUBLE)/NULLIF(CAST(" + BA + " AS DOUBLE),0)*100,0) AS disc_pct, "
             "COUNT(*) AS orders, AVG(" + A + ") AS avg_revenue "
             "FROM " + T + " "
             "WHERE " + BA + " > 0 " + delivered_and + " "
             "GROUP BY 1 ORDER BY 1"),
            "How discount % affects order volume and revenue",
        ),
        (
            "revenue concentration pareto analysis",
            ("SELECT " + SE + ", SUM(" + A + ") AS revenue "
             "FROM " + T + " " + delivered_filter + " "
             "GROUP BY " + SE + " ORDER BY revenue DESC"),
            "Seller revenue distribution for Pareto/Gini analysis",
        ),
        (
            "status breakdown cancellation return rate",
            ("SELECT " + ST + ", COUNT(*) AS n, "
             "COUNT(*)*100.0/SUM(COUNT(*)) OVER() AS pct "
             "FROM " + T + " GROUP BY " + ST + " ORDER BY n DESC"),
            "Order status distribution including cancellation/return rates",
        ),
        (
            "daily order volume trend",
            ("SELECT " + D + ", COUNT(*) AS orders, SUM(" + A + ") AS revenue "
             "FROM " + T + " GROUP BY " + D + " ORDER BY " + D),
            "Daily order count and revenue trend",
        ),
    ]


def _real_estate_patterns(table: str, schema: dict) -> List[tuple]:
    T    = f'"{table}"'
    PR   = f'"{schema["price"]}"'       if schema.get("price")    else "sale_price"
    SQ   = f'"{schema["sqft"]}"'        if schema.get("sqft")     else "sqft"
    BD   = f'"{schema["bedrooms"]}"'    if schema.get("bedrooms") else "bedrooms"
    NB   = f'"{schema["neighborhood"]}"' if schema.get("neighborhood") else "neighborhood"
    YB   = f'"{schema["year_built"]}"'  if schema.get("year_built") else "year_built"
    PT   = f'"{schema["property_type"]}"' if schema.get("property_type") else "property_type"
    ZP   = f'"{schema["zip"]}"'         if schema.get("zip")      else "zip_code"
    DT   = f'"{schema["date"]}"'        if schema.get("date")     else "list_date"

    return [
        (
            "price per sqft by neighborhood",
            f"SELECT {NB}, AVG({PR}) AS avg_price, "
            f"AVG({PR}/NULLIF({SQ},0)) AS avg_price_per_sqft, "
            f"COUNT(*) AS listings "
            f"FROM {T} WHERE {SQ} > 0 "
            f"GROUP BY {NB} ORDER BY avg_price_per_sqft DESC",
            "Price per sqft distribution by neighborhood",
        ),
        (
            "price distribution by bedroom count",
            f"SELECT {BD}, AVG({PR}) AS avg_price, MEDIAN({PR}) AS median_price, "
            f"COUNT(*) AS n "
            f"FROM {T} WHERE {BD} IS NOT NULL "
            f"GROUP BY {BD} ORDER BY {BD}",
            "How price varies with number of bedrooms",
        ),
        (
            "newer vs older properties price comparison",
            f"SELECT CASE WHEN {YB} >= 2000 THEN 'Post-2000' "
            f"WHEN {YB} >= 1980 THEN '1980-1999' ELSE 'Pre-1980' END AS era, "
            f"AVG({PR}) AS avg_price, COUNT(*) AS n "
            f"FROM {T} WHERE {YB} IS NOT NULL "
            f"GROUP BY 1 ORDER BY avg_price DESC",
            "Price comparison across construction eras",
        ),
        (
            "property type price comparison",
            f"SELECT {PT}, AVG({PR}) AS avg_price, "
            f"MEDIAN({PR}) AS median_price, COUNT(*) AS n "
            f"FROM {T} WHERE {PT} IS NOT NULL "
            f"GROUP BY {PT} ORDER BY avg_price DESC",
            "Average price by property type",
        ),
        (
            "listings trend over time",
            f"SELECT {DT}, COUNT(*) AS listings, AVG({PR}) AS avg_price "
            f"FROM {T} WHERE {DT} IS NOT NULL "
            f"GROUP BY {DT} ORDER BY {DT}",
            "Listing volume and average price over time",
        ),
        (
            "top zip codes by median price",
            f"SELECT {ZP}, MEDIAN({PR}) AS median_price, COUNT(*) AS n "
            f"FROM {T} WHERE {ZP} IS NOT NULL "
            f"GROUP BY {ZP} HAVING COUNT(*) >= 5 "
            f"ORDER BY median_price DESC LIMIT 20",
            "Most expensive zip codes by median listing price",
        ),
        (
            "price per sqft overall vs by type simpson paradox check",
            f"SELECT {PT}, AVG({PR}) AS avg_price, "
            f"AVG({PR}/NULLIF({SQ},0)) AS avg_psqft, "
            f"COUNT(*) AS n, "
            f"SUM({SQ}*1.0)/COUNT(*) AS avg_sqft "
            f"FROM {T} WHERE {SQ}>0 AND {PT} IS NOT NULL "
            f"GROUP BY {PT} ORDER BY avg_psqft DESC",
            "Overall vs per-type price per sqft — detect Simpson's paradox",
        ),
        (
            "overpriced vs underpriced properties",
            f"SELECT {NB}, {BD}, {SQ}, {PR}, "
            f"{PR}/NULLIF({SQ},0) AS actual_psqft "
            f"FROM {T} WHERE {SQ} > 0 ORDER BY actual_psqft DESC LIMIT 50",
            "Properties ranked by price per sqft to identify outliers",
        ),
    ]


def _financial_patterns(table: str, schema: dict) -> List[tuple]:
    T   = f'"{table}"'
    TK  = f'"{schema["ticker"]}"'   if schema.get("ticker")  else "ticker"
    CL  = f'"{schema["close"]}"'    if schema.get("close")   else "close"
    VO  = f'"{schema["volume"]}"'   if schema.get("volume")  else "volume"
    DT  = f'"{schema["date"]}"'     if schema.get("date")    else "trade_date"
    RE  = f'"{schema["returns"]}"'  if schema.get("returns") else "returns"

    return [
        (
            "daily returns by ticker",
            f"SELECT {TK}, {DT}, "
            f"({CL} - LAG({CL}) OVER(PARTITION BY {TK} ORDER BY {DT})) / "
            f"NULLIF(LAG({CL}) OVER(PARTITION BY {TK} ORDER BY {DT}),0) AS daily_return "
            f"FROM {T} ORDER BY {DT}",
            "Daily percentage returns by ticker",
        ),
        (
            "price trend over time",
            f"SELECT {DT}, {TK}, {CL}, {VO} "
            f"FROM {T} ORDER BY {DT}",
            "Price and volume time series by ticker",
        ),
        (
            "volatility by ticker",
            f"SELECT {TK}, STDDEV({CL}) AS price_stddev, AVG({CL}) AS avg_close, "
            f"STDDEV({CL})/AVG({CL})*100 AS cv_pct, COUNT(*) AS n "
            f"FROM {T} GROUP BY {TK} ORDER BY cv_pct DESC",
            "Price volatility (coefficient of variation) by ticker",
        ),
        (
            "top volume days",
            f"SELECT {DT}, {TK}, {VO}, {CL} "
            f"FROM {T} ORDER BY {VO} DESC LIMIT 20",
            "Highest trading volume days",
        ),
    ]


def _hr_patterns(table: str, schema: dict) -> List[tuple]:
    T   = f'"{table}"'
    DE  = f'"{schema["department"]}"' if schema.get("department") else "department"
    SA  = f'"{schema["salary"]}"'     if schema.get("salary")    else "salary"
    HD  = f'"{schema["hire_date"]}"'  if schema.get("hire_date") else "hire_date"
    TI  = f'"{schema["title"]}"'      if schema.get("title")     else "job_title"
    LV  = f'"{schema["level"]}"'      if schema.get("level")     else "seniority"

    return [
        (
            "salary by department",
            f"SELECT {DE}, AVG({SA}) AS avg_salary, MEDIAN({SA}) AS median_salary, "
            f"MIN({SA}) AS min_salary, MAX({SA}) AS max_salary, COUNT(*) AS headcount "
            f"FROM {T} WHERE {SA} IS NOT NULL "
            f"GROUP BY {DE} ORDER BY avg_salary DESC",
            "Salary distribution by department",
        ),
        (
            "headcount hiring trend over time",
            f"SELECT DATE_TRUNC('month', {HD}::DATE) AS hire_month, "
            f"COUNT(*) AS new_hires, {DE} "
            f"FROM {T} WHERE {HD} IS NOT NULL "
            f"GROUP BY 1, {DE} ORDER BY 1",
            "Monthly hiring trend by department",
        ),
        (
            "compensation by level or seniority",
            f"SELECT {LV}, AVG({SA}) AS avg_salary, COUNT(*) AS n "
            f"FROM {T} GROUP BY {LV} ORDER BY avg_salary DESC",
            "Average salary by seniority/band",
        ),
        (
            "job title count and salary",
            f"SELECT {TI}, COUNT(*) AS headcount, AVG({SA}) AS avg_salary "
            f"FROM {T} GROUP BY {TI} ORDER BY headcount DESC LIMIT 20",
            "Headcount and salary by job title",
        ),
    ]


def _generic_patterns(table: str, schema: dict) -> List[tuple]:
    """Structural patterns that work for any dataset with a date + numeric column."""
    T  = f'"{table}"'
    AM = f'"{schema["amount"]}"' if schema.get("amount") else None
    DT = f'"{schema["date"]}"'   if schema.get("date")   else None
    GR = f'"{schema["group"]}"'  if schema.get("group")  else None

    patterns = []

    if DT and AM:
        patterns.append((
            "metric trend over time",
            f"SELECT {DT}, SUM({AM}) AS total, AVG({AM}) AS avg, COUNT(*) AS n "
            f"FROM {T} WHERE {DT} IS NOT NULL "
            f"GROUP BY {DT} ORDER BY {DT}",
            f"Aggregate {schema.get('amount','metric')} over time",
        ))

    if GR and AM:
        patterns.append((
            "metric by group category",
            f"SELECT {GR}, SUM({AM}) AS total, AVG({AM}) AS avg, COUNT(*) AS n "
            f"FROM {T} WHERE {GR} IS NOT NULL "
            f"GROUP BY {GR} ORDER BY total DESC LIMIT 20",
            f"Aggregate {schema.get('amount','metric')} by {schema.get('group','group')}",
        ))
        patterns.append((
            "top n records by metric",
            f"SELECT * FROM {T} ORDER BY {AM} DESC LIMIT 20",
            f"Top records ranked by {schema.get('amount','metric')}",
        ))

    if not patterns:
        patterns.append((
            "row count and basic stats",
            f"SELECT COUNT(*) AS total_rows FROM {T}",
            "Basic row count for any table",
        ))

    return patterns


# ── Schema introspection helper ───────────────────────────────────────────────
def _introspect_table(con, table: str, domain: str) -> dict:
    """
    Extract the most relevant column names for each semantic role
    based on the detected domain.
    """
    try:
        desc = con.execute(f"DESCRIBE \"{table}\"").df()
    except Exception:
        return {}

    col_name_col = desc.columns[0]
    col_type_col = desc.columns[1] if len(desc.columns) > 1 else desc.columns[0]

    cols_info = {
        str(row[col_name_col]): str(row[col_type_col]).lower()
        for _, row in desc.iterrows()
    }
    cols = list(cols_info.keys())
    cols_lower = {c.lower(): c for c in cols}

    def find(keywords, prefer_numeric=False):
        """Return first column whose name contains any keyword."""
        for kw in keywords:
            for cl, orig in cols_lower.items():
                if kw in cl:
                    ct = cols_info.get(orig, "")
                    if prefer_numeric and not any(
                        t in ct for t in ["int", "float", "double", "decimal", "real", "numeric"]
                    ):
                        continue
                    return orig
        return None

    schema: dict = {}

    if domain == "ecommerce":
        schema["date"]         = find(["order_date", "purchase_date", "transaction_date", "date"])
        schema["amount"]       = find(["final_amount", "total_amount", "amount", "revenue", "price", "value"], True)
        schema["category"]     = find(["category", "product_type", "segment", "dept"])
        schema["customer"]     = find(["customer_id", "client_id", "user_id", "buyer_id"])
        schema["seller"]       = find(["seller_id", "vendor_id", "merchant_id"])
        schema["status"]       = find(["status", "state", "order_status"])
        schema["platform"]     = find(["platform", "channel", "source", "medium"])
        schema["discount"]     = find(["discount", "rebate", "markdown"], True)
        schema["base_amount"]  = find(["base_amount", "list_price", "original_price", "mrp"], True)
        schema["rating"]       = find(["rating", "score", "review_score", "stars"], True)

    elif domain == "real_estate":
        schema["price"]        = find(["sale_price", "list_price", "price", "value", "amount"], True)
        schema["sqft"]         = find(["sqft", "sq_ft", "square_feet", "area", "living_area"], True)
        schema["bedrooms"]     = find(["bedrooms", "beds", "bedroom_count"], True)
        schema["bathrooms"]    = find(["bathrooms", "baths", "bathroom_count"], True)
        schema["year_built"]   = find(["year_built", "built_year", "construction_year"], True)
        schema["neighborhood"] = find(["neighborhood", "area", "district", "community", "suburb"])
        schema["property_type"]= find(["property_type", "home_type", "type", "style"])
        schema["zip"]          = find(["zip", "zip_code", "postal", "postcode"])
        schema["date"]         = find(["list_date", "sale_date", "date", "close_date"])

    elif domain == "financial":
        schema["ticker"]   = find(["ticker", "symbol", "stock", "instrument"])
        schema["close"]    = find(["close", "adj_close", "price", "last"], True)
        schema["volume"]   = find(["volume", "vol", "shares"], True)
        schema["date"]     = find(["date", "trade_date", "timestamp", "time"])
        schema["returns"]  = find(["return", "ret", "daily_return", "pct_change"], True)

    elif domain == "hr":
        schema["department"] = find(["department", "dept", "team", "division", "unit"])
        schema["salary"]     = find(["salary", "compensation", "pay", "wage", "income"], True)
        schema["hire_date"]  = find(["hire_date", "start_date", "join_date", "employment_date"])
        schema["title"]      = find(["title", "job_title", "role", "position"])
        schema["level"]      = find(["level", "seniority", "band", "grade", "tier"])

    # Generic fallbacks shared across domains
    if not schema.get("amount"):
        schema["amount"] = find(["amount", "value", "revenue", "price", "cost", "total"], True)
    if not schema.get("date"):
        schema["date"]   = find(["date", "time", "period", "created", "updated"])
    if not schema.get("group"):
        schema["group"]  = find(["type", "category", "status", "class", "segment", "tier", "group"])

    return {k: v for k, v in schema.items() if v is not None}


# ── Main seeding function ─────────────────────────────────────────────────────
def seed_for_dataset_type(con) -> str:
    """
    Called after each dataset upload/connection swap.
    Detects the domain, clears stale patterns, and seeds domain-appropriate
    SQL patterns using the REAL column names from the uploaded data.

    Returns the detected domain string.
    """
    global _CURRENT_DATASET_TYPE

    # 1. Detect domain
    domain = detect_dataset_type(con)
    _CURRENT_DATASET_TYPE = domain

    # 2. Clear any patterns from a different domain (prevent contamination)
    try:
        existing = l4_get_all()
        ids_to_delete = [
            doc_id for doc_id, meta in zip(
                existing.get("ids", []), existing.get("metadatas", [])
            )
            if meta.get("dataset_type", "generic") not in (domain, "generic")
        ]
        if ids_to_delete:
            l4_delete(ids_to_delete)
            print(f"  [Memory] Cleared {len(ids_to_delete)} stale patterns "
                  f"(previous domain)")
    except Exception as exc:
        print(f"  [Memory] Pattern cleanup warning: {exc}")

    # 3. Find the main data table
    try:
        tables_df = con.execute("SHOW TABLES").df()
        tables = [t for t in tables_df.iloc[:, 0].tolist()
                  if not any(s in t.lower() for s in ("modified", "tmp", "temp"))]
    except Exception:
        print("  [Memory] Could not list tables for seeding")
        return domain

    if not tables:
        return domain

    # Pick the "main" table
    table = tables[0]
    domain_table_hints = {
        "ecommerce":   ["order", "sale", "transaction"],
        "real_estate": ["listing", "property", "house", "home", "realt"],
        "financial":   ["stock", "trade", "price", "ticker"],
        "hr":          ["employee", "staff", "headcount", "people"],
        "healthcare":  ["patient", "claim", "visit"],
        "saas":        ["event", "session", "user"],
    }
    for hint in domain_table_hints.get(domain, []):
        for t in tables:
            if hint in t.lower():
                table = t
                break

    # 4. Introspect real column names
    schema = _introspect_table(con, table, domain)
    print(f"  [Memory] Seeding patterns for domain='{domain}', "
          f"table='{table}', schema_keys={list(schema.keys())}")

    # 5. Build and store domain-specific patterns
    if domain == "ecommerce":
        patterns = _ecommerce_patterns(table, schema)
    elif domain == "real_estate":
        patterns = _real_estate_patterns(table, schema)
    elif domain == "financial":
        patterns = _financial_patterns(table, schema)
    elif domain == "hr":
        patterns = _hr_patterns(table, schema)
    else:
        patterns = _generic_patterns(table, schema)

    # Always add generic fallback patterns on top
    generic_extras = _generic_patterns(table, schema)
    for pt, sql, desc in generic_extras:
        # Only add if not already covered
        if not any(p[0] == pt for p in patterns):
            patterns.append((pt, sql, desc))

    for pt, sql, desc in patterns:
        l4_store(pt, sql, desc, dataset_type=domain)

    print(f"  [Memory] Seeded {len(patterns)} L4 patterns for '{domain}'")
    return domain


# ── One-time startup cleanup ──────────────────────────────────────────────────
def _cleanup_legacy_patterns() -> None:
    """
    Remove any patterns seeded before the multi-domain system was introduced
    (those referencing hardcoded e-commerce tables like 'orders' with no
    dataset_type tag or wrong tag).
    """
    LEGACY_TABLES = {
        "orders", "products", "customers", "sellers", "inventory",
        "payments", "shipments", "reviews", "cart",
    }
    try:
        res = l4_get_all()
        ids_to_delete = []
        for doc_id, meta in zip(res.get("ids", []), res.get("metadatas", [])):
            sql_upper = meta.get("sql_template", "").upper()
            # Pattern references a hardcoded table AND has no dataset_type metadata
            has_hardcoded = any(
                f"FROM {t.upper()}" in sql_upper or f"JOIN {t.upper()}" in sql_upper
                for t in LEGACY_TABLES
            )
            no_type_tag = "dataset_type" not in meta
            if has_hardcoded and no_type_tag:
                ids_to_delete.append(doc_id)
        if ids_to_delete:
            l4_delete(ids_to_delete)
            print(f"  [Memory] Cleaned {len(ids_to_delete)} legacy untagged L4 patterns")
    except Exception as exc:
        print(f"  [Memory] Legacy cleanup warning (non-fatal): {exc}")


_cleanup_legacy_patterns()

print(f"L2 episodes: {l2_count()} | L4 patterns: {l4_count()}")
print(f"Embedding dim: {len(embed('test'))}")
