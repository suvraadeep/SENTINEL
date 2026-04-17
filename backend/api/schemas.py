"""Pydantic request / response models for all SENTINEL API endpoints."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Provider / Auth
# ---------------------------------------------------------------------------
class ProviderConfigRequest(BaseModel):
    provider: str = Field(..., description="openai | google | anthropic | groq | nvidia")
    api_key: str = Field(..., min_length=4)
    main_model: str
    fast_model: str


class ProviderConfigResponse(BaseModel):
    valid: bool
    provider: str
    main_model: str
    fast_model: str
    models: List[Dict[str, str]] = []
    error: Optional[str] = None


class ModelsResponse(BaseModel):
    provider: str
    models: List[Dict[str, str]]
    default_main: str
    default_fast: str


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    dataset: Optional[str] = None   # filename hint from frontend for dataset-aware routing


class ChartItem(BaseModel):
    title: str
    html: str


class QueryResponse(BaseModel):
    query: str
    intent: str = ""
    sql: Optional[str] = None
    sql_result_preview: Optional[List[Dict[str, Any]]] = None
    aqp_ci: Optional[Dict[str, Any]] = None
    charts: List[ChartItem] = []
    insights: str = ""
    chart_explanations: str = ""
    rca_result: Optional[Dict[str, Any]] = None
    forecast_result: Optional[Dict[str, Any]] = None
    anomaly_result: Optional[Dict[str, Any]] = None
    math_result: Optional[Dict[str, Any]] = None
    memory_info: Dict[str, Any] = {}
    error: Optional[str] = None
    duration_ms: int = 0


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
class UploadResponse(BaseModel):
    success: bool
    filename: str
    row_count: int
    tables: List[str]
    primary_table: str
    columns: Dict[str, str]
    date_col: Optional[str] = None
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    schema_preview: str = ""
    dataset_count: int = 1            # total datasets loaded so far
    all_tables: List[str] = []        # every table in the DB after this upload
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
class MemoryStats(BaseModel):
    l2: Dict[str, Any]
    l3: Dict[str, Any]
    l4: Dict[str, Any]


class L2Episode(BaseModel):
    question: str
    sql: str
    result_summary: str
    score: float
    timestamp: str


class L4Pattern(BaseModel):
    problem_type: str
    sql_template: str
    example_query: str


class L3GraphData(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# DataLab
# ---------------------------------------------------------------------------
class DatasetInfo(BaseModel):
    filename: str
    tables: List[str]
    row_count: int
    date_min: Optional[str] = None
    date_max: Optional[str] = None


class DataLabPreviewResponse(BaseModel):
    table: str
    columns: List[str]
    dtypes: Dict[str, str]
    row_count: int
    rows: List[Dict[str, Any]]   # up to 500 rows
    numeric_summary: Dict[str, Any] = {}


class DataLabOperationRequest(BaseModel):
    table: str
    operation: str   # "filter" | "aggregate" | "sort" | "sample" | "describe"
    params: Dict[str, Any] = {}


class DataLabOperationResponse(BaseModel):
    success: bool
    table: str
    operation: str
    columns: List[str] = []
    rows: List[Dict[str, Any]] = []
    row_count: int = 0
    message: str = ""
    error: Optional[str] = None


class DataLabSqlRequest(BaseModel):
    sql: str = Field(..., min_length=1, max_length=10000)


class DataLabSqlResponse(BaseModel):
    success: bool
    columns: List[str] = []
    rows: List[Dict[str, Any]] = []
    row_count: int = 0
    error: Optional[str] = None


class DataLabTransformRequest(BaseModel):
    table: str
    prompt: str = Field(..., min_length=1, max_length=2000)
    current_sql: Optional[str] = None   # chained view SQL from previous transforms


class DataLabTransformResponse(BaseModel):
    success: bool
    sql: str = ""
    columns: List[str] = []
    rows: List[Dict[str, Any]] = []
    row_count: int = 0
    error: Optional[str] = None


class DataLabPlotRequest(BaseModel):
    table: str
    prompt: str = Field(..., min_length=1, max_length=1000)
    current_sql: Optional[str] = None


class DataLabPlotResponse(BaseModel):
    success: bool
    charts: List[Dict[str, str]] = []   # [{title, html}]
    code: str = ""
    error: Optional[str] = None


class DataLabAutoPlotResponse(BaseModel):
    charts: List[Dict[str, str]] = []


class DataLabSchemaColumn(BaseModel):
    name: str
    dtype: str
    null_count: int = 0
    null_pct: float = 0.0
    unique_count: int = 0
    sample_values: List[Any] = []
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    mean_val: Optional[Any] = None


class DataLabSchemaResponse(BaseModel):
    table: str
    row_count: int
    col_count: int
    columns: List[DataLabSchemaColumn] = []
    memory_mb: float = 0.0


class DataLabIdentifyRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)


class DataLabIdentifyResponse(BaseModel):
    table: Optional[str] = None
    dataset: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
    ambiguous: bool = False
    candidates: List[str] = []


class DataLabTransformV2Request(BaseModel):
    """Enhanced transform request supporting multi-dataset and pandas/numpy."""
    table: str
    prompt: str = Field(..., min_length=1, max_length=2000)
    current_sql: Optional[str] = None          # chained SQL from prior SQL-based steps
    prior_step_codes: List[str] = []            # pandas codes from all prior steps (in order)
    use_pandas: bool = True                    # prefer pandas/numpy over SQL


class DataLabTransformV2Response(BaseModel):
    success: bool
    mode: str = "pandas"                  # "pandas" | "sql"
    code: str = ""                        # Python code shown to user
    sql: str = ""                         # SQL equivalent if available
    columns: List[str] = []
    rows: List[Dict[str, Any]] = []
    row_count: int = 0
    error: Optional[str] = None
    verifier_notes: str = ""             # notes from SQL verifier


class DatasetRemoveResponse(BaseModel):
    success: bool
    filename: str
    dropped_tables: List[str] = []
    remaining_datasets: List[str] = []
    error: Optional[str] = None


class DataLabSchemaQueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)


class DataLabSchemaQueryResponse(BaseModel):
    success: bool
    code: str = ""
    result_json: str = ""          # JSON-serialised result
    format: str = "json"           # "json" | "table"
    mode: str = "hardcoded"        # "hardcoded" | "llm"
    error: Optional[str] = None


class DataLabPromoteRequest(BaseModel):
    """Promote a DataLab transform result to a new DuckDB table for Intelligence queries."""
    table: str                                    # base table name
    version_name: Optional[str] = None            # custom name or auto-generated
    prior_step_codes: List[str] = []              # pandas codes to replay
    current_sql: Optional[str] = None             # SQL from prior SQL-based steps


class DataLabPromoteResponse(BaseModel):
    success: bool
    new_table: str = ""
    filename: str = ""                            # registered dataset filename
    row_count: int = 0
    error: Optional[str] = None


class DataLabDropTableResponse(BaseModel):
    success: bool
    table: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    initialized: bool
    has_custom_data: bool
