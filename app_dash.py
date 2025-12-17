import base64
import io
import re
import time
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, callback, dcc, html, dash_table, ctx, no_update

# Uses your existing file snowflake_client.py
from snowflake_client import resultScrapingData


# =========================
# In-memory cache (local dev)
# =========================
# OFFLINE_CACHE[country] = {"ts": float_epoch, "data": list[dict]}
OFFLINE_CACHE: dict[str, dict] = {}
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes


# --------- Helpers for reading & cleaning data --------- #

def build_comp_key_mapping(offline_keys: list[str], looker_keys: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    offline_set = list(dict.fromkeys(offline_keys))  # unique, preserve order

    for lk in looker_keys:
        candidates = []
        for ok in offline_set:
            shorter, longer = (ok, lk) if len(ok) <= len(lk) else (lk, ok)
            if len(shorter) >= 4 and longer.startswith(shorter):
                candidates.append(ok)
        mapping[lk] = candidates[0] if len(candidates) == 1 else lk

    return mapping


def parse_contents(contents: Optional[str]) -> pd.DataFrame:
    if contents is None:
        return pd.DataFrame(columns=["Competitor", "SKU"])

    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.StringIO(decoded.decode(encoding)))
        except UnicodeDecodeError:
            continue

    return pd.DataFrame(columns=["Competitor", "SKU"])


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps extra columns if present (e.g. URL, row_count).
    Normalizes column names to expected ones.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Competitor", "SKU"])

    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    df = df.rename(
        columns={
            "competitor": "Competitor",
            "sku": "SKU",
            "url": "URL",
            "row_count": "row_count",
            "country": "Country",
        }
    )

    expected = {"Competitor", "SKU"}
    if not expected.issubset(df.columns):
        return pd.DataFrame(columns=["Competitor", "SKU"])

    df["Competitor"] = df["Competitor"].astype(str).str.strip()
    df["SKU"] = df["SKU"].astype(str).str.strip()

    # Keep duplicates handling consistent
    df = df.dropna(subset=["Competitor", "SKU"]).drop_duplicates(subset=["Competitor", "SKU"])
    return df


def split_invalid_links(off_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Invalid links are offline rows where row_count == 1.
    Returns (invalid_df, msg).
    """
    if off_df is None or off_df.empty:
        return pd.DataFrame(columns=["Competitor", "SKU", "URL"]), None

    if "row_count" not in off_df.columns:
        return pd.DataFrame(columns=["Competitor", "SKU", "URL"]), "No row_count column provided"

    rc = pd.to_numeric(off_df["row_count"], errors="coerce")
    invalid_df = off_df[rc == 1].copy()

    if "URL" not in invalid_df.columns:
        invalid_df["URL"] = ""

    return invalid_df, None


def apply_remove_invalid_toggle(off_df: pd.DataFrame, remove_invalid: bool) -> pd.DataFrame:
    """
    If remove_invalid is True and row_count exists: keep only row_count > 1
    Otherwise: return full off_df.
    """
    if off_df is None or off_df.empty:
        return off_df

    if not remove_invalid:
        return off_df

    if "row_count" not in off_df.columns:
        return off_df

    rc = pd.to_numeric(off_df["row_count"], errors="coerce")
    return off_df[rc > 1].copy()


# --------- Competitor name normalization / alignment --------- #

_COUNTRY_TOKENS = {
    "de", "fr", "es", "it", "nl", "uk", "gb", "be", "ch",
    "at", "ie", "pl", "pt", "cz", "dk", "se", "no", "fi",
    "us", "ca", "au"
}


def normalize_competitor_name(name: str) -> str:
    s = str(name).lower().strip()
    s = re.sub(r"[._\-()+,]", " ", s)
    s = re.sub(r"\s+", " ", s)

    tokens = s.split()
    tokens = [t for t in tokens if t not in _COUNTRY_TOKENS]
    if not tokens:
        tokens = [s]
    return "".join(tokens).strip()


def align_competitors(off_df: pd.DataFrame, look_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    off_df = off_df.copy()
    look_df = look_df.copy()

    off_df["comp_key"] = off_df["Competitor"].apply(normalize_competitor_name)
    look_df["comp_key"] = look_df["Competitor"].apply(normalize_competitor_name)

    offline_keys = sorted(off_df["comp_key"].dropna().unique())
    looker_keys = sorted(look_df["comp_key"].dropna().unique())
    key_mapping = build_comp_key_mapping(offline_keys, looker_keys)
    look_df["comp_key"] = look_df["comp_key"].map(key_mapping).fillna(look_df["comp_key"])

    off_keys = set(off_df["comp_key"].dropna().unique())
    look_keys = set(look_df["comp_key"].dropna().unique())
    common_keys = off_keys & look_keys
    offline_only_keys = off_keys - look_keys

    unmatched_offline = (
        off_df[off_df["comp_key"].isin(offline_only_keys)][["Competitor", "comp_key"]]
        .drop_duplicates()
        .sort_values("Competitor")
        .reset_index(drop=True)
    )

    off_df = off_df[off_df["comp_key"].isin(common_keys)].copy()
    look_df = look_df[look_df["comp_key"].isin(common_keys)].copy()

    return off_df, look_df, unmatched_offline


# --------- Metrics / charts --------- #

def compute_metrics(off_df: pd.DataFrame, look_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    off_df, look_df, unmatched_offline = align_competitors(off_df, look_df)

    look_slim = look_df[["comp_key", "SKU", "Competitor"]].rename(columns={"Competitor": "Competitor_Looker"})

    merged = off_df.merge(
        look_slim,
        on=["comp_key", "SKU"],
        how="left",
        indicator=True,
    )
    merged["FoundInLooker"] = merged["_merge"] == "both"

    comp_stats = (
        merged.groupby("Competitor", dropna=False)
        .agg(
            offline_skus=("SKU", "nunique"),
            found_in_looker=("FoundInLooker", "sum"),
        )
        .reset_index()
    )
    comp_stats["missing"] = comp_stats["offline_skus"] - comp_stats["found_in_looker"]
    comp_stats["loss_pct"] = (
        (comp_stats["missing"] / comp_stats["offline_skus"])
        .replace([pd.NA, float("inf")], 0)
        * 100
    ).fillna(0).round(2)

    total_offline = len(off_df["SKU"])
    total_looker = len(look_df["SKU"])
    missing_global = total_offline - total_looker
    loss_pct_global = round((missing_global / total_offline) * 100, 2) if total_offline else 0.0

    global_stats = pd.DataFrame(
        {
            "metric": [
                "Offline SKUs",
                "Looker SKUs",
                "Missing vs Looker",
                "Loss %",
                "Offline Competitors",
                "Looker Competitors",
            ],
            "value": [
                total_offline,
                total_looker,
                missing_global,
                loss_pct_global,
                off_df["Competitor"].nunique(),
                look_df["Competitor"].nunique(),
            ],
        }
    )

    return merged, comp_stats, global_stats, unmatched_offline


def metric_cards(stats: pd.DataFrame):
    cards = []
    for _, row in stats.iterrows():
        cards.append(
            html.Div(
                [
                    html.Div(row["metric"], className="metric-label"),
                    html.Div(f"{row['value']}", className="metric-value"),
                ],
                className="metric-card",
            )
        )
    return cards


def comp_stack_chart(comp_stats: pd.DataFrame):
    if comp_stats.empty:
        return {}

    long_df = comp_stats.melt(
        id_vars=["Competitor", "offline_skus", "loss_pct"],
        value_vars=["found_in_looker", "missing"],
        var_name="status",
        value_name="sku_count",
    )

    fig = px.bar(
        long_df.sort_values(["loss_pct", "Competitor"], ascending=[False, True]),
        x="Competitor",
        y="sku_count",
        color="status",
        barmode="stack",
        labels={"sku_count": "Number of SKUs", "status": "Status"},
        hover_data={"offline_skus": True, "loss_pct": True, "status": True, "sku_count": True},
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Competitor (Offline)",
        yaxis_title="Offline SKUs (Found + Missing)",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#111827"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E5E7EB")
    return fig


def invalid_links_bar_chart(invalid_df: pd.DataFrame, selected_comp: str | None):
    if invalid_df is None or invalid_df.empty:
        return {}

    counts = (
        invalid_df.groupby("Competitor", dropna=False)
        .size()
        .reset_index(name="invalid_count")
        .sort_values("invalid_count", ascending=False)
    )

    if selected_comp:
        counts["is_selected"] = counts["Competitor"].eq(selected_comp)
        counts["opacity"] = counts["is_selected"].map({True: 1.0, False: 0.25})
    else:
        counts["opacity"] = 1.0

    fig = px.bar(
        counts,
        x="Competitor",
        y="invalid_count",
        labels={"invalid_count": "Invalid links (row_count = 1)"},
    )
    fig.update_traces(marker={"opacity": counts["opacity"].tolist()})

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Competitor",
        yaxis_title="Invalid links",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#111827"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E5E7EB")
    return fig


def table_columns(df: pd.DataFrame):
    return [{"name": c, "id": c} for c in df.columns]


# --------- Shared DataTable style --------- #

COMMON_TABLE_PROPS = dict(
    sort_action="native",
    filter_action="native",
    style_table={
        "overflowX": "auto",
        "borderRadius": "10px",
        "overflow": "hidden",
        "marginTop": "8px",
    },
    style_header={
        "backgroundColor": "#F3F4F6",
        "color": "#111827",
        "fontWeight": "600",
        "border": "none",
    },
    style_cell={
        "padding": "8px 10px",
        "borderBottom": "1px solid #E5E7EB",
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "fontSize": "13px",
        "whiteSpace": "nowrap",
        "textOverflow": "ellipsis",
        "maxWidth": 0,
        "color": "#111827",
        "backgroundColor": "#FFFFFF",
    },
    style_data_conditional=[
        {"if": {"row_index": "odd"}, "backgroundColor": "#FAFAFA"},
        {"if": {"state": "active"}, "backgroundColor": "#EFF6FF", "border": "1px solid #2563EB"},
        {"if": {"state": "selected"}, "backgroundColor": "#DBEAFE", "border": "1px solid #2563EB"},
    ],
)


# =========================
# Dash app
# =========================

COUNTRIES = ["GB", "ES", "US", "DE", "IT", "AU", "CA", "FR"]

app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Store(id="offline-data"),
        dcc.Store(id="looker-data"),
        dcc.Store(id="offline-invalid-data"),

        html.H1("Match Coverage Dashboard", className="app-title"),

        html.H3("Offline Source", className="section-title"),
        html.Div(
            [
                dcc.RadioItems(
                    id="offline-source",
                    options=[
                        {"label": "Upload CSV", "value": "upload"},
                        {"label": "Fetch from Snowflake", "value": "snowflake"},
                    ],
                    value="upload",
                    inline=True,
                ),
            ],
            className="card",
            style={"padding": "10px 12px"},
        ),

        # Upload offline container
        html.Div(
            [
                html.H4("Offline DB CSV (Competitor, SKU, URL, optional row_count)", className="section-title"),
                dcc.Upload(
                    id="upload-offline",
                    children=html.Div(["Drag and Drop or ", html.B("Select File")]),
                    className="upload-box",
                ),
            ],
            id="offline-upload-container",
            className="uploader-card",
            style={"marginTop": "10px"},
        ),

        # Snowflake fetch container (with loading spinner)
        html.Div(
            [
                html.H4("Fetch Offline from Snowflake", className="section-title"),
                dcc.Dropdown(
                    id="country-dropdown",
                    options=[{"label": c, "value": c} for c in COUNTRIES],
                    value="DE",
                    clearable=False,
                ),
                dcc.Loading(
                    type="default",
                    children=html.Div(
                        [
                            html.Button("Fetch", id="btn-fetch", n_clicks=0, style={"marginTop": "8px"}),
                            html.Div(
                                id="snowflake-fetch-status",
                                style={"marginTop": "8px", "fontSize": "13px"},
                            ),
                        ]
                    ),
                ),
                html.Div(
                    "Note: externalbrowser auth opens your browser for Auth0 SSO (first time or when session expires).",
                    style={"marginTop": "6px", "fontSize": "12px", "opacity": 0.8},
                ),
            ],
            id="snowflake-fetch-container",
            className="card",
            style={"padding": "12px", "marginTop": "10px", "display": "none"},
        ),

        html.H3("Looker Input", className="section-title"),
        html.Div(
            [
                html.H4("Looker CSV (Competitor, SKU)", className="section-title"),
                dcc.Upload(
                    id="upload-looker",
                    children=html.Div(["Drag and Drop or ", html.B("Select File")]),
                    className="upload-box",
                ),
            ],
            className="uploader-card",
        ),

        html.Div(id="upload-status", className="upload-status", style={"marginTop": "8px"}),
        html.Div(id="upload-summary", className="upload-summary", style={"marginTop": "4px"}),

        # Toggle (default ON)
        html.Div(
            [
                html.H3("Data Filters", className="section-title"),
                html.Div(
                    [
                        dcc.Checklist(
                            id="toggle-remove-invalid",
                            options=[{"label": "Remove invalid links (row_count > 1)", "value": "on"}],
                            value=["on"],
                            labelStyle={"display": "inline-block", "marginRight": "12px"},
                        ),
                    ],
                    className="card",
                    style={"padding": "10px 12px"},
                ),
            ],
            style={"marginTop": "12px"},
        ),

        html.H3("Overall", className="section-title"),
        html.Div(id="global-cards", className="metrics-row"),

        html.H3("Distinct Values (Offline vs Looker)", className="section-title"),
        html.Div(id="distinct-cards", className="metrics-row"),

        html.H3("By Competitor (Offline)", className="section-title"),
        html.Div(
            dcc.Loading(type="default", children=dcc.Graph(id="comp-volume-chart")),
            className="card",
        ),

        html.Div(
            [
                dcc.Loading(
                    type="default",
                    children=dash_table.DataTable(
                        id="comp-table",
                        data=[],
                        columns=[],
                        **COMMON_TABLE_PROPS,
                    ),
                )
            ],
            className="card",
            style={"marginTop": "16px"},
        ),

        html.H3("Competitor Drilldown", className="section-title"),
        html.Div(
            [
                dcc.Dropdown(
                    id="competitor-dropdown",
                    options=[],
                    placeholder="Select competitor",
                )
            ],
            className="card",
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.H4("SKUs Found in Looker", className="subsection-title"),
                        dcc.Loading(
                            type="default",
                            children=dash_table.DataTable(
                                id="found-table",
                                data=[],
                                columns=[{"name": "SKU", "id": "SKU"}],
                                **COMMON_TABLE_PROPS,
                                page_size=12,
                            ),
                        ),
                    ],
                    className="card",
                ),
                html.Div(
                    [
                        html.H4("SKUs Missing in Looker", className="subsection-title"),
                        dcc.Loading(
                            type="default",
                            children=dash_table.DataTable(
                                id="missing-table",
                                data=[],
                                columns=[{"name": "SKU", "id": "SKU"}, {"name": "URL", "id": "URL"}],
                                **COMMON_TABLE_PROPS,
                                page_size=12,
                                export_format="csv",
                            ),
                        ),
                    ],
                    className="card",
                ),
            ],
            className="two-column-row",
        ),

        html.H3("Top Missing SKUs (Global)", className="section-title"),
        html.Div(
            dcc.Loading(
                type="default",
                children=dash_table.DataTable(
                    id="top-missing",
                    data=[],
                    columns=[],
                    **COMMON_TABLE_PROPS,
                    page_size=20,
                ),
            ),
            className="card",
        ),

        # --------- Invalid links section --------- #
        html.H3("Invalid link SKUs", className="section-title"),
        html.Div(
            [
                html.Div(id="invalid-rowcount-msg", className="upload-summary", style={"marginBottom": "8px"}),

                dcc.Loading(type="default", children=dcc.Graph(id="invalid-links-chart")),

                dcc.Dropdown(
                    id="invalid-competitor-dropdown",
                    options=[],
                    placeholder="Select competitor (invalid links)",
                ),

                dcc.Loading(
                    type="default",
                    children=dash_table.DataTable(
                        id="invalid-sku-table",
                        data=[],
                        columns=[{"name": "SKU", "id": "SKU"}, {"name": "URL", "id": "URL"}],
                        **COMMON_TABLE_PROPS,
                        page_size=15,
                        export_format="csv",
                    ),
                ),
            ],
            className="card",
            style={"padding": "12px"},
        ),
    ],
    className="app-container",
)


# Show/hide the correct offline source UI
@callback(
    Output("offline-upload-container", "style"),
    Output("snowflake-fetch-container", "style"),
    Input("offline-source", "value"),
)
def toggle_offline_source_ui(source):
    if source == "snowflake":
        return {"display": "none"}, {"padding": "12px", "marginTop": "10px", "display": "block"}
    return {"marginTop": "10px", "display": "block"}, {"display": "none"}


# Load Looker data from upload (always)
@callback(
    Output("looker-data", "data"),
    Input("upload-looker", "contents"),
    prevent_initial_call=False,
)
def load_looker(contents):
    df_c = clean_df(parse_contents(contents))
    return df_c.to_dict("records")


def _cache_get(country: str) -> list[dict] | None:
    entry = OFFLINE_CACHE.get(country)
    if not entry:
        return None
    if (time.time() - entry["ts"]) > CACHE_TTL_SECONDS:
        OFFLINE_CACHE.pop(country, None)
        return None
    return entry["data"]


def _cache_set(country: str, data: list[dict]) -> None:
    OFFLINE_CACHE[country] = {"ts": time.time(), "data": data}


# Load Offline data from either Upload OR Snowflake (with cache + loading status)
@callback(
    Output("offline-data", "data"),
    Output("upload-status", "children"),
    Output("snowflake-fetch-status", "children"),
    Input("upload-offline", "contents"),
    Input("btn-fetch", "n_clicks"),
    Input("offline-source", "value"),
    State("country-dropdown", "value"),
    State("upload-offline", "filename"),
    prevent_initial_call=False,
)
def load_offline(off_contents, n_fetch, source, country, off_filename):
    trig = ctx.triggered_id

    # Switching sources: keep existing offline-data, just instruct user
    if trig == "offline-source":
        if source == "snowflake":
            return no_update, "Offline source changed to Snowflake.", "Select a country and click Fetch."
        return no_update, "Offline source changed to Upload.", ""

    # Upload path
    if source == "upload":
        if off_contents is None:
            return no_update, "Waiting for offline upload...", ""
        df_c = clean_df(parse_contents(off_contents))
        name = off_filename or "offline.csv"
        return df_c.to_dict("records"), f"Offline loaded from upload: {name} ({len(df_c)} rows)", ""

    # Snowflake path: only fetch on button click
    if source == "snowflake":
        if trig != "btn-fetch":
            return no_update, "Snowflake mode enabled.", "Select a country and click Fetch."

        country = (country or "").strip().upper()
        if country not in COUNTRIES:
            return no_update, "Snowflake mode enabled.", f"Invalid country '{country}'."

        # Cache hit
        cached = _cache_get(country)
        if cached is not None:
            return cached, f"Offline loaded from cache for {country} ({len(cached)} rows)", f"Cache hit (TTL {CACHE_TTL_SECONDS//60}m)."

        # Cache miss → call your function
        try:
            df_c = resultScrapingData(country)
            df_c = clean_df(df_c)
            records = df_c.to_dict("records")
            _cache_set(country, records)
            return records, f"Offline loaded from Snowflake for {country} ({len(records)} rows)", "Fetched from Snowflake and cached."
        except Exception as e:
            return no_update, "Snowflake fetch failed.", f"{type(e).__name__}: {e}"

    return no_update, "Waiting...", ""


@callback(
    Output("global-cards", "children"),
    Output("comp-volume-chart", "figure"),
    Output("comp-table", "data"),
    Output("comp-table", "columns"),
    Output("competitor-dropdown", "options"),
    Output("competitor-dropdown", "value"),
    Output("top-missing", "data"),
    Output("top-missing", "columns"),
    Output("distinct-cards", "children"),
    Output("upload-summary", "children"),
    Output("offline-invalid-data", "data"),
    Output("invalid-competitor-dropdown", "options"),
    Output("invalid-competitor-dropdown", "value"),
    Output("invalid-rowcount-msg", "children"),
    Input("offline-data", "data"),
    Input("looker-data", "data"),
    Input("toggle-remove-invalid", "value"),
)
def update_views(off_data, look_data, toggle_value):
    off_df = pd.DataFrame(off_data or [])
    look_df = pd.DataFrame(look_data or [], columns=["Competitor", "SKU"])

    if off_df.empty or look_df.empty:
        empty_stats = pd.DataFrame({"metric": ["Offline rows", "Looker rows"], "value": [len(off_df), len(look_df)]})
        return (
            metric_cards(empty_stats),
            {},
            [],
            [],
            [],
            None,
            [],
            [],
            metric_cards(empty_stats),
            "Upload/fetch Offline and upload Looker to compute metrics.",
            [],
            [],
            None,
            "",
        )

    remove_invalid = bool(toggle_value and "on" in toggle_value)

    invalid_df, row_count_msg = split_invalid_links(off_df)
    off_df_for_metrics = apply_remove_invalid_toggle(off_df, remove_invalid)

    merged, comp_stats, global_stats, unmatched_offline = compute_metrics(off_df_for_metrics, look_df)

    cards = metric_cards(global_stats)
    comp_volume_fig = comp_stack_chart(comp_stats)

    comp_cols = table_columns(comp_stats) if not comp_stats.empty else []
    comp_data = comp_stats.to_dict("records")

    competitors = sorted(merged["Competitor"].dropna().unique())
    dropdown_options = [{"label": c, "value": c} for c in competitors]
    dropdown_value = competitors[0] if competitors else None

    missing = merged[~merged["FoundInLooker"]]
    top_missing = (
        missing.groupby("SKU")
        .agg(
            missing_count=("Competitor", "nunique"),
            competitors=("Competitor", lambda x: ", ".join(sorted(x.unique()))),
        )
        .reset_index()
        .sort_values("missing_count", ascending=False)
        .head(20)
    )
    top_cols = table_columns(top_missing) if not top_missing.empty else []
    top_data = top_missing.to_dict("records")

    distinct_stats = pd.DataFrame(
        {
            "metric": ["Offline Distinct SKUs", "Looker Distinct SKUs"],
            "value": [
                off_df_for_metrics["SKU"].nunique() if "SKU" in off_df_for_metrics.columns else 0,
                look_df["SKU"].nunique() if "SKU" in look_df.columns else 0,
            ],
        }
    )
    distinct_cards = metric_cards(distinct_stats)

    offline_rows = len(off_df_for_metrics)
    looker_rows = len(look_df)
    unmatched_names = sorted(unmatched_offline["Competitor"].unique())
    if unmatched_names:
        preview = ", ".join(unmatched_names[:10])
        extra = len(unmatched_names) - 10
        if extra > 0:
            preview += f", +{extra} more"
        summary = (
            f"Offline rows (after filter): {offline_rows} | Looker rows: {looker_rows} | "
            f"Offline-only competitors ({len(unmatched_names)}): {preview}"
        )
    else:
        summary = (
            f"Offline rows (after filter): {offline_rows} | Looker rows: {looker_rows} | "
            "All offline competitors have a match in Looker."
        )

    if row_count_msg:
        invalid_opts = []
        invalid_val = None
        invalid_msg = row_count_msg
    else:
        inv_comps = sorted(invalid_df["Competitor"].dropna().unique())
        invalid_opts = [{"label": c, "value": c} for c in inv_comps]
        invalid_val = inv_comps[0] if inv_comps else None
        invalid_msg = f"{len(invalid_df)} invalid rows (row_count = 1)."

    return (
        cards,
        comp_volume_fig,
        comp_data,
        comp_cols,
        dropdown_options,
        dropdown_value,
        top_data,
        top_cols,
        distinct_cards,
        summary,
        invalid_df.to_dict("records"),
        invalid_opts,
        invalid_val,
        invalid_msg,
    )


@callback(
    Output("found-table", "data"),
    Output("missing-table", "data"),
    Input("competitor-dropdown", "value"),
    State("offline-data", "data"),
    State("looker-data", "data"),
    State("toggle-remove-invalid", "value"),
)
def update_drilldown(selected_comp, off_data, look_data, toggle_value):
    if not selected_comp:
        return [], []

    off_df = pd.DataFrame(off_data or [])
    look_df = pd.DataFrame(look_data or [], columns=["Competitor", "SKU"])

    if off_df.empty or look_df.empty:
        return [], []

    remove_invalid = bool(toggle_value and "on" in toggle_value)
    off_df_for_metrics = apply_remove_invalid_toggle(off_df, remove_invalid)

    merged, _, _, _ = compute_metrics(off_df_for_metrics, look_df)
    subset = merged[merged["Competitor"] == selected_comp].copy()

    found = subset[subset["FoundInLooker"]]["SKU"].drop_duplicates().sort_values()
    found_data = [{"SKU": sku} for sku in found]

    if "URL" in subset.columns:
        missing_df = (
            subset[~subset["FoundInLooker"]][["SKU", "URL"]]
            .drop_duplicates()
            .sort_values("SKU")
        )
        missing_data = missing_df.to_dict("records")
    else:
        missing = subset[~subset["FoundInLooker"]]["SKU"].drop_duplicates().sort_values()
        missing_data = [{"SKU": sku} for sku in missing]

    return found_data, missing_data


@callback(
    Output("invalid-sku-table", "data"),
    Output("invalid-links-chart", "figure"),
    Input("invalid-competitor-dropdown", "value"),
    State("offline-invalid-data", "data"),
)
def update_invalid_section(selected_comp, invalid_data):
    inv_df = pd.DataFrame(invalid_data or [], columns=["Competitor", "SKU", "URL"])

    fig = invalid_links_bar_chart(inv_df, selected_comp)

    if inv_df.empty or not selected_comp:
        return [], fig

    subset = inv_df[inv_df["Competitor"] == selected_comp].copy()
    if "URL" not in subset.columns:
        subset["URL"] = ""

    subset = subset[["SKU", "URL"]].drop_duplicates().sort_values("SKU")
    return subset.to_dict("records"), fig


server = app.server

if __name__ == "__main__":
    app.run()
