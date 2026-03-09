import base64
import io
import re
import time
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, callback, dcc, html, dash_table, ctx, no_update

from snowflake_client import resultScrapingData


# ============================================================
# Config
# ============================================================

COUNTRIES = ["GB", "ES", "US", "DE", "IT", "AU", "CA", "FR"]

OFFLINE_CACHE: dict[str, dict] = {}
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes

ALL_COMP = "__ALL__"  # value used for "All competitors" in dropdown

# Tokens to strip from competitor names (country suffixes etc.)
_COUNTRY_TOKENS = {
    "de", "fr", "es", "it", "nl", "uk", "gb", "be", "ch",
    "at", "ie", "pl", "pt", "cz", "dk", "se", "no", "fi",
    "us", "ca", "au"
}


# ============================================================
# CSV parsing / cleaning
# ============================================================

def apply_apparel_suffix_to_looker(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    """
    If enabled, append '__apparel' to all Looker competitor names
    before alignment / joining.
    """
    if df is None or df.empty or not enabled:
        return df

    df_c = df.copy()
    if "Competitor" not in df_c.columns:
        return df_c

    df_c["Competitor"] = (
        df_c["Competitor"]
        .astype(str)
        .str.strip()
        .apply(lambda x: f"{x}__apparel" if x else x)
    )
    return df_c


def filter_offline_weeks_to_looker(off_df: pd.DataFrame, look_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only offline rows where scrape_week exists in looker.
    Looker is the source of truth for the weeks we analyze.
    """
    if off_df is None or off_df.empty:
        return off_df
    if look_df is None or look_df.empty:
        # If looker is empty, nothing to match weeks against
        return off_df.iloc[0:0]

    if "scrape_week" not in off_df.columns or "scrape_week" not in look_df.columns:
        # If either side doesn't have scrape_week, can't do week intersection safely
        return off_df.iloc[0:0]

    off_c = off_df.copy()
    look_c = look_df.copy()

    # Normalize to consistent string keys (avoid object/date mix)
    off_c["scrape_week"] = off_c["scrape_week"].astype("string").fillna("").astype(str).str.strip()
    look_c["scrape_week"] = look_c["scrape_week"].astype("string").fillna("").astype(str).str.strip()

    look_weeks = set(look_c.loc[look_c["scrape_week"] != "", "scrape_week"].unique().tolist())
    if not look_weeks:
        return off_c.iloc[0:0]

    return off_c[off_c["scrape_week"].isin(look_weeks)].copy()



def parse_contents(contents: Optional[str]) -> pd.DataFrame:
    if contents is None:
        return pd.DataFrame()
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.StringIO(decoded.decode(encoding)))
        except UnicodeDecodeError:
            continue

    return pd.DataFrame()


def _to_week_str(x) -> str:
    """Normalize to YYYY-MM-DD string (date only)."""
    dt = pd.to_datetime(x, errors="coerce")
    if pd.isna(dt):
        return ""
    return dt.normalize().date().isoformat()


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep extra columns if present (URL, row_count, Country, scrape_week).
    Normalize common column names across sources.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Competitor", "SKU"])

    df_c = df.copy()
    df_c = df_c.rename(columns={c: c.strip().lower() for c in df_c.columns})

    df_c = df_c.rename(
        columns={
            "competitor": "Competitor",
            "sku": "SKU",
            "url": "URL",
            "row_count": "row_count",
            "country": "Country",
            "scrape_week": "scrape_week",
            "scraped_date": "scraped_date",
            "scraped week week": "scrape_week",  # Looker
        }
    )

    if not {"Competitor", "SKU"}.issubset(df_c.columns):
        return pd.DataFrame(columns=["Competitor", "SKU"])

    df_c["Competitor"] = df_c["Competitor"].astype(str).str.strip()
    df_c["SKU"] = df_c["SKU"].astype(str).str.strip()

    if "scrape_week" in df_c.columns:
        df_c["scrape_week"] = df_c["scrape_week"].apply(_to_week_str)
        df_c = df_c[df_c["scrape_week"].astype(str).str.len() > 0]

    df_c = df_c.dropna(subset=["Competitor", "SKU"])

    # Dedupe: include scrape_week if present
    if "scrape_week" in df_c.columns:
        df_c = df_c.drop_duplicates(subset=["Competitor", "SKU", "scrape_week"])
    else:
        df_c = df_c.drop_duplicates(subset=["Competitor", "SKU"])

    return df_c


def derive_selected_country_from_df(df: pd.DataFrame) -> str:
    if df is None or df.empty or "Country" not in df.columns:
        return ""
    s = df["Country"].dropna().astype(str).str.strip()
    if s.empty:
        return ""
    try:
        return s.value_counts().idxmax()
    except Exception:
        return s.iloc[0]


# ============================================================
# Invalid links filtering
# ============================================================

def split_invalid_links(off_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Invalid links are offline rows where row_count == 1."""
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
    """If remove_invalid is True and row_count exists: keep only row_count > 1."""
    if off_df is None or off_df.empty or not remove_invalid or "row_count" not in off_df.columns:
        return off_df
    rc = pd.to_numeric(off_df["row_count"], errors="coerce")
    return off_df[rc > 1].copy()


# ============================================================
# Competitor normalization / alignment
# ============================================================

def normalize_competitor_name(name: str) -> str:
    """
    Normalize competitor into a stable key.

    NOTE: hyphen is escaped to avoid regex "bad range" errors.
    """
    s = str(name).lower().strip()
    s = re.sub(r"[._()+,\-]", " ", s)   # <- safe hyphen
    s = re.sub(r"\s+", " ", s)

    tokens = s.split()
    tokens = [t for t in tokens if t not in _COUNTRY_TOKENS]
    if not tokens:
        tokens = [s]
    return "".join(tokens).strip()


def build_comp_key_mapping(offline_keys: list[str], looker_keys: list[str]) -> dict[str, str]:
    """
    Maps Looker keys to Offline keys when there is a single unambiguous prefix match.
    """
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

def enforce_merge_key_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "comp_key" not in df.columns:
        return df
    df["comp_key"] = df["comp_key"].astype("string").fillna("").astype(str)

    if "SKU" in df.columns:
        df["SKU"] = df["SKU"].astype("string").fillna("").astype(str)
    return df




def align_competitors(off_df: pd.DataFrame, look_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Adds comp_key to both dfs, aligns Looker keys to Offline keys, then keeps only common keys.
    Returns (off_aligned, look_aligned, unmatched_offline_competitors).
    """
    off_df = off_df.copy()
    look_df = look_df.copy()

    off_df["comp_key"] = off_df["Competitor"].apply(normalize_competitor_name)
    look_df["comp_key"] = look_df["Competitor"].apply(normalize_competitor_name)


    off_df = enforce_merge_key_types(off_df)
    look_df = enforce_merge_key_types(look_df)

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


# ============================================================
# Core status tables
# ============================================================

def _weeks_last_n(off_df: pd.DataFrame, last_n_weeks: int) -> list[str]:
    weeks = sorted(off_df["scrape_week"].dropna().unique())
    return weeks[-last_n_weeks:] if len(weeks) > last_n_weeks else weeks


def _build_status_table(off_df: pd.DataFrame, look_df: pd.DataFrame, weeks: list[str]) -> pd.DataFrame:
    """
    For each offline (scrape_week, comp_key, SKU) -> found(bool) if also in looker same week.
    """
    off_pairs = off_df[off_df["scrape_week"].isin(weeks)][["scrape_week", "comp_key", "SKU"]].drop_duplicates()

    look_pairs = look_df[look_df["scrape_week"].isin(weeks)][["scrape_week", "comp_key", "SKU"]].drop_duplicates()
    look_pairs["found"] = True

    status = off_pairs.merge(look_pairs, on=["scrape_week", "comp_key", "SKU"], how="left")
    status["found"] = status["found"].fillna(False)
    return status


# ============================================================
# Latest week metrics (existing behavior)
# ============================================================

def compute_metrics_latest(off_df_latest: pd.DataFrame, look_df_latest: pd.DataFrame):

    off_df_latest, look_df_latest, unmatched_offline = align_competitors(off_df_latest, look_df_latest)

    off_df_latest = enforce_merge_key_types(off_df_latest)
    look_df_latest = enforce_merge_key_types(look_df_latest)

    look_slim = look_df_latest[["comp_key", "SKU", "Competitor"]].rename(columns={"Competitor": "Competitor_Looker"})

    merged = off_df_latest.merge(
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

    total_offline = len(off_df_latest["SKU"])
    total_looker = len(look_df_latest["SKU"])
    missing_global = total_offline - total_looker
    loss_pct_global = round((missing_global / total_offline) * 100, 2) if total_offline else 0.0

    global_stats = pd.DataFrame(
        {
            "metric": [
                "Offline SKUs (latest week)",
                "Looker SKUs (latest week)",
                "Missing vs Looker (latest week)",
                "Loss % (latest week)",
                "Offline Competitors (latest week)",
                "Looker Competitors (latest week)",
            ],
            "value": [
                total_offline,
                total_looker,
                missing_global,
                loss_pct_global,
                off_df_latest["Competitor"].nunique(),
                look_df_latest["Competitor"].nunique(),
            ],
        }
    )

    return merged, comp_stats, global_stats, unmatched_offline


# ============================================================
# Historical series (loss % only)
# ============================================================

def compute_weekly_loss_series(off_df: pd.DataFrame, look_df: pd.DataFrame, last_n_weeks: int = 4) -> pd.DataFrame:
    """
    Returns one row per week:
      scrape_week, offline_pairs, found_pairs, missing_pairs, loss_pct
    """
    if "scrape_week" not in off_df.columns or "scrape_week" not in look_df.columns:
        return pd.DataFrame(columns=["scrape_week", "offline_pairs", "found_pairs", "missing_pairs", "loss_pct"])

    off_a, look_a, _ = align_competitors(off_df, look_df)
    weeks = _weeks_last_n(off_a, last_n_weeks)
    if not weeks:
        return pd.DataFrame(columns=["scrape_week", "offline_pairs", "found_pairs", "missing_pairs", "loss_pct"])

    status = _build_status_table(off_a, look_a, weeks)

    weekly = (
        status.groupby("scrape_week", dropna=False)
        .agg(
            offline_pairs=("SKU", "size"),
            found_pairs=("found", "sum"),
        )
        .reset_index()
        .sort_values("scrape_week")
    )
    weekly["missing_pairs"] = weekly["offline_pairs"] - weekly["found_pairs"]
    weekly["loss_pct"] = (weekly["missing_pairs"] / weekly["offline_pairs"]).replace([pd.NA, float("inf")], 0).fillna(0) * 100
    weekly["loss_pct"] = weekly["loss_pct"].round(2)

    return weekly


# ============================================================
# Change detection (new / recovered / churned)
# ============================================================

def compute_new_pairs_over_time(off_df: pd.DataFrame, last_n_weeks: int = 4) -> pd.DataFrame:
    """
    New pairs each week (excluding the first week in the window).

    We treat the earliest week as baseline, so we DON'T plot it.
    For week i>0: new_pairs = pairs in week i not seen in any previous week in the window.
    """
    if off_df is None or off_df.empty or "scrape_week" not in off_df.columns:
        return pd.DataFrame(columns=["scrape_week", "new_pairs"])

    weeks = _weeks_last_n(off_df, last_n_weeks)
    if len(weeks) < 2:
        # Not enough history to compute "new" beyond a baseline
        return pd.DataFrame(columns=["scrape_week", "new_pairs"])

    df_c = off_df[off_df["scrape_week"].isin(weeks)].copy()
    df_c["comp_key"] = df_c["Competitor"].apply(normalize_competitor_name)
    df_c = df_c[["scrape_week", "comp_key", "SKU"]].drop_duplicates()

    weeks_sorted = sorted(weeks)

    # Baseline: first week pairs are "known", but not counted as "new"
    base_week = weeks_sorted[0]
    base_pairs = df_c[df_c["scrape_week"] == base_week][["comp_key", "SKU"]]
    seen_prev: set[tuple[str, str]] = set(map(tuple, base_pairs.values.tolist()))

    rows = []
    for w in weeks_sorted[1:]:
        week_pairs = df_c[df_c["scrape_week"] == w][["comp_key", "SKU"]]
        pairs = set(map(tuple, week_pairs.values.tolist()))
        new_pairs = pairs - seen_prev
        rows.append({"scrape_week": w, "new_pairs": len(new_pairs)})
        seen_prev |= pairs

    return pd.DataFrame(rows)



def compute_recovered_churn_global(off_df: pd.DataFrame, look_df: pd.DataFrame, last_n_weeks: int = 4) -> pd.DataFrame:
    """
    For each week (starting from the 2nd in the window):
      recovered_pairs = missing prev week -> found this week
      churned_pairs   = found prev week   -> missing this week
    """
    if "scrape_week" not in off_df.columns or "scrape_week" not in look_df.columns:
        return pd.DataFrame(columns=["scrape_week", "recovered_pairs", "churned_pairs"])

    off_a, look_a, _ = align_competitors(off_df, look_df)
    weeks = _weeks_last_n(off_a, last_n_weeks)
    if len(weeks) < 2:
        return pd.DataFrame(columns=["scrape_week", "recovered_pairs", "churned_pairs"])

    status = _build_status_table(off_a, look_a, weeks)

    rows = []
    for i in range(1, len(weeks)):
        w_prev = weeks[i - 1]
        w_curr = weeks[i]

        prev_s = status[status["scrape_week"] == w_prev][["comp_key", "SKU", "found"]].drop_duplicates()
        curr_s = status[status["scrape_week"] == w_curr][["comp_key", "SKU", "found"]].drop_duplicates()

        common = prev_s.merge(curr_s, on=["comp_key", "SKU"], how="inner", suffixes=("_prev", "_curr"))

        recovered = ((common["found_prev"] == False) & (common["found_curr"] == True)).sum()
        churned = ((common["found_prev"] == True) & (common["found_curr"] == False)).sum()

        rows.append({"scrape_week": w_curr, "recovered_pairs": int(recovered), "churned_pairs": int(churned)})

    return pd.DataFrame(rows)


def compute_competitor_weekly_churn(off_df: pd.DataFrame, look_df: pd.DataFrame, last_n_weeks: int = 4) -> pd.DataFrame:
    """
    Per-competitor recovered/churned per week (week = current week in the transition).
    """
    if "scrape_week" not in off_df.columns or "scrape_week" not in look_df.columns:
        return pd.DataFrame(columns=["scrape_week", "Competitor", "churned_pairs", "recovered_pairs"])

    off_a, look_a, _ = align_competitors(off_df, look_df)
    weeks = _weeks_last_n(off_a, last_n_weeks)
    if len(weeks) < 2:
        return pd.DataFrame(columns=["scrape_week", "Competitor", "churned_pairs", "recovered_pairs"])

    status = _build_status_table(off_a, look_a, weeks)

    comp_map = (
        off_a[["comp_key", "Competitor"]]
        .drop_duplicates()
        .groupby("comp_key")["Competitor"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    rows = []
    for i in range(1, len(weeks)):
        w_prev = weeks[i - 1]
        w_curr = weeks[i]

        prev_s = status[status["scrape_week"] == w_prev][["comp_key", "SKU", "found"]].drop_duplicates()
        curr_s = status[status["scrape_week"] == w_curr][["comp_key", "SKU", "found"]].drop_duplicates()

        common = prev_s.merge(curr_s, on=["comp_key", "SKU"], how="inner", suffixes=("_prev", "_curr"))

        common["churned"] = (common["found_prev"] == True) & (common["found_curr"] == False)
        common["recovered"] = (common["found_prev"] == False) & (common["found_curr"] == True)

        grp = common.groupby("comp_key").agg(
            churned_pairs=("churned", "sum"),
            recovered_pairs=("recovered", "sum"),
        ).reset_index()

        grp["scrape_week"] = w_curr
        grp["Competitor"] = grp["comp_key"].map(comp_map).fillna(grp["comp_key"])

        rows.append(grp[["scrape_week", "Competitor", "churned_pairs", "recovered_pairs"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["scrape_week", "Competitor", "churned_pairs", "recovered_pairs"])


# ============================================================
# Latest-week "new SKUs table" (your previous behavior)
# ============================================================

def compute_new_skus_latest(off_df: pd.DataFrame, last_n_weeks: int = 4) -> pd.DataFrame:
    """
    New SKUs in latest week vs previous weeks (within window).
    """
    if off_df is None or off_df.empty or "scrape_week" not in off_df.columns:
        return pd.DataFrame(columns=["scrape_week", "Country", "Competitor", "SKU", "URL"])

    weeks = _weeks_last_n(off_df, last_n_weeks)
    if not weeks:
        return pd.DataFrame(columns=["scrape_week", "Country", "Competitor", "SKU", "URL"])

    latest_week = weeks[-1]
    prev_weeks = weeks[:-1]

    df_latest = off_df[off_df["scrape_week"] == latest_week].copy()
    df_prev = off_df[off_df["scrape_week"].isin(prev_weeks)].copy()

    df_latest["comp_key"] = df_latest["Competitor"].apply(normalize_competitor_name)
    df_prev["comp_key"] = df_prev["Competitor"].apply(normalize_competitor_name)

    latest_pairs = df_latest[["comp_key", "SKU"]].drop_duplicates()
    prev_pairs = df_prev[["comp_key", "SKU"]].drop_duplicates()

    new_pairs = latest_pairs.merge(prev_pairs, on=["comp_key", "SKU"], how="left", indicator=True)
    new_pairs = new_pairs[new_pairs["_merge"] == "left_only"][["comp_key", "SKU"]]

    out = df_latest.merge(new_pairs, on=["comp_key", "SKU"], how="inner")

    if "URL" not in out.columns:
        out["URL"] = ""
    if "Country" not in out.columns:
        out["Country"] = ""

    out["scrape_week"] = latest_week
    out = out[["scrape_week", "Country", "Competitor", "SKU", "URL"]].drop_duplicates()
    out = out.sort_values(["Competitor", "SKU"])

    return out


# ============================================================
# Charts
# ============================================================

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


def loss_history_line(weekly: pd.DataFrame):
    if weekly is None or weekly.empty:
        return {}
    fig = px.line(
        weekly,
        x="scrape_week",
        y="loss_pct",
        markers=True,
        labels={"scrape_week": "Scrape week (Saturday)", "loss_pct": "Loss %"},
        title="Loss % over time (last weeks)",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#111827"),
    )
    fig.update_xaxes(type="category")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E5E7EB")
    return fig


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
        title="Found vs Missing by Competitor (latest week)",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
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
        title="Invalid links by competitor",
    )
    fig.update_traces(marker={"opacity": counts["opacity"].tolist()})
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Competitor",
        yaxis_title="Invalid links",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#111827"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E5E7EB")
    return fig


def new_skus_history_chart(new_df: pd.DataFrame):
    if new_df is None or new_df.empty:
        return {}
    fig = px.bar(
        new_df,
        x="scrape_week",
        y="new_pairs",
        labels={"scrape_week": "Week (Saturday)", "new_pairs": "New SKUs"},
        title="New SKUs over time",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#111827"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_xaxes(type="category")
    fig.update_yaxes(gridcolor="#E5E7EB")
    return fig


def recovered_churned_bar(df_rc: pd.DataFrame, title: str):
    if df_rc is None or df_rc.empty:
        return {}
    long_df = df_rc.melt(
        id_vars=["scrape_week"],
        value_vars=["recovered_pairs", "churned_pairs"],
        var_name="metric",
        value_name="count",
    )
    fig = px.bar(
        long_df,
        x="scrape_week",
        y="count",
        color="metric",
        barmode="group",
        labels={"scrape_week": "Week (Saturday)", "count": "Count", "metric": ""},
        title=title,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#111827"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_xaxes(type="category")
    fig.update_yaxes(gridcolor="#E5E7EB")
    return fig


def competitor_churn_heatmap(comp_churn: pd.DataFrame, top_n: int = 40):
    """
    Heatmap of churned_pairs by Competitor x scrape_week (week is current week of transition).
    """
    if comp_churn is None or comp_churn.empty:
        return {}

    df_c = comp_churn.copy()
    totals = df_c.groupby("Competitor")["churned_pairs"].sum().reset_index(name="tot_churn")
    keep = set(totals.sort_values("tot_churn", ascending=False).head(top_n)["Competitor"])
    df_c = df_c[df_c["Competitor"].isin(keep)].copy()

    pivot = df_c.pivot_table(
        index="Competitor",
        columns="scrape_week",
        values="churned_pairs",
        aggfunc="sum",
        fill_value=0,
    )

    fig = px.imshow(
        pivot,
        labels=dict(x="Week (Saturday)", y="Competitor", color="Churned SKUs"),
        aspect="auto",
        title="Churn Heatmap (Found → Missing) — top competitors by churn",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#111827"),
        title_font=dict(size=14),
    )
    return fig


# ============================================================
# Table styling
# ============================================================

def table_columns(df: pd.DataFrame):
    return [{"name": c, "id": c} for c in df.columns]


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


# ============================================================
# Dash app layout
# ============================================================

app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Store(id="offline-data"),
        dcc.Store(id="looker-data"),
        dcc.Store(id="offline-invalid-data"),
        dcc.Store(id="selected-country"),

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

        html.Div(
            [
                html.H4("Offline DB CSV (Competitor, SKU, URL, optional row_count, optional scrape_week)", className="section-title"),
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
                            html.Div(id="snowflake-fetch-status", style={"marginTop": "8px", "fontSize": "13px"}),
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
                html.H4("Looker CSV (Competitor, SKU, Scraped Week Week)", className="section-title"),
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
                        dcc.Checklist(
                            id="toggle-apparel",
                            options=[{"label": "Apparel mode (append '__apparel' to all Looker competitors)", "value": "on"}],
                            value=[],
                            labelStyle={"display": "inline-block", "marginRight": "12px", "marginTop": "8px"},
                        ),
                    ],
                    className="card",
                    style={"padding": "10px 12px"},
                ),
            ],
            style={"marginTop": "12px"},
        ),

        # -----------------------
        # Latest week section
        # -----------------------
        html.H3("Overall (latest week)", className="section-title"),
        html.Div(id="latest-week-note", className="upload-summary", style={"marginBottom": "8px"}),
        html.Div(id="global-cards", className="metrics-row"),

        html.H3("Historical (loss %, last 4 weeks)", className="section-title"),
        html.Div(
            [
                dcc.Loading(type="default", children=dcc.Graph(id="loss-history-chart")),
            ],
            className="card",
            style={"padding": "12px"},
        ),

        # -----------------------
        # Change detection tabs
        # -----------------------
        html.H3("Change Detection (last 4 weeks)", className="section-title"),
        html.Div(
            [
                dcc.Tabs(
                    id="change-tabs",
                    value="global",
                    children=[
                        dcc.Tab(
                            label="Global",
                            value="global",
                            children=[
                                dcc.Loading(type="default", children=dcc.Graph(id="new-skus-history-chart")),
                                dcc.Loading(type="default", children=dcc.Graph(id="recovered-churned-global-chart")),
                            ],
                        ),
                        dcc.Tab(
                            label="By Competitor",
                            value="competitor",
                            children=[
                                html.Div(
                                    [
                                        html.H4("Recovered / Churned trend", className="subsection-title"),
                                        dcc.Dropdown(
                                            id="change-competitor-dropdown",
                                            options=[{"label": "All competitors", "value": ALL_COMP}],
                                            value=ALL_COMP,
                                            clearable=False,
                                        ),
                                        dcc.Loading(type="default", children=dcc.Graph(id="recovered-churned-competitor-chart")),
                                    ],
                                    className="card",
                                    style={"padding": "12px"},
                                ),
                            ],
                        ),
                    ],
                )
            ],
            className="card",
            style={"padding": "12px"},
        ),

        # -----------------------
        # Distinct cards
        # -----------------------
        html.H3("Distinct Values (latest week)", className="section-title"),
        html.Div(id="distinct-cards", className="metrics-row"),

        # -----------------------
        # By competitor latest week
        # -----------------------
        html.H3("By Competitor (latest week)", className="section-title"),
        html.Div(dcc.Loading(type="default", children=dcc.Graph(id="comp-volume-chart")), className="card"),

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

        # -----------------------
        # Drilldown
        # -----------------------
        html.H3("Competitor Drilldown (latest week)", className="section-title"),
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
                        html.H4("SKUs Found in Looker (latest week)", className="subsection-title"),
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
                        html.H4("SKUs Missing in Looker (latest week)", className="subsection-title"),
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

        # -----------------------
        # New SKUs latest week (table)
        # -----------------------
        html.H3("New SKUs (latest week vs previous 3 weeks)", className="section-title"),
        html.Div(
            [
                html.Div(id="new-skus-note", className="upload-summary", style={"marginBottom": "8px"}),
                dcc.Loading(
                    type="default",
                    children=dash_table.DataTable(
                        id="new-skus-table",
                        data=[],
                        columns=[],
                        **COMMON_TABLE_PROPS,
                        page_size=20,
                        export_format="csv",
                    ),
                ),
            ],
            className="card",
            style={"padding": "12px"},
        ),

        # -----------------------
        # Top missing (latest week)
        # -----------------------
        html.H3("Top Missing SKUs (latest week)", className="section-title"),
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

        # -----------------------
        # Invalid links
        # -----------------------
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
                html.H4("All invalid links (all competitors)", className="subsection-title", style={"marginTop": "14px"}),
                dcc.Loading(
                    type="default",
                    children=dash_table.DataTable(
                        id="invalid-all-table",
                        data=[],
                        columns=[],
                        **COMMON_TABLE_PROPS,
                        page_size=20,
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


# ============================================================
# UI callbacks
# ============================================================

@callback(
    Output("offline-upload-container", "style"),
    Output("snowflake-fetch-container", "style"),
    Input("offline-source", "value"),
)
def toggle_offline_source_ui(source):
    if source == "snowflake":
        return {"display": "none"}, {"padding": "12px", "marginTop": "10px", "display": "block"}
    return {"marginTop": "10px", "display": "block"}, {"display": "none"}


@callback(
    Output("looker-data", "data"),
    Input("upload-looker", "contents"),
    prevent_initial_call=False,
)
def load_looker(contents):
    df_c = clean_df(parse_contents(contents))
    df_c = df_c.drop(columns=["comp_key"], errors="ignore")
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


@callback(
    Output("offline-data", "data"),
    Output("upload-status", "children"),
    Output("snowflake-fetch-status", "children"),
    Output("selected-country", "data"),
    Input("upload-offline", "contents"),
    Input("btn-fetch", "n_clicks"),
    Input("offline-source", "value"),
    State("country-dropdown", "value"),
    State("upload-offline", "filename"),
    prevent_initial_call=False,
)
def load_offline(off_contents, n_fetch, source, country, off_filename):
    trig = ctx.triggered_id

    if trig == "offline-source":
        if source == "snowflake":
            return no_update, "Offline source changed to Snowflake.", "Select a country and click Fetch.", no_update
        return no_update, "Offline source changed to Upload.", "", no_update

    if source == "upload":
        if off_contents is None:
            return no_update, "Waiting for offline upload...", "", no_update
        df_c = clean_df(parse_contents(off_contents))
        df_c = df_c.drop(columns=["comp_key"], errors="ignore")
        name = off_filename or "offline.csv"
        selected_country = derive_selected_country_from_df(df_c)
        return df_c.to_dict("records"), f"Offline loaded from upload: {name} ({len(df_c)} rows)", "", selected_country

    if source == "snowflake":
        if trig != "btn-fetch":
            return no_update, "Snowflake mode enabled.", "Select a country and click Fetch.", no_update

        country = (country or "").strip().upper()
        if country not in COUNTRIES:
            return no_update, "Snowflake mode enabled.", f"Invalid country '{country}'.", no_update

        cached = _cache_get(country)
        if cached is not None:
            return (
                cached,
                f"Offline loaded from cache for {country} ({len(cached)} rows)",
                f"Cache hit (TTL {CACHE_TTL_SECONDS//60}m).",
                country,
            )

        try:
            df_c = resultScrapingData(country)
            df_c = clean_df(df_c)
            df_c = df_c.drop(columns=["comp_key"], errors="ignore")
            records = df_c.to_dict("records")
            _cache_set(country, records)
            return (
                records,
                f"Offline loaded from Snowflake for {country} ({len(records)} rows)",
                "Fetched from Snowflake and cached.",
                country,
            )
        except Exception as e:
            return no_update, "Snowflake fetch failed.", f"{type(e).__name__}: {e}", no_update

    return no_update, "Waiting...", "", no_update


# ============================================================
# Main compute callback (global + tables + charts)
# ============================================================

@callback(
    Output("latest-week-note", "children"),
    Output("global-cards", "children"),
    Output("loss-history-chart", "figure"),

    # Change detection (global)
    Output("new-skus-history-chart", "figure"),
    Output("recovered-churned-global-chart", "figure"),

    # Latest competitor chart/table
    Output("comp-volume-chart", "figure"),
    Output("comp-table", "data"),
    Output("comp-table", "columns"),
    Output("competitor-dropdown", "options"),
    Output("competitor-dropdown", "value"),

    # Top missing
    Output("top-missing", "data"),
    Output("top-missing", "columns"),

    # Distinct cards + summary
    Output("distinct-cards", "children"),
    Output("upload-summary", "children"),

    # Invalid section
    Output("offline-invalid-data", "data"),
    Output("invalid-competitor-dropdown", "options"),
    Output("invalid-competitor-dropdown", "value"),
    Output("invalid-rowcount-msg", "children"),
    Output("invalid-all-table", "data"),
    Output("invalid-all-table", "columns"),

    # New SKUs table
    Output("new-skus-note", "children"),
    Output("new-skus-table", "data"),
    Output("new-skus-table", "columns"),

    # Change competitor dropdown options (for tab)
    Output("change-competitor-dropdown", "options"),
    Output("change-competitor-dropdown", "value"),

    Input("offline-data", "data"),
    Input("looker-data", "data"),
    Input("toggle-remove-invalid", "value"),
    Input("selected-country", "data"),
    Input("toggle-apparel", "value"),
)
def update_views(off_data, look_data, toggle_value, selected_country,apparel_value):
    off_df = pd.DataFrame(off_data or [])
    look_df = pd.DataFrame(look_data or [])

    apparel_enabled = bool(apparel_value and "on" in apparel_value)
    look_df = apply_apparel_suffix_to_looker(look_df, apparel_enabled)

    # Empty state
    if off_df.empty or look_df.empty:
        empty_stats = pd.DataFrame({"metric": ["Offline rows", "Looker rows"], "value": [len(off_df), len(look_df)]})
        empty_cards = metric_cards(empty_stats)
        return (
            "Upload/fetch Offline and upload Looker to compute metrics.",
            empty_cards,
            {},
            {}, {}, {},
            {},
            [], [], [], None,
            [], [],
            empty_cards,
            "Upload/fetch Offline and upload Looker to compute metrics.",
            [],
            [], None, "",
            [], [],
            "",
            [], [],
            [{"label": "All competitors", "value": ALL_COMP}],
            ALL_COMP,
        )

    remove_invalid = bool(toggle_value and "on" in toggle_value)

    # Invalid links
    invalid_df, row_count_msg = split_invalid_links(off_df)

    # Filter offline for metrics
    off_df_for_metrics = apply_remove_invalid_toggle(off_df, remove_invalid)
    off_df_for_metrics = filter_offline_weeks_to_looker(off_df_for_metrics, look_df)

    # Latest week selection
    if "scrape_week" in off_df_for_metrics.columns and off_df_for_metrics["scrape_week"].notna().any():
        latest_week = sorted(off_df_for_metrics["scrape_week"].dropna().unique())[-1]
        note = f"Latest scraping week (Saturday): {latest_week}"
        off_latest = off_df_for_metrics[off_df_for_metrics["scrape_week"] == latest_week].copy()
        look_latest = look_df[look_df["scrape_week"] == latest_week].copy() if "scrape_week" in look_df.columns else look_df.copy()
    else:
        note = "No scrape_week column found in Offline; treating data as latest-week only."
        off_latest = off_df_for_metrics.copy()
        look_latest = look_df.copy()

    # Latest week metrics
    merged, comp_stats, global_stats, unmatched_offline = compute_metrics_latest(off_latest, look_latest)
    cards = metric_cards(global_stats)

    # Distinct cards (latest)
    distinct_stats = pd.DataFrame(
        {
            "metric": ["Offline Distinct SKUs (latest)", "Looker Distinct SKUs (latest)"],
            "value": [
                off_latest["SKU"].nunique() if "SKU" in off_latest.columns else 0,
                look_latest["SKU"].nunique() if "SKU" in look_latest.columns else 0,
            ],
        }
    )
    distinct_cards = metric_cards(distinct_stats)

    # Latest competitor chart/table
    comp_volume_fig = comp_stack_chart(comp_stats)
    comp_cols = table_columns(comp_stats) if not comp_stats.empty else []
    comp_data = comp_stats.to_dict("records")

    competitors = sorted(merged["Competitor"].dropna().unique())
    dropdown_options = [{"label": c, "value": c} for c in competitors]
    dropdown_value = competitors[0] if competitors else None

    # Top missing (latest)
    missing_latest = merged[~merged["FoundInLooker"]]
    top_missing = (
        missing_latest.groupby("SKU")
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

    # Loss % history (last 4 weeks)
    weekly_loss = compute_weekly_loss_series(off_df_for_metrics, look_df, last_n_weeks=4)
    loss_fig = loss_history_line(weekly_loss)

    # Change detection charts (need scrape_week in both)
    if "scrape_week" in off_df_for_metrics.columns and "scrape_week" in look_df.columns:
        # align is done inside these functions where needed
        new_df = compute_new_pairs_over_time(off_df_for_metrics, last_n_weeks=4)
        rc_global = compute_recovered_churn_global(off_df_for_metrics, look_df, last_n_weeks=4)
        comp_churn = compute_competitor_weekly_churn(off_df_for_metrics, look_df, last_n_weeks=4)

        new_fig = new_skus_history_chart(new_df)
        rc_fig = recovered_churned_bar(rc_global, "Recovered vs Churned (global) — week over week")

        # competitor dropdown options for By Competitor tab
        comp_list = sorted(comp_churn["Competitor"].dropna().unique()) if not comp_churn.empty else []
        change_opts = [{"label": "All competitors", "value": ALL_COMP}] + [{"label": c, "value": c} for c in comp_list]
        change_default = ALL_COMP
    else:
        new_fig = {}
        rc_fig = {}
        change_opts = [{"label": "All competitors", "value": ALL_COMP}]
        change_default = ALL_COMP

    # Upload summary
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

    # Invalid section
    if row_count_msg:
        invalid_opts = []
        invalid_val = None
        invalid_msg = row_count_msg
        invalid_all_data = []
        invalid_all_cols = []
    else:
        inv_comps = sorted(invalid_df["Competitor"].dropna().unique())
        invalid_opts = [{"label": c, "value": c} for c in inv_comps]
        invalid_val = inv_comps[0] if inv_comps else None
        invalid_msg = f"{len(invalid_df)} invalid rows (row_count = 1)."

        invalid_all_df = invalid_df.copy()
        if "URL" not in invalid_all_df.columns:
            invalid_all_df["URL"] = ""
        invalid_all_df["Country"] = (selected_country or "").strip()
        invalid_all_df = invalid_all_df[["Country", "Competitor", "SKU", "URL"]].drop_duplicates()
        invalid_all_df = invalid_all_df.sort_values(["Competitor", "SKU"], ascending=[True, True])
        invalid_all_cols = table_columns(invalid_all_df)
        invalid_all_data = invalid_all_df.to_dict("records")

    # New SKUs latest table
    if "scrape_week" in off_df_for_metrics.columns:
        new_skus_df = compute_new_skus_latest(off_df_for_metrics, last_n_weeks=4)
        if new_skus_df.empty:
            new_note = "No new SKUs found for latest week (or not enough history)."
            new_data = []
            new_cols = []
        else:
            new_note = f"New SKUs in {new_skus_df['scrape_week'].iloc[0]}: {len(new_skus_df)}"
            new_cols = table_columns(new_skus_df)
            new_data = new_skus_df.to_dict("records")
    else:
        new_note = "Offline has no scrape_week column → cannot compute new SKUs."
        new_data = []
        new_cols = []

    return (
        note,
        cards,
        loss_fig,
        new_fig,
        rc_fig,
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
        invalid_all_data,
        invalid_all_cols,
        new_note,
        new_data,
        new_cols,
        change_opts,
        change_default,
    )


# ============================================================
# Change detection: competitor trend chart callback
# ============================================================

@callback(
    Output("recovered-churned-competitor-chart", "figure"),
    Input("change-competitor-dropdown", "value"),
    State("offline-data", "data"),
    State("looker-data", "data"),
    State("toggle-remove-invalid", "value"),
    State("toggle-apparel", "value"),
)
def update_change_comp_chart(selected_comp, off_data, look_data, toggle_value, apparel_value):
    off_df = pd.DataFrame(off_data or [])
    look_df = pd.DataFrame(look_data or [])
    apparel_enabled = bool(apparel_value and "on" in apparel_value)
    look_df = apply_apparel_suffix_to_looker(look_df, apparel_enabled)

    if off_df.empty or look_df.empty or "scrape_week" not in off_df.columns or "scrape_week" not in look_df.columns:
        return {}

    remove_invalid = bool(toggle_value and "on" in toggle_value)
    off_df_for_metrics = apply_remove_invalid_toggle(off_df, remove_invalid)
    off_df_for_metrics = filter_offline_weeks_to_looker(off_df_for_metrics, look_df)

    comp_churn = compute_competitor_weekly_churn(off_df_for_metrics, look_df, last_n_weeks=4)
    if comp_churn.empty:
        return {}

    if not selected_comp or selected_comp == ALL_COMP:
        rc_global = comp_churn.groupby("scrape_week")[["recovered_pairs", "churned_pairs"]].sum().reset_index()
        return recovered_churned_bar(rc_global, "Recovered vs Churned (global) — week over week")

    df_c = comp_churn[comp_churn["Competitor"] == selected_comp].copy()
    if df_c.empty:
        return {}

    # Bar grouped (like global) but per competitor
    df_c = df_c.sort_values("scrape_week")
    fig = recovered_churned_bar(df_c[["scrape_week", "recovered_pairs", "churned_pairs"]],
                                f"Recovered vs Churned — {selected_comp}")
    return fig


# ============================================================
# Drilldown: latest week found/missing tables
# ============================================================

@callback(
    Output("found-table", "data"),
    Output("missing-table", "data"),
    Input("competitor-dropdown", "value"),
    State("offline-data", "data"),
    State("looker-data", "data"),
    State("toggle-remove-invalid", "value"),
    State("toggle-apparel", "value"),
)
def update_drilldown(selected_comp, off_data, look_data, toggle_value, apparel_value):
    if not selected_comp:
        return [], []

    off_df = pd.DataFrame(off_data or [])
    look_df = pd.DataFrame(look_data or [])

    if off_df.empty or look_df.empty:
        return [], []
    
    apparel_enabled = bool(apparel_value and "on" in apparel_value)
    look_df = apply_apparel_suffix_to_looker(look_df, apparel_enabled)

    remove_invalid = bool(toggle_value and "on" in toggle_value)
    off_df_for_metrics = apply_remove_invalid_toggle(off_df, remove_invalid)

    # Latest week only
    if "scrape_week" in off_df_for_metrics.columns and off_df_for_metrics["scrape_week"].notna().any():
        latest_week = sorted(off_df_for_metrics["scrape_week"].dropna().unique())[-1]
        off_latest = off_df_for_metrics[off_df_for_metrics["scrape_week"] == latest_week].copy()
        look_latest = look_df[look_df["scrape_week"] == latest_week].copy() if "scrape_week" in look_df.columns else look_df.copy()
    else:
        off_latest = off_df_for_metrics.copy()
        look_latest = look_df.copy()

    merged, _, _, _ = compute_metrics_latest(off_latest, look_latest)
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


# ============================================================
# Invalid links section callback
# ============================================================

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


# ============================================================
# Entrypoint
# ============================================================

server = app.server

if __name__ == "__main__":
    app.run()
