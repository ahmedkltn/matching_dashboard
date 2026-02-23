# snowflake_client.py
import os
from contextlib import contextmanager

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

SCRAPE_WEEK_GRACE_DAYS = 5


@contextmanager
def snowflake_connection():
    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        authenticator="externalbrowser",
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE"),
    )
    try:
        yield conn
    finally:
        conn.close()


def query_df(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """
    Runs a query and returns a dataframe.

    NOTE:
    When using pd.read_sql with DBAPI param style, any literal % in SQL must be escaped as %%.
    """
    with snowflake_connection() as conn:
        return pd.read_sql(sql, conn, params=params)


RESULT_SCRAPING_DATA_SQL = f"""
WITH cfg AS (
    SELECT {SCRAPE_WEEK_GRACE_DAYS} AS grace_days
),

country_cfg AS (
    SELECT
        UPPER(%s) AS country,
        CASE
            WHEN UPPER(%s) IN ('ES', 'FR', 'DE', 'IT', 'UK', 'GB') THEN 'NPL'
            WHEN UPPER(%s) IN ('CA', 'US') THEN 'NADM'
            ELSE 'AU/NZ'
        END AS channel
),

NP_Status AS (
    SELECT
        COALESCE(np_sku, product_id) AS np_sku,
        COALESCE(status, v2_status)  AS status
    FROM (
        SELECT
            inn.np_sku,
            inn.status,
            v2.product_id,
            v2.v2_status
        FROM (
            SELECT DISTINCT
                t2.npproductcode AS np_sku,
                CASE
                    WHEN t2.npproductcode LIKE 'VEN%%' THEN t1.itemstatus
                    WHEN t2.npproductcode LIKE 'VDS%%' THEN t1.itemstatus
                    ELSE t1.productfamilysummarystatus
                END AS status,
                t1.channel
            FROM NATPEN_LAKE_PROD.PIM.PIM_PRODUCTCHANNELS t1
            JOIN NATPEN_LAKE_PROD.PIM.PIM_PRODUCTS t2
                ON t1.PRODUCTID = t2.PRODUCTID
            JOIN country_cfg cfg2
                ON t1.channel = cfg2.channel
        ) inn
        FULL JOIN (
            SELECT DISTINCT
                product_id,
                status AS v2_status,
                channel AS channel1
            FROM NATPEN_LAKE_PROD.PRODUCTS.V2_PRODUCT_STATUS_SNAPSHOT
        ) v2
            ON inn.np_sku = v2.product_id
           AND inn.channel = v2.channel1
    )
),

/* -------- Weekly snapshots aligned to Saturday --------
   IMPORTANT: apply "grace" shift BEFORE bucketing so late scrapes still count for prior week.
   Without this, a scrape on Tue 27 buckets to Saturday 31, even if it "belongs" to Saturday 24.
*/
weekly_scrapes AS (
    SELECT
        d.COUNTRY,
        d.COMPETITOR,

        /* Saturday bucket with grace shift */
        DATEADD(
            day,
            5,
            DATE_TRUNC('WEEK', DATEADD(day, -cfg.grace_days, d.SCRAPED_DATE))
        ) AS scrape_week_sat,

        /* pick the latest scrape timestamp within that bucket */
        MAX(d.SCRAPED_DATE) AS week_scraped_date
    FROM MCP.MARKETPLACE.COMPETITOR_MAPPED_DATA d
    CROSS JOIN cfg
    WHERE d.COUNTRY = %s
    GROUP BY 1, 2, 3
),

/* Rank weekly snapshots (by scrape timestamp) per competitor */
ranked_weeks AS (
    SELECT
        COUNTRY,
        COMPETITOR,
        scrape_week_sat,
        week_scraped_date,
        DENSE_RANK() OVER (
            PARTITION BY COUNTRY, COMPETITOR
            ORDER BY week_scraped_date DESC
        ) AS week_rank
    FROM weekly_scrapes
),

dates_per_competitor AS (
    SELECT
        COUNTRY,
        COMPETITOR,
        MAX(CASE WHEN week_rank = 1 THEN week_scraped_date END) AS latest_date,
        MAX(CASE WHEN week_rank = 2 THEN week_scraped_date END) AS previous_date
    FROM ranked_weeks
    GROUP BY COUNTRY, COMPETITOR
),

chosen_date AS (
    SELECT
        COUNTRY,
        COMPETITOR,
        CASE
            WHEN latest_date >= DATEADD(day, -5, DATE_TRUNC('WEEK', CURRENT_DATE()))
                THEN previous_date
            ELSE latest_date
        END AS target_scraped_date
    FROM dates_per_competitor
),

/* Keep only weekly snapshots up to target_scraped_date, take last 4 */
last_n_weeks AS (
    SELECT
        rw.COUNTRY,
        rw.COMPETITOR,
        rw.scrape_week_sat,
        rw.week_scraped_date,
        ROW_NUMBER() OVER (
            PARTITION BY rw.COUNTRY, rw.COMPETITOR
            ORDER BY rw.week_scraped_date DESC
        ) AS rn
    FROM ranked_weeks rw
    JOIN chosen_date cd
      ON rw.COUNTRY = cd.COUNTRY
     AND rw.COMPETITOR = cd.COMPETITOR
     AND rw.week_scraped_date <= cd.target_scraped_date
),

snapshots AS (
    SELECT
        COUNTRY,
        COMPETITOR,
        scrape_week_sat,
        week_scraped_date
    FROM last_n_weeks
    WHERE rn <= 4
),

count_sku AS (
    SELECT
        COMPETITOR,
        ATTRIBUTES:SKU::STRING AS SKU,
        DATE(SCRAPED_DATE) AS SCRAPED_DATE,
        COUNT(*) AS row_count
    FROM MCP.MARKETPLACE.COMPETITOR_MAPPED_DATA d
    WHERE d.COUNTRY = %s
    GROUP BY COMPETITOR, SKU, DATE(SCRAPED_DATE)
)

SELECT DISTINCT
    d.COMPETITOR,
    d.COUNTRY,
    s.scrape_week_sat AS scrape_week,

    s.week_scraped_date AS scraped_date,

    d.ATTRIBUTES:SKU::STRING      AS SKU,
    nps.status                    AS STATUS,
    d.ATTRIBUTES:URL::STRING      AS URL,
    cs.row_count
FROM MCP.MARKETPLACE.COMPETITOR_MAPPED_DATA d
JOIN snapshots s
  ON d.COUNTRY = s.COUNTRY
 AND d.COMPETITOR = s.COMPETITOR
 AND d.SCRAPED_DATE = s.week_scraped_date
LEFT JOIN NP_Status nps
  ON d.ATTRIBUTES:SKU::STRING = nps.np_sku
JOIN count_sku cs
  ON d.ATTRIBUTES:SKU::STRING = cs.SKU
 AND d.COMPETITOR = cs.COMPETITOR
 AND DATE(d.SCRAPED_DATE) = cs.SCRAPED_DATE
WHERE d.COUNTRY = %s
  AND nps.status != 'Disabled'
  AND nps.status NOT ILIKE '%%not%%'
ORDER BY
    s.week_scraped_date DESC,
    SKU,
    COMPETITOR
"""


def resultScrapingData(country: str) -> pd.DataFrame:
    country = (country or "").strip().upper()
    if not country:
        raise ValueError("country must be a non-empty string like 'DE'")

    # Params order matches %s occurrences:
    # country_cfg: UPPER(%s), CASE UPPER(%s), CASE UPPER(%s)
    # weekly_scrapes: WHERE d.COUNTRY = %s
    # count_sku: WHERE d.COUNTRY = %s
    # final WHERE d.COUNTRY = %s
    params = (country, country, country, country, country, country)

    df_c = query_df(RESULT_SCRAPING_DATA_SQL, params=params)

    rename_map = {
        "COMPETITOR": "Competitor",
        "SKU": "SKU",
        "URL": "URL",
        "ROW_COUNT": "row_count",
        "COUNTRY": "Country",
        "STATUS": "STATUS",
        "SCRAPE_WEEK": "scrape_week",
        "SCRAPED_DATE": "scraped_date",
    }
    df_c = df_c.rename(columns={k: v for k, v in rename_map.items() if k in df_c.columns})

    return df_c
