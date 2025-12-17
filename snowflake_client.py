import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from contextlib import contextmanager

load_dotenv()


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
    When using pd.read_sql with DBAPI parameter style, any literal % in SQL must be escaped as %%.
    """
    with snowflake_connection() as conn:
        return pd.read_sql(sql, conn, params=params)


RESULT_SCRAPING_DATA_SQL = """
WITH NP_Status AS (
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
                npproductcode AS np_sku,
                CASE
                    WHEN npproductcode LIKE 'VEN%%' THEN itemstatus
                    WHEN npproductcode LIKE 'VDS%%' THEN itemstatus
                    ELSE productfamilysummarystatus
                END AS status,
                channel
            FROM NATPEN_LAKE_PROD.PIM.PIM_PRODUCTCHANNELS t1
            JOIN NATPEN_LAKE_PROD.PIM.PIM_PRODUCTS t2
                ON t1.PRODUCTID = t2.PRODUCTID
            WHERE channel IN ('NPL', 'NADM', 'AU/NZ')
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

ranked_scrapes AS (
    SELECT DISTINCT
        COUNTRY,
        COMPETITOR,
        SCRAPED_DATE,
        DENSE_RANK() OVER (
            PARTITION BY COUNTRY, COMPETITOR
            ORDER BY SCRAPED_DATE DESC
        ) AS scrape_rank
    FROM MCP.MARKETPLACE.COMPETITOR_MAPPED_DATA
    WHERE COUNTRY = %s
),

dates_per_competitor AS (
    SELECT
        COUNTRY,
        COMPETITOR,
        MAX(CASE WHEN scrape_rank = 1 THEN SCRAPED_DATE END) AS latest_date,
        MAX(CASE WHEN scrape_rank = 2 THEN SCRAPED_DATE END) AS previous_date
    FROM ranked_scrapes
    GROUP BY COUNTRY, COMPETITOR
),

chosen_date AS (
    SELECT
        COUNTRY,
        COMPETITOR,
        CASE
            WHEN latest_date >= DATEADD(day, -5, DATE_TRUNC('week', CURRENT_DATE()))
                THEN previous_date
            ELSE latest_date
        END AS target_scraped_date
    FROM dates_per_competitor
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
    d.ATTRIBUTES:SKU::STRING      AS SKU,
    nps.status                   AS STATUS,
    d.ATTRIBUTES:URL::STRING     AS URL,
    cs.row_count,
    cd.target_scraped_date
FROM MCP.MARKETPLACE.COMPETITOR_MAPPED_DATA d
JOIN chosen_date cd
    ON d.COUNTRY = cd.COUNTRY
   AND d.COMPETITOR = cd.COMPETITOR
   AND d.SCRAPED_DATE = cd.target_scraped_date
LEFT JOIN NP_Status nps
    ON d.ATTRIBUTES:SKU::STRING = nps.np_sku
JOIN count_sku cs
  ON d.ATTRIBUTES:SKU = cs.SKU
 AND d.COMPETITOR = cs.COMPETITOR
 AND cd.target_scraped_date = cs.SCRAPED_DATE
WHERE d.COUNTRY = %s
  AND nps.status != 'Disabled'
  AND nps.status NOT ILIKE '%%not%%'
ORDER BY SKU, COMPETITOR
"""


def resultScrapingData(country: str) -> pd.DataFrame:
    country = (country or "").strip().upper()
    if not country:
        raise ValueError("country must be a non-empty string like 'DE'")

    params = (country, country, country)
    df_c = query_df(RESULT_SCRAPING_DATA_SQL, params=params)

    rename_map = {
        "COMPETITOR": "Competitor",
        "SKU": "SKU",
        "URL": "URL",
        "ROW_COUNT": "row_count",
        "COUNTRY": "Country",
        "TARGET_SCRAPED_DATE": "target_scraped_date",
        "STATUS": "STATUS",
    }
    df_c = df_c.rename(columns={k: v for k, v in rename_map.items() if k in df_c.columns})

    return df_c
