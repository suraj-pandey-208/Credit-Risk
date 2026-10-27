import os
import json
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ============================================================
# LOAD .ENV
# ============================================================

load_dotenv()


# ============================================================
# DATABASE CONFIG
# ============================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "credit_risk")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ============================================================
# DATABASE URL
# ============================================================

if not DB_PASSWORD:
    raise RuntimeError(
        "DB_PASSWORD is missing. "
        "Please create a .env file in the project root."
    )


DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}@"
    f"{DB_HOST}:{DB_PORT}/"
    f"{DB_NAME}"
)


# ============================================================
# SQLALCHEMY ENGINE
# ============================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True
)


# ============================================================
# CREATE TABLE
# ============================================================

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_events (
    id BIGSERIAL PRIMARY KEY,

    request_id VARCHAR(64)
        UNIQUE NOT NULL,

    applicant_id VARCHAR(128),

    created_at TIMESTAMPTZ
        NOT NULL,

    risk_probability DOUBLE PRECISION
        NOT NULL,

    threshold DOUBLE PRECISION
        NOT NULL,

    risk_band VARCHAR(32)
        NOT NULL,

    decision VARCHAR(64)
        NOT NULL,

    input_payload JSONB,

    drivers JSONB
);
"""


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db():

    with engine.begin() as conn:

        conn.execute(
            text(CREATE_SQL)
        )


# ============================================================
# DATABASE HEALTH
# ============================================================

def db_health():

    try:

        with engine.connect() as conn:

            conn.execute(
                text("SELECT 1")
            )

        return "connected"

    except Exception as e:

        return f"unavailable: {e}"


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    result,
    payload
):

    with engine.begin() as conn:

        conn.execute(
            text(
                """
                INSERT INTO prediction_events
                (
                    request_id,
                    applicant_id,
                    created_at,
                    risk_probability,
                    threshold,
                    risk_band,
                    decision,
                    input_payload,
                    drivers
                )
                VALUES
                (
                    :request_id,
                    :applicant_id,
                    :created_at,
                    :risk_probability,
                    :threshold,
                    :risk_band,
                    :decision,
                    CAST(:input_payload AS JSONB),
                    CAST(:drivers AS JSONB)
                )
                ON CONFLICT (request_id)
                DO NOTHING
                """
            ),
            {
                "request_id":
                    result["request_id"],

                "applicant_id":
                    result["applicant_id"],

                "created_at":
                    datetime.now(timezone.utc),

                "risk_probability":
                    float(
                        result["risk_probability"]
                    ),

                "threshold":
                    float(
                        result["policy_threshold"]
                    ),

                "risk_band":
                    result["risk_band"],

                "decision":
                    result["decision"],

                "input_payload":
                    json.dumps(
                        payload,
                        default=str
                    ),

                "drivers":
                    json.dumps(
                        result.get(
                            "top_model_drivers",
                            []
                        ),
                        default=str
                    ),
            }
        )


# ============================================================
# FETCH RECENT PREDICTIONS
# ============================================================

def fetch_recent_predictions(
    limit=50
):

    with engine.connect() as conn:

        rows = conn.execute(
            text(
                """
                SELECT
                    created_at,
                    applicant_id,
                    risk_probability,
                    threshold,
                    risk_band,
                    decision
                FROM prediction_events
                ORDER BY id DESC
                LIMIT :limit
                """
            ),
            {
                "limit": int(limit)
            }
        ).mappings().all()

    return [
        dict(row)
        for row in rows
    ]