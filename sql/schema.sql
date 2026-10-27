CREATE TABLE IF NOT EXISTS prediction_events (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(64) UNIQUE NOT NULL,
    applicant_id VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL,
    risk_probability DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    risk_band VARCHAR(32) NOT NULL,
    decision VARCHAR(64) NOT NULL,
    input_payload JSONB,
    drivers JSONB
);
