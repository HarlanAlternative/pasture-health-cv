CREATE TABLE IF NOT EXISTS inferences (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    bbox            FLOAT[],
    date_from       DATE,
    date_to         DATE,
    health_score    FLOAT,
    ndvi_mean       FLOAT,
    ndvi_std        FLOAT,
    class_healthy   FLOAT,
    class_stressed  FLOAT,
    class_bare      FLOAT,
    class_water     FLOAT,
    inference_ms    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_inferences_created_at ON inferences(created_at);
