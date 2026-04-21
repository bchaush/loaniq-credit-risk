-- LoanIQ Feature Engineering
-- Runs on the applications table, produces model_features view

CREATE VIEW IF NOT EXISTS model_features AS
SELECT
    SK_ID_CURR,
    TARGET,

    -- === INCOME & DEBT ===
    AMT_INCOME_TOTAL,
    AMT_CREDIT,
    AMT_ANNUITY,
    AMT_GOODS_PRICE,

    ROUND(AMT_CREDIT / NULLIF(AMT_INCOME_TOTAL, 0), 4)     AS debt_to_income,
    ROUND(AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0), 4)    AS annuity_to_income,
    ROUND(AMT_CREDIT / NULLIF(AMT_ANNUITY, 0), 1)          AS loan_term_implied,
    ROUND(AMT_GOODS_PRICE / NULLIF(AMT_CREDIT, 0), 4)      AS ltv_ratio,

    -- === AGE & EMPLOYMENT ===
    ROUND(DAYS_BIRTH / 365.25, 1)                           AS age_years,
    ROUND(DAYS_EMPLOYED / 365.25, 1)                        AS employed_years,
    ROUND(DAYS_EMPLOYED / NULLIF(DAYS_BIRTH, 0), 4)         AS employment_to_age_ratio,

    CASE WHEN DAYS_EMPLOYED IS NULL THEN 1 ELSE 0 END       AS is_unemployed,

    -- === CREDIT SCORE FLAGS ===
    EXT_SOURCE_1,
    EXT_SOURCE_2,
    EXT_SOURCE_3,
    ROUND(
        COALESCE(EXT_SOURCE_1, 0) +
        COALESCE(EXT_SOURCE_2, 0) +
        COALESCE(EXT_SOURCE_3, 0)
    , 4)                                                    AS ext_score_sum,

    CASE WHEN EXT_SOURCE_2 < 0.3 THEN 1 ELSE 0 END         AS low_ext_score_2,
    CASE WHEN EXT_SOURCE_3 < 0.3 THEN 1 ELSE 0 END         AS low_ext_score_3,

    -- === FAMILY & HOUSING ===
    CNT_CHILDREN,
    CNT_FAM_MEMBERS,
    FLAG_OWN_CAR,
    FLAG_OWN_REALTY,

    CASE WHEN CNT_CHILDREN > 2 THEN 1 ELSE 0 END           AS many_children,

    -- === REGION & BEHAVIOR ===
    REGION_RATING_CLIENT,
    REG_CITY_NOT_WORK_CITY,
    FLAG_DOCUMENT_3,
    COALESCE(AMT_REQ_CREDIT_BUREAU_YEAR, 0)                 AS credit_inquiries_year,

    CASE WHEN AMT_REQ_CREDIT_BUREAU_YEAR > 3
         THEN 1 ELSE 0 END                                  AS high_inquiry_flag,

    -- === CATEGORICAL (keep as-is for one-hot encoding in Python) ===
    NAME_INCOME_TYPE,
    NAME_EDUCATION_TYPE,
    NAME_FAMILY_STATUS,
    NAME_HOUSING_TYPE,
    OCCUPATION_TYPE,
    ORGANIZATION_TYPE

FROM applications;