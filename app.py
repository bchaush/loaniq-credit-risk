import streamlit as st
import pandas as pd
import numpy as np
import json, sys, os
import re

sys.path.insert(0, os.path.dirname(__file__))
from model.explainer import full_assessment, score_applicant

st.set_page_config(
    page_title="LoanIQ — Underwriting",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

with open("model/metadata.json") as f:
    metadata = json.load(f)

# ── Markdown → HTML helper ─────────────────────────────────────────
def md_to_html(text: str) -> str:
    """Convert basic markdown to HTML for safe injection into st.markdown HTML blocks."""
    if not text:
        return ""
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Newlines → <br>
    text = text.replace('\n', '<br>')
    return text

# ── Full design system ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ─── Reset ─── */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] { background: #0a0c11 !important; }
[data-testid="stHeader"], [data-testid="stSidebar"],
[data-testid="stToolbar"], footer { display: none !important; }

.block-container {
    max-width: 1140px !important;
    padding: 1.5rem 1.5rem 4rem !important;
    margin: 0 auto !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* ─── Typography ─── */
h1, h2, h3, h4, p, label, div {
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* ─── Input elements ─── */
[data-testid="stNumberInput"] input,
[data-baseweb="select"] > div,
[data-baseweb="input"] input {
    background: #141820 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 6px !important;
    color: #e8eaf2 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
}

[data-baseweb="select"] > div:hover,
[data-testid="stNumberInput"] input:focus {
    border-color: #252d3d !important;
    box-shadow: none !important;
}

/* Select dropdowns */
[data-baseweb="select"] > div {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
}

/* ─── Tabs ─── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e2535 !important;
    gap: 0 !important;
    margin-bottom: 1.5rem !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
    color: #555c73 !important;
    padding: 8px 18px !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #e8eaf2 !important;
    border-bottom-color: #3b82f6 !important;
    font-weight: 500 !important;
}

/* ─── Buttons ─── */
.stButton > button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 7px !important;
    font-size: 13px !important;
    width: 100% !important;
    transition: opacity .15s !important;
    letter-spacing: -.01em !important;
}
[data-testid="baseButton-primary"] {
    background: #3b82f6 !important;
    border: none !important;
    color: #fff !important;
    padding: 10px 0 !important;
}
[data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: 1px solid #252d3d !important;
    color: #8b90a8 !important;
    padding: 8px 0 !important;
}
[data-testid="baseButton-primary"]:hover { opacity: .88 !important; }
[data-testid="baseButton-secondary"]:hover {
    background: #141820 !important;
    color: #e8eaf2 !important;
}

/* ─── Metrics ─── */
[data-testid="stMetric"] {
    background: #141820 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 8px !important;
    padding: .75rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 9px !important;
    text-transform: uppercase !important;
    letter-spacing: .07em !important;
    color: #555c73 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    font-size: 18px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    color: #e8eaf2 !important;
}

/* ─── Warning ─── */
[data-testid="stAlert"] {
    background: #1a1200 !important;
    border: 1px solid #78350f !important;
    border-radius: 6px !important;
    color: #f59e0b !important;
    font-size: 12px !important;
    padding: .5rem .875rem !important;
}

/* ─── Spinner ─── */
.stSpinner { color: #3b82f6 !important; }

/* ─── Checkbox ─── */
[data-testid="stCheckbox"] label {
    color: #8b90a8 !important;
    font-size: 13px !important;
}

/* ─── Divider ─── */
hr { border-color: #1e2535 !important; margin: .75rem 0 !important; }

/* ─── Custom components ─── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.125rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid #1e2535;
}
.brand {
    display: flex; align-items: center; gap: 10px;
    font-size: 14px; font-weight: 500; color: #e8eaf2;
    font-family: 'IBM Plex Sans', sans-serif;
}
.brand-mark {
    width: 28px; height: 28px; background: #3b82f6;
    border-radius: 5px; display: flex; align-items: center;
    justify-content: center; color: #fff; font-size: 11px;
    font-family: 'IBM Plex Mono', monospace; font-weight: 500;
}
.perf-badge {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    color: #555c73; background: #0f1219; border: 1px solid #1e2535;
    border-radius: 4px; padding: 4px 10px;
}
.perf-badge b { color: #3b82f6; font-weight: 500; }

.workflow-header {
    background: #0f1219; border: 1px solid #1e2535;
    border-radius: 10px; padding: 1.125rem 1.375rem;
    margin-bottom: 1.25rem;
}
.wh-title {
    font-size: 15px; font-weight: 600; color: #e8eaf2;
    letter-spacing: -.02em; margin-bottom: 3px;
}
.wh-sub { font-size: 12px; color: #555c73; margin-bottom: .875rem; }
.steps { display: flex; align-items: center; gap: 0; }
.step { display: flex; align-items: center; gap: 8px; }
.step-num {
    width: 22px; height: 22px; border-radius: 50%;
    border: 1px solid #252d3d; display: flex;
    align-items: center; justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #555c73; flex-shrink: 0;
}
.step.s-done .step-num { background:#0a1f12;border-color:#22c55e;color:#22c55e; }
.step.s-active .step-num { background:#3b82f6;border-color:#3b82f6;color:#fff; }
.step-label { font-size: 11px; color: #555c73; white-space: nowrap; }
.step.s-done .step-label { color: #4ade80; }
.step.s-active .step-label { color: #93c5fd; }
.step-line { width: 28px; height: 1px; background: #1e2535; margin: 0 6px; flex-shrink: 0; }

.section-card {
    background: #0f1219; border: 1px solid #1e2535;
    border-radius: 10px; padding: 1.125rem 1.375rem;
    margin-bottom: .875rem;
}
.section-title {
    font-size: 11px; font-weight: 600; color: #8b90a8;
    text-transform: uppercase; letter-spacing: .08em;
}
.section-desc { font-size: 11px; color: #555c73; margin-top: 2px; margin-bottom: .875rem; }

.derived-bar {
    background: #0f1219; border: 1px solid #1e2535;
    border-radius: 10px; padding: .875rem 1.375rem;
    margin-bottom: .875rem;
}
.db-title {
    font-size: 10px; font-weight: 500; color: #555c73;
    text-transform: uppercase; letter-spacing: .07em;
    margin-bottom: .625rem;
}

/* ─── Await panel ─── */
.await-panel {
    background: #0f1219; border: 1px solid #1e2535;
    border-radius: 10px; padding: 1.375rem;
    margin-bottom: .875rem;
}
.await-label {
    font-size: 10px; font-weight: 500; text-transform: uppercase;
    letter-spacing: .07em; color: #555c73; margin-bottom: 1rem;
}
.await-decision {
    border-radius: 8px; border: 1px dashed #1e2535;
    padding: 1.5rem; text-align: center; margin-bottom: .875rem;
}
.await-ghost {
    font-size: 28px; font-weight: 600; color: #1e2535;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -.04em; margin-bottom: .375rem;
}
.await-ghost-sub { font-size: 11px; color: #555c73; }
.ghost-metric {
    background: #141820; border: 1px dashed #1e2535;
    border-radius: 7px; padding: .75rem 1rem;
}
.ghost-bar {
    height: 18px; background: #141820;
    border-radius: 3px; border: 1px solid #1e2535;
}
.ghost-line {
    height: 8px; background: #141820;
    border-radius: 2px; margin-bottom: 5px;
}
.await-explain {
    background: #141820; border-radius: 7px;
    padding: .875rem; border: 1px dashed #1e2535;
    margin-top: .875rem;
}
.await-explain-title {
    font-size: 9px; text-transform: uppercase; letter-spacing: .07em;
    color: #555c73; margin-bottom: .5rem;
}
.await-hint {
    text-align: center; font-size: 11px; color: #555c73;
    margin-top: 1rem; line-height: 1.5;
}

/* ─── Decision hero ─── */
.decision-hero {
    border-radius: 10px; padding: 1.375rem; margin-bottom: .875rem;
}
.dh-declined { background: #1a0a0a; border: 1px solid #7f1d1d; }
.dh-approved { background: #0a1a0f; border: 1px solid #14532d; }
.dh-review   { background: #1a1200; border: 1px solid #78350f; }

.dh-eyebrow {
    font-size: 9px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .1em; margin-bottom: .625rem;
}
.dh-declined .dh-eyebrow { color: #7f1d1d; }
.dh-approved .dh-eyebrow { color: #14532d; }
.dh-review   .dh-eyebrow { color: #78350f; }

.dh-label {
    font-size: 34px; font-weight: 600; letter-spacing: -.04em;
    font-family: 'IBM Plex Mono', monospace; line-height: 1;
}
.dh-declined .dh-label { color: #ef4444; }
.dh-approved .dh-label { color: #22c55e; }
.dh-review   .dh-label { color: #f59e0b; }

.dh-tier { font-size: 12px; margin-top: 4px; margin-bottom: 1rem; }
.dh-declined .dh-tier { color: #f09595; }
.dh-approved .dh-tier { color: #86efac; }
.dh-review   .dh-tier { color: #fcd34d; }

.dh-metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 1rem; }
.dh-metric { border-radius: 7px; padding: .75rem 1rem; }
.dh-declined .dh-metric { background: rgba(239,68,68,.07); }
.dh-approved .dh-metric { background: rgba(34,197,94,.07); }
.dh-review   .dh-metric { background: rgba(245,158,11,.07); }

.dh-metric-label {
    font-size: 9px; text-transform: uppercase; letter-spacing: .07em;
    margin-bottom: 4px;
}
.dh-declined .dh-metric-label { color: #7f1d1d; }
.dh-approved .dh-metric-label { color: #14532d; }
.dh-review   .dh-metric-label { color: #78350f; }

.dh-metric-value {
    font-size: 22px; font-weight: 500;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: -.02em;
}
.dh-declined .dh-metric-value { color: #ef4444; }
.dh-approved .dh-metric-value { color: #22c55e; }
.dh-review   .dh-metric-value { color: #f59e0b; }

.thresh-label {
    display: flex; justify-content: space-between;
    font-size: 9px; font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 4px;
}
.dh-declined .thresh-label { color: #7f1d1d; }
.dh-approved .thresh-label { color: #14532d; }
.dh-review   .thresh-label { color: #78350f; }

.thresh-track { height: 6px; border-radius: 3px; }
.dh-declined .thresh-track { background: rgba(239,68,68,.12); }
.dh-approved .thresh-track { background: rgba(34,197,94,.12); }
.dh-review   .thresh-track { background: rgba(245,158,11,.12); }

.thresh-fill { height: 100%; border-radius: 3px; }
.dh-declined .thresh-fill { background: #ef4444; }
.dh-approved .thresh-fill { background: #22c55e; }
.dh-review   .thresh-fill { background: #f59e0b; }

.thresh-zones {
    display: flex; justify-content: space-between;
    font-size: 9px; font-family: 'IBM Plex Mono', monospace;
    color: #555c73; margin-top: 4px;
}

.explain-wrap { margin-top: .875rem; }
.explain-eyebrow {
    font-size: 9px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .08em; margin-bottom: .5rem;
}
.dh-declined .explain-eyebrow { color: #7f1d1d; }
.dh-approved .explain-eyebrow { color: #14532d; }
.dh-review   .explain-eyebrow { color: #78350f; }

.explain-body {
    font-size: 12.5px; line-height: 1.65; color: #8b90a8;
    border-left: 2px solid; padding-left: .875rem;
}
.dh-declined .explain-body { border-color: #7f1d1d; }
.dh-approved .explain-body { border-color: #14532d; }
.dh-review   .explain-body { border-color: #78350f; }

/* ─── Support card ─── */
.support-card {
    background: #0f1219; border: 1px solid #1e2535;
    border-radius: 10px; padding: 1rem 1.375rem;
    margin-bottom: .875rem;
}
.sc-title {
    font-size: 10px; font-weight: 500; text-transform: uppercase;
    letter-spacing: .07em; color: #555c73; margin-bottom: .75rem;
}
.sc-subtitle {
    font-size: 10px; color: #555c73; margin-top: -.5rem;
    margin-bottom: .625rem;
}
.sc-metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; margin-bottom: .875rem; }
.sc-metric { background: #141820; border-radius: 6px; padding: .625rem .875rem; }
.sc-label { font-size: 9px; text-transform: uppercase; letter-spacing: .06em; color: #555c73; }
.sc-value { font-size: 15px; font-weight: 500; font-family: 'IBM Plex Mono', monospace; color: #e8eaf2; margin-top: 2px; }

.feat-row { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }
.feat-name {
    font-size: 10px; font-family: 'IBM Plex Mono', monospace;
    color: #555c73; width: 130px; flex-shrink: 0;
    text-align: right; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
}
.feat-bg { flex: 1; height: 4px; background: #1e2535; border-radius: 2px; overflow: hidden; }
.feat-fill { height: 100%; border-radius: 2px; }
.feat-val {
    font-size: 10px; font-family: 'IBM Plex Mono', monospace;
    color: #555c73; width: 36px; text-align: right; flex-shrink: 0;
}

@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
    .steps { flex-wrap: wrap; gap: 6px; }
    .step-line { display: none; }
}
</style>
""", unsafe_allow_html=True)

# ─── Topbar ────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <div class="brand">
        <div class="brand-mark">IQ</div>
        LoanIQ
        <span style="color:#1e2535;margin:0 2px">/</span>
        <span style="font-size:12px;color:#555c73;font-weight:400">Underwriting</span>
    </div>
    <div class="perf-badge">ROC-AUC <b>{metadata['roc_auc']}</b> &nbsp;·&nbsp; {metadata['n_train']:,} training samples</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Single applicant", "Batch upload", "Model"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE APPLICANT
# ══════════════════════════════════════════════════════════════════
with tab1:

    if "step" not in st.session_state:
        st.session_state.step = 1
    if "result" not in st.session_state:
        st.session_state.result = None

    step = st.session_state.step

    def step_class(n):
        if n < step: return "s-done"
        if n == step: return "s-active"
        return ""

    def step_num_display(n):
        return "✓" if n < step else str(n)

    st.markdown(f"""
    <div class="workflow-header">
        <div class="wh-title">Single Applicant Assessment</div>
        <div class="wh-sub">Enter applicant data, run the model, and review the AI-generated lending rationale.</div>
        <div class="steps">
            <div class="step {step_class(1)}">
                <div class="step-num">{step_num_display(1)}</div>
                <div class="step-label">Complete applicant profile</div>
            </div>
            <div class="step-line"></div>
            <div class="step {step_class(2)}">
                <div class="step-num">{step_num_display(2)}</div>
                <div class="step-label">Run assessment</div>
            </div>
            <div class="step-line"></div>
            <div class="step {step_class(3)}">
                <div class="step-num">{step_num_display(3)}</div>
                <div class="step-label">Review decision &amp; explanation</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.1, 1], gap="medium")

    # ── LEFT: Intake ───────────────────────────────────────────────
    with left:

        # ── FIX 1: Removed the orphaned standalone section-card div.
        #    Each section now has exactly ONE header, rendered inside
        #    its st.container() block via the inline style div below.

        # Financial Profile
        with st.container():
            st.markdown("""
            <div style="background:#0f1219;border:1px solid #1e2535;border-radius:10px;padding:1.125rem 1.375rem;margin-bottom:.5rem">
                <div class="section-title">Financial Profile</div>
                <div class="section-desc">Core affordability and loan structure inputs.</div>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                amt_income  = st.number_input("Annual income ($)",  10000, 1000000, 60000, 5000)
                amt_annuity = st.number_input("Annual payment ($)", 1000, 200000,   12000, 500)
            with c2:
                amt_credit  = st.number_input("Loan amount ($)",    5000, 2000000,  180000, 5000)
                amt_goods   = st.number_input("Goods price ($)",    5000, 2000000,  170000, 5000)

        # Credit Profile
        with st.container():
            st.markdown("""
            <div style="background:#0f1219;border:1px solid #1e2535;border-radius:10px;padding:1.125rem 1.375rem;margin-bottom:.5rem">
                <div class="section-title">Credit Profile</div>
                <div class="section-desc">External credit indicators and recent credit behavior.</div>
            </div>
            """, unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: ext_source_1 = st.number_input("Bureau score 1", 0.0, 1.0, 0.50, 0.01, format="%.2f")
            with c2: ext_source_2 = st.number_input("Bureau score 2", 0.0, 1.0, 0.45, 0.01, format="%.2f")
            with c3: ext_source_3 = st.number_input("Bureau score 3", 0.0, 1.0, 0.50, 0.01, format="%.2f")
            flags = []
            if ext_source_2 < 0.3: flags.append("Bureau score 2")
            if ext_source_3 < 0.3: flags.append("Bureau score 3")
            if flags:
                st.warning(f"{'  ·  '.join(flags)} below 0.30 threshold — strong default predictor")
            c1, c2 = st.columns(2)
            with c1: credit_inq    = st.number_input("Credit inquiries (12 mo)", 0, 20, 1)
            with c2: region_rating = st.selectbox("Region rating", [1, 2, 3])

        # Personal Profile
        with st.container():
            st.markdown("""
            <div style="background:#0f1219;border:1px solid #1e2535;border-radius:10px;padding:1.125rem 1.375rem;margin-bottom:.5rem">
                <div class="section-title">Personal Profile</div>
                <div class="section-desc">Basic household and demographic context.</div>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age_years    = st.number_input("Age (years)", 18, 75, 35)
                cnt_children = st.number_input("Children", 0, 10, 0)
                cnt_fam      = st.number_input("Family members", 1, 15, 2)
            with c2:
                education     = st.selectbox("Education", [
                    "Secondary / secondary special", "Higher education",
                    "Incomplete higher", "Lower secondary", "Academic degree"])
                family_status = st.selectbox("Family status", [
                    "Married", "Single / not married", "Civil marriage",
                    "Separated", "Widow"])
                housing_type  = st.selectbox("Housing type", [
                    "House / apartment", "With parents", "Municipal apartment",
                    "Rented apartment", "Office apartment", "Co-op apartment"])
            c1, c2 = st.columns(2)
            with c1: flag_own_car    = st.checkbox("Owns a car")
            with c2: flag_own_realty = st.checkbox("Owns real estate", value=True)

        # Employment
        with st.container():
            st.markdown("""
            <div style="background:#0f1219;border:1px solid #1e2535;border-radius:10px;padding:1.125rem 1.375rem;margin-bottom:.5rem">
                <div class="section-title">Employment</div>
                <div class="section-desc">Employment stability and labor profile.</div>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                income_type    = st.selectbox("Income type", [
                    "Working", "Commercial associate", "Pensioner",
                    "State servant", "Unemployed", "Student"])
                employed_years = st.number_input("Years employed", 0.0, 40.0, 5.0, 0.5)
            with c2:
                occupation = st.selectbox("Occupation", [
                    "Laborers", "Core staff", "Accountants", "Managers",
                    "Drivers", "Sales staff", "Medicine staff",
                    "High skill tech staff", "Secretaries", "Unknown"])
                org_type   = st.selectbox("Organization type", [
                    "Business Entity Type 3", "School", "Government",
                    "Medicine", "Self-employed", "Construction", "Other"])
            reg_city_work = st.checkbox("Lives in different city than works")

        # ── Derived metrics strip ──────────────────────────────────
        dti       = round(amt_credit / max(amt_income, 1), 2)
        a2i       = round(amt_annuity / max(amt_income, 1), 3)
        ltv       = round(amt_goods / max(amt_credit, 1), 3)
        loan_term = round(amt_credit / max(amt_annuity, 1), 0)

        st.markdown(f"""
        <div class="derived-bar">
            <div class="db-title">Derived risk metrics</div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem">
                <div style="background:#141820;border-radius:6px;padding:.5rem .75rem;text-align:center">
                    <div style="font-size:9px;text-transform:uppercase;letter-spacing:.06em;color:#555c73">Debt / income</div>
                    <div style="font-size:15px;font-weight:500;font-family:'IBM Plex Mono',monospace;color:#e8eaf2;margin-top:2px">{dti:.1f}x</div>
                </div>
                <div style="background:#141820;border-radius:6px;padding:.5rem .75rem;text-align:center">
                    <div style="font-size:9px;text-transform:uppercase;letter-spacing:.06em;color:#555c73">Annuity / inc</div>
                    <div style="font-size:15px;font-weight:500;font-family:'IBM Plex Mono',monospace;color:#e8eaf2;margin-top:2px">{a2i:.1%}</div>
                </div>
                <div style="background:#141820;border-radius:6px;padding:.5rem .75rem;text-align:center">
                    <div style="font-size:9px;text-transform:uppercase;letter-spacing:.06em;color:#555c73">Loan-to-value</div>
                    <div style="font-size:15px;font-weight:500;font-family:'IBM Plex Mono',monospace;color:#e8eaf2;margin-top:2px">{ltv:.1%}</div>
                </div>
                <div style="background:#141820;border-radius:6px;padding:.5rem .75rem;text-align:center">
                    <div style="font-size:9px;text-transform:uppercase;letter-spacing:.06em;color:#555c73">Implied term</div>
                    <div style="font-size:15px;font-weight:500;font-family:'IBM Plex Mono',monospace;color:#e8eaf2;margin-top:2px">{int(loan_term)} mo</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        ca, cb = st.columns([2, 1])
        with ca:
            run_full  = st.button("Run full assessment + AI explanation",
                                  type="primary", use_container_width=True)
        with cb:
            run_quick = st.button("Quick score", use_container_width=True)

    # ── RIGHT: Results panel ───────────────────────────────────────
    with right:

        is_unemployed = 1 if income_type == "Unemployed" else 0
        emp_age_r     = round((employed_years * 365) / max(age_years * 365.25, 1), 4)
        ext_sum       = round(ext_source_1 + ext_source_2 + ext_source_3, 4)
        low_ext2      = 1 if ext_source_2 < 0.3 else 0
        low_ext3      = 1 if ext_source_3 < 0.3 else 0
        many_ch       = 1 if cnt_children > 2 else 0
        high_inq      = 1 if credit_inq > 3 else 0

        applicant = {
            "AMT_INCOME_TOTAL": amt_income, "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity, "AMT_GOODS_PRICE": amt_goods,
            "debt_to_income": dti, "annuity_to_income": a2i,
            "loan_term_implied": loan_term, "ltv_ratio": ltv,
            "age_years": float(age_years), "employed_years": employed_years,
            "employment_to_age_ratio": emp_age_r, "is_unemployed": is_unemployed,
            "EXT_SOURCE_1": ext_source_1, "EXT_SOURCE_2": ext_source_2,
            "EXT_SOURCE_3": ext_source_3, "ext_score_sum": ext_sum,
            "low_ext_score_2": low_ext2, "low_ext_score_3": low_ext3,
            "CNT_CHILDREN": cnt_children, "CNT_FAM_MEMBERS": cnt_fam,
            "FLAG_OWN_CAR": int(flag_own_car), "FLAG_OWN_REALTY": int(flag_own_realty),
            "many_children": many_ch, "REGION_RATING_CLIENT": region_rating,
            "REG_CITY_NOT_WORK_CITY": int(reg_city_work), "FLAG_DOCUMENT_3": 1,
            "credit_inquiries_year": credit_inq, "high_inquiry_flag": high_inq,
            "NAME_INCOME_TYPE": income_type, "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS": family_status, "NAME_HOUSING_TYPE": housing_type,
            "OCCUPATION_TYPE": occupation, "ORGANIZATION_TYPE": org_type,
        }

        if run_full or run_quick:
            st.session_state.step = 3
            if run_full:
                with st.spinner("Running model and generating explanation..."):
                    res = full_assessment(applicant)
            else:
                res = score_applicant(applicant)
                res["explanation"] = None
            st.session_state.result = res
        elif st.session_state.result is None:
            st.session_state.step = max(1, step)

        result = st.session_state.result

        if result is None:
            # ── AWAITING STATE ──────────────────────────────────────
            st.markdown("""
            <div class="await-panel">
                <div class="await-label">Awaiting assessment</div>
                <div class="await-decision">
                    <div class="await-ghost">— — —</div>
                    <div class="await-ghost-sub">Decision will appear here after running the model</div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-bottom:.875rem">
                    <div class="ghost-metric">
                        <div style="font-size:9px;text-transform:uppercase;letter-spacing:.07em;color:#555c73;margin-bottom:8px">Risk score</div>
                        <div class="ghost-bar"></div>
                    </div>
                    <div class="ghost-metric">
                        <div style="font-size:9px;text-transform:uppercase;letter-spacing:.07em;color:#555c73;margin-bottom:8px">Default probability</div>
                        <div class="ghost-bar" style="width:65%"></div>
                    </div>
                </div>
                <div class="await-explain">
                    <div class="await-explain-title">AI decision explanation</div>
                    <div class="ghost-line" style="width:88%"></div>
                    <div class="ghost-line" style="width:72%"></div>
                    <div class="ghost-line" style="width:80%"></div>
                    <div class="ghost-line" style="width:58%"></div>
                </div>
                <div class="await-hint">
                    Complete the applicant profile and press<br>
                    <strong style="color:#8b90a8">Run full assessment</strong> to generate a decision.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # ── RESULT STATE ────────────────────────────────────────
            prob  = result["default_probability"]
            score = result["risk_score"]
            dec   = result["decision"]
            tier  = result["risk_tier"]
            expl  = result.get("explanation", "")

            css   = {"APPROVED": "approved", "DECLINED": "declined", "REVIEW": "review"}[dec]
            bar_w = f"{score / 10:.1f}%"
            bureau_avg = round(ext_sum / 3, 2)

            # ── FIX 2: Convert Claude's markdown to HTML before injecting.
            #    This prevents raw **bold** and \n from appearing in the UI.
            explanation_html = ""
            if expl:
                expl_rendered = md_to_html(expl)
                explanation_html = f"""
                <div class="explain-wrap">
                    <div class="explain-eyebrow">AI decision explanation</div>
                    <div class="explain-body">{expl_rendered}</div>
                </div>"""

            st.markdown(f"""
            <div class="decision-hero dh-{css}">
                <div class="dh-eyebrow">Credit decision</div>
                <div class="dh-label">{dec}</div>
                <div class="dh-tier">{tier}</div>
                <div class="dh-metric-grid">
                    <div class="dh-metric">
                        <div class="dh-metric-label">Risk score</div>
                        <div class="dh-metric-value">{score}<span style="font-size:13px;opacity:.5"> /1000</span></div>
                    </div>
                    <div class="dh-metric">
                        <div class="dh-metric-label">Default probability</div>
                        <div class="dh-metric-value">{prob:.1%}</div>
                    </div>
                </div>
                <div class="thresh-label"><span>0</span><span>Approved ≥ 850</span><span>1000</span></div>
                <div class="thresh-track">
                    <div class="thresh-fill" style="width:{bar_w}"></div>
                </div>
                <div class="thresh-zones"><span>High risk</span><span>Review</span><span>Approved</span></div>
                {explanation_html}
            </div>
            """, unsafe_allow_html=True)

            # Supporting metrics + feature importance
            st.markdown(f"""
            <div class="support-card">
                <div class="sc-title">Supporting risk metrics</div>
                <div class="sc-metric-grid">
                    <div class="sc-metric"><div class="sc-label">Debt-to-income</div><div class="sc-value">{dti:.2f}x</div></div>
                    <div class="sc-metric"><div class="sc-label">Annuity / income</div><div class="sc-value">{a2i:.1%}</div></div>
                    <div class="sc-metric"><div class="sc-label">Loan-to-value</div><div class="sc-value">{ltv:.1%}</div></div>
                    <div class="sc-metric"><div class="sc-label">Bureau avg</div><div class="sc-value">{bureau_avg:.2f}</div></div>
                </div>
                <div class="sc-title">Top model drivers</div>
                <div class="sc-subtitle">Based on historical training importance</div>
                <div class="feat-row"><span class="feat-name">low_ext_score_3</span><div class="feat-bg"><div class="feat-fill" style="width:100%;background:#3b82f6"></div></div><span class="feat-val">0.148</span></div>
                <div class="feat-row"><span class="feat-name">ext_score_sum</span><div class="feat-bg"><div class="feat-fill" style="width:70%;background:#3b82f6"></div></div><span class="feat-val">0.104</span></div>
                <div class="feat-row"><span class="feat-name">EXT_SOURCE_3</span><div class="feat-bg"><div class="feat-fill" style="width:45%;background:#60a5fa"></div></div><span class="feat-val">0.066</span></div>
                <div class="feat-row"><span class="feat-name">EXT_SOURCE_2</span><div class="feat-bg"><div class="feat-fill" style="width:34%;background:#60a5fa"></div></div><span class="feat-val">0.050</span></div>
                <div class="feat-row"><span class="feat-name">low_ext_score_2</span><div class="feat-bg"><div class="feat-fill" style="width:32%;background:#60a5fa"></div></div><span class="feat-val">0.047</span></div>
                <div class="feat-row"><span class="feat-name">NAME_EDUCATION</span><div class="feat-bg"><div class="feat-fill" style="width:29%;background:#93c5fd"></div></div><span class="feat-val">0.042</span></div>
                <div class="feat-row"><span class="feat-name">is_unemployed</span><div class="feat-bg"><div class="feat-fill" style="width:26%;background:#93c5fd"></div></div><span class="feat-val">0.039</span></div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Reset assessment", use_container_width=True):
                st.session_state.result = None
                st.session_state.step = 1
                st.rerun()


# ══════════════════════════════════════════════════════════════════
# TAB 2 — BATCH UPLOAD
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="background:#0f1219;border:1px solid #1e2535;border-radius:10px;padding:1.375rem;margin-bottom:.875rem">
        <div class="section-title">Batch Loan Screening</div>
    </div>
    """, unsafe_allow_html=True)
    st.info("Upload a CSV with pre-engineered features to score multiple applications at once.")
    st.markdown("**Required columns:** `debt_to_income`, `annuity_to_income`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, `age_years`, `ext_score_sum`")

    sample_df = pd.DataFrame([{
        "debt_to_income": 3.0, "annuity_to_income": 0.20,
        "EXT_SOURCE_1": 0.50, "EXT_SOURCE_2": 0.45, "EXT_SOURCE_3": 0.50,
        "ext_score_sum": 1.45, "low_ext_score_2": 0, "low_ext_score_3": 0,
        "age_years": 35, "employed_years": 5.0, "is_unemployed": 0, "ltv_ratio": 0.94,
        "AMT_INCOME_TOTAL": 60000, "AMT_CREDIT": 180000
    }] * 3)
    st.download_button("Download sample CSV template", sample_df.to_csv(index=False),
                       "loaniq_sample.csv", "text/csv")
    uploaded = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_batch):,} applications")
        if st.button("Run batch scoring", type="primary"):
            results, prog = [], st.progress(0)
            for i, row in df_batch.iterrows():
                results.append(score_applicant(row.to_dict()))
                prog.progress((i + 1) / len(df_batch))
            df_out = pd.concat([df_batch.reset_index(drop=True), pd.DataFrame(results)], axis=1)
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Total",    len(df_out))
            b2.metric("Approved", (df_out.decision == "APPROVED").sum())
            b3.metric("Review",   (df_out.decision == "REVIEW").sum())
            b4.metric("Declined", (df_out.decision == "DECLINED").sum())
            st.dataframe(df_out, use_container_width=True)
            st.download_button("Download results", df_out.to_csv(index=False), "results.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — MODEL
# ══════════════════════════════════════════════════════════════════
with tab3:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC",          metadata["roc_auc"])
    m2.metric("Training samples", f"{metadata['n_train']:,}")
    m3.metric("Test samples",     f"{metadata['n_test']:,}")
    m4.metric("Default rate",     f"{metadata['default_rate']:.1%}")
    st.markdown(f"""
    <div style="background:#0f1219;border:1px solid #1e2535;border-radius:10px;padding:1.125rem 1.375rem;margin-top:.875rem">
        <div class="section-title" style="margin-bottom:.75rem">Model details</div>
        <div style="font-size:12.5px;color:#8b90a8;line-height:1.7">
            <b style="color:#c8cbe0">Algorithm:</b> XGBoost with early stopping (242 rounds)<br>
            <b style="color:#c8cbe0">Class balancing:</b> scale_pos_weight = 11.4<br>
            <b style="color:#c8cbe0">Features:</b> {metadata['n_features']} engineered from SQL pipeline<br>
            <b style="color:#c8cbe0">Explainability:</b> Claude AI plain-English rationale per decision
        </div>
    </div>
    """, unsafe_allow_html=True)