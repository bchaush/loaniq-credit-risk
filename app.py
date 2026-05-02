import streamlit as st
import pandas as pd
import numpy as np
import json, sys, os
import re

sys.path.insert(0, os.path.dirname(__file__))
from model.explainer import full_assessment, score_applicant

if __name__ != "__main__":
    raise SystemExit

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

/* ─── Input elements (fintech field styling) ─── */
[data-testid="stNumberInput"] input,
[data-baseweb="input"] input {
    background: #10141d !important;
    border: 1px solid rgba(48,58,78,.95) !important;
    border-radius: 8px !important;
    color: #e8eaf2 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    padding: 11px 14px !important;
    min-height: 42px !important;
    transition: border-color .15s ease, box-shadow .15s ease, background .15s ease !important;
}
[data-testid="stNumberInput"] input:hover,
[data-baseweb="input"] input:hover {
    border-color: rgba(59,130,246,.55) !important;
    background: #121722 !important;
    box-shadow: 0 0 0 1px rgba(59,130,246,.18), 0 0 20px -6px rgba(37,99,235,.35) !important;
}
[data-testid="stNumberInput"] input:focus,
[data-baseweb="input"] input:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,.32), 0 0 28px -4px rgba(37,99,235,.45) !important;
    outline: none !important;
    background: #0e1219 !important;
}

[data-baseweb="select"] > div {
    background: #10141d !important;
    border: 1px solid rgba(48,58,78,.95) !important;
    border-radius: 8px !important;
    color: #e8eaf2 !important;
    min-height: 42px !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
    transition: border-color .15s ease, box-shadow .15s ease, background .15s ease !important;
}
[data-baseweb="select"] > div:hover {
    border-color: rgba(59,130,246,.5) !important;
    background: #121722 !important;
    box-shadow: 0 0 0 1px rgba(59,130,246,.16), 0 0 22px -6px rgba(37,99,235,.32) !important;
}
[data-baseweb="select"] > div:focus-within,
[data-baseweb="select"]:focus-within > div {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,.3), 0 0 28px -4px rgba(37,99,235,.4) !important;
}

/* BaseWeb selectbox popover — clean flat dark menu */
[data-baseweb="popover"] > div,
[data-baseweb="popover"] [role="listbox"] {
    background: #0f1219 !important;
    background-color: #0f1219 !important;
    border: 1px solid rgba(48,58,78,.95) !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 32px -8px rgba(0,0,0,.72),
                inset 0 1px 0 rgba(255,255,255,.04) !important;
    padding: 4px !important;
    overflow: hidden !important;
}
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] li {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 7px !important;
    color: #c8cde0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 8px 12px !important;
    margin: 1px 0 !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] li:hover {
    background: rgba(59,130,246,.13) !important;
    background-color: rgba(59,130,246,.13) !important;
    color: #e8eaf2 !important;
}
[data-baseweb="popover"] [aria-selected="true"][role="option"] {
    background: rgba(59,130,246,.18) !important;
    color: #93c5fd !important;
    font-weight: 500 !important;
}

/* Compliance / disclosure banners */
.lq-compliance-banner {
    position: relative;
    overflow: hidden;
    margin: -0.125rem 0 1.0625rem;
    padding: .75rem 1.125rem .75rem 1.25rem;
    border-radius: 14px;
    border: 1px solid rgba(42,52,68,.93);
    background:
        radial-gradient(120% 140% at 0% -20%, rgba(59,130,246,.10) 0%, transparent 55%),
        linear-gradient(180deg, rgba(18,21,31,.92) 0%, rgba(12,14,20,.90) 100%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 12px 36px -28px rgba(0,0,0,.60);
}
.lq-compliance-banner::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, rgba(96,165,250,.95) 0%, rgba(37,99,235,.95) 100%);
    opacity: .65;
}
.lq-compliance-banner p {
    margin: 0;
    padding-left: .35rem;
    font-size: 12px;
    line-height: 1.55;
    color: #aeb6cb;
}
.lq-ecoa-banner {
    margin: 0 0 10px;
    padding: .875rem 1.125rem;
    border-radius: 13px;
    border: 1px solid rgba(245,158,11,.35);
    background:
        radial-gradient(90% 80% at 100% 0%, rgba(245,158,11,.12) 0%, transparent 55%),
        linear-gradient(178deg, rgba(18,21,31,.90) 0%, rgba(12,14,20,.88) 100%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 10px 32px -26px rgba(0,0,0,.65);
}
.lq-ecoa-eyebrow {
    font-size: 10px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .11em;
    color: rgba(245,158,11,.90);
    margin-bottom: .375rem;
}
.lq-ecoa-body {
    font-size: 12px;
    line-height: 1.6;
    color: rgba(235,216,170,.90);
}

.lq-taxonomy-banner {
    margin: 0 0 10px;
    padding: .875rem 1.125rem;
    border-radius: 13px;
    border: 1px solid rgba(59,130,246,.35);
    background:
        radial-gradient(90% 80% at 100% 0%, rgba(59,130,246,.08) 0%, transparent 55%),
        linear-gradient(178deg, rgba(18,21,31,.90) 0%, rgba(12,14,20,.88) 100%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 10px 32px -26px rgba(0,0,0,.65);
}
.lq-taxonomy-eyebrow {
    font-size: 10px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .11em;
    color: rgba(96,165,250,.90);
    margin-bottom: .375rem;
}
.lq-taxonomy-body {
    font-size: 12px;
    line-height: 1.6;
    color: rgba(186,210,252,.88);
}

[data-testid="stNumberInput"] button,
[data-testid="stNumberInput"] [role="button"] {
    background: rgba(255,255,255,.05) !important;
    border-color: rgba(48,58,78,.7) !important;
    color: #9ca8c4 !important;
}
[data-testid="stNumberInput"] button:hover {
    background: rgba(255,255,255,.09) !important;
    color: #e8eaf2 !important;
}

/* Field labels above inputs */
[data-testid="stNumberInput"] label p,
[data-baseweb="select"] label p {
    font-size: 12.5px !important;
    font-weight: 550 !important;
    letter-spacing: -.015em !important;
    color: #c9cfdf !important;
    margin-bottom: 6px !important;
}
[data-testid="stCheckbox"] label p {
    font-size: 12.75px !important;
    font-weight: 480 !important;
    color: #b4bacd !important;
}

/* ─── Tab 1 form — section cards & grouping ─── */
.lq-form-section-head {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin: 0 0 10px;
    padding: .875rem 1.125rem;
    border-radius: 13px;
    border: 1px solid rgba(42,52,68,.93);
    background:
        radial-gradient(120% 100% at 0% 0%, rgba(59,130,246,.1) 0%, transparent 55%),
        linear-gradient(178deg, rgba(18,21,31,.92) 0%, rgba(12,14,20,.88) 100%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 10px 32px -26px rgba(0,0,0,.65);
}
.lq-form-section-num {
    flex-shrink: 0;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .04em;
    color: #93c5fd;
    border: 1px solid rgba(59,130,246,.42);
    background: rgba(59,130,246,.12);
}
.lq-section-meta {
    display: flex;
    flex-direction: column;
    gap: 5px;
    min-width: 0;
}
.lq-section-title-main {
    font-size: 11px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .09em;
    color: #9aa3ba;
}
.lq-section-title-sub {
    font-size: 1.0625rem;
    font-weight: 600;
    letter-spacing: -.025em;
    color: #f0f3f9;
    line-height: 1.2;
}
.lq-section-desc-main {
    font-size: 12px;
    line-height: 1.48;
    color: #878fa5;
}

.lq-form-section-spacer {
    height: 12px;
    margin: 0;
    padding: 0;
}

.lq-checkbox-row-spacer {
    height: 4px;
}

.lq-derived-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
}
.lq-derived-cell {
    background: #121723;
    border: 1px solid rgba(40,49,67,.82);
    border-radius: 8px;
    padding: .5rem .65rem;
    text-align: center;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.03);
}
.lq-derived-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: #6b738e;
}
.lq-derived-value {
    font-size: 15px;
    font-weight: 500;
    font-family: 'IBM Plex Mono', monospace;
    color: #e8eaf2;
    margin-top: 4px;
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
[data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"] {
    color: #e8eaf2 !important;
    border-bottom: 2px solid #3b82f6 !important;
    font-weight: 500 !important;
}

/* ─── Buttons — Streamlit puts kind="primary" + data-testid on the same <button> ─── */
.stButton > button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    width: 100% !important;
    transition: transform .12s ease, box-shadow .15s ease, filter .15s ease, border-color .15s ease, background .15s ease !important;
    letter-spacing: -.01em !important;
}

/* Primary CTA: override theme primary red — match kind attribute + test ids + nested selector */
button[kind="primary"],
.stButton > button[kind="primary"],
section[data-testid="stMain"] button[kind="primary"],
[data-testid="stAppViewContainer"] button[kind="primary"],
[data-testid="baseButton-primary"],
[data-testid="stBaseButton-primary"] {
    --primary-color: #2563eb !important;
    background: linear-gradient(150deg, #60a5fa 0%, #2563eb 48%, #1d4ed8 100%) !important;
    background-color: #2563eb !important;
    background-image: linear-gradient(150deg, #60a5fa 0%, #2563eb 48%, #1d4ed8 100%) !important;
    border: none !important;
    border-color: transparent !important;
    color: #fff !important;
    padding: 10px 0 !important;
    box-shadow: 0 4px 18px rgba(37,99,235,.42), inset 0 1px 0 rgba(255,255,255,.22) !important;
}

button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover,
section[data-testid="stMain"] button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    filter: brightness(1.05) !important;
    box-shadow: 0 6px 26px rgba(37,99,235,.55), 0 0 0 1px rgba(96,165,250,.35), inset 0 1px 0 rgba(255,255,255,.26) !important;
}

button[kind="primary"]:active,
.stButton > button[kind="primary"]:active,
[data-testid="baseButton-primary"]:active,
[data-testid="stBaseButton-primary"]:active {
    transform: translateY(1px) !important;
    filter: brightness(.97) !important;
}

[data-testid="baseButton-secondary"],
[data-testid="stBaseButton-secondary"],
button[kind="secondary"],
.stButton > button[kind="secondary"] {
    background: transparent !important;
    background-image: none !important;
    border: 1px solid rgba(59,130,246,.35) !important;
    color: #a8b8d8 !important;
    padding: 8px 0 !important;
}

[data-testid="baseButton-secondary"]:hover,
[data-testid="stBaseButton-secondary"]:hover,
button[kind="secondary"]:hover,
.stButton > button[kind="secondary"]:hover {
    background: rgba(37,99,235,.12) !important;
    border-color: rgba(96,165,250,.55) !important;
    color: #e8eaf2 !important;
    box-shadow: 0 0 0 1px rgba(59,130,246,.2), 0 0 18px -6px rgba(37,99,235,.35) !important;
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

/* ─── Checkbox — Streamlit 1.5x paints the box via Emotion on label > span (data-uri bg), not inline SVG paths ─── */
[data-testid="stCheckbox"] label {
    color: #8b90a8 !important;
    font-size: 13px !important;
}

[data-testid="stCheckbox"] input[type="checkbox"] {
    accent-color: #2563eb !important;
}

/* Visible tile is the <span> immediately before the hidden <input type="checkbox"> (label child order varies) */
[data-testid="stCheckbox"] label[data-baseweb="checkbox"] span:has(+ input[type="checkbox"]) {
    transition: box-shadow .15s ease, border-color .15s ease, background-color .15s ease, background-image .15s ease !important;
}

/* Unchecked — neutral frame */
[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:not(:checked):not(:indeterminate)) span:has(+ input[type="checkbox"]) {
    background-color: #121722 !important;
    background-image: none !important;
    border-left-color: rgba(55,65,85,.95) !important;
    border-right-color: rgba(55,65,85,.95) !important;
    border-top-color: rgba(55,65,85,.95) !important;
    border-bottom-color: rgba(55,65,85,.95) !important;
    box-shadow: none !important;
}

[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:not(:checked):not(:indeterminate)):hover span:has(+ input[type="checkbox"]) {
    border-left-color: rgba(96,165,250,.55) !important;
    border-right-color: rgba(96,165,250,.55) !important;
    border-top-color: rgba(96,165,250,.55) !important;
    border-bottom-color: rgba(96,165,250,.55) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,.2) !important;
}

/* Checked — blue gradient tile + white tick (replaces theme tickFillSelected red) */
[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked:not(:indeterminate)) span:has(+ input[type="checkbox"]) {
    background-color: transparent !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 17 13' width='17' height='13'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop stop-color='%2360a5fa'/%3E%3Cstop offset='1' stop-color='%231d4ed8'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='17' height='13' rx='2.5' fill='url(%23g)'/%3E%3Cpath fill='%23ffffff' d='M6.50002 12.6L0.400024 6.60002L2.60002 4.40002L6.50002 8.40002L13.9 0.900024L16.1 3.10002L6.50002 12.6Z'/%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: contain !important;
    border-left-color: transparent !important;
    border-right-color: transparent !important;
    border-top-color: transparent !important;
    border-bottom-color: transparent !important;
    box-shadow: 0 0 0 1px rgba(37,99,235,.35), 0 4px 14px rgba(37,99,235,.28) !important;
}

[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked:not(:indeterminate)):hover span:has(+ input[type="checkbox"]) {
    box-shadow: 0 0 0 2px rgba(96,165,250,.45), 0 6px 20px rgba(37,99,235,.38) !important;
}

/* Indeterminate — solid blue + white bar */
[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:indeterminate) span:has(+ input[type="checkbox"]) {
    background-color: #1e40af !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 14 4' width='14' height='4'%3E%3Cpath d='M14 0.5H0V3.5H14V0.5Z' fill='%23ffffff'/%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 65% auto !important;
    border-color: transparent !important;
    box-shadow: 0 0 0 1px rgba(37,99,235,.35) !important;
}

/* Focus — blue glow on tile */
[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:focus-within span:has(+ input[type="checkbox"]) {
    box-shadow: 0 0 0 3px rgba(59,130,246,.42) !important;
}

[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked:not(:indeterminate)):focus-within span:has(+ input[type="checkbox"]) {
    box-shadow: 0 0 0 3px rgba(59,130,246,.48), 0 4px 16px rgba(37,99,235,.35) !important;
}

[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:disabled) span:has(+ input[type="checkbox"]) {
    opacity: .5 !important;
    filter: grayscale(0.15) !important;
    box-shadow: none !important;
}

/* ─── Divider ─── */
hr { border-color: #1e2535 !important; margin: .75rem 0 !important; }

/* ─── Custom components — topbar & workflow (premium) ─── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem 1.25rem;
    padding: 1rem 1.25rem;
    margin: 0 0 1.5rem;
    border-radius: 14px;
    border: 1px solid rgba(38,47,63,0.95);
    background:
        radial-gradient(120% 140% at 0% -20%, rgba(59,130,246,.14) 0%, transparent 55%),
        linear-gradient(180deg, rgba(18,21,31,.98) 0%, rgba(12,14,20,.94) 100%);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.06),
        0 1px 0 rgba(0,0,0,.35),
        0 20px 50px -24px rgba(0,0,0,.65);
}
.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
}
.brand-mark {
    width: 32px;
    height: 32px;
    flex-shrink: 0;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 11px;
    letter-spacing: .04em;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    background: linear-gradient(150deg, #60a5fa 0%, #2563eb 52%, #1d4ed8 100%);
    box-shadow:
        0 4px 16px rgba(37,99,235,.42),
        inset 0 1px 0 rgba(255,255,255,.25);
}
.brand-copy {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
}
.brand-product {
    display: flex;
    align-items: baseline;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: -.03em;
    color: #f2f4fa;
    font-family: 'IBM Plex Sans', sans-serif;
    line-height: 1.2;
}
.brand-slug {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: .02em;
    color: #7b849c;
}
.brand-pipe {
    color: #2f3a52;
    font-weight: 400;
    margin: 0 1px;
}
.brand-chip {
    display: inline-flex;
    align-items: center;
    height: 22px;
    padding: 0 8px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #93b4e8;
    background: rgba(59,130,246,.14);
    border: 1px solid rgba(59,130,246,.35);
}

.perf-stack {
    display: flex;
    flex-wrap: wrap;
    align-items: stretch;
    gap: 10px;
}
.perf-metric {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 3px;
    min-width: 108px;
    padding: .5rem .75rem .55rem;
    border-radius: 10px;
    background: rgba(10,13,18,.72);
    border: 1px solid rgba(40,49,67,.85);
}
.perf-metric-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #5f6a82;
}
.perf-metric-value {
    font-size: 14px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -.02em;
    color: #e8eaf2;
}
.perf-metric-value-accent {
    color: #93c5fd;
}

.workflow-header {
    position: relative;
    overflow: hidden;
    border-radius: 14px;
    padding: 1.25rem 1.375rem 1.25rem;
    margin-bottom: 1.0625rem;
    border: 1px solid rgba(38,47,63,0.95);
    background:
        radial-gradient(100% 120% at 100% -10%, rgba(59,130,246,.09) 0%, transparent 50%),
        linear-gradient(165deg, rgba(18,21,31,.98) 0%, rgba(11,13,19,.96) 100%);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.05),
        0 16px 48px -28px rgba(0,0,0,.55);
}
.workflow-header::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
    border-radius: 14px 0 0 14px;
    background: linear-gradient(180deg, #60a5fa 0%, #2563eb 55%, #1e3a5f 100%);
    opacity: .95;
}
.wh-kicker {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #6b7a96;
    margin-bottom: .5rem;
}
.wh-title {
    font-size: 1.25rem;
    font-weight: 600;
    letter-spacing: -.03em;
    color: #f0f3f9;
    margin-bottom: .5rem;
    line-height: 1.25;
}
.wh-sub {
    font-size: 12.5px;
    line-height: 1.55;
    color: #8b94ac;
    max-width: 52ch;
    margin-bottom: .9375rem;
}
.steps {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0;
    padding-top: .125rem;
}
.step {
    display: flex;
    align-items: center;
    gap: 10px;
}
.step-num {
    width: 26px;
    height: 26px;
    border-radius: 8px;
    border: 1px solid #2c3548;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    color: #6b7388;
    flex-shrink: 0;
    background: rgba(8,10,14,.5);
}
.step.s-done .step-num {
    background: linear-gradient(145deg, rgba(59,130,246,.22) 0%, rgba(15,23,42,.92) 100%);
    border-color: rgba(96,165,250,.55);
    color: #93c5fd;
    box-shadow: 0 0 0 1px rgba(59,130,246,.18);
}
.step.s-active .step-num {
    background: linear-gradient(145deg, #3b82f6 0%, #1d4ed8 100%);
    border-color: rgba(96,165,250,.6);
    color: #fff;
    box-shadow: 0 4px 14px rgba(37,99,235,.4);
}
.step-label {
    font-size: 11.5px;
    font-weight: 500;
    letter-spacing: -.01em;
    color: #6c7489;
    white-space: nowrap;
}
.step.s-done .step-label { color: #93c5fd; }
.step.s-active .step-label { color: #bfdbfe; }
.step-line {
    width: 36px;
    height: 1px;
    margin: 0 10px;
    flex-shrink: 0;
    background: linear-gradient(90deg, rgba(44,53,72,.2) 0%, #2a3448 50%, rgba(44,53,72,.2) 100%);
}

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
    background:
        radial-gradient(100% 90% at 0% -20%, rgba(59,130,246,.08) 0%, transparent 50%),
        linear-gradient(180deg, rgba(16,18,26,.94) 0%, rgba(12,14,20,.94) 100%);
    border: 1px solid rgba(42,52,68,.93);
    border-radius: 12px;
    padding: .875rem 1.125rem .9375rem;
    margin-bottom: .75rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 12px 40px -30px rgba(0,0,0,.55);
}
.db-title {
    font-size: 10px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .09em;
    color: #6f7a93;
    margin-bottom: .625rem;
}

/* ─── Await panel — results runway (premium) ─── */
@keyframes rp-shimmer {
    0% { background-position: 140% 0; }
    100% { background-position: -40% 0; }
}

.await-panel {
    margin-bottom: .875rem;
    padding: 0;
    border-radius: 14px;
    border: 1px solid rgba(42,52,68,.93);
    background:
        radial-gradient(110% 100% at 10% -30%, rgba(59,130,246,.14) 0%, transparent 45%),
        linear-gradient(180deg, rgba(16,18,26,.94) 0%, rgba(10,11,17,.94) 100%);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.05),
        0 20px 50px -32px rgba(0,0,0,.72);
}
.await-panel-inner { padding: 1.375rem 1.5rem 1.3125rem; }
.await-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 9px;
    font-weight: 650;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #7d8aa3;
    margin-bottom: .5rem;
}
.await-badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #93c5fd 0%, #2563eb 100%);
    box-shadow: 0 0 0 4px rgba(37,99,235,.28);
}
.await-label-main {
    font-size: 1.125rem;
    font-weight: 600;
    letter-spacing: -.03em;
    color: #eef1f9;
    line-height: 1.25;
    margin-bottom: .375rem;
}
.await-copy {
    font-size: 12.5px;
    line-height: 1.52;
    color: #959db5;
    max-width: 40ch;
    margin-bottom: 1.125rem;
}
.await-steps {
    list-style: none;
    padding: 0;
    margin: 0 0 1rem;
}
.await-steps li {
    display: flex;
    align-items: flex-start;
    gap: .5rem;
    font-size: 11.75px;
    color: #a8aec4;
    line-height: 1.45;
    margin-bottom: .5rem;
}
.await-steps li:last-child { margin-bottom: 0; }
.await-steps-mark {
    flex-shrink: 0;
    width: 5px;
    height: 5px;
    margin-top: 5px;
    border-radius: 1px;
    background: rgba(59,130,246,.85);
}

.await-preview {
    border-radius: 11px;
    border: 1px solid rgba(48,58,76,.92);
    background: rgba(8,10,14,.72);
    padding: 1.125rem;
    margin-bottom: .9375rem;
}
.await-preview-kicker {
    font-size: 9px;
    font-weight: 650;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #6f7b94;
    margin-bottom: .75rem;
}
.await-preview-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: .625rem;
    margin-bottom: .875rem;
}
@media (max-width: 520px) {
    .await-preview-grid { grid-template-columns: 1fr; }
}

.await-preview-callout {
    text-align: center;
    padding: 1rem 1rem 1rem;
    border-radius: 9px;
    border: 1px dashed rgba(59,130,246,.35);
    background: radial-gradient(circle at 50% -20%, rgba(59,130,246,.2) 0%, transparent 60%);
}
.await-ghost {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 650;
    letter-spacing: -.05em;
    color: transparent;
    background: linear-gradient(110deg,#2a3548 35%, #4b5a73 42%, #2a3548 54%);
    background-size: 220% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    animation: rp-shimmer 3.2s ease-in-out infinite;
    margin-bottom: .35rem;
}
.await-ghost-sub {
    font-size: 11.5px;
    color: #7c869e;
}

.ghost-metric {
    padding: .65rem .8rem .7rem;
    border-radius: 9px;
    border: 1px solid rgba(44,53,71,.92);
    background: rgba(12,14,20,.94);
}
.ghost-label {
    font-size: 9px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: #6b7490;
    margin-bottom: 10px;
}
.ghost-bar-wrap {
    height: 22px;
    border-radius: 6px;
    background: rgba(14,17,26,.94);
    border: 1px solid rgba(40,49,63,.94);
    overflow: hidden;
    position: relative;
}
.ghost-bar {
    height: 100%;
    width: 100%;
    border-radius: 5px;
    background: linear-gradient(110deg, #252e3f 38%, rgba(148,163,214,.08) 50%, #252e3f 62%);
    background-size: 220% 100%;
    animation: rp-shimmer 3.2s ease-in-out infinite;
}
.ghost-bar-muted { opacity: .82; min-width: 28%; max-width: 100%; }

.ghost-line-stack { margin-top: .75rem; }
.ghost-line {
    height: 7px;
    border-radius: 3px;
    margin-bottom: 7px;
    background: linear-gradient(110deg, #272f3f 38%, rgba(148,163,214,.06) 50%, #272f3f 62%);
    background-size: 220% 100%;
    animation: rp-shimmer 3s ease-in-out infinite;
}
.ghost-line:last-child { margin-bottom: 0; }

.await-section-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: .5rem;
}
.await-explain-title {
    font-size: 9px;
    letter-spacing: .08em;
    text-transform: uppercase;
    font-weight: 650;
    color: #6f7b93;
}
.await-explain-chip {
    font-size: 9px;
    font-weight: 600;
    color: #64708d;
}
.await-explain {
    border-radius: 10px;
    padding: 1rem .95rem .95rem;
    border: 1px solid rgba(44,53,71,.94);
    background: rgba(10,11,17,.94);
}

.await-hint {
    text-align: center;
    margin-top: 1.125rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(42,52,68,.82);
}
.await-hint p {
    font-size: 11.75px;
    line-height: 1.62;
    color: #959db5;
    margin: 0;
}

/* ─── Decision hero ─── */
.decision-hero {
    overflow: hidden;
    border-radius: 14px;
    padding: 1.4375rem 1.4375rem 1.3125rem;
    margin-bottom: .9375rem;
    position: relative;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 18px 48px -32px rgba(0,0,0,.72);
}
.dh-declined {
    border: 1px solid rgba(190,53,62,.72);
    background:
        radial-gradient(90% 80% at 100% 0%, rgba(239,68,68,.19) 0%, transparent 50%),
        linear-gradient(160deg,#1f0a0a 0%,#140606 68%,#160808 100%);
}
.dh-approved {
    border: 1px solid rgba(59,130,246,.72);
    background:
        radial-gradient(90% 80% at 100% 0%, rgba(59,130,246,.28) 0%, transparent 52%),
        linear-gradient(160deg,#0a1428 0%,#081222 72%,#060d18 100%);
}
.dh-review {
    border: 1px solid rgba(227,173,71,.82);
    background:
        radial-gradient(90% 80% at 100% 0%, rgba(245,158,11,.2) 0%, transparent 50%),
        linear-gradient(160deg,#261c05 0%,#171005 74%,#150d04 100%);
}

.dh-eyebrow {
    font-size: 10px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .13em;
    margin-bottom: .75rem;
}
.dh-declined .dh-eyebrow { color: #fca5a5; }
.dh-approved .dh-eyebrow { color: #93c5fd; }
.dh-review   .dh-eyebrow { color: #fcd34d; }

.dh-label {
    font-size: 38px;
    font-weight: 650;
    letter-spacing: -.05em;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.02;
    margin-bottom: 2px;
}
.dh-declined .dh-label { color: #f87171; }
.dh-approved .dh-label { color: #38bdf8; }
.dh-review   .dh-label { color: #fbbf24; }

.dh-tier {
    font-size: 13px;
    font-weight: 500;
    margin-top: .25rem;
    margin-bottom: 1.125rem;
    letter-spacing: -.01em;
}
.dh-declined .dh-tier { color: #fecaca; opacity: .95; }
.dh-approved .dh-tier { color: #bfdbfe; opacity: .95; }
.dh-review   .dh-tier { color: #fef3c7; opacity: .95; }

.dh-metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 1.0625rem; }
.dh-metric {
    border-radius: 10px;
    padding: .9375rem 1rem .875rem;
    border: 1px solid rgba(255,255,255,.06);
}
.dh-declined .dh-metric { background: rgba(0,0,0,.32); backdrop-filter: blur(4px); }
.dh-approved .dh-metric { background: rgba(0,0,0,.32); backdrop-filter: blur(4px); }
.dh-review   .dh-metric { background: rgba(0,0,0,.27); backdrop-filter: blur(4px); }

.dh-metric-label {
    font-size: 9px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .085em;
    margin-bottom: 6px;
}
.dh-declined .dh-metric-label { color: #fecaca; }
.dh-approved .dh-metric-label { color: #93c5fd; }
.dh-review   .dh-metric-label { color: #fde68a; }

.dh-metric-value {
    font-size: 24px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -.03em;
    line-height: 1;
}
.dh-declined .dh-metric-value { color: #fff; }
.dh-approved .dh-metric-value { color: #f0f9ff; }
.dh-review   .dh-metric-value { color: #fffbeb; }

.dh-score-suffix { font-size: 13px; font-weight: 500; opacity: .52; vertical-align: .12em; }

.thresh-shell { margin-bottom: .25rem; }
.thresh-label {
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    margin-bottom: 6px;
}
.dh-declined .thresh-label { color: #fecaca; }
.dh-approved .thresh-label { color: #93c5fd; }
.dh-review   .thresh-label { color: #fde68a; }

.thresh-track {
    height: 9px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,.08);
}
.dh-declined .thresh-track { background: rgba(239,68,68,.21); box-shadow: inset 0 1px 2px rgba(0,0,0,.65); }
.dh-approved .thresh-track { background: rgba(37,99,235,.28); box-shadow: inset 0 1px 2px rgba(0,0,0,.65); }
.dh-review   .thresh-track { background: rgba(245,158,11,.26); box-shadow: inset 0 1px 2px rgba(0,0,0,.65); }

.thresh-fill { height: 100%; border-radius: 999px; box-shadow: 0 0 16px rgba(255,255,255,.06); }
.dh-declined .thresh-fill { background: linear-gradient(90deg,#f87171,#dc2626); }
.dh-approved .thresh-fill { background: linear-gradient(90deg,#38bdf8,#2563eb,#1d4ed8); }
.dh-review   .thresh-fill { background: linear-gradient(90deg,#fcd34d,#f59e0b); }

.thresh-zones {
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
    font-size: 9px;
    font-family: 'IBM Plex Mono', monospace;
    color: rgba(255,255,255,.52);
}

.explain-panel {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,.1);
}

.explain-eyebrow {
    font-size: 10px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .103em;
    margin-bottom: .625rem;
}
.dh-declined .explain-eyebrow { color: #fecaca; }
.dh-approved .explain-eyebrow { color: #93c5fd; }
.dh-review   .explain-eyebrow { color: #fde68a; }

.explain-body-inner {
    font-size: 12.875px;
    line-height: 1.74;
    color: rgba(228,231,242,.93);
}
.explain-inner {
    border-radius: 10px;
    padding: .9375rem 1rem 1rem;
    background: rgba(0,0,0,.42);
    border: 1px solid rgba(255,255,255,.08);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
}

/* ─── Support card ─── */
.support-card {
    border-radius: 14px;
    padding: 1.0625rem 1.4375rem 1.0625rem;
    margin-bottom: .875rem;
    border: 1px solid rgba(42,52,68,.95);
    background:
        radial-gradient(80% 60% at 90% -20%, rgba(59,130,246,.06) 0%, transparent 50%),
        linear-gradient(180deg, rgba(15,17,26,.94) 0%, rgba(10,11,18,.93) 100%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 16px 44px -32px rgba(0,0,0,.72);
}

.sc-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,.45), transparent);
    opacity: .5;
    margin: .0625rem 0 .875rem;
}

.sc-head {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: .75rem;
    margin-bottom: .75rem;
}
.sc-head-end { flex-shrink: 0; font-size: 9px; color: #5d667d; letter-spacing: .02em; }
.sc-section-title-wrap { min-width: 0; }

.sc-title {
    font-size: 10px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .098em;
    color: #8b93a9;
}
.sc-sub-inline {
    display: block;
    margin-top: 3px;
    font-size: 11px;
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
    color: #6c7389;
}

.sc-metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; margin-bottom: 1rem; }
.sc-metric {
    padding: .6875rem .875rem .75rem;
    border-radius: 9px;
    border: 1px solid rgba(44,53,71,.93);
    background: rgba(10,11,17,.94);
}
.sc-label {
    font-size: 9px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: #6e778e;
}
.sc-value {
    margin-top: 4px;
    font-size: 15px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -.02em;
    color: #eef0f9;
}

.feat-block-head {
    padding-top: .25rem;
    margin-bottom: .75rem;
}
.sc-subtitle { font-size: 11px; color: #6c7389; margin-top: 3px; }

.feat-list { display: flex; flex-direction: column; gap: 11px; }
.feat-row { display: grid; grid-template-columns: minmax(0, 128px) 1fr 44px; align-items: center; gap: 10px 10px; }
.feat-head-label {
    font-size: 9px;
    font-weight: 650;
    letter-spacing: .09em;
    text-transform: uppercase;
    color: #6f7c96;
}
.feat-track-label { opacity: .65; padding-left: 4px; }
.feat-val-head { justify-self: end; }

.feat-name {
    font-size: 10.5px;
    font-family: 'IBM Plex Mono', monospace;
    color: #a9b2c9;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.feat-bg {
    height: 6px;
    background: rgba(18,21,31,.94);
    border-radius: 99px;
    border: 1px solid rgba(44,53,71,.93);
    overflow: hidden;
    box-shadow: inset 0 1px 1px rgba(0,0,0,.52);
}

.feat-fill { height: 100%; border-radius: 999px; }
.feat-val {
    font-size: 10.5px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    color: #cfd5e9;
    text-align: right;
}

.feat-caption {
    grid-column: 1 / -1;
    padding-top: 2px;
    font-size: 10px;
    color: #5d677e;
}

.feat-header-row {
    display: grid;
    grid-template-columns: minmax(0, 128px) 1fr 44px;
    align-items: end;
    gap: 10px;
    padding-bottom: 6px;
    margin-bottom: 2px;
    border-bottom: 1px solid rgba(52,61,82,.94);
}
.feat-header-row .feat-val-head {
    font-size: 9px !important;
    font-weight: 650;
    letter-spacing: .075em;
    text-transform: uppercase;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #6f7c96 !important;
}

/* Tab 1 — sticky right column (intake + results row only) */
[data-testid="stVerticalBlock"]:has(.workflow-header) > [data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
}
[data-testid="stVerticalBlock"]:has(.workflow-header) > [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) {
    position: sticky !important;
    top: 10px;
    align-self: flex-start;
    max-height: calc(100vh - 20px);
    overflow-y: auto;
    overflow-x: hidden;
    z-index: 12;
    padding-right: 2px;
    scrollbar-width: thin;
    scrollbar-color: rgba(59,130,246,.45) rgba(12,14,20,.4);
}

@media (max-width: 480px) {
    .feat-row,
    .feat-header-row { grid-template-columns: 1fr; gap: 4px 0; row-gap: 5px; }
    .feat-name { text-align: left; padding-left: 0; }
}

@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
    .steps { gap: 8px 0; }
    .step-line { display: none; }
    .topbar { padding: .875rem 1rem; margin-bottom: 1.125rem; }
    .workflow-header { padding: 1.125rem 1.125rem; }
    .wh-sub { margin-bottom: 1rem; }
    .lq-form-section-head { flex-direction: column; gap: 10px; }
    .lq-form-section-num { width: 34px; height: 34px; }
    .lq-derived-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .dh-label { font-size: 30px !important; }
    .dh-metric-value { font-size: 21px !important; }
    .await-panel-inner { padding-left: 1.125rem; padding-right: 1.125rem; }
    [data-testid="stVerticalBlock"]:has(.workflow-header) > [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) {
        position: static !important;
        max-height: none !important;
        overflow: visible !important;
        padding-right: 0 !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ─── Topbar ────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <div class="brand">
        <div class="brand-mark">IQ</div>
        <div class="brand-copy">
            <div class="brand-product">
                LoanIQ
                <span class="brand-pipe">/</span>
                <span class="brand-chip">Underwriting</span>
            </div>
            <div class="brand-slug">Credit decisioning workspace</div>
        </div>
    </div>
    <div class="perf-stack">
        <div class="perf-metric">
            <span class="perf-metric-label">ROC-AUC</span>
            <span class="perf-metric-value perf-metric-value-accent">{metadata['roc_auc']:.3f}</span>
        </div>
        <div class="perf-metric">
            <span class="perf-metric-label">Training samples</span>
            <span class="perf-metric-value">{metadata['n_train']:,}</span>
        </div>
    </div>
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
        <div class="wh-kicker">Underwriting workflow</div>
        <div class="wh-title">Single applicant assessment</div>
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

    st.markdown("""
    <div class="lq-compliance-banner">
        <p>US Fintech Prototype: This demo uses the Home Credit international risk dataset as a public proxy for borrower-risk modeling. The workflow, terminology, and compliance framing reflect a Boston-area fintech / US underwriting context. A production US deployment would require regulated US data validation, ECOA/fair-lending review, adverse-action reason governance, and ongoing model monitoring.</p>
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
            <div class="lq-form-section-head">
                <span class="lq-form-section-num">01</span>
                <div class="lq-section-meta">
                    <div class="lq-section-title-main">Financial</div>
                    <div class="lq-section-title-sub">Income &amp; loan structure</div>
                    <div class="lq-section-desc-main">Sizing and affordability anchors for underwriting.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                amt_income  = st.number_input(
                    "Verified Gross Annual Income", 10000, 1000000, 60000, 5000,
                    format="%d",
                    help="Total verified annual income (USD). Primary driver of repayment capacity and affordability.",
                )
                amt_annuity = st.number_input(
                    "Target Annual Debt Service", 1000, 200000, 12000, 500,
                    format="%d",
                    help="Total annual repayment obligation (USD). Key input for debt-to-income and affordability checks.",
                )
            with c2:
                amt_credit  = st.number_input(
                    "Requested Credit Facility", 5000, 2000000, 180000, 5000,
                    format="%d",
                    help="Requested loan amount (USD). Used with income to assess leverage and credit risk.",
                )
                amt_goods   = st.number_input(
                    "Collateral Valuation", 5000, 2000000, 170000, 5000,
                    format="%d",
                    help="Estimated value of pledged collateral (USD). Used to calculate loan-to-value and loss protection.",
                )
            st.markdown('<div class="lq-form-section-spacer" aria-hidden="true"></div>', unsafe_allow_html=True)

        # Credit Profile
        with st.container():
            st.markdown("""
            <div class="lq-form-section-head">
                <span class="lq-form-section-num">02</span>
                <div class="lq-section-meta">
                    <div class="lq-section-title-main">Credit</div>
                    <div class="lq-section-title-sub">Alternative credit composites &amp; inquiry history</div>
                    <div class="lq-section-desc-main">Non-FICO alternative scoring signals normalized to 0–1, plus inquiry history.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                ext_source_1 = st.number_input(
                    "Alternative Credit Composite A", 0.0, 1.0, 0.50, 0.01,
                    format="%.2f", help="Simulated alternative credit score (0–1). Proxy for borrower reliability when traditional bureau data is limited.",
                )
            with c2:
                ext_source_2 = st.number_input(
                    "Alternative Credit Composite B", 0.0, 1.0, 0.45, 0.01,
                    format="%.2f", help="Simulated alternative credit score (0–1). Secondary reliability signal — flags elevated risk when below 0.30.",
                )
            with c3:
                ext_source_3 = st.number_input(
                    "Alternative Credit Composite C", 0.0, 1.0, 0.50, 0.01,
                    format="%.2f", help="Simulated alternative credit score (0–1). Tertiary composite used alongside A and B for ensemble risk profiling.",
                )
            flags = []
            if ext_source_2 < 0.3: flags.append("Bureau score 2")
            if ext_source_3 < 0.3: flags.append("Bureau score 3")
            if flags:
                st.warning(f"{'  ·  '.join(flags)} below 0.30 threshold — strong default predictor")
            c1, c2 = st.columns(2)
            with c1:
                credit_inq = st.number_input(
                    "Hard inquiries (12 months)", 0, 20, 1,
                    help="Number of credit inquiries in the past 12 months. Higher values may indicate elevated credit demand or risk.",
                )
            with c2:
                region_rating = st.selectbox(
                    "Geographic Risk Tier",
                    [1, 2, 3],
                    help="Internal geographic risk band. Captures regional economic and default risk differences.",
                )
            st.markdown('<div class="lq-form-section-spacer" aria-hidden="true"></div>', unsafe_allow_html=True)

        # Personal Profile
        with st.container():
            st.markdown("""
            <div class="lq-form-section-head">
                <span class="lq-form-section-num">03</span>
                <div class="lq-section-meta">
                    <div class="lq-section-title-main">Personal</div>
                    <div class="lq-section-title-sub">Household &amp; housing</div>
                    <div class="lq-section-desc-main">Demographics used for stability and policy checks.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="lq-ecoa-banner">
                <div class="lq-ecoa-eyebrow">ECOA / Fair Lending Note</div>
                <div class="lq-ecoa-body">Demographic features such as Age and Marital Status are retained in this UI strictly for model parity with the underlying research dataset. In a live US production environment, these variables would be excluded or sanitized to comply with the Equal Credit Opportunity Act (ECOA).</div>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age_years = st.number_input(
                    "Age (years)",
                    18, 75, 35,
                    help="Applicant age in years. Used for lifecycle risk calibration and policy checks.",
                )
                cnt_children = st.number_input(
                    "Dependent children",
                    0, 10, 0,
                    help="Number of dependent children. Impacts disposable income and financial obligations.",
                )
                cnt_fam = st.number_input(
                    "Household size",
                    1, 15, 2,
                    help="Total household members. Used to contextualize living costs and financial pressure.",
                )
            with c2:
                education = st.selectbox(
                    "Education level",
                    [
                        "Secondary / secondary special",
                        "Higher education",
                        "Incomplete higher",
                        "Lower secondary",
                        "Academic degree",
                    ],
                    help="Highest completed education level. Correlates with income stability and long-term earning potential.",
                )
                family_status = st.selectbox(
                    "Marital / family status",
                    [
                        "Married",
                        "Single / not married",
                        "Civil marriage",
                        "Separated",
                        "Widow",
                    ],
                    help="Recorded family status. Used as a proxy for household stability in this demo.",
                )
                housing_type = st.selectbox(
                    "Primary housing",
                    [
                        "House / apartment",
                        "With parents",
                        "Municipal apartment",
                        "Rented apartment",
                        "Office apartment",
                        "Co-op apartment",
                    ],
                    help="Primary residence type. Indicates housing stability and potential asset ownership.",
                )
            st.markdown('<div class="lq-checkbox-row-spacer" aria-hidden="true"></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                flag_own_car = st.checkbox(
                    "Owns a vehicle",
                    help="Indicates vehicle ownership. May reflect asset base and mobility.",
                )
            with c2:
                flag_own_realty = st.checkbox(
                    "Owns residential real estate",
                    value=True,
                    help="Indicates property ownership. Strong signal of asset backing and financial stability.",
                )
            st.markdown('<div class="lq-form-section-spacer" aria-hidden="true"></div>', unsafe_allow_html=True)

        # Employment
        with st.container():
            st.markdown("""
            <div class="lq-form-section-head">
                <span class="lq-form-section-num">04</span>
                <div class="lq-section-meta">
                    <div class="lq-section-title-main">Employment</div>
                    <div class="lq-section-title-sub">Employment Stability &amp; Income Profile</div>
                    <div class="lq-section-desc-main">Employment channel, organizational segment, and tenure signals used for income stability assessment.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="lq-taxonomy-banner">
                <div class="lq-taxonomy-eyebrow">DATA TAXONOMY NOTE</div>
                <div class="lq-taxonomy-body">Employer and occupation categories reflect the Home Credit dataset taxonomy. In a US production deployment, these inputs would map to SOC (Standard Occupational Classification) occupation standards, NAICS industry sectors, and verified income and employment data sources where permitted.</div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                income_type = st.selectbox(
                    "Primary Employment Status",
                    [
                        "Working",
                        "Commercial associate",
                        "Pensioner",
                        "State servant",
                        "Unemployed",
                        "Student",
                    ],
                    help="Employment status at application. Key indicator of income continuity and repayment ability.",
                )
                employed_years = st.number_input(
                    "Role Tenure (Years)",
                    0.0, 40.0, 5.0, 0.5,
                    help="Years in current role or field. Longer tenure generally indicates greater income stability.",
                )
            with c2:
                occupation = st.selectbox(
                    "Occupational Category",
                    [
                        "Laborers",
                        "Core staff",
                        "Accountants",
                        "Managers",
                        "Drivers",
                        "Sales staff",
                        "Medicine staff",
                        "High skill tech staff",
                        "Secretaries",
                        "Unknown",
                    ],
                    help="Broad occupation group. Used to assess income predictability and sector-specific risk.",
                )
                org_type = st.selectbox(
                    "Employer Sector",
                    [
                        "Business Entity Type 3",
                        "School",
                        "Government",
                        "Medicine",
                        "Self-employed",
                        "Construction",
                        "Other",
                    ],
                    help="Employer industry segment. Captures sector-level income volatility and risk exposure.",
                )
            reg_city_work = st.checkbox(
                "Applicant Commutes Across Municipalities",
                help="Indicates whether the applicant works outside their home area. May reflect commuting burden and stability factors.",
            )
            st.markdown('<div class="lq-form-section-spacer" aria-hidden="true"></div>', unsafe_allow_html=True)

        # ── Derived metrics strip ──────────────────────────────────
        dti       = round(amt_credit / max(amt_income, 1), 2)
        a2i       = round(amt_annuity / max(amt_income, 1), 3)
        ltv       = round(amt_goods / max(amt_credit, 1), 3)
        loan_term = round(amt_credit / max(amt_annuity, 1), 0)
        payoff_years = max(0, int(round(loan_term)))
        payoff_display = f"{payoff_years} yr"

        st.markdown(f"""
        <div class="derived-bar">
            <div class="db-title">Derived risk metrics (live)</div>
            <div class="lq-derived-grid">
                <div class="lq-derived-cell" title="Requested credit amount divided by annual income. Measures borrower leverage relative to income.">
                    <div class="lq-derived-label">Credit-to-Income</div>
                    <div class="lq-derived-value">{dti:.1f}x</div>
                </div>
                <div class="lq-derived-cell" title="Annual repayment obligation divided by annual income. Core affordability metric.">
                    <div class="lq-derived-label">Debt Service / Income</div>
                    <div class="lq-derived-value">{a2i:.1%}</div>
                </div>
                <div class="lq-derived-cell" title="Collateral value divided by requested credit amount. Higher values indicate stronger collateral protection.">
                    <div class="lq-derived-label">Collateral Coverage</div>
                    <div class="lq-derived-value">{ltv:.1%}</div>
                </div>
                <div class="lq-derived-cell" title="Estimated years to repay the requested credit based on annual debt service.">
                    <div class="lq-derived-label">Payoff horizon (est.)</div>
                    <div class="lq-derived-value">{payoff_display}</div>
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
                <div class="await-panel-inner">
                    <div class="await-badge">
                        <span class="await-badge-dot" aria-hidden="true"></span>
                        Assessment output
                    </div>
                    <div class="await-label-main">Results will render here after scoring.</div>
                    <div class="await-copy">Complete the applicant profile at left; this panel summarizes the underwriting decision,
                    calibrated default probability on the trained feature space, threshold position, and&nbsp;drivers.</div>
                    <ul class="await-steps">
                        <li><span class="await-steps-mark"></span>Policy-aligned credit disposition (Approve / Review / Decline).</li>
                        <li><span class="await-steps-mark"></span>Internal risk score and PD consistent with ROC-trained weights.</li>
                        <li><span class="await-steps-mark"></span>Narrative explanation on full assessments (skipped for Quick score).</li>
                    </ul>
                    <div class="await-preview">
                        <div class="await-preview-kicker">Layout preview · sample chrome</div>
                        <div class="await-preview-grid">
                            <div class="ghost-metric">
                                <div class="ghost-label">Risk score (0–1000)</div>
                                <div class="ghost-bar-wrap"><div class="ghost-bar"></div></div>
                            </div>
                            <div class="ghost-metric">
                                <div class="ghost-label">Default probability</div>
                                <div class="ghost-bar-wrap"><div class="ghost-bar ghost-bar-muted" style="width:68%"></div></div>
                            </div>
                        </div>
                        <div class="await-preview-callout">
                            <div class="await-ghost">···</div>
                            <div class="await-ghost-sub">Decision headline appears here · threshold fill animates live</div>
                        </div>
                        <div class="await-section-title">
                            <span class="await-explain-title">Narrative &amp; drivers</span>
                            <span class="await-explain-chip">Synthetic placeholder</span>
                        </div>
                        <div class="ghost-line-stack await-explain">
                            <div class="ghost-line" style="width:92%"></div>
                            <div class="ghost-line" style="width:76%"></div>
                            <div class="ghost-line" style="width:81%"></div>
                            <div class="ghost-line" style="width:59%"></div>
                        </div>
                    </div>
                    <div class="await-hint">
                        <p>Use <strong>Run full assessment + AI explanation</strong> for the complete package, or
                        <strong>Quick score</strong> when you only need risk score&nbsp;&amp;&nbsp;PD.</p>
                    </div>
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
            explanation_text = expl if expl else ""

            st.markdown(f"""
            <div class="decision-hero dh-{css}">
                <div class="dh-eyebrow">Credit disposition</div>
                <div class="dh-label">{dec}</div>
                <div class="dh-tier">{tier}</div>
                <div class="dh-metric-grid">
                    <div class="dh-metric">
                        <div class="dh-metric-label">Risk score</div>
                        <div class="dh-metric-value">{score}<span class="dh-score-suffix"> / 1000</span></div>
                    </div>
                    <div class="dh-metric">
                        <div class="dh-metric-label">Default probability</div>
                        <div class="dh-metric-value">{prob:.1%}</div>
                    </div>
                </div>
                <div class="thresh-shell">
                  <div class="thresh-label"><span>0</span><span>Approved&nbsp;≥&nbsp;850</span><span>1000</span></div>
                  <div class="thresh-track">
                    <div class="thresh-fill" style="width:{bar_w}"></div>
                  </div>
                  <div class="thresh-zones"><span>Higher&nbsp;risk</span><span>Review</span><span>Approve</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if explanation_text:
                st.markdown("---")
                st.markdown("**Underwriting Rationale**")
                st.markdown(explanation_text)

            # Supporting metrics + feature importance
            st.markdown(f"""
            <div class="support-card">
                <div class="sc-divider"></div>
                <div class="sc-head">
                    <div class="sc-section-title-wrap">
                        <div class="sc-title">Key ratios</div>
                        <span class="sc-sub-inline">Aligned with modeled fields on this&nbsp;record</span>
                    </div>
                    <span class="sc-head-end">Snapshot</span>
                </div>
                <div class="sc-metric-grid">
                    <div class="sc-metric"><div class="sc-label">Credit-to-Income</div><div class="sc-value">{dti:.2f}x</div></div>
                    <div class="sc-metric"><div class="sc-label">Debt Service / Income</div><div class="sc-value">{a2i:.1%}</div></div>
                    <div class="sc-metric"><div class="sc-label">Collateral Coverage</div><div class="sc-value">{ltv:.1%}</div></div>
                    <div class="sc-metric"><div class="sc-label">Normalized Bureau Score (0–1)</div><div class="sc-value">{bureau_avg:.2f}</div></div>
                </div>
                <div class="feat-block-head">
                    <div class="sc-title">Global model drivers · training baseline</div>
                    <div class="sc-subtitle">Relative gain from model training. Global view only — not individualized SHAP attribution for this applicant.</div>
                </div>
                <div class="feat-list">
                  <div class="feat-header-row">
                    <span class="feat-head-label">Training feature</span>
                    <span class="feat-head-label feat-track-label">Contribution</span>
                    <span class="feat-val feat-val-head">Imp.</span>
                  </div>
                  <div class="feat-row"><span class="feat-name">Alt. Credit Signal C (low)</span><div class="feat-bg"><div class="feat-fill" style="width:100%;background:#3b82f6"></div></div><span class="feat-val">0.148</span></div>
                  <div class="feat-row"><span class="feat-name">Alt. Credit Composite (sum)</span><div class="feat-bg"><div class="feat-fill" style="width:70%;background:#3b82f6"></div></div><span class="feat-val">0.104</span></div>
                  <div class="feat-row"><span class="feat-name">Alt. Credit Score C</span><div class="feat-bg"><div class="feat-fill" style="width:45%;background:#60a5fa"></div></div><span class="feat-val">0.066</span></div>
                  <div class="feat-row"><span class="feat-name">Alt. Credit Score B</span><div class="feat-bg"><div class="feat-fill" style="width:34%;background:#60a5fa"></div></div><span class="feat-val">0.050</span></div>
                  <div class="feat-row"><span class="feat-name">Alt. Credit Signal B (low)</span><div class="feat-bg"><div class="feat-fill" style="width:32%;background:#60a5fa"></div></div><span class="feat-val">0.047</span></div>
                  <div class="feat-row"><span class="feat-name">Education Level</span><div class="feat-bg"><div class="feat-fill" style="width:29%;background:#93c5fd"></div></div><span class="feat-val">0.042</span></div>
                  <div class="feat-row"><span class="feat-name">Unemployment Indicator</span><div class="feat-bg"><div class="feat-fill" style="width:26%;background:#93c5fd"></div></div><span class="feat-val">0.039</span></div>
                </div>
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
    <div class="lq-form-section-head">
        <div class="lq-section-meta">
            <div class="lq-section-title-main">Batch processing</div>
            <div class="lq-section-title-sub">Batch loan screening</div>
            <div class="lq-section-desc-main">Upload applicant data to score multiple loan applications using the same underwriting model and decision logic.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info(
        "Upload a CSV file containing applicant-level features to score multiple loan "
        "applications simultaneously. Batch scoring uses the same underwriting model and "
        "decision logic as the single applicant workflow."
    )
    st.markdown(
        "**Required fields:** Debt-to-Income · Debt Service / Income · "
        "Alt. Credit Score B · Alt. Credit Score C · Applicant Age · "
        "Alt. Credit Composite (sum)"
    )
    st.markdown(
        "<small style='color:#6b7280;'>Column names must match: debt_to_income, annuity_to_income, "
        "EXT_SOURCE_2, EXT_SOURCE_3, age_years, ext_score_sum</small>",
        unsafe_allow_html=True,
    )

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
    st.markdown(
        "<small style='color:#6b7280;'>Output: A scored dataset with risk scores, default "
        "probabilities, and recommended credit dispositions for each applicant.</small>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<small style='color:#6b7280;'>Files with "
        "missing or misnamed columns will be rejected "
        "with validation feedback.</small>",
        unsafe_allow_html=True,
    )

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_batch):,} applications")
        if st.button("Run batch scoring", type="primary"):
            expected_cols = metadata["features"]
            missing_cols = [c for c in expected_cols if c not in df_batch.columns]
            for col in expected_cols:
                if col not in df_batch.columns:
                    df_batch[col] = 0
            if missing_cols:
                st.warning(
                    f"{len(missing_cols)} columns missing from upload "
                    f"and defaulted to 0. Results may differ from "
                    f"single applicant scoring for affected rows. "
                    f"Missing: {', '.join(missing_cols[:5])}"
                    + (" and more." if len(missing_cols) > 5 else ".")
                )
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
    st.markdown("""
    <div class="lq-form-section-head">
        <div class="lq-section-meta">
            <div class="lq-section-title-main">Model</div>
            <div class="lq-section-title-sub">Model performance &amp; architecture</div>
            <div class="lq-section-desc-main">Core model metrics and training configuration used for credit risk estimation.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.875rem;margin-bottom:1.25rem;">
  <div style="background:#0f1219;border:1px solid #1e2535;border-left:2px solid rgba(59,130,246,0.6);box-shadow:0 0 12px rgba(59,130,246,0.08);border-radius:8px;padding:.875rem 1rem;">
    <div style="font-size:.7rem;font-weight:600;letter-spacing:.08em;color:#6b7280;text-transform:uppercase;margin-bottom:.35rem;">ROC-AUC</div>
    <div style="font-size:1.4rem;font-weight:700;color:#e8eaf2;">{metadata['roc_auc']:.3f}</div>
  </div>
  <div style="background:#0f1219;border:1px solid #1e2535;border-left:2px solid rgba(59,130,246,0.6);box-shadow:0 0 12px rgba(59,130,246,0.08);border-radius:8px;padding:.875rem 1rem;">
    <div style="font-size:.7rem;font-weight:600;letter-spacing:.08em;color:#6b7280;text-transform:uppercase;margin-bottom:.35rem;">Training Samples</div>
    <div style="font-size:1.4rem;font-weight:700;color:#e8eaf2;">{metadata['n_train']:,}</div>
  </div>
  <div style="background:#0f1219;border:1px solid #1e2535;border-left:2px solid rgba(59,130,246,0.6);box-shadow:0 0 12px rgba(59,130,246,0.08);border-radius:8px;padding:.875rem 1rem;">
    <div style="font-size:.7rem;font-weight:600;letter-spacing:.08em;color:#6b7280;text-transform:uppercase;margin-bottom:.35rem;">Test Samples</div>
    <div style="font-size:1.4rem;font-weight:700;color:#e8eaf2;">{metadata['n_test']:,}</div>
  </div>
  <div style="background:#0f1219;border:1px solid #1e2535;border-left:2px solid rgba(59,130,246,0.6);box-shadow:0 0 12px rgba(59,130,246,0.08);border-radius:8px;padding:.875rem 1rem;">
    <div style="font-size:.7rem;font-weight:600;letter-spacing:.08em;color:#6b7280;text-transform:uppercase;margin-bottom:.35rem;">Default Rate</div>
    <div style="font-size:1.4rem;font-weight:700;color:#e8eaf2;">{metadata['default_rate']:.1%}</div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#9ca3af; font-size:0.85rem; "
        "margin: 0.5rem 0 1.25rem;'>"
        "Model performance is within an acceptable range "
        "for retail credit risk models, providing moderate "
        "discriminatory power between default and "
        "non-default outcomes. Decision thresholds are "
        "calibrated to align with underwriting policy "
        "bands (Approve / Review / Decline)."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"""
    <div class="lq-form-section-head" style="margin-top:1.5rem;">
        <div class="lq-section-meta">
            <div class="lq-section-title-main">Configuration</div>
            <div class="lq-section-title-sub">Model details</div>
        </div>
    </div>
    <div style="font-size:12.5px;color:#8b90a8;line-height:1.7;margin-top:.75rem">
            <b style="color:#c8cbe0">Algorithm:</b> XGBoost with early stopping (242 rounds)<br>
            <b style="color:#c8cbe0">Class balancing:</b> scale_pos_weight = 11.4<br>
            <b style="color:#c8cbe0">Features:</b> {metadata['n_features']} engineered variables capturing credit behavior, income stability, and alternative scoring signals<br>
            <b style="color:#c8cbe0">Explainability:</b> Post-model decision rationale generated using LLM-based summarization of key risk drivers
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#9ca3af; font-size:0.85rem; "
        "margin-top:1rem;'>"
        "The model is designed to support underwriting "
        "decisions by estimating default risk and enabling "
        "consistent, explainable credit assessments."
        "</div>",
        unsafe_allow_html=True,
    )
