import hashlib
import re
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from doc_processing import analyze_invoices, clear_cache, process_invoices
from model_gateway import invoke_llm

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Expense Tracker",
    page_icon="🏛️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F1F5F9;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.hero-banner {
    background: linear-gradient(135deg, #0F172A 0%, #1D4ED8 100%);
    border-radius: 16px;
    padding: 48px 40px 40px 40px;
    margin-bottom: 32px;
    color: #FFFFFF;
}
.hero-banner h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    color: #FFFFFF;
}
.hero-banner p {
    font-size: 1.05rem;
    color: #CBD5E1;
    margin: 0 0 16px 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    color: #E2E8F0;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 500;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-card .metric-label {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.metric-card .metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #0F172A;
}

.section-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 28px 24px;
    margin-bottom: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* Summary card */
.summary-card {
    background: #FFFFFF;
    border: 1px solid #C7D7FE;
    border-left: 5px solid #3B82F6;
    border-radius: 12px;
    padding: 24px 28px;
    margin-top: 16px;
    box-shadow: 0 2px 8px rgba(59,130,246,0.07);
}
.summary-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.summary-header .icon {
    font-size: 1.4rem;
}
.summary-header .title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1D4ED8;
}
.summary-stat-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}
.summary-stat {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #1E40AF;
    font-weight: 600;
}
.summary-body p {
    font-size: 0.95rem;
    color: #374151;
    line-height: 1.75;
    margin: 0 0 10px 0;
}
.summary-body p:last-child {
    margin-bottom: 0;
}

/* Advice card */
.advice-card {
    background: #FFFFFF;
    border: 1px solid #D1FAE5;
    border-left: 5px solid #10B981;
    border-radius: 12px;
    padding: 24px 28px;
    margin-top: 16px;
    box-shadow: 0 2px 8px rgba(16,185,129,0.07);
}
.advice-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.advice-header .icon {
    font-size: 1.4rem;
}
.advice-header .title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #065F46;
}
.advice-audience {
    display: inline-block;
    background: #D1FAE5;
    border: 1px solid #6EE7B7;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.76rem;
    color: #065F46;
    font-weight: 600;
    margin-bottom: 14px;
}
.advice-body p {
    font-size: 0.95rem;
    color: #374151;
    line-height: 1.75;
    margin: 0 0 12px 0;
}
.advice-body p:last-child {
    margin-bottom: 0;
}

/* Tracking snapshot card */
.snapshot-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.88rem;
}
.snapshot-period {
    font-weight: 600;
    color: #0F172A;
}
.snapshot-amount {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1D4ED8;
}
.snapshot-meta {
    color: #64748B;
    font-size: 0.8rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "summary_stats" not in st.session_state:
    st.session_state.summary_stats = None
if "advice" not in st.session_state:
    st.session_state.advice = None
if "advice_audience" not in st.session_state:
    st.session_state.advice_audience = None
if "beginner_mode" not in st.session_state:
    st.session_state.beginner_mode = False
if "processed_hashes" not in st.session_state:
    st.session_state.processed_hashes = set()
if "budget_total" not in st.session_state:
    st.session_state.budget_total = 5000
if "budget_office_supplies" not in st.session_state:
    st.session_state.budget_office_supplies = 2000
if "budget_equipment" not in st.session_state:
    st.session_state.budget_equipment = 1500
if "budget_services" not in st.session_state:
    st.session_state.budget_services = 800
if "budget_utilities" not in st.session_state:
    st.session_state.budget_utilities = 700
# Tracking: list of snapshot dicts
if "tracking_snapshots" not in st.session_state:
    st.session_state.tracking_snapshots = []


# ---------------------------------------------------------------------------
# Helper: _build_stats
# ---------------------------------------------------------------------------
def _build_stats(df: pd.DataFrame) -> dict:
    total_amount = df["Amount"].sum()
    num_items = len(df)
    cat_breakdown = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    vendor_totals = df.groupby("Vendor")["Amount"].sum().sort_values(ascending=False)
    top_vendor = vendor_totals.index[0] if len(vendor_totals) else "N/A"
    top_vendor_amount = float(vendor_totals.iloc[0]) if len(vendor_totals) else 0.0
    top_cat = cat_breakdown.index[0] if len(cat_breakdown) else "N/A"
    date_range_str = "unknown date range"
    avg_daily = None
    try:
        dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
        if len(dates) >= 2:
            min_d = dates.min()
            max_d = dates.max()
            days = max((max_d - min_d).days, 1)
            avg_daily = total_amount / days
            date_range_str = f"{min_d.strftime('%b %d, %Y')} — {max_d.strftime('%b %d, %Y')}"
        elif len(dates) == 1:
            date_range_str = dates.iloc[0].strftime("%b %d, %Y")
    except Exception:
        pass
    return {
        "total_amount": total_amount,
        "num_items": num_items,
        "top_cat": top_cat,
        "top_vendor": top_vendor,
        "top_vendor_amount": top_vendor_amount,
        "date_range_str": date_range_str,
        "avg_daily": avg_daily,
        "cat_breakdown": cat_breakdown,
    }


# ---------------------------------------------------------------------------
# Helper: generate_summary
# ---------------------------------------------------------------------------
def generate_summary(df: pd.DataFrame) -> tuple:
    stats = _build_stats(df)
    total_amount = stats["total_amount"]
    num_items = stats["num_items"]
    date_range_str = stats["date_range_str"]
    avg_daily = stats["avg_daily"]
    cat_breakdown = stats["cat_breakdown"]
    top_vendor = stats["top_vendor"]
    top_vendor_amount = stats["top_vendor_amount"]

    cat_lines = ", ".join(
        f"{cat}: ${val:,.2f}" for cat, val in cat_breakdown.items()
    )
    dtype_breakdown = df.groupby("Doc Type")["Amount"].sum()
    dtype_lines = ", ".join(
        f"{dt}: ${val:,.2f}" for dt, val in dtype_breakdown.items()
    )
    avg_daily_str = f"Average daily spend: ${avg_daily:,.2f}." if avg_daily else ""

    prompt = (
        "You are a professional expense analyst.\n\n"
        "Write exactly 3 short sentences. Cover only:\n"
        "1. Total spend and date range.\n"
        "2. Largest spending category and top vendor.\n"
        "3. One specific actionable recommendation to reduce costs.\n\n"
        "Do not restate every number. Do not use markdown. Do not use bullet points. "
        "Do not use headers. Do not use bold text. "
        "Do not add preamble, commentary, self-evaluation, or revision notes. "
        "Return plain text only.\n\n"
        f"Data:\n"
        f"- Total amount: ${total_amount:,.2f}\n"
        f"- Line items: {num_items}\n"
        f"- Date range: {date_range_str}\n"
        f"- {avg_daily_str}\n"
        f"- Category breakdown: {cat_lines}\n"
        f"- Top vendor: {top_vendor} (${top_vendor_amount:,.2f})\n"
        f"- Document type breakdown: {dtype_lines}\n"
    )

    raw = invoke_llm(prompt, max_new_tokens=300)
    cleaned = re.sub(r"\*\*|\*|##|^- ", "", raw, flags=re.MULTILINE).strip()

    # split into sentences for card display
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    return cleaned, stats, sentences


# ---------------------------------------------------------------------------
# Helper: render_summary_card
# ---------------------------------------------------------------------------
def render_summary_card(summary_text: str, stats: dict, sentences: list):
    stat_chips = ""
    stat_chips += f'<span class="summary-stat">📅 {stats["date_range_str"]}</span>'
    stat_chips += f'<span class="summary-stat">💵 ${stats["total_amount"]:,.2f} total</span>'
    stat_chips += f'<span class="summary-stat">🧾 {stats["num_items"]} items</span>'
    stat_chips += f'<span class="summary-stat">🏷️ Top: {stats["top_cat"]}</span>'
    stat_chips += f'<span class="summary-stat">🏢 {stats["top_vendor"]}</span>'
    if stats["avg_daily"]:
        stat_chips += f'<span class="summary-stat">📈 ${stats["avg_daily"]:,.2f}/day</span>'

    paragraphs_html = "".join(f"<p>{s}</p>" for s in sentences) if sentences else f"<p>{summary_text}</p>"

    st.markdown(
        f"""
<div class="summary-card">
    <div class="summary-header">
        <span class="icon">📊</span>
        <span class="title">AI Expense Summary</span>
    </div>
    <div class="summary-stat-row">{stat_chips}</div>
    <div class="summary-body">{paragraphs_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helper: generate_advice
# ---------------------------------------------------------------------------
_AUDIENCE_CONFIGS = {
    "Business Executive": {
        "tone": (
            "You are a CFO-level financial strategist briefing a C-suite executive. "
            "Be concise, ROI-focused, and strategic. Use business language."
        ),
        "format": (
            "Write exactly 4 short paragraphs:\n"
            "1. Executive spending assessment with key risk indicators.\n"
            "2. Top 2-3 cost reduction levers with estimated impact.\n"
            "3. Vendor portfolio and contract renegotiation opportunities.\n"
            "4. The single highest-priority action for this fiscal period.\n\n"
            "No markdown, no bullets, no bold, no headers. Plain text only."
        ),
    },
    "Department Manager": {
        "tone": (
            "You are a senior financial analyst advising a department manager "
            "who controls a team budget. Be practical and specific."
        ),
        "format": (
            "Write exactly 4 paragraphs:\n"
            "1. Overall budget health and areas of concern.\n"
            "2. Specific overspending categories and what to do about them.\n"
            "3. Vendor consolidation or process improvements to reduce spend.\n"
            "4. One concrete action to take this month.\n\n"
            "No markdown, no bullets, no bold, no headers. Plain text only."
        ),
    },
    "Casual User": {
        "tone": (
            "You are a helpful financial friend giving practical money advice. "
            "Be friendly, clear, and to the point. Use everyday language."
        ),
        "format": (
            "Write exactly 3 paragraphs:\n"
            "1. A quick summary of how the spending looks overall.\n"
            "2. The one biggest area to watch and a simple tip to spend less there.\n"
            "3. One easy thing to do this week to save money.\n\n"
            "No markdown, no bullets, no bold, no headers. Plain text only."
        ),
    },
    "Beginner": {
        "tone": (
            "You are a patient, encouraging financial coach talking to someone "
            "who has never tracked expenses before. Avoid all jargon. "
            "If you must use a financial term, explain it immediately in brackets."
        ),
        "format": (
            "Write exactly 5 short paragraphs:\n"
            "1. What these expenses are in simple everyday terms.\n"
            "2. Which area costs the most and why that matters for daily life.\n"
            "3. Whether the spending looks healthy or worrying, explained simply.\n"
            "4. Two very easy things the reader can do right now to spend less.\n"
            "5. One encouraging sentence about taking control of money.\n\n"
            "No markdown, no bullets, no bold, no headers. Plain text only. "
            "Write as if talking to a friend."
        ),
    },
    "Student / Non-profit": {
        "tone": (
            "You are a budget advisor helping a student organization or non-profit "
            "make the most of very limited funds. Be resourceful and practical."
        ),
        "format": (
            "Write exactly 4 paragraphs:\n"
            "1. How the budget is being used and any warning signs.\n"
            "2. The best opportunities to cut costs without losing quality.\n"
            "3. Free or low-cost alternatives for the highest-spend areas.\n"
            "4. One specific frugal win to pursue this week.\n\n"
            "No markdown, no bullets, no bold, no headers. Plain text only."
        ),
    },
}


def generate_advice(df: pd.DataFrame, audience: str = "Department Manager") -> str:
    stats = _build_stats(df)
    total_amount = stats["total_amount"]
    num_items = stats["num_items"]
    cat_breakdown = stats["cat_breakdown"]
    top_vendor = stats["top_vendor"]
    top_vendor_amount = stats["top_vendor_amount"]
    top_dtype = (
        df.groupby("Doc Type")["Amount"].sum().sort_values(ascending=False).index[0]
        if "Doc Type" in df.columns and not df["Doc Type"].isna().all()
        else "N/A"
    )

    cat_lines = "\n".join(
        f"  - {cat}: ${val:,.2f}" for cat, val in cat_breakdown.items()
    )

    category_budgets = {
        "Office Supplies": float(st.session_state.budget_office_supplies),
        "Equipment": float(st.session_state.budget_equipment),
        "Services": float(st.session_state.budget_services),
        "Utilities": float(st.session_state.budget_utilities),
    }
    over_budget_cats = []
    for cat, bgt in category_budgets.items():
        spent = float(df[df["Doc Type"] == cat]["Amount"].sum()) if "Doc Type" in df.columns else 0.0
        if bgt > 0 and spent > bgt:
            over_budget_cats.append(f"{cat} (spent ${spent:,.2f}, budget ${bgt:,.2f})")

    over_budget_str = (
        "Over-budget categories: " + ", ".join(over_budget_cats)
        if over_budget_cats
        else "No categories currently over budget."
    )

    cfg = _AUDIENCE_CONFIGS.get(audience, _AUDIENCE_CONFIGS["Department Manager"])
    prompt = (
        f"{cfg['tone']}\n\n"
        f"{cfg['format']}\n\n"
        f"Expense data:\n"
        f"- Total spend: ${total_amount:,.2f} across {num_items} line items\n"
        f"- Highest-spend category: {cat_breakdown.index[0] if len(cat_breakdown) else 'N/A'}\n"
        f"- Top vendor: {top_vendor} (${top_vendor_amount:,.2f})\n"
        f"- Dominant document type: {top_dtype}\n"
        f"- {over_budget_str}\n"
        f"- Category breakdown:\n{cat_lines}\n"
    )

    raw = invoke_llm(prompt, max_new_tokens=500)
    cleaned = re.sub(r"\*\*|\*|##|^- ", "", raw, flags=re.MULTILINE).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Helper: render_advice_card
# ---------------------------------------------------------------------------
def render_advice_card(advice_text: str, audience: str):
    paragraphs = [p.strip() for p in advice_text.split("\n") if p.strip()]
    paragraphs_html = "".join(f"<p>{p}</p>" for p in paragraphs)
    st.markdown(
        f"""
<div class="advice-card">
    <div class="advice-header">
        <span class="icon">💡</span>
        <span class="title">AI Expense Advice</span>
    </div>
    <span class="advice-audience">For: {audience}</span>
    <div class="advice-body">{paragraphs_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Budget Settings")

    st.session_state.budget_total = st.number_input(
        "Total Department Budget ($)",
        min_value=0,
        value=st.session_state.budget_total,
        step=100,
    )

    st.markdown("**Per-Category Budgets**")

    st.session_state.budget_office_supplies = st.number_input(
        "Office Supplies budget ($)",
        min_value=0,
        value=st.session_state.budget_office_supplies,
        step=100,
    )
    st.session_state.budget_equipment = st.number_input(
        "Equipment budget ($)",
        min_value=0,
        value=st.session_state.budget_equipment,
        step=100,
    )
    st.session_state.budget_services = st.number_input(
        "Services budget ($)",
        min_value=0,
        value=st.session_state.budget_services,
        step=100,
    )
    st.session_state.budget_utilities = st.number_input(
        "Utilities budget ($)",
        min_value=0,
        value=st.session_state.budget_utilities,
        step=100,
    )

    st.markdown("---")
    st.subheader("Advice Audience")
    audience_choice = st.selectbox(
        "Who is this advice for?",
        options=list(_AUDIENCE_CONFIGS.keys()),
        index=list(_AUDIENCE_CONFIGS.keys()).index(
            st.session_state.get("advice_audience") or "Department Manager"
        ),
        help="Choose your role to get advice written specifically for you.",
    )
    st.session_state.advice_audience = audience_choice

    st.markdown("---")
    st.subheader("Display Options")
    st.session_state.beginner_mode = st.toggle(
        "Beginner Mode",
        value=st.session_state.beginner_mode,
        help="Turn this on for simple, jargon-free explanations.",
    )
    if st.session_state.beginner_mode:
        st.info("Beginner Mode is ON. Switch audience to 'Beginner' for matching advice.")

    st.markdown("---")
    st.subheader("Spending Tracker")
    tracking_period = st.selectbox(
        "Track by period",
        options=["Weekly", "Monthly", "Quarterly"],
        key="tracking_period_select",
    )
    if st.button("Save Snapshot", use_container_width=True):
        if st.session_state.df is not None and not st.session_state.df.empty:
            snap_total = float(st.session_state.df["Amount"].sum())
            snap_items = len(st.session_state.df)
            snap_top_cat = (
                st.session_state.df.groupby("Category")["Amount"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
                if "Category" in st.session_state.df.columns
                else "N/A"
            )
            st.session_state.tracking_snapshots.append({
                "date": date.today().isoformat(),
                "period": tracking_period,
                "total": snap_total,
                "items": snap_items,
                "top_cat": snap_top_cat,
            })
            st.success("Snapshot saved!")
        else:
            st.warning("No data to snapshot yet.")

    if st.button("Clear Snapshots", use_container_width=True):
        st.session_state.tracking_snapshots = []
        st.rerun()

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero-banner">
    <h1>🏛️ AI Expense Tracker</h1>
    <p>Upload PDF receipts and invoices — AI extracts, categorises, and analyses your expenses automatically.</p>
    <span class="hero-badge">Powered by IBM watsonx.ai</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# File uploader
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF receipts or invoices (up to 10 files)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files and len(uploaded_files) > 10:
    st.error("Maximum 10 files allowed. Only the first 10 will be processed.")
    uploaded_files = uploaded_files[:10]

# ---------------------------------------------------------------------------
# Action buttons row
# ---------------------------------------------------------------------------
col_submit, col_analyze, col_summary, col_advice, col_export, col_clear = st.columns(
    [1, 1, 1.4, 1.2, 1.4, 1]
)

with col_submit:
    submit_clicked = st.button("Submit", type="primary", use_container_width=True)

with col_analyze:
    analyze_clicked = st.button("Analyze", use_container_width=True)

with col_summary:
    summary_clicked = st.button("Generate Summary", use_container_width=True)

with col_advice:
    advice_clicked = st.button("Get Advice", use_container_width=True)

with col_export:
    if st.session_state.df is not None and not st.session_state.df.empty:
        csv_data = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="Export CSV",
            data=csv_data,
            file_name="expenses.csv",
            mime="text/csv",
            use_container_width=True,
        )

with col_clear:
    clear_clicked = st.button("Clear All", use_container_width=True)

# ---------------------------------------------------------------------------
# Clear All
# ---------------------------------------------------------------------------
if clear_clicked:
    st.session_state.df = None
    st.session_state.summary = None
    st.session_state.summary_stats = None
    st.session_state.advice = None
    st.session_state.advice_audience = None
    st.session_state.processed_hashes = set()
    clear_cache()
    st.rerun()

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
if submit_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file before submitting.")
    else:
        new_files = []
        for f in uploaded_files:
            file_hash = hashlib.md5(f.getvalue()).hexdigest()
            if file_hash not in st.session_state.processed_hashes:
                new_files.append(f)

        if not new_files:
            st.warning("All uploaded files have already been processed.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(completed, total, filename):
                progress_bar.progress(int(completed / total * 100))
                status_text.text(
                    f"Processing file {completed} of {total}: {filename}..."
                )

            df_new, debug_info = process_invoices(
                new_files, max_workers=2, progress_callback=progress_callback
            )

            progress_bar.empty()
            status_text.empty()

            if df_new is not None and not df_new.empty:
                if st.session_state.df is None:
                    st.session_state.df = df_new
                else:
                    st.session_state.df = pd.concat(
                        [st.session_state.df, df_new], ignore_index=True
                    )

            failed_names = {
                fname
                for fname, msg in debug_info.items()
                if msg.startswith("ERROR")
            }
            for f in new_files:
                if f.name not in failed_names:
                    file_hash = hashlib.md5(f.getvalue()).hexdigest()
                    st.session_state.processed_hashes.add(file_hash)

            st.session_state.summary = None
            st.session_state.summary_stats = None
            st.session_state.advice = None

            success_count = len(new_files) - len(failed_names)
            if success_count > 0:
                st.success(f"Successfully processed {success_count} file(s).")
            if failed_names:
                st.warning(
                    "The following files could not be processed: "
                    + ", ".join(sorted(failed_names))
                )

# ---------------------------------------------------------------------------
# Results section
# ---------------------------------------------------------------------------
if st.session_state.df is not None and not st.session_state.df.empty:
    df = st.session_state.df

    total_amount = df["Amount"].sum()
    files_processed = len(st.session_state.processed_hashes)
    num_items = len(df)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
<div class="metric-card">
    <div class="metric-label">Files Processed</div>
    <div class="metric-value">{files_processed}</div>
</div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
<div class="metric-card">
    <div class="metric-label">Line Items</div>
    <div class="metric-value">{num_items}</div>
</div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
<div class="metric-card">
    <div class="metric-label">Total Amount</div>
    <div class="metric-value">${total_amount:,.2f}</div>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 24px'></div>", unsafe_allow_html=True)

    # Budget summary
    total_budget = float(st.session_state.budget_total)
    budget_pct = (total_amount / total_budget * 100) if total_budget > 0 else 0.0
    progress_val = min(budget_pct / 100.0, 1.0)

    st.markdown("#### Budget Overview")
    st.progress(progress_val)
    st.caption(
        f"Spent ${total_amount:,.2f} of ${total_budget:,.2f} budget ({budget_pct:.1f}%)"
    )

    if total_amount > total_budget:
        over_by = total_amount - total_budget
        st.error(f"⚠️ Over budget by ${over_by:,.2f}")
    elif budget_pct >= 80:
        st.warning("⚠️ Approaching budget limit")
    else:
        st.success("✅ Within budget")

    # Per-category alerts
    st.markdown("#### Category Budget Status")
    category_budgets = {
        "Office Supplies": float(st.session_state.budget_office_supplies),
        "Equipment": float(st.session_state.budget_equipment),
        "Services": float(st.session_state.budget_services),
        "Utilities": float(st.session_state.budget_utilities),
    }

    for cat, cat_budget in category_budgets.items():
        cat_spent = float(
            df[df["Doc Type"] == cat]["Amount"].sum()
            if "Doc Type" in df.columns
            else 0.0
        )
        if cat_spent > cat_budget:
            st.error(
                f"⚠️ {cat} over budget: spent ${cat_spent:,.2f} of ${cat_budget:,.2f}"
            )
        else:
            cat_pct = (cat_spent / cat_budget) if cat_budget > 0 else 0.0
            st.caption(f"{cat}: ${cat_spent:,.2f} / ${cat_budget:,.2f}")
            st.progress(min(cat_pct, 1.0))

    st.markdown("<div style='margin-top: 24px'></div>", unsafe_allow_html=True)

    # DataFrame
    display_col_map = {
        "📅 Date": "Date",
        "🏢 Vendor": "Vendor",
        "📄 Doc Type": "Doc Type",
        "🏷️ Category": "Category",
        "📝 Description": "Description",
        "💱 Currency": "Currency",
        "💵 Amount": "Amount",
    }

    display_df = pd.DataFrame()
    for emoji_col, raw_col in display_col_map.items():
        if raw_col in df.columns:
            display_df[emoji_col] = df[raw_col]
        else:
            display_df[emoji_col] = ""

    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------
if analyze_clicked:
    if st.session_state.df is None or st.session_state.df.empty:
        st.warning("Please upload and submit receipts first.")
    else:
        cat_budgets = {
            "Office Supplies": float(st.session_state.budget_office_supplies),
            "Equipment": float(st.session_state.budget_equipment),
            "Services": float(st.session_state.budget_services),
            "Utilities": float(st.session_state.budget_utilities),
        }
        figures = analyze_invoices(st.session_state.df, category_budgets=cat_budgets)
        vendor_chart = figures[0]
        category_chart = figures[1]
        doc_type_chart = figures[2]
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("Expenses by Vendor")
            st.plotly_chart(vendor_chart, use_container_width=True)
        with chart_col2:
            st.subheader("Expenses by Category")
            st.plotly_chart(category_chart, use_container_width=True)
        st.subheader("Expenses by Document Type")
        st.plotly_chart(doc_type_chart, use_container_width=True)
        if len(figures) == 4:
            budget_chart = figures[3]
            st.subheader("Budget vs. Actual by Category")
            st.plotly_chart(budget_chart, use_container_width=True)

# ---------------------------------------------------------------------------
# Generate Summary
# ---------------------------------------------------------------------------
if summary_clicked:
    if st.session_state.df is None or st.session_state.df.empty:
        st.warning("Please upload and submit receipts first.")
    else:
        with st.spinner("Generating AI summary..."):
            text, stats, sentences = generate_summary(st.session_state.df)
            st.session_state.summary = text
            st.session_state.summary_stats = (stats, sentences)

if st.session_state.summary and st.session_state.summary_stats:
    stats, sentences = st.session_state.summary_stats
    render_summary_card(st.session_state.summary, stats, sentences)

# ---------------------------------------------------------------------------
# Get Advice
# ---------------------------------------------------------------------------
if advice_clicked:
    if st.session_state.df is None or st.session_state.df.empty:
        st.warning("Please upload and submit receipts first.")
    else:
        audience = st.session_state.advice_audience or "Department Manager"
        with st.spinner(f"Generating advice for {audience}..."):
            st.session_state.advice = generate_advice(
                st.session_state.df, audience=audience
            )

if st.session_state.advice:
    audience_label = st.session_state.advice_audience or "Department Manager"
    render_advice_card(st.session_state.advice, audience_label)

    advice_export = (
        f"AI Expense Advice\n"
        f"Audience: {audience_label}\n"
        f"{'=' * 40}\n\n"
        f"{st.session_state.advice}\n"
    )
    st.download_button(
        label="Export Advice as TXT",
        data=advice_export,
        file_name="expense_advice.txt",
        mime="text/plain",
    )

# ---------------------------------------------------------------------------
# Spending Tracker
# ---------------------------------------------------------------------------
if st.session_state.tracking_snapshots:
    st.markdown("---")
    st.markdown("#### 📅 Spending Tracker")

    period_filter = st.selectbox(
        "Filter snapshots by period",
        options=["All", "Weekly", "Monthly", "Quarterly"],
        key="tracker_filter",
    )

    snapshots = st.session_state.tracking_snapshots
    if period_filter != "All":
        snapshots = [s for s in snapshots if s["period"] == period_filter]

    if not snapshots:
        st.info(f"No {period_filter.lower()} snapshots saved yet.")
    else:
        # trend: show totals over time
        snap_df = pd.DataFrame(snapshots)
        snap_df["date"] = pd.to_datetime(snap_df["date"])
        snap_df = snap_df.sort_values("date")

        import plotly.graph_objects as go
        fig_track = go.Figure()
        fig_track.add_trace(go.Scatter(
            x=snap_df["date"],
            y=snap_df["total"],
            mode="lines+markers",
            marker=dict(size=8, color="#3B82F6"),
            line=dict(color="#3B82F6", width=2),
            name="Total Spend",
        ))
        fig_track.update_layout(
            font=dict(family="Inter, sans-serif", size=13),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Snapshot Date"),
            yaxis=dict(title="Total Spend ($)"),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_track, use_container_width=True)

        # snapshot cards
        for snap in reversed(snapshots):
            st.markdown(
                f"""
<div class="snapshot-card">
    <div>
        <div class="snapshot-period">{snap["period"]} snapshot</div>
        <div class="snapshot-meta">Saved {snap["date"]} &nbsp;·&nbsp; {snap["items"]} items &nbsp;·&nbsp; Top: {snap["top_cat"]}</div>
    </div>
    <div class="snapshot-amount">${snap["total"]:,.2f}</div>
</div>
""",
                unsafe_allow_html=True,
            )
