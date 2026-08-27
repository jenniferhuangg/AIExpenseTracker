import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import plotly.graph_objects as go
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from model_gateway import invoke_llm

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_pipeline_options = PdfPipelineOptions()
_pipeline_options.do_ocr = True
_pipeline_options.do_table_structure = True

_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)}
)

_file_cache: dict = {}

_CHART_COLORS = {
    "Office Supplies": "#3B82F6",
    "Equipment": "#A855F7",
    "Services": "#10B981",
    "Utilities": "#F59E0B",
}

_BASE_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", size=13),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def _pdf_bytes_to_markdown(pdf_bytes: bytes) -> str:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        result = _converter.convert(tmp_path)
        return result.document.export_to_markdown()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Document type detection
# ---------------------------------------------------------------------------

_TYPE_KEYWORDS = {
    "office_supplies": [
        "office", "supplies", "paper", "pens", "folders", "stationery",
        "toner", "ink", "printer", "desk", "chair", "filing",
    ],
    "equipment": [
        "equipment", "computer", "laptop", "monitor", "keyboard", "mouse",
        "hardware", "software", "technology", "device", "machinery",
    ],
    "services": [
        "services", "consulting", "maintenance", "repair", "cleaning",
        "security", "professional", "contractor", "vendor", "support",
    ],
    "utilities": [
        "utilities", "electricity", "water", "gas", "internet", "phone",
        "telecommunications", "energy", "power", "heating", "cooling",
    ],
}


def _detect_doc_type(filename: str, text: str) -> str:
    filename_lower = filename.lower()
    text_lower = text.lower()
    scores = {}
    for doc_type, keywords in _TYPE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in filename_lower:
                score += 3
            if kw in text_lower:
                score += 1
        scores[doc_type] = score
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "generic"
    return best


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _get_extraction_prompt(doc_type: str, text: str) -> str:
    step2 = (
        "\nSTEP 2: Now convert the table above into a JSON array. "
        "Each row becomes a JSON object with these fields:\n"
        "- date (from Date column)\n"
        "- vendor (from Vendor column)\n"
        "- doc_type (leave as empty string)\n"
        "- category (from Category column)\n"
        "- description (from Description column)\n"
        "- currency (from Currency column)\n"
        "- amount (from Amount column)\n"
        "- confidence (set to 0.9)\n"
        "\nReturn ONLY the JSON array with no markdown, no code fences, no explanation:\n"
        "[\""
    )

    rules = (
        "Rules:\n"
        "- One line per charge\n"
        "- Date format: YYYY-MM-DD (or leave empty if not found)\n"
        "- Amount: numeric only (no currency symbols)\n"
        "- Include header row\n"
        "- Use | to separate columns\n"
    )

    rules_services = (
        "Rules:\n"
        "- One line per item/charge\n"
        "- Date format: YYYY-MM-DD (or leave empty if not found)\n"
        "- Amount: numeric only (no currency symbols)\n"
        "- Include header row\n"
        "- Use | to separate columns\n"
    )

    if doc_type == "office_supplies":
        categories = (
            "Categories: Paper Products, Writing Instruments, Filing & Storage, "
            "Desk Accessories, Printer Supplies, Technology Accessories, Furniture, "
            "Taxes & Fees, Shipping, Miscellaneous"
        )
        return (
            "Analyze this office supplies invoice and extract all charges. "
            "Ignore any [image] tags.\n\n"
            "STEP 1: Create a table with these columns separated by | (pipe):\n"
            "Date | Vendor | Category | Description | Currency | Amount\n\n"
            + categories + "\n\n"
            + rules + "\n"
            "Document:\n" + text + "\n\n"
            "Table:\n"
            + step2
        )

    if doc_type == "equipment":
        categories = (
            "Categories: Computer Hardware, Software Licenses, Peripherals, "
            "Networking Equipment, Maintenance, Installation, Taxes & Fees, Miscellaneous"
        )
        return (
            "Analyze this equipment invoice and extract all charges. "
            "Ignore any [image] tags.\n\n"
            "STEP 1: Create a table with these columns separated by | (pipe):\n"
            "Date | Vendor | Category | Description | Currency | Amount\n\n"
            + categories + "\n\n"
            + rules + "\n"
            "Document:\n" + text + "\n\n"
            "Table:\n"
            + step2
        )

    if doc_type == "services":
        categories = (
            "Categories: Consulting, Maintenance, Repair, Cleaning, Security, "
            "Professional Services, Contractor Fees, Taxes & Fees, Miscellaneous"
        )
        return (
            "Analyze this services invoice and extract all charges. "
            "Ignore any [image] tags.\n\n"
            "STEP 1: Create a table with these columns separated by | (pipe):\n"
            "Date | Vendor | Category | Description | Currency | Amount\n\n"
            + categories + "\n\n"
            + rules_services + "\n"
            "Document:\n" + text + "\n\n"
            "Table:\n"
            + step2
        )

    if doc_type == "utilities":
        categories = (
            "Categories: Electricity, Water, Gas, Internet, Telephone, "
            "Telecommunications, Energy, Taxes & Fees, Service Charges, Miscellaneous"
        )
        return (
            "Analyze this utilities invoice and extract all charges. "
            "Ignore any [image] tags.\n\n"
            "STEP 1: Create a table with these columns separated by | (pipe):\n"
            "Date | Vendor | Category | Description | Currency | Amount\n\n"
            + categories + "\n\n"
            + rules + "\n"
            "Document:\n" + text + "\n\n"
            "Table:\n"
            + step2
        )

    # generic
    categories = (
        "Categories: Office Supplies, Equipment, Services, Utilities, Maintenance, "
        "Professional Services, Technology, Taxes & Fees, Miscellaneous"
    )
    return (
        "Analyze this invoice and extract all charges. "
        "Ignore any [image] tags.\n\n"
        "STEP 1: Create a table with these columns separated by | (pipe):\n"
        "Date | Vendor | Category | Description | Currency | Amount\n\n"
        + categories + "\n\n"
        + rules + "\n"
        "Document:\n" + text + "\n\n"
        "Table:\n"
        + step2
    )


# ---------------------------------------------------------------------------
# Vendor extraction
# ---------------------------------------------------------------------------

_HEADER_KEYWORDS = {"invoice", "folio", "date", "page", "guest", "number", "charges", "credits", "description"}


def _extract_vendor_from_text(text: str) -> str:
    lines = text.splitlines()[:10]
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(kw in lower for kw in _HEADER_KEYWORDS):
            continue
        if sum(c.isalpha() for c in stripped) >= 1 and len(stripped) >= 3:
            return stripped
    # fallback: first sequence of capitalized words in top lines
    for line in lines:
        match = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", line)
        if match:
            return match.group(1)
    return "Unknown"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_DOCTYPE_MAP = {
    "office_supplies": "Office Supplies",
    "equipment": "Equipment",
    "services": "Services",
    "utilities": "Utilities",
    "generic": "",
}


def _normalize_expenses(rows: list, filename: str, text: str) -> list:
    doc_type_key = _detect_doc_type(filename, text)
    doc_type_label = _DOCTYPE_MAP.get(doc_type_key, "")
    vendor_fallback = None
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        vendor = str(row.get("vendor", "")).strip()
        if not vendor or vendor.lower() == "unknown":
            if vendor_fallback is None:
                vendor_fallback = _extract_vendor_from_text(text)
            vendor = vendor_fallback
        amount_raw = row.get("amount", 0)
        amount = _parse_amount(amount_raw) if not isinstance(amount_raw, float) else abs(amount_raw)
        confidence_raw = row.get("confidence", 0.9)
        try:
            confidence = float(confidence_raw)
        except (ValueError, TypeError):
            confidence = 0.0
        normalized.append({
            "date": str(row.get("date", "")).strip(),
            "vendor": vendor,
            "doc_type": doc_type_label,
            "category": str(row.get("category", "")).strip(),
            "description": str(row.get("description", "")).strip(),
            "currency": str(row.get("currency", "")).strip(),
            "amount": amount,
            "confidence": confidence,
        })
    return normalized


# ---------------------------------------------------------------------------
# Amount parsing
# ---------------------------------------------------------------------------

def _parse_amount(amount_str) -> float:
    if isinstance(amount_str, (int, float)):
        return abs(float(amount_str))
    s = str(amount_str).strip()
    # remove currency symbols
    s = re.sub(r"[$€£¥₹]", "", s).strip()
    if not s:
        return 0.0
    # remove leading/trailing signs for later abs()
    s = s.lstrip("+-")
    try:
        # European decimal: 1.234,56 or 1,5
        if re.match(r"^\d{1,3}(\.\d{3})*(,\d+)$", s):
            s = s.replace(".", "").replace(",", ".")
        # comma as decimal separator with no thousands: 1,5
        elif re.match(r"^\d+,\d+$", s):
            s = s.replace(",", ".")
        else:
            # standard: remove thousands commas
            s = s.replace(",", "")
        return abs(float(s))
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_json_from_llm(llm_output: str) -> list:
    # prepend "[" because prompt ends with [" to prime LLM
    text = "[" + llm_output

    # strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Strategy 1: brace-depth scanning for array boundary
    try:
        start = text.find("[")
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start: i + 1]
                        parsed = json.loads(candidate)
                        if isinstance(parsed, list):
                            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: extract individual {...} objects
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                fragment = text[start: i + 1]
                try:
                    obj = json.loads(fragment)
                    if isinstance(obj, dict):
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass
    if objects:
        return objects

    # Strategy 3: full parse fallback
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    return []


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    global _file_cache
    _file_cache = {}


# ---------------------------------------------------------------------------
# Single-file processing
# ---------------------------------------------------------------------------

def _process_single_file(filename: str, pdf_bytes: bytes) -> tuple:
    md5 = hashlib.md5(pdf_bytes).hexdigest()
    if md5 in _file_cache:
        rows = _file_cache[md5]
        debug = "cached"
        return rows, debug

    text = _pdf_bytes_to_markdown(pdf_bytes)
    doc_type = _detect_doc_type(filename, text)
    prompt = _get_extraction_prompt(doc_type, text)
    llm_output = invoke_llm(prompt, max_new_tokens=4096)
    raw_rows = _parse_json_from_llm(llm_output)
    rows = _normalize_expenses(raw_rows, filename, text)

    debug_parts = [
        f"doc_type={doc_type}",
        f"text_len={len(text)}",
        f"llm_output_len={len(llm_output)}",
        f"raw_rows={len(raw_rows)}",
        f"normalized_rows={len(rows)}",
    ]
    debug = " | ".join(debug_parts)

    _file_cache[md5] = rows
    return rows, debug


# ---------------------------------------------------------------------------
# Public: process_invoices
# ---------------------------------------------------------------------------

def process_invoices(uploaded_files, max_workers: int = 2, progress_callback=None):
    all_rows = []
    debug_info = {}
    total = len(uploaded_files)
    completed = 0
    lock_completed = [0]  # mutable container for thread-safe counter

    def _task(uf):
        filename = uf.name
        pdf_bytes = uf.read()
        rows, debug = _process_single_file(filename, pdf_bytes)
        return filename, rows, debug

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_task, uf): uf.name for uf in uploaded_files}
        for future in as_completed(future_map):
            fname = future_map[future]
            lock_completed[0] += 1
            completed = lock_completed[0]
            try:
                filename, rows, debug = future.result()
                if len(rows) == 0:
                    debug_info[filename] = f"ERROR: 0 rows extracted — {debug}"
                else:
                    debug_info[filename] = debug
                    all_rows.extend(rows)
            except Exception as exc:
                debug_info[fname] = f"ERROR: {exc}"
            if progress_callback:
                progress_callback(completed, total, fname)

    if not all_rows:
        return pd.DataFrame(columns=["Date", "Vendor", "Doc Type", "Category", "Description", "Currency", "Amount", "Confidence"]), debug_info

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={
        "date": "Date",
        "vendor": "Vendor",
        "doc_type": "Doc Type",
        "category": "Category",
        "description": "Description",
        "currency": "Currency",
        "amount": "Amount",
        "confidence": "Confidence",
    })
    return df[["Date", "Vendor", "Doc Type", "Category", "Description", "Currency", "Amount", "Confidence"]], debug_info


# ---------------------------------------------------------------------------
# Public: analyze_invoices
# ---------------------------------------------------------------------------

def analyze_invoices(df, category_budgets: dict = None):
    transparent = "rgba(0,0,0,0)"

    # 1. Horizontal bar: total by vendor
    vendor_totals = df.groupby("Vendor")["Amount"].sum().sort_values(ascending=True)
    fig1 = go.Figure(go.Bar(
        x=vendor_totals.values,
        y=vendor_totals.index,
        orientation="h",
        marker_color="#3B82F6",
    ))
    fig1.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(title="Total Amount"),
        yaxis=dict(title=""),
    )

    # 2. Donut: by category
    cat_totals = df.groupby("Category")["Amount"].sum()
    fig2 = go.Figure(go.Pie(
        labels=cat_totals.index,
        values=cat_totals.values,
        hole=0.4,
    ))
    fig2.update_layout(
        font=dict(family="Inter, sans-serif", size=13),
        plot_bgcolor=transparent,
        paper_bgcolor=transparent,
    )

    # 3. Bar: by doc type
    dtype_totals = df.groupby("Doc Type")["Amount"].sum()
    bar_colors = [_CHART_COLORS.get(dt, "#6B7280") for dt in dtype_totals.index]
    fig3 = go.Figure(go.Bar(
        x=dtype_totals.index,
        y=dtype_totals.values,
        marker_color=bar_colors,
    ))
    fig3.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(title=""),
        yaxis=dict(title="Total Amount"),
    )

    if category_budgets is None:
        return fig1, fig2, fig3

    # 4. Grouped bar: budgeted vs actual
    categories = ["Office Supplies", "Equipment", "Services", "Utilities"]
    actuals = []
    budgeted = []
    actual_colors = []

    for cat in categories:
        actual = float(
            df[df["Doc Type"] == cat]["Amount"].sum()
            if "Doc Type" in df.columns and cat in df["Doc Type"].values
            else 0.0
        )
        budget_val = float(category_budgets.get(cat, 0))
        actuals.append(actual)
        budgeted.append(budget_val)
        actual_colors.append("#EF4444" if actual > budget_val else "#3B82F6")

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        name="Budgeted",
        x=categories,
        y=budgeted,
        marker_color="#94A3B8",
    ))
    fig4.add_trace(go.Bar(
        name="Actual",
        x=categories,
        y=actuals,
        marker_color=actual_colors,
    ))
    fig4.update_layout(
        title="Budget vs. Actual by Category",
        barmode="group",
        font=dict(family="Inter, sans-serif", size=13),
        plot_bgcolor=transparent,
        paper_bgcolor=transparent,
        xaxis=dict(title=""),
        yaxis=dict(title="Amount"),
        legend=dict(orientation="h", y=1.1),
    )

    return fig1, fig2, fig3, fig4
