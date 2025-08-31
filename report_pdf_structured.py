# report_pdf_structured.py — clean text + logo cover + wrapped tables + platform summary pages
import os
import re
import json  # NEW
import argparse
import datetime as dt
import tempfile
import shutil
import warnings
import numpy as np

import pandas as pd
from fpdf import FPDF

import glob
from colorama import Fore, Style, init as colorama_init


# silence fpdf2 deprecation noise from older examples
warnings.filterwarnings("ignore", category=DeprecationWarning, module="fpdf")

# ---- matplotlib: non-interactive backend BEFORE pyplot
from matplotlib import use as mpl_use
mpl_use("Agg")
import matplotlib.pyplot as plt

try:
    from fontTools.ttLib import TTFont
except Exception:
    TTFont = None


# ---------- Defaults ----------
DEFAULT_OUT_DIR = r"C:\fyp\FYP_HateSpeechModel\pdf_reports"

DEFAULT_FB_PRED = r"C:\fyp\FYP_HateSpeechModel\hatespeech_prediction\facebook_hate_predictions.csv"
DEFAULT_IG_PRED = r"C:\fyp\FYP_HateSpeechModel\hatespeech_prediction\instagram_hate_predictions.csv"
DEFAULT_MS_PRED = r"C:\fyp\FYP_HateSpeechModel\hatespeech_prediction\messenger_hate_predictions.csv"

DEFAULT_FB_HASH = r"C:\fyp\FYP_HateSpeechModel\evidence_hash_reports\facebook_hash_report.csv"
DEFAULT_IG_HASH = r"C:\fyp\FYP_HateSpeechModel\evidence_hash_reports\instagram_hash_report.csv"
DEFAULT_MS_HASH = r"C:\fyp\FYP_HateSpeechModel\evidence_hash_reports\messenger_hash_report.csv"

DEFAULT_FONT = r"C:\fyp\FYP_HateSpeechModel\fonts\static\NotoSansSinhala-Regular.ttf"
DEFAULT_LOGO = r"C:\fyp\FYP_HateSpeechModel\assets\hawkeye_logo.png"


# ---------- Text cleaning helpers ----------
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F900-\U0001F9FF]")
SUFFIX_JUNK_PATTERNS = [
    r"reverb_db[^\s]*",
    r"local_?message_?pe",
    r"direct\.db\s+messages(?:_content)?",
    r"fts\.db\s+messages(?:_content)?",
    r"crypto_db[^\s]*",
    r"omnistore_[^\s]*",
    r"enigma\.db",
    r"\(\(\s*0\s*B.*$",
]
SINHALA_RANGE = r"\u0D80-\u0DFF"
PRINTABLE_KEEP = re.compile(
    r"[^\t\n\r A-Za-z"
    r"\u2000-\u206F"
    r"\u0020-\u007E"
    r"\u0D80-\u0DFF"
    r"]"
)


def strip_emojis(s: str) -> str:
    return _EMOJI_RE.sub("", str(s or ""))


def resolve_logo_path(cli_logo: str | None) -> str | None:
    """Try common locations so --logo is optional."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        cli_logo,
        DEFAULT_LOGO,
        os.path.join(base_dir, "assets", "hawkeye_logo.png"),
        os.path.join(os.getcwd(), "assets", "hawkeye_logo.png"),
        os.path.join(os.getcwd(), "hawkeye_logo.png"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def ensure_static_ttf(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Unicode TTF font not found at: {path}")
    if "VariableFont" in os.path.basename(path):
        raise ValueError("Variable font detected. Use a static TTF.")
    if TTFont is not None:
        try:
            tt = TTFont(path)
            if "fvar" in tt:
                raise ValueError("Variable font detected (has 'fvar' table). Use a static TTF.")
        except Exception as e:
            raise ValueError(f"Unsupported/invalid TTF: {e}")
    return path


def _bytes_literal_to_text(s: str) -> str:
    if s.startswith(("b'", 'b"')) and s.endswith(("'", '"')):
        try:
            obj = eval(s)
            if isinstance(obj, (bytes, bytearray)):
                return obj.decode("utf-8", "ignore")
        except Exception:
            return s[2:].strip("\"'")
    return s


def _strip_prefix_noise(s: str) -> str:
    m = re.search(rf"[{SINHALA_RANGE}]+|[A-Za-z]{{2,}}", s)
    return s[m.start():] if m else s


def _strip_suffix_noise(s: str) -> str:
    for pat in SUFFIX_JUNK_PATTERNS:
        m = re.search(pat, s, flags=re.I)
        if m:
            s = s[:m.start()].rstrip()
    return s


def sanitize_text(x) -> str:
    s = strip_emojis(str(x or ""))
    s = _CTRL_RE.sub(" ", s)
    s = PRINTABLE_KEEP.sub("", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def clean_text(x) -> str:
    s = str(x or "")
    s = _bytes_literal_to_text(s)
    s = strip_emojis(s)
    s = _CTRL_RE.sub(" ", s)
    s = _strip_prefix_noise(s)
    s = PRINTABLE_KEEP.sub("", s)
    s = _strip_suffix_noise(s)
    s = s.replace("_", " ")
    s = re.sub(r"[ \t]+", " ", s).strip(" \n\r\t")
    return s


def soft_wrap(s: str) -> str:
    s = str(s or "")
    zw = "\u200b"
    return s.replace("\\", "\\" + zw).replace("/", "/" + zw).replace("_", "_" + zw).replace("-", "-" + zw)

# --- helpers for All Texts page ---

def _nice_datetime(s: str) -> str:
    """Return 'DD/MM/YYYY' if parseable; else original string."""
    try:
        ts = pd.to_datetime(str(s), errors="coerce")
        if pd.notna(ts):
            return ts.strftime("%d/%m/%Y")
    except Exception:
        pass
    return str(s or "")


def _reduce_code_blobs(s: str) -> str:
    """Hide long code-like blobs (e.g., hashes/base64) with an ellipsis."""
    return re.sub(r"[A-Za-z0-9+/=_-]{40,}", "…", str(s or ""))

def _parse_any_epoch(v):
    """Return pandas.Timestamp or NaT from numbers like epoch ms/sec."""
    try:
        x = pd.to_numeric(v, errors="coerce")
        if pd.isna(x):
            return pd.NaT
        # ms vs sec heuristics
        if x >= 1e12:      # clearly milliseconds
            return pd.to_datetime(x, unit="ms")
        if 1e9 <= x < 1e12:  # seconds (1970+)
            return pd.to_datetime(x, unit="s")
    except Exception:
        pass
    return pd.NaT

def _format_ts(ts):
    if ts is None or pd.isna(ts):
        return "-"
    return ts.strftime("%d/%m/%Y")

def extract_msg_datetime(row: pd.Series) -> str:
    """
    Try many common fields to find a timestamp, convert/format it nicely.
    Returns 'HH:MM, DD/MM/YYYY' or '-'.
    """
    # Most likely fields first
    candidates = [
        "created_at", "date_time", "datetime",
        "timestamp_ms", "message_timestamp", "msg_timestamp",
        "timestamp", "sent_at", "time_sent", "delivered_at",
        "received_at", "server_time", "client_time",
        "date_sent", "time", "date"
    ]

    # 1) Direct single-field parse (epoch or string)
    for c in candidates:
        if c in row:
            v = row.get(c)
            if str(v).strip() in ("", "-", "0", "None", "nan", "NaN"):
                continue
            # epoch numbers?
            ts = _parse_any_epoch(v)
            if pd.notna(ts):
                return _format_ts(ts)
            # free-form datetime string
            ts = pd.to_datetime(str(v), errors="coerce")
            if pd.notna(ts):
                return _format_ts(ts)

    # 2) Combine separate date + time columns if present
    if ("date" in row) and ("time" in row):
        combo = f"{row.get('date','')} {row.get('time','')}"
        ts = pd.to_datetime(combo, errors="coerce")
        if pd.notna(ts):
            return _format_ts(ts)

    return "-"

# ---------- Auto-detect helpers (usernames & device) ----------

def _most_common_nonempty(series) -> str:
    try:
        s = pd.Series(series).astype(str).str.strip()
        s = s[(s.notna()) & (s.ne(""))]
        if s.empty:
            return ""
        return s.value_counts().idxmax()
    except Exception:
        return ""

def infer_username(df) -> str:
    if df is None or df.empty:
        return ""
    for col in ["username", "user_name", "sender_username", "profile_username", "handle", "owner_username",
                "author", "author_name", "sender", "sender_name", "from", "from_name", "profile_name",
                "account_name", "screen_name", "name", "full_name"]:
        if col in df.columns:
            u = _most_common_nonempty(df[col])
            if u:
                u = str(u).strip().strip('"').strip("'")
                if u.startswith("@"):
                    u = u[1:]
                if "/" in u:
                    u = u.rstrip("/").split("/")[-1]
                return u
    if "source_database" in df.columns:
        pat = re.compile(r"[\\/]+([A-Za-z0-9._-]{3,})\.(?:db|sqlite3?)$", re.I)
        for v in df["source_database"].dropna():
            m = pat.search(str(v))
            if m:
                return m.group(1)
    return ""

def infer_device_from_dfs(dfs) -> str:
    candidates = ["device_model", "device", "model", "phone_model", "product_model",
                  "ro_product_model", "ro_product_name", "manufacturer", "device_name"]
    for df in dfs:
        if df is None or df.empty:
            continue
        for col in candidates:
            if col in df.columns:
                val = _most_common_nonempty(df[col])
                if val:
                    return val
    for p in [
        os.path.join(os.getcwd(), "artifacts", "device_info.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "device_info.json"),
        "device_info.json", "device_info.txt",
    ]:
        if os.path.isfile(p):
            try:
                if p.lower().endswith(".json"):
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for k in ("model", "device_model", "device", "name"):
                        if data.get(k):
                            return str(data[k])
                else:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                return line
            except Exception:
                pass
    return ""


# ---------- Data ----------

def load_df(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    for col in ("text", "source_database", "source_table", "created_at", "date_time"):
        if col in df.columns:
            df[col] = df[col].fillna("").map(str)
    if "predicted_label" in df.columns:
        df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce").fillna(0).astype(int)
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    if "text" in df.columns:
        df["text"] = df["text"].map(clean_text)
    return df


def compute_stats(df):
    if df is None or df.empty:
        return dict(total=0, hate=0, safe=0, hate_pct=0.0, safe_pct=0.0)
    total = len(df)
    hate = int((df.get("predicted_label", 0) == 1).sum())
    safe = total - hate
    hate_pct = round((hate / total * 100.0), 1) if total else 0.0
    return dict(total=total, hate=hate, safe=safe, hate_pct=hate_pct, safe_pct=round(100.0 - hate_pct, 1))


def flagged_rows(df, top_n=20):
    if df is None or df.empty:
        return pd.DataFrame(columns=["confidence", "text", "source_database", "source_table", "created_at", "date_time"])
    flg = df[df.get("predicted_label", 0) == 1].copy()
    if "confidence" in flg.columns:
        flg = flg.sort_values("confidence", ascending=False)
    return flg.head(top_n)


def all_rows_sample(df, top_n=25):
    if df is None or df.empty:
        return pd.DataFrame(columns=["predicted_label", "text", "source_database", "source_table", "created_at", "date_time"])
    out = df.copy()
    if "text" in out.columns:
        out["__len"] = out["text"].str.len()
        out = out.sort_values("__len", ascending=False).drop(columns="__len")
    return out.head(top_n)


# ---------- PDF ----------

class ReportPDF(FPDF):
    def __init__(self, font_path):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(True, margin=15)
        self.add_font("UNI", "", font_path)
        self.add_font("UNI", "B", font_path)
        self.set_font("UNI", "", 12)

    def _eff_width(self) -> float:
        return self.w - self.l_margin - self.r_margin

    def full_width_multicell(self, h: float, txt: str):
        self.set_x(self.l_margin)
        safe = sanitize_text(str(txt or "")).replace("\r", " ").replace("\n", " ")
        self.multi_cell(self._eff_width(), h, safe if safe else "-")

    # ---------- wrapping helpers ----------

    def _wrap_for_cell(self, w, txt, max_lines=2, line_h=6):
        safe = sanitize_text(txt if txt is not None else "-")
        lines = self.multi_cell(w, line_h, safe, dry_run=True, output="LINES")
        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines]
            eff = w - 2 * self.c_margin
            last = lines[-1]
            while last and self.get_string_width(last + "…") > eff:
                last = last[:-1]
            lines[-1] = (last + "…") if last else "…"
        return [l if l else "-" for l in lines]

    def table_row_wrapped(self, widths, cells, aligns=None, line_h=6, border=1, wrap_cols=None, default_max_lines=2):
        if aligns is None:
            aligns = ["L"] * len(widths)
        wrap_cols = wrap_cols or {}
        col_lines = []
        for i, (w, val) in enumerate(zip(widths, cells)):
            if i in wrap_cols:
                max_lines = wrap_cols.get(i, default_max_lines)
                lines = self._wrap_for_cell(w, val, max_lines=max_lines, line_h=line_h)
            else:
                lines = [sanitize_text(val if val is not None else "-")]
            col_lines.append(lines)
        n_lines = max(len(ls) for ls in col_lines)
        row_h = n_lines * line_h
        x0, y0 = self.get_x(), self.get_y()
        for w, lines, align in zip(widths, col_lines, aligns):
            x, y = self.get_x(), self.get_y()
            if border:
                self.rect(x, y, w, row_h)
            for j, line in enumerate(lines):
                self.set_xy(x + self.c_margin, y + j * line_h)
                self.cell(w - 2 * self.c_margin, line_h, line, border=0, align=align)
            self.set_xy(x + w, y)
        self.set_xy(x0, y0 + row_h)

    def table_row(self, widths, cells, border=1, aligns=None, line_h=7):
        aligns = aligns or ["L"] * len(widths)
        self.set_x(self.l_margin)
        for w, val, align in zip(widths, cells, aligns):
            self.cell(w, line_h, sanitize_text(val), border=border, align=align)
        self.ln(line_h)

    # ---------- small utilities ----------

    def hr(self, pad=2):
        y = self.get_y() + pad
        self.set_draw_color(180, 180, 180)
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.set_draw_color(0, 0, 0)
        self.set_y(y + pad)

    def kv_inline(self, label, value, lw=None, ln=15, bullet=False, bullet_w=4):
        """
        Key : value on a single line (left label, right value).
        - lw: optional fixed label width (mm). If None, width is computed to fit the label.
        - ln: line height (mm).
        """
        label_txt = f"{label} :"
        value_txt = sanitize_text(value) if value else "-"

        if lw is None:
            lw = self.get_string_width(label_txt) + 2 * self.c_margin + 3

        if bullet:
            self.cell(bullet_w, ln, "•", align="C")

        self.set_font("UNI", "", 11)
        self.cell(lw, ln, label_txt, align="L")
        self.cell(self._eff_width() - lw, ln, value_txt, align="L", new_x="LMARGIN", new_y="NEXT")

    # ---------- cover (page 1) ----------

    def cover_page(self, today_str: str, logo_path: str | None = None):
        self.add_page()
        self.set_y(20)
        self.set_text_color(0, 0, 0)
        self.set_font("UNI", "B", 50)

        def center_line(text: str, line_h: float):
            self.cell(0, line_h, text.upper(), new_x="LMARGIN", new_y="NEXT", align="C")

        center_line("CYBERBULLYING", 20)
        center_line("DETECTION REPORT", 20)

        self.ln(20)
        self.set_font("UNI", "B", 16)
        w = min(160, self._eff_width())
        x = (self.w - w) / 2
        self.set_xy(x, self.get_y() + 6)
        self.multi_cell(w, 6, "Digital Forensic Analysis on Social Media Platforms", align="C")
        self.set_x(x)
        self.multi_cell(w, 6, "of Instagram, Facebook, Messenger", align="C")

        if logo_path and os.path.exists(logo_path):
            self.ln(25)
            logo_w = 90
            logo_x = (self.w - logo_w) / 2
            self.image(logo_path, x=logo_x, y=self.get_y(), w=logo_w)
            self.set_y(self.get_y() + logo_w + 4)

        self.set_font("UNI", "B", 15)
        bottom_offset = 30
        self.set_y(self.h - bottom_offset)
        self.cell(0, 6, today_str, align="C")

    # ---------- overview (page 2) ----------

    def summary_table(self, totals):
        widths = [50, 33, 55, 55]
        self.set_font("UNI", "B", 12)
        self.table_row(widths, ["Social Media Platform", "Total of Texts",
                                "Amount of Hate Speech", "Amount of Safe Speech"])
        self.set_font("UNI", "", 12)
        for name in ["Facebook", "Instagram", "Messenger"]:
            s = totals.get(name, dict(total=0, hate=0, safe=0, hate_pct=0.0, safe_pct=0.0))
            self.table_row(widths, [name, s["total"], f"{s['hate']} = {s['hate_pct']}%", f"{s['safe']} = {s['safe_pct']}%"])

    def grouped_bars_vertical(self, totals, tmpdir, hate_color="#06b6d4", safe_color="#5be7e7"):
        path = os.path.join(tmpdir, "summary_bar.png")

        labels = ["Facebook", "Instagram", "Messenger"]
        hate = [totals.get(k, {}).get("hate", 0) for k in labels]
        safe = [totals.get(k, {}).get("safe", 0) for k in labels]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        ax.bar(x - width/2, hate, width, label="Hate Speech", color=hate_color)
        ax.bar(x + width/2, safe, width, label="Safe Speech", color=safe_color)

        ax.set_xticks(x, labels)
        ax.set_ylabel("Count")
        ax.set_ylim(bottom=0)
        ax.set_title("Report Summary")
        ax.legend(loc="upper center", ncol=2, frameon=False)

        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        self.ln(8)
        self.image(path, w=180)
        self.ln(4)

    def overview_page(self, args, totals, tmpdir):
        self.add_page()
        self.set_font("UNI", "B", 18)
        self.cell(0, 9, "Cyberbullying - Hate Speech Detection Report",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(8)

        self.set_font("UNI", "", 12)
        # labels = ["Case Number", "Investigator", "Device Name", "Owner of the Device"]  # (old, with device)
        labels = ["Case Number", "Investigator", "Owner of the Device"]  # device removed from display
        label_w = max(self.get_string_width(f"{lbl} :") for lbl in labels) + 2*self.c_margin + 3
        self.kv_inline("Case Number", args.case_id or "-", lw=label_w, ln=15, bullet=False)
        self.kv_inline("Investigator", args.investigator or "-", lw=label_w, ln=15, bullet=False)
        # self.kv_inline("Device Name", args.device_id or "-", lw=label_w, ln=15, bullet=True)  # HIDDEN
        self.kv_inline("Owner of the Device", args.owner or "-", lw=label_w, ln=15, bullet=False)

        self.ln(8)
        self.hr(pad=4)
        self.ln(8)

        self.set_font("UNI", "B", 16)
        self.cell(0, 7, "Report Summary", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(10)

        self.summary_table(totals)
        self.grouped_bars_vertical(totals, tmpdir)

    #------------overall device summary with table + pie (page 3)-------------------

    def overall_device_summary_page(self, totals, tmpdir):
        """
        Page 3: Overall Hate Speech Detected from the Device
        - Table with per-platform % (hate / safe)
        - 'Total' row with overall % across all platforms
        - Pie chart with overall hate vs safe
        """
        # compute counts and overall percentages 
        labels = ["Facebook", "Instagram", "Messenger"]
        per = {k: totals.get(k, {"total": 0, "hate": 0, "safe": 0, "hate_pct": 0.0, "safe_pct": 0.0}) for k in labels}

        total_counts = sum(per[k]["total"] for k in labels)
        total_hate   = sum(per[k]["hate"]  for k in labels)
        total_safe   = sum(per[k]["safe"]  for k in labels)

        overall_hate_pct = round((total_hate / total_counts * 100.0), 1) if total_counts else 0.0
        overall_safe_pct = round(100.0 - overall_hate_pct, 1) if total_counts else 0.0

        # draw page
        self.add_page()
        self.set_font("UNI", "B", 18)
        self.cell(0, 9, "Overall Hate Speech Detected from the Device",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(20)

        # table header
        widths = [80, 55, 55]  # Platform | Hate % | Safe %
        line_h = 15
        self.set_font("UNI", "B", 12)
        self.table_row(widths, ["Social Media Platform", "Hate Speech", "Safe Speech"], line_h=line_h)

        # table rows
        self.set_font("UNI", "", 12)
        for name in labels:
            hp = per[name]["hate_pct"]
            sp = per[name]["safe_pct"]
            self.table_row(widths, [name, f"{hp}%", f"{sp}%"], line_h=line_h)

        # total row (overall)
        self.set_font("UNI", "B", 12)
        self.table_row(widths, ["Total", f"{overall_hate_pct}%", f"{overall_safe_pct}%"], line_h=line_h)
        self.set_font("UNI", "", 12)

        # overall pie chart
        fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=300)
        wedges, texts, autotexts = ax.pie(
            [max(total_hate, 0), max(total_safe, 0)],
            startangle=90,
            colors=["#06b6d4", "#5be7e7"],
            labels=None,
            autopct=lambda p: f"{p:.1f}%",
            pctdistance=0.75,
            textprops={"fontsize": 9}
        )
        centre = plt.Circle((0, 0), 0.58, fc="white")
        ax.add_artist(centre)
        ax.axis("equal")
        ax.legend(["Hate Speech", "Safe Speech"], loc="upper center",
                  bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

        pie_path = os.path.join(tmpdir, "overall_device_pie.png")
        fig.tight_layout()
        fig.savefig(pie_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        # place the image centered
        self.ln(15)
        img_w = 180
        img_w = min(img_w, self._eff_width())
        x = self.l_margin + (self._eff_width() - img_w) / 2
        self.image(pie_path, x=x, w=img_w)
        self.ln(10)


    # ---------- platform SUMMARY page (4,8,12) ----------

    def platform_summary_table(self, platform_name, stats):
        widths = [50, 33, 55, 55]
        self.set_font("UNI", "B", 12)
        self.table_row(
            widths,
            ["Social Media Platform", "Total of Texts", "Amount of Hate Speech ", "Amount of Safe Speech"]
        )
        self.set_font("UNI", "", 12)
        hate_txt = f"{stats['hate']} = {stats['hate_pct']}%"
        safe_txt = f"{stats['safe']} = {stats['safe_pct']}%"
        self.table_row(widths, [platform_name, stats["total"], hate_txt, safe_txt])

    def platform_summary_page(self, platform_name, username, when_str, stats, tmpdir):
        self.add_page()
        self.set_font("UNI", "B", 18)
        self.cell(0, 9, "Cyberbullying - Hate Speech Detection", align="C",
                  new_x="LMARGIN", new_y="NEXT")
        self.ln(6)

        self.set_font("UNI", "", 12)
        kv_labels = ["Social Media Platform", "Social Media Profile", "Report Generated Date"]
        label_w = max(self.get_string_width(f"{t} :") for t in kv_labels) + 2*self.c_margin + 3
        self.kv_inline("Social Media Platform", platform_name, lw=label_w, ln=15)
        self.kv_inline("Social Media Profile", username or "-", lw=label_w, ln=15)
        self.kv_inline("Report Generated Date", when_str, lw=label_w, ln=15)

        self.ln(6)
        self.hr(pad=4)
        self.ln(6)

        self.set_font("UNI", "B", 16)
        self.cell(0, 7, f"{platform_name} Summary", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(6)
        self.platform_summary_table(platform_name, stats)

        # Pie chart (centered)
        hate = stats.get("hate", 0)
        safe = stats.get("safe", 0)
        data = [hate, safe]
        colors = ["#06b6d4", "#5be7e7"]

        fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=300)
        wedges, texts, autotexts = ax.pie(
            data, startangle=90, colors=colors, labels=None,
            autopct=lambda p: f"{p:.1f}%", pctdistance=0.75, textprops={"fontsize": 9}
        )
        centre = plt.Circle((0, 0), 0.58, fc="white")
        ax.add_artist(centre)
        ax.axis("equal")
        ax.legend(["Hate Speech", "Safe Speech"], loc="upper center",
                  bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

        pie_path = os.path.join(tmpdir, f"{platform_name.lower()}_pie.png")
        fig.tight_layout()
        fig.savefig(pie_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        self.ln(8)
        img_w = 150
        img_w = min(img_w, self._eff_width())
        x = self.l_margin + (self._eff_width() - img_w) / 2
        self.image(pie_path, x=x, w=img_w)
        self.ln(4)

   # ---------- All Texts Extracted pages (5,9,13) ----------

    def all_texts_pages(self, platform_name: str, df: pd.DataFrame):
        """
        Render ALL extracted rows for the given platform.
        Uses fixed layout like your screenshot, wraps key columns to 2 lines,
        and automatically paginates.
        """
        widths = [6, 26, 60, 25, 40, 40]  # #, Hate/Safe, Text, DateTime, DB, Table
        line_h = 8

        def add_page_with_header():
            self.add_page()
            # Title
            self.set_font("UNI", "B", 13)
            self.cell(0, 7, f"All Texts Extracted {platform_name}", align="C",
                    new_x="LMARGIN", new_y="NEXT")
            self.ln(4)
            # Header row
            self.set_font("UNI", "B", 10)
            self.table_row(
                widths,
                ["#", "Hate or Safe", "Text", "Date", "Source DB", "Source Table"],
                line_h=line_h
            )
            self.set_font("UNI", "", 10)

        add_page_with_header()

        # No rows case
        if df is None or df.empty:
            self.full_width_multicell(6, "No rows.")
            return

        row_no = 1
        for _, r in df.reset_index(drop=True).iterrows():
            # Each row will be at most 2 lines tall for wrapped cols
            required_h = 5 * line_h
            # If we’re too close to the bottom, start a new page with header
            if self.get_y() + required_h > (self.h - self.b_margin):
                add_page_with_header()

            label = "Hate" if int(r.get("predicted_label", 0)) == 1 else "Safe"
            txt = clean_text(r.get("text", ""))  # <- show the full (cleaned) message
            dtm = extract_msg_datetime(r)
            sdb = soft_wrap(r.get("source_database", ""))
            stb = soft_wrap(r.get("source_table", ""))

            self.table_row_wrapped(
                widths,
                [row_no, label, txt, dtm, sdb, stb],
                aligns=["C", "L", "L", "L", "L", "L"],
                line_h=line_h,
                wrap_cols={2: 3, 3: 3, 4: 3, 5: 3},  # Text / Date / DB / Table
                default_max_lines=3
            )
            row_no += 1

    # ---------- Flagged (Hate) Texts pages (6,10,14) ----------

    def flagged_texts_pages(self, platform_name: str, df: pd.DataFrame, top_n: int = 25):
        """
        Render ONLY model-flagged hate texts (predicted_label == 1).
        Texts are cleaned for human readability and long code-ish blobs are
        trimmed with an ellipsis.
        """
        # Filter to hate only
        if df is None or df.empty:
            flg = pd.DataFrame(columns=df.columns if df is not None else [])
        else:
            flg = df[df.get("predicted_label", 0) == 1].copy()
            if "confidence" in flg.columns:
                flg = flg.sort_values("confidence", ascending=False)
        if top_n and len(flg) > top_n:
            flg = flg.head(top_n)

        widths = [7, 30, 60, 28, 38, 34]   # #, Confidence, Text, DateTime, DB, Table
        line_h = 8

        def header_page():
            self.add_page()
            self.set_font("UNI", "B", 13)
            self.cell(0, 7, f"{platform_name} Flagged Messages", align="C",
                      new_x="LMARGIN", new_y="NEXT")
            self.ln(4)
            self.set_font("UNI", "B", 10)
            self.table_row(
                widths,
                ["#", "Confidence Rate", "Text", "Date", "Source DB", "Source Table"],
                line_h=line_h
            )
            self.set_font("UNI", "", 10)

        header_page()

        if flg.empty:
            self.full_width_multicell(6, "No flagged messages.")
            return

        row_no = 1
        for _, r in flg.reset_index(drop=True).iterrows():
            # keep page nicely paginated
            required_h = 5 * line_h
            if self.get_y() + required_h > (self.h - self.b_margin):
                header_page()
        

            # confidence as percentage (if present)
            conf = "-"
            try:
                conf = f"{float(r.get('confidence', 0))*100:.1f}%"
            except Exception:
                pass

            # human-readable, code-lite text
            txt = clean_text(r.get("text", ""))
            txt = _reduce_code_blobs(txt)

            dtm = extract_msg_datetime(r)
            sdb = soft_wrap(r.get("source_database", ""))
            stb = soft_wrap(r.get("source_table", ""))

            self.table_row_wrapped(
                widths,
                [row_no, conf, txt, dtm, sdb, stb],
                aligns=["C", "C", "L", "L", "L", "L"],
                line_h=line_h,
                wrap_cols={2: 3, 3: 3, 4: 3, 5: 3},
                default_max_lines=3
            )
            row_no += 1

    # ---------- appendix: database SHA-256 hashes pages (7,11,15) ----------

    def appendix_hashes_page(self, platform_name: str, hash_df: pd.DataFrame | None):
        """
        Appendix : Database SHA-256 Hashes
        - Two columns (Databases | Hashes)
        - Auto-wrap long values
        - Auto-add new pages when space runs out, repeating the header/table header
        """
        widths = [95, 95]          # two equal columns
        line_h = 7                  # line height used everywhere
        wrap_cfg = {0: 2, 1: 2}     # wrap both columns to max 2 lines

        def _page_header():
            """Title, subtitle, blurb, then table header."""
            self.add_page()
            self.set_font("UNI", "B", 16)
            self.cell(0, 8, "Appendix : Database SHA-256 Hashes", align="C",
                    new_x="LMARGIN", new_y="NEXT")
            self.ln(6)

            self.set_font("UNI", "", 12)
            self.cell(0, 6, f"Platform: {platform_name}", new_x="LMARGIN", new_y="NEXT")
            self.ln(6)

            # table header
            self.set_font("UNI", "B", 11)
            self.table_row(widths, ["Databases", "Hashes"], line_h=line_h)
            self.set_font("UNI", "", 10)

        def _ensure_space_for(row_h: float):
            """Start a new page with headers if the next row won't fit."""
            if self.get_y() + row_h > (self.h - self.b_margin):
                _page_header()

        # start first page
        _page_header()

        # Rows
        if hash_df is None or hash_df.empty:
            self.full_width_multicell(6, "No hashes CSV provided.")
        else:
            # pick sensible columns
            db_col = next((c for c in ["database_file", "file", "database", "db"] if c in hash_df.columns),
                        hash_df.columns[0])
            hash_col = next((c for c in ["sha256_hash", "sha256", "hash", "sha_256"] if c in hash_df.columns),
                            (hash_df.columns[1] if len(hash_df.columns) > 1 else hash_df.columns[0]))

            for _, r in hash_df.iterrows():
                db_val = soft_wrap(r.get(db_col, ""))
                sha_val = soft_wrap(r.get(hash_col, ""))

                # Pre-compute wrapped lines to know the row height *before* drawing
                l_db  = self._wrap_for_cell(widths[0], db_val,  max_lines=wrap_cfg[0], line_h=line_h)
                l_sha = self._wrap_for_cell(widths[1], sha_val, max_lines=wrap_cfg[1], line_h=line_h)
                row_h = max(len(l_db), len(l_sha)) * line_h

                _ensure_space_for(row_h)

                # Now actually render the row (wrapping applied inside)
                self.table_row_wrapped(
                    widths, [db_val, sha_val],
                    aligns=["L", "L"], line_h=line_h, border=1,
                    wrap_cols=wrap_cfg, default_max_lines=2
                )

        # Bottom note
        self.ln(6)
        self.set_text_color(110, 110, 110)
        self.set_font("UNI", "", 10)
        self.full_width_multicell(
            5,
            "Note: Below is an original source-file list of extracted data displayed in the"
            "report for quick reference. The full CSV remains the authoritative record. " 
            "Furthermore, this report summarizes automated analysis results. "
            "Always refer to the original evidence for deep, sensitive, or critical investigations and decisions."
        )
        self.set_text_color(0, 0, 0)
        self.set_font("UNI", "", 12)

# ---------- Build ----------
def parse_args():
    p = argparse.ArgumentParser(description="Generate structured PDF report.")
    p.add_argument("--config", help="Optional path to JSON config from Hawkeye wrapper", default="")

    p.add_argument("--case-id", default="")
    p.add_argument("--investigator", default="")
    # p.add_argument("--device-id", default="")  # <<< DISABLED: we no longer accept/show device id
    p.add_argument("--owner", default="")

    p.add_argument("--fb-username", default="")
    p.add_argument("--ig-username", default="")
    p.add_argument("--ms-username", default="")

    p.add_argument("--fb-pred", default=DEFAULT_FB_PRED)
    p.add_argument("--ig-pred", default=DEFAULT_IG_PRED)
    p.add_argument("--ms-pred", default=DEFAULT_MS_PRED)

    p.add_argument("--fb-hash", default=DEFAULT_FB_HASH)
    p.add_argument("--ig-hash", default=DEFAULT_IG_HASH)
    p.add_argument("--ms-hash", default=DEFAULT_MS_HASH)

    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--font", default=DEFAULT_FONT)
    p.add_argument("--logo", default="", help="Optional path to a PNG/JPG logo for the cover page")
    p.add_argument("--top-n", type=int, default=25)
    return p.parse_args()


def merge_config(args):
    """If --config is supplied, merge it over CLI defaults (CLI still wins)."""
    if not args.config:
        return args
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if getattr(args, k, "") in ("", None):
                setattr(args, k, v)
    except Exception:
        pass
    return args


def prompt_if_missing(args):
    """Interactive prompts for the fields you wanted if missing."""
    try:
        if not args.case_id:
            args.case_id = input(Fore.YELLOW + "\n 📂 Enter the Case ID : ").strip()
        if not args.investigator:
            args.investigator = input(Fore.YELLOW + "\n 🕵🏻️ Enter the name of the Investigator : ").strip()
        if not args.owner:
            args.owner = input(Fore.YELLOW + "\n 👩🏻‍💼 Enter the Owner's name of the mobile device : ").strip()
    except EOFError:
        pass
    return args


def build_report(args):
    os.makedirs(args.out_dir, exist_ok=True)
    args.font = ensure_static_ttf(args.font)

    # load data
    fb = load_df(args.fb_pred)
    ig = load_df(args.ig_pred)
    ms = load_df(args.ms_pred)

    # NEW: load hash CSVs for appendix pages
    fb_hash = load_df(args.fb_hash)
    ig_hash = load_df(args.ig_hash)
    ms_hash = load_df(args.ms_hash)

    # auto-fill usernames/device if blank
    if not args.fb_username:
        args.fb_username = infer_username(fb) or ""
    if not args.ig_username:
        args.ig_username = infer_username(ig) or ""
    if not args.ms_username:
        args.ms_username = infer_username(ms) or ""
    # if not args.device_id:
    #     args.device_id = infer_device_from_dfs([fb, ig, ms]) or "UNKNOWN-DEVICE"  # <<< DISABLED

    # prompt for human fields if needed
    args = prompt_if_missing(args)

    fb_stats = compute_stats(fb)
    ig_stats = compute_stats(ig)
    ms_stats = compute_stats(ms)
    totals = {"Facebook": fb_stats, "Instagram": ig_stats, "Messenger": ms_stats}

    today = dt.datetime.now()
    today_str = today.strftime("%d of %B, %Y")
    when = today.strftime("%d/%m/%Y")
    ts_name = today.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"Structured_Report_{ts_name}.pdf")
    logo_path = resolve_logo_path(args.logo)

    tmp = tempfile.mkdtemp(prefix="pdf_tmp_")
    try:
        pdf = ReportPDF(args.font)

        # Page 1 (cover)
        pdf.cover_page(today_str, logo_path=logo_path)

        # Page 2 (overview)
        pdf.overview_page(args, totals, tmp)

        # Page 3 (overall device summary with table + pie)
        pdf.overall_device_summary_page(totals, tmp)

        # Pages 4,8,12 (summary) + 5,9,13 (all texts) + 6,10,14 (Flaged texts)
        # Facebook
        pdf.platform_summary_page("Facebook",  args.fb_username, when, fb_stats, tmp)
        #pdf.all_texts_pages("Facebook", fb)
        pdf.flagged_texts_pages("Facebook", fb, top_n=args.top_n)     
        pdf.appendix_hashes_page("Facebook", fb_hash)
        
        # Instagram
        pdf.platform_summary_page("Instagram", args.ig_username, when, ig_stats, tmp)
        #pdf.all_texts_pages("Instagram", ig)
        pdf.flagged_texts_pages("Instagram", ig, top_n=args.top_n)   
        pdf.appendix_hashes_page("Instagram", ig_hash)

        # Messenger
        pdf.platform_summary_page("Messenger", args.ms_username, when, ms_stats, tmp)
        #pdf.all_texts_pages("Messenger", ms)
        pdf.flagged_texts_pages("Messenger", ms, top_n=args.top_n)  
        pdf.appendix_hashes_page("Messenger", ms_hash)  
       
        # Optional detailed pages (keep disabled if you want the simple flow)
        # pdf.platform_section("Facebook",  args.fb_username, when, fb_stats, all_rows_sample(fb, args.top_n), flagged_rows(fb, args.top_n), load_df(args.fb_hash))
        # pdf.platform_section("Instagram", args.ig_username, when, ig_stats, all_rows_sample(ig, args.top_n), flagged_rows(ig, args.top_n), load_df(args.ig_hash))
        # pdf.platform_section("Messenger", args.ms_username, when, ms_stats, all_rows_sample(ms, args.top_n), flagged_rows(ms, args.top_n), load_df(args.ms_hash))

        pdf.output(out_path)
        print(Fore.GREEN + "\n✅✅✅ Report Generation Completed Successfully")
        print(Fore.GREEN + "\n📍 Report Saved Location → ")
        print(Fore.YELLOW + f"\n         👉🏻 {out_path}")
        print(Fore.BLUE + """\n
_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._."""+ Style.RESET_ALL)
        print(Fore.MAGENTA +"\n👋🏻  Quiting.....")
        print(Fore.CYAN +"""\n                                    
|==============================================================================================|
|8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888|
|==============================================================================================|
|88|                                                                                        |88|
|88|     :::    :::     :::     :::       ::: :::    ::: :::::::::: :::   ::: ::::::::::    |88|
|88|     :+:    :+:   :+: :+:   :+:       :+: :+:   :+:  :+:        :+:   :+: :+:           |88|
|88|     +:+    +:+  +:+   +:+  +:+       +:+ +:+  +:+   +:+         +:+ +:+  +:+           |88|
|88|     +#++:++#++ +#++:++#++: +#+  +:+  +#+ +#++:++    +#++:++#     +#++:   +#++:++#      |88|
|88|     +#+    +#+ +#+     +#+ +#+ +#+#+ +#+ +#+  +#+   +#+           +#+    +#+           |88|
|88|     #+#    #+# #+#     #+#  #+#+# #+#+#  #+#   #+#  #+#           #+#    #+#           |88|
|88|     ###    ### ###     ###   ###   ###   ###    ### ##########    ###    ##########    |88|
|88|                                                                                        |88|
|==============================================================================================|
|8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888|
|888                                                                                        888|
|888            ANDROID FORENSIC TOOL FOR HARSH SPEECH DETECTION ON SOCIAL MEDIA            888|
|888                                                                                        888|
|8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888|
|==============================================================================================|
    """)
        print(Fore.MAGENTA +"""
::::::::::: :::    :::     :::     ::::    ::: :::    :::    :::   :::  ::::::::  :::    ::: :::
    :+:     :+:    :+:   :+: :+:   :+:+:   :+: :+:   :+:     :+:   :+: :+:    :+: :+:    :+: :+:
    +:+     +:+    +:+  +:+   +:+  :+:+:+  +:+ +:+  +:+       +:+ +:+  +:+    +:+ +:+    +:+ +:+
    +#+     +#++:++#++ +#++:++#++: +#+ +:+ +#+ +#++:++         +#++:   +#+    +:+ +#+    +:+ +:+
    +#+     +#+    +#+ +#+     +#+ +#+  +#+#+# +#+  +#+         +#+    +#+    +#+ +#+    +#+ 
    #+#     #+#    #+# #+#     #+# #+#   #+#+# #+#   #+#        #+#    #+#    #+# #+#    #+# #+#
    ###     ###    ### ###     ### ###    #### ###    ###       ###     ########   ########  ###
        """+ Style.RESET_ALL)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    args = parse_args()
    args = merge_config(args)
    build_report(args)


if __name__ == "__main__":
    main()
