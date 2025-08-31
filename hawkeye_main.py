import os
import sqlite3
import pandas as pd
import subprocess
import tarfile
import hashlib
import warnings
import re
import unicodedata
import glob
import sys
from colorama import Fore, Style, init as colorama_init
import os, sys, json, tempfile, subprocess, datetime as _dt


# ML imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# =========================
# Setup
# =========================
warnings.simplefilter(action='ignore', category=FutureWarning)
colorama_init(autoreset=True)

OUTPUT_EXTRACT = r"C:\fyp\FYP_HateSpeechModel\android_media_extraction"
OUTPUT_FB_CSV = r"C:\fyp\FYP_HateSpeechModel\extracted_csv\facebook_all_text.csv"
OUTPUT_IG_CSV = r"C:\fyp\FYP_HateSpeechModel\extracted_csv\instagram_all_text.csv"
OUTPUT_MSG_CSV = r"C:\fyp\FYP_HateSpeechModel\extracted_csv\messenger_all_text.csv"
OUTPUT_FB_HASH_REPORT = r"C:\fyp\FYP_HateSpeechModel\evidence_hash_reports\facebook_hash_report.csv"
OUTPUT_IG_HASH_REPORT = r"C:\fyp\FYP_HateSpeechModel\evidence_hash_reports\instagram_hash_report.csv"
OUTPUT_MSG_HASH_REPORT = r"C:\fyp\FYP_HateSpeechModel\evidence_hash_reports\messenger_hash_report.csv"
    
# Where to save model prediction outputs
OUTPUT_FB_PRED = r"C:\fyp\FYP_HateSpeechModel\hatespeech_prediction\facebook_hate_predictions.csv"
OUTPUT_IG_PRED = r"C:\fyp\FYP_HateSpeechModel\hatespeech_prediction\instagram_hate_predictions.csv"
OUTPUT_MSG_PRED = r"C:\fyp\FYP_HateSpeechModel\hatespeech_prediction\messenger_hate_predictions.csv"

# Model folder
MODEL_PATH = r".\saved_model"

# Keywords for text/timestamp columns
TEXT_KEYWORDS = ["text", "message", "body", "caption", "comment", "title", "content", "description"]
TIME_KEYWORDS = [
    "time", "date", "timestamp", "created", "creation","sent", "received", "delivered", "start_time",
    "client_time", "server_time","message_time", "modified", "updated", "executed_time"
]

# =========================
# Utility
# =========================
#Connecting to the Android Device

def run_cmd(cmd):
    """Run shell command and return (stdout, stderr, returncode)."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def adb_connect(target):
    """Connect to ADB target and verify device connection. Prints success on connect."""
    out, err, rc = run_cmd(f'adb connect {target}')
    combined = f"{out}\n{err}".strip()
    success = ("connected to" in combined.lower()) or ("already connected to" in combined.lower())

    if success:
        devs_out, _, _ = run_cmd("adb devices")
        if target.split(":")[0] in devs_out or target in devs_out:
            print(Fore.GREEN + "\n✅✅✅ Android Mobile Phone is Successfully Connected via ADB..." + Style.RESET_ALL)
            return True

    print(Fore.RED + "\n❌ ADB connection failed. Output:\n" + combined + Style.RESET_ALL)
    return False

def prompt_for_ip(default_port=5555):
    """
    Ask user for device IP (optionally with :port). Returns normalized 'ip:port' string.
    - If user enters '192.168.1.23' -> returns '192.168.1.23:5555'
    - If user enters '192.168.1.23:5566' -> returns as-is
    """
    while True:
        ip = input(Fore.YELLOW + "\nEnter the IP of the Android device or IP:PORT (e.g. 192.168.xx.xx or 192.168.xx.xx:xxxx): " + Style.RESET_ALL).strip()
        if not ip:
            print(Fore.RED + "\nIP is required. Try again (or press Ctrl+C to quit)." + Style.RESET_ALL)
            continue

        # Basic IPv4 + optional :port validation
        m = re.match(r'^(\d{1,3}(?:\.\d{1,3}){3})(?::(\d{1,5}))?$', ip)
        if not m:
            print(Fore.RED + "\nInvalid format. Please enter like 192.168.x.x or 192.168.x.x:port" + Style.RESET_ALL)
            continue

        host, port = m.group(1), m.group(2)
        # Quick octet sanity check (0-255)
        try:
            if not all(0 <= int(o) <= 255 for o in host.split(".")):
                raise ValueError
        except ValueError:
            print(Fore.RED + "\nInvalid IP octets. Each should be 0-255." + Style.RESET_ALL)
            continue

        if port is None:
            port = str(default_port)

        return f"{host}:{port}"

#Generating Hash Reports

def sha256sum(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def generate_hash_report(folder_path, report_csv):
    hash_records = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".db"):
                file_path = os.path.join(root, file)
                file_hash = sha256sum(file_path)
                hash_records.append({"database_file": file, "sha256_hash": file_hash})
    os.makedirs(os.path.dirname(report_csv), exist_ok=True)
    if hash_records:
        df = pd.DataFrame(hash_records)
        df.to_csv(report_csv, index=False, encoding="utf-8")
        print(Fore.GREEN + f"\n✅ Hash report saved: {report_csv}" + Style.RESET_ALL)
    else:
        print(Fore.RED + "\n‼️ No database files found for hashing." + Style.RESET_ALL)

#Pulling the data from Android Device

def pull_sdcard_app_data(package_name, base_out_dir, label):
    """
    Create tar on device (/sdcard/Download/{label}_data.tar),
    pull to PC (base_out_dir), list DBs, then extract ONLY .db files.
    Returns (success:bool, db_root_dir:str|None)
    """
    os.makedirs(base_out_dir, exist_ok=True)

    tar_filename = f"{label}_data.tar"  # device name
    sdcard_tar = f"/sdcard/Download/{tar_filename}"
    local_tar = os.path.join(base_out_dir, f"{label}_data_data.tar")  # keep your current naming style

    print(Fore.CYAN + f"\n♻️ Creating TAR for {label} on device..." + Style.RESET_ALL)
    create_tar_cmd = f"adb shell su -c 'tar --create --file={sdcard_tar} /data/data/{package_name}'"
    out, err, rc = run_cmd(create_tar_cmd)

    print(Fore.CYAN + f"\n♻️ Pulling TAR for {label} to PC..." + Style.RESET_ALL)
    out, err, rc = run_cmd(f'adb pull {sdcard_tar} "{local_tar}"')
    if rc != 0 or not os.path.exists(local_tar):
        print(Fore.RED + f"\n[‼️] Failed to pull TAR for {label}.\n{out}\n{err}" + Style.RESET_ALL)
        return False, None

    # show DBs found inside tar (good for verification)
    #print(Fore.CYAN + f"\n🔎 Listing .db files inside {os.path.basename(local_tar)} ..." + Style.RESET_ALL)
    #db_members = list_databases_in_tar(local_tar)
    #if db_members:
        #for m in db_members[:12]:
            #print("   •", m)
        #if len(db_members) > 12:
            #print(f"   • ... (+{len(db_members)-12} more)")
    #else:
        #print(Fore.YELLOW + "   (no .db files reported by tar index)" + Style.RESET_ALL)

    # extract ONLY .db files (avoid Windows-invalid cache names)
    print(Fore.CYAN + f"\n♻️ Extracting ONLY .db files for {label}..." + Style.RESET_ALL)
    db_out_dir = os.path.join(base_out_dir, "databases_extracted")
    extracted_count = extract_databases_from_tar(local_tar, db_out_dir)
    if extracted_count == 0:
        print(Fore.YELLOW + f"\n⚠️ No .db files extracted for {label}." + Style.RESET_ALL)
        return False, None

    print(Fore.GREEN + f"\n✅ Extracted {extracted_count} database file(s) to {db_out_dir}" + Style.RESET_ALL)
    return True, db_out_dir
    
INVALID_WIN_CHARS = '<>:"/\\|?*'

def sanitize_path_component(name: str) -> str:
    """Make a filename safe for Windows."""
    cleaned = ''.join('_' if ch in INVALID_WIN_CHARS else ch for ch in name)
    return cleaned.strip().rstrip('.')

def safe_join(base: str, *parts: str) -> str:
    """Join and normalize, preventing path traversal."""
    joined = os.path.join(base, *parts)
    norm = os.path.normpath(joined)
    if not os.path.abspath(norm).startswith(os.path.abspath(base)):
        raise ValueError("Blocked unsafe path traversal")
    return norm

def list_databases_in_tar(local_tar: str):
    """Return a list of .db member paths inside the tar."""
    dbs = []
    with tarfile.open(local_tar, "r") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.lower().endswith(".db"):
                dbs.append(m.name)
    return sorted(dbs)

#Extracting the .dbs from the .tar file

def extract_databases_from_tar(local_tar: str, out_dir: str) -> int:
    """
    Extract ONLY .db files from local_tar into out_dir,
    preserving a sanitized subpath. Returns count extracted.
    """
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    with tarfile.open(local_tar, "r") as tar:
        for m in tar.getmembers():
            if not (m.isfile() and m.name.lower().endswith(".db")):
                continue
            # normalize and sanitize every path component
            rel = m.name.lstrip("./")
            parts = [sanitize_path_component(p) for p in rel.split("/")]
            target = safe_join(out_dir, *parts)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            try:
                fobj = tar.extractfile(m)
                if fobj is None:
                    continue
                with open(target, "wb") as wf:
                    wf.write(fobj.read())
                count += 1
            except Exception as e:
                print(Fore.YELLOW + f"\n⚠️ Skipped {m.name}: {e}" + Style.RESET_ALL)
    return count

def find_tar(out_dir: str, label: str):
    """
    Find the app tar file we just pulled.
    Accepts fb_data.tar / fb_data_data.tar / any *.tar in folder.
    """
    candidates = [
        os.path.join(out_dir, f"{label}_data.tar"),
        os.path.join(out_dir, f"{label}_data_data.tar"),
    ] + glob.glob(os.path.join(out_dir, "*.tar"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None
    
def _parse_any_epoch(v):
    """
    Try to parse numbers like epoch ms/sec; otherwise return pandas.NaT.
    """
    try:
        x = pd.to_numeric(v, errors="coerce")
        if pd.isna(x):
            return pd.NaT
        # milliseconds vs seconds heuristics
        if x >= 1e12:
            return pd.to_datetime(x, unit="ms", errors="coerce")
        if 1e9 <= x < 1e12:
            return pd.to_datetime(x, unit="s", errors="coerce")
    except Exception:
        pass
    return pd.NaT


def _first_nice_datetime(row: pd.Series, timeish_cols: list[str]) -> str:
    """
    From a row and a list of timeish column names, return a nice date string:
    - Prefer epoch millis/seconds if present
    - Otherwise try free-form datetime parsing
    - If nothing usable: return "-"
    Output format: "YYYY-MM-DD" (you can switch to "%H:%M, %d/%m/%Y" if you prefer)
    """
    for c in timeish_cols:
        if c not in row:
            continue
        v = row.get(c, "")
        if pd.isna(v) or str(v).strip() in ("", "0", "None", "nan", "NaN"):
            continue

        ts = _parse_any_epoch(v)
        if pd.notna(ts):
            return ts.strftime("%Y-%m-%d")

        # try parsing free-form date strings
        ts = pd.to_datetime(str(v), errors="coerce")
        if pd.notna(ts):
            return ts.strftime("%Y-%m-%d")

    # fallbacks: if there are explicit 'date' + 'time' split columns
    if ("date" in row) and ("time" in row):
        combo = f"{row.get('date','')} {row.get('time','')}"
        ts = pd.to_datetime(combo, errors="coerce")
        if pd.notna(ts):
            return ts.strftime("%Y-%m-%d")

    return "-"

# =========================
# Database Parsing
# =========================

def extract_text_from_db(db_path):
    """Extract rows from tables, prioritizing text fields but falling back to all fields if no match."""
    extracted_rows = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall()]

        for table in tables:
            try:
                cursor.execute(f"PRAGMA table_info({table})")
                col_names = [c[1] for c in cursor.fetchall()]
                low = [c.lower() for c in col_names]

                text_cols = [c for c in col_names if any(k in c.lower() for k in TEXT_KEYWORDS)]
                time_cols = [c for c in col_names if any(k in c.lower() for k in TIME_KEYWORDS)]

                if text_cols:
                    selected_cols = list(dict.fromkeys(text_cols + time_cols))  # keep order unique
                else:
                    selected_cols = col_names  # fallback: all columns

                if not selected_cols:
                    continue

                query = f"SELECT {', '.join(selected_cols)} FROM {table}"
                df = pd.read_sql_query(query, conn)

                # Best-effort unify timestamps into one 'date_time' column
                if time_cols:
                    # Compute one nice string per row
                    df["date_time"] = df.apply(lambda r: _first_nice_datetime(r, time_cols), axis=1)
                elif {"date", "time"}.issubset(set(col_names)):
                    df["date_time"] = df.apply(lambda r: _first_nice_datetime(r, ["date", "time"]), axis=1)
                else:
                    df["date_time"] = "-"

                df["source_database"] = os.path.basename(db_path)
                df["source_table"] = table
                extracted_rows.append(df)

            except Exception:
                continue

        conn.close()
    except Exception as e:
        print(Fore.RED + f"\n‼️ Error reading {db_path}: {e}" + Style.RESET_ALL)
    return extracted_rows

def parse_all_databases(folder_path, output_csv):
    """\nParse all .db files in a folder and save merged CSV."""
    all_dfs = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".db"):
                db_path = os.path.join(root, file)
                print(Fore.CYAN + f"   ▫️ Parsing : {file}" + Style.RESET_ALL)
                extracted_data = extract_text_from_db(db_path)
                all_dfs.extend(extracted_data)

    non_empty_dfs = [df for df in all_dfs if not df.empty]
    if non_empty_dfs:
        merged_df = pd.concat(non_empty_dfs, ignore_index=True)
        merged_df.to_csv(output_csv, index=False, encoding="utf-8")
        print(Fore.GREEN + f"\n✅ Saved merged CSV: {output_csv}" + Style.RESET_ALL)
    else:
        print(Fore.RED + "\n‼️ No usable data found." + Style.RESET_ALL)

# =========================
# Model & Inference Helpers
# =========================

def load_model(model_path):
    print(Fore.CYAN + "\n♻️ Loading hate-speech model..." + Style.RESET_ALL)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^\wඅ-ෆාැෑුූිීෙේෛොෝෘෲෟාං ]+", " ", text)
    return text.strip().lower()

def keyword_flag(text: str) -> bool:
    keywords = [
        # Singlish
        "gon","balla", "ballo","pissu","umbata","nari","wesi","haraka","thambiya","baduwak","modaya", "gona" , "besikaya", "haththa", "cari", "kari", "harakek", 
        "puka","mada","fuck","idiot","lamayek","moda","buruwa","hutta","utto", "hutto","kolukaraya","gona", "gon gani","gani", "pakaya" , "bitch" , "pissek"
        # Sinhala
        "ගොන්","බල්ලා","පිස්සු","මෝඩයෙක්","පිස්සෙක්","පිස්සූ","හරකෙක්","නරකයා","වඳිල්ල","පකයා","අල්ලගමු","මුහුණ","බල්ලෝ","උඹට","නරි","වෙසි",
        "වේසිය","හරකා","තම්බියා","බඩුවක්","මෝඩයා","ගොනා","බේසිකයා","හැත්ත","කැරි","පුක","ෆක්","ළමයෙක්","මෝඩ","බූරුවා","හුත්ත","උත්තෝ",
        "හුත්තෝ","කොලුකාරයා","ගොනා","ගොන් ගෑණි","පකයා","බිච්","ගෑණි"
    ]
    normalized = unicodedata.normalize("NFC", str(text).lower())
    return any(kw in normalized for kw in keywords)

def predict_batch(df: pd.DataFrame, tokenizer, model):
    """
    Runs detection on a dataframe containing multiple potential text columns.
    Returns a dataframe with columns: text, predicted_label, confidence, source_database, source_table
    """
    if df.empty:
        return pd.DataFrame(columns=["text","predicted_label","confidence","source_database","source_table","date_time"])

    # choose text-like columns
    textish_cols = [c for c in df.columns if any(k in c.lower() for k in TEXT_KEYWORDS)]
    if not textish_cols:
        textish_cols = [c for c in df.columns if df[c].dtype == object]

    # time-like columns to feed our formatter if unified column is missing
    timeish_cols = [c for c in df.columns if any(k in c.lower() for k in TIME_KEYWORDS)]

    # flatten to one "text" column
    texts, times, src_db, src_tbl = [], [], [], []
    for _, row in df.iterrows():
        pieces = []
        for c in textish_cols:
            val = row.get(c, "")
            if isinstance(val, str) and val.strip():
                pieces.append(val)
        if not pieces:
            continue

        combined = " | ".join(pieces)
        texts.append(combined)
        src_db.append(row.get("source_database", ""))
        src_tbl.append(row.get("source_table", ""))

        if "date_time" in df.columns:
            dt_val = row.get("date_time", "-")
            dt_val = "-" if pd.isna(dt_val) else str(dt_val)
        else:
            dt_val = _first_nice_datetime(row, timeish_cols) if timeish_cols else "-"
        times.append(dt_val)

    if not texts:
        return pd.DataFrame(columns=["text","predicted_label","confidence","source_database","source_table","date_time"])

    # Fallback first, then model
    cleaned = [clean_text(t) for t in texts]
    fallback_mask = [keyword_flag(t) for t in texts]

    non_fb_idx = [i for i, m in enumerate(fallback_mask) if not m]
    
    if non_fb_idx:
        enc = tokenizer([cleaned[i] for i in non_fb_idx], return_tensors="pt",
                        truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=1).cpu()
            pred_classes = torch.argmax(probs, dim=1).tolist()
            pred_conf = probs.max(dim=1).values.tolist()

    rows = []
    j = 0
    for i, text in enumerate(texts):
        if fallback_mask[i]:
            lbl, conf = 1, 0.99
        else:
            lbl, conf = int(pred_classes[j]), float(pred_conf[j])
            j += 1
        rows.append([text, lbl, conf, times[i], src_db[i], src_tbl[i]])

    return pd.DataFrame(rows, columns=[
        "text", "predicted_label", "confidence",
        "date_time", "source_database", "source_table"
    ])

    #out_df = pd.DataFrame(rows, columns=["text","predicted_label","confidence"]) 
    #out_df["source_database"] = src_db
    #out_df["source_table"] = src_tbl
    #return out_df

def run_detection_on_csv(input_csv: str, output_csv: str, tokenizer, model):
    if not os.path.exists(input_csv):
        print(Fore.YELLOW + f"\n⚠️ Skipping predictions: CSV not found -> {input_csv}" + Style.RESET_ALL)
        return 0, 0

    print(Fore.CYAN + f"\n🔎 Running hate-speech detection on: {input_csv}" + Style.RESET_ALL)
    df = pd.read_csv(input_csv, encoding="utf-8", low_memory=False)
    pred_df = predict_batch(df, tokenizer, model)

    if pred_df.empty:
        print(Fore.YELLOW + f"\n⚠️ No text-like data found in {input_csv}." + Style.RESET_ALL)
        return 0, 0

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pred_df.to_csv(output_csv, index=False, encoding="utf-8")
    total = len(pred_df)
    toxic = int((pred_df["predicted_label"] == 1).sum())
    print(Fore.GREEN + f"\n✅ Predictions saved: {output_csv}" + Style.RESET_ALL)
    print(Fore.MAGENTA + f"\n📊 Summary -> total: {total}, hate_speech: {toxic}, safe: {total - toxic}" + Style.RESET_ALL)
    return total, toxic

# =========================
# Banner
# =========================
def print_banner():
    print(Fore.MAGENTA +"""\n
               .                                                                 .        
             _/|\                                                               /|\_      
            _|H/':                           _______                           :'\H|_     
           _\HH|  `-._                     .'\_   _/'.                     _.-'  |HH/_    
          _\HH\       HH".--.-....________.'/(v.-.v)\'.________....-.--."HH       /HH/_   
         _\Hb\.       Cb  CHHb  Cb_   '-.HH| _/_|_\_ |HH.-'   _dD  dHHD  dD       ./dH/_  
         'C.HH'._      '   'oHb   'H.    \H\  \ V /  //H/    .H'   dHo'  '      _.'HH.D'  
          'C.HHH`o._          ''-..__     |H\_ '-' _/H|     __..-''          _,o'HHH.D'   
            'C.HHHH`o.__                 .'H/   T   \H'.                 __,o'HHHH.D'     
               '-HHHHHHH"o.___       _.-'HH|  V | V  |HH'-._       ___.o"HHHHHHH-'        
                  'C.HHHHHHHHbiooooidHHH..H|    |    |H..HHHbiooooidHHHHHHHH.D'           
                      '"CHHHHHHHHHHHHHH/  |H|  \|/  |H|  \HHHHHHHHHHHHHHD"'               
                           ''::::''         H_|___|_H         ''::::''                     
                                       .\}.("   |   ").{/.                                
                                       (/' '\.  |  ./' '\)                                
                                      .'\    '  |  '   ,/'.                               
                                   { }|  '.     |     .'  |{ }                            
                                  .iH|-._  \    |    /  _.-|Hi.                           
                                 :oHH|'-._  |   |   |  _.-'|HHo:                          
                                    ''|   ''|---|---|''   |''                             
                                       .   /    |    \   .                                
                                        \.'     |     './                                   
                                          '--.__|__.--'    
    """+ Style.RESET_ALL) 
    print(Fore.CYAN +"""                                    
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
    """+ Style.RESET_ALL)

def find_db_root(folder):
    for r, dirs, _ in os.walk(folder):
        if os.path.basename(r) == "databases":
            return r
    return folder

#==========================
# report generation
#==========================
def ask_yes_no(q: str) -> bool:
        a = input(q + " [Y/N][Yes/No]: ").strip().lower()
        return a in ("y", "yes", "Y")

def report_main():
        if not ask_yes_no(Fore.GREEN + "\n👀 Do you need the final hate-speech detection report as a PDF?" + Style.RESET_ALL):
            print(Fore.MAGENTA +"\n👍🏻 Okay skipping PDF generation...")
            print(Fore.BLUE +"\n👋🏻  Quiting.....")
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
            return
        
        print(Fore.BLUE + """\n
_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._."""+ Style.RESET_ALL)

        print(Fore.BLUE + """\n 
                   .88888888888888888888.               .88888888888888888888.
                .888                    888.         .888                    888.
              .88                          88.     .88                          88.
                     .88888888888888888.                 .88888888888888888.
                   .88   .000000000.   88.             .88   .000000000.   88.
                 .88    000 00000 000    88.         .88    000 00000 000    88. 
               .88     000 000 000 000     88.     .88     000 000 000 000     88.
                 '88    000 00000 000    88'         '88    000 00000 000    88'
                   '88   '000000000'   88'             '88   '000000000'   88'
                     '88888888888888888'                 '88888888888888888'              
                    
|==============================================================================================|
|                               OUTPUT RESULTS PDF GENERATION                                  |
|==============================================================================================|"""+ Style.RESET_ALL)

        case_id      = input(Fore.YELLOW + "\n📂 Enter the Case ID :  " + Style.RESET_ALL).strip()
        investigator = input(Fore.YELLOW + "\n🕵🏻️ Enter the name of the Investigator : " + Style.RESET_ALL).strip()
        owner        = input(Fore.YELLOW + "\n👩🏻‍💼 Enter the Owner's name of the mobile device : " + Style.RESET_ALL).strip()

        print(Fore.BLUE + """\n
_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._."""+ Style.RESET_ALL)

        # Leave device_id and usernames blank → the report script will auto-detect.
        cfg = {
            "case_id": case_id,
            "investigator": investigator,
            "owner": owner,
            "fb_username": "",        # auto
            "ig_username": "",        # auto
            "ms_username": ""         # auto
            # You can add/override paths here if you want, or rely on defaults in the report script.
        }

        cfg_path = os.path.join(tempfile.gettempdir(), "hawkeye_report_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        # If you run from a venv, swap "python" with the full path to env/Scripts/python.exe
        cmd = [sys.executable, "report_pdf_structured.py", "--config", cfg_path]
        print(Fore.MAGENTA +"\n♻️ Running Hawkeye PDF Generator ..... ", "" .join(cmd))
        subprocess.run(cmd, check=True)

# =========================
# Main Workflow
# =========================
def main():
    # Show banner
    print_banner()

    # Connect to device
    target = prompt_for_ip(default_port=5555)
    if not adb_connect(target):
        return
    os.environ["ANDROID_SERIAL"] = target

    print(Fore.BLUE + """\n
_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._."""+ Style.RESET_ALL)

    print(Fore.BLUE + """\n             
                  .88888888888888888888.               .88888888888888888888.
               .888                    888.         .888                    888.
             .88                          88.     .88                          88.
                    .88888888888888888.                 .88888888888888888.
                  .88   .000000000.   88.             .88   .000000000.   88.
                .88    000 00000 000    88.         .88    000 00000 000    88. 
              .88     000 000 000 000     88.     .88     000 000 000 000     88.
                '88    000 00000 000    88'         '88    000 00000 000    88'
                  '88   '000000000'   88'             '88   '000000000'   88'
                    '88888888888888888'                 '88888888888888888'
          
|==============================================================================================|
|                                Social Media Data Extraction                                  |
|==============================================================================================|"""+ Style.RESET_ALL)

    # === Facebook Extraction ===
    print(Fore.MAGENTA + "\n📍 Extracting Facebook Databases..." + Style.RESET_ALL)
    fb_ok, fb_db_root = pull_sdcard_app_data("com.facebook.katana", OUTPUT_EXTRACT, "fb")
    if fb_ok and fb_db_root:
        print(Fore.BLUE + "\n📍 Generating SHA-256 Hash Report..." + Style.RESET_ALL)
        generate_hash_report(fb_db_root, OUTPUT_FB_HASH_REPORT)
        print(Fore.MAGENTA + "\n📍 Parsing Facebook Databases..." + Style.RESET_ALL)
        parse_all_databases(fb_db_root, OUTPUT_FB_CSV)
    
    print(Fore.BLUE + """\n
_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._."""+ Style.RESET_ALL)

    # === Instagram Extraction ===
    print(Fore.MAGENTA + "\n📍 Extracting Instagram Databases..." + Style.RESET_ALL)
    ig_ok, ig_db_root = pull_sdcard_app_data("com.instagram.android", OUTPUT_EXTRACT, "ig")
    if ig_ok and ig_db_root:
        print(Fore.BLUE + "\n📍 Generating SHA-256 Hash Report..." + Style.RESET_ALL)
        generate_hash_report(ig_db_root, OUTPUT_IG_HASH_REPORT)
        print(Fore.MAGENTA + "\n[📍IG] Parsing Instagram Databases..." + Style.RESET_ALL)
        parse_all_databases(ig_db_root, OUTPUT_IG_CSV)
    
    print(Fore.BLUE + """\n
_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._."""+ Style.RESET_ALL)

    # === Messenger Extraction ===
    print(Fore.MAGENTA + "\n📍 Extracting Messenger Databases..." + Style.RESET_ALL)
    msg_ok, msg_db_root = pull_sdcard_app_data("com.facebook.orca", OUTPUT_EXTRACT, "ms")
    if msg_ok and msg_db_root:
        print(Fore.BLUE + "\n📍 Generating SHA-256 Hash Report..." + Style.RESET_ALL)
        generate_hash_report(msg_db_root, OUTPUT_MSG_HASH_REPORT)
        print(Fore.MAGENTA + "\n📍Parsing Messenger Databases..." + Style.RESET_ALL)
        parse_all_databases(msg_db_root, OUTPUT_MSG_CSV)

    print(Fore.GREEN + "\n✅✅✅ Extraction, Hashing, and Parsing Completed!" + Style.RESET_ALL)

    print(Fore.BLUE + """\n
_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._."""+ Style.RESET_ALL)

    # === Hate Speech Detection ===
    print(Fore.BLUE + """\n
                  .88888888888888888888.               .88888888888888888888.
               .888                    888.         .888                    888.
             .88                          88.     .88                          88.
                    .88888888888888888.                 .88888888888888888.
                  .88   .000000000.   88.             .88   .000000000.   88.
                .88    000 00000 000    88.         .88    000 00000 000    88. 
              .88     000 000 000 000     88.     .88     000 000 000 000     88.
                '88    000 00000 000    88'         '88    000 00000 000    88'
                  '88   '000000000'   88'             '88   '000000000'   88'
                    '88888888888888888'                 '88888888888888888'
          
|==============================================================================================|
|                                   Hate Speech Detection                                      |
|==============================================================================================|""" + Style.RESET_ALL)

    tokenizer, model = load_model(MODEL_PATH)
    fb_total, fb_toxic = run_detection_on_csv(OUTPUT_FB_CSV, OUTPUT_FB_PRED, tokenizer, model)
    ig_total, ig_toxic = run_detection_on_csv(OUTPUT_IG_CSV, OUTPUT_IG_PRED, tokenizer, model)
    msg_total, msg_toxic = run_detection_on_csv(OUTPUT_MSG_CSV, OUTPUT_MSG_PRED, tokenizer, model)

    print(Fore.BLUE + """\n
_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._."""+ Style.RESET_ALL)

    # === Final Summary ===
    print(Fore.BLUE + """\n
                  .88888888888888888888.               .88888888888888888888.
               .888                    888.         .888                    888.
             .88                          88.     .88                          88.
                    .88888888888888888.                 .88888888888888888.
                  .88   .000000000.   88.             .88   .000000000.   88.
                .88    000 00000 000    88.         .88    000 00000 000    88. 
              .88     000 000 000 000     88.     .88     000 000 000 000     88.
                '88    000 00000 000    88'         '88    000 00000 000    88'
                  '88   '000000000'   88'             '88   '000000000'   88'
                    '88888888888888888'                 '88888888888888888'
          
|==============================================================================================|
|                               Hate Speech Detection Summary                                  |
|==============================================================================================|""" + Style.RESET_ALL)
    print(Fore.BLUE + """\n
    Facebook :""" + Style.RESET_ALL)
    print(Fore.MAGENTA + f"""  
        👉🏻 Total = {fb_total} 
        👉🏻 Hate Speech = {fb_toxic}
        👉🏻 Safe = {fb_total - fb_toxic}""" + Style.RESET_ALL)
    print(Fore.BLUE + """\n
    Instagram :""" + Style.RESET_ALL)
    print(Fore.MAGENTA + f""" 
        👉🏻 Total = {ig_total}
        👉🏻 Hate Speech = {ig_toxic}
        👉🏻 Safe = {ig_total - ig_toxic}""" + Style.RESET_ALL)
    print(Fore.BLUE + """\n
    Messenger :""" + Style.RESET_ALL)
    print(Fore.MAGENTA + f""" 
        👉🏻 Total = {msg_total}
        👉🏻 Hate Speech = {msg_toxic}
        👉🏻 Safe = {msg_total - msg_toxic}""" + Style.RESET_ALL)
    
    print(Fore.BLUE + """\n
_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._."""+ Style.RESET_ALL)

    print(Fore.GREEN + "\n✅✅✅ Extractions and Hate Speech Detection All Completed!" + Style.RESET_ALL)
    print(Fore.GREEN + "\n🎯 Saved Locations of the Generated Reports :" + Style.RESET_ALL)
    print(Fore.YELLOW +"\n   👉🏻 " + OUTPUT_FB_PRED)
    print(Fore.YELLOW +"\n   👉🏻 " + OUTPUT_IG_PRED)
    print(Fore.YELLOW +"\n   👉🏻 " + OUTPUT_MSG_PRED)

    print(Fore.BLUE + """\n
_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._."""+ Style.RESET_ALL)
    
    # === Final report ===
    report_main()

if __name__ == "__main__":
    # your existing main workflow:
    main()

   


