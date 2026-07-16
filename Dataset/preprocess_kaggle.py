"""
Kaggle Data Preprocessor — csv.DictReader for huge summary file
================================================================
Uses csv.DictReader (fast C implementation) for the 4.8GB summary file.
"""

import os, re, gc, csv
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))

POSTINGS = os.path.join(DATASET_DIR, "linkedin_job_postings.csv")
SKILLS   = os.path.join(DATASET_DIR, "job_skills.csv")
SUMMARY  = os.path.join(DATASET_DIR, "job_summary.csv")
OUTPUT   = os.path.join(DATA_DIR, "it_jobs_processed.csv")

CHUNK = 200_000

_STATE_MAP = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}
# Reverse map: abbreviations like "ca", "ny" → "CA", "NY"
_ABBREV_MAP = {v.lower(): v for v in _STATE_MAP.values()}

IT_KW = ["software", "engineer", "developer", "data scien", "machine learning",
         "ai ", "devops", "cloud", "full stack", "frontend", "backend",
         "mobile", "security", "qa ", "test", "sre", "platform", "infrastructure",
         "python", "java", "javascript", "react", "node", "golang"]

SKILL_KW = {
    "programming": ["python", "java", "javascript", "typescript", "c++", "c#", "ruby", "golang", "rust", "php"],
    "cloud":       ["aws", "azure", "gcp", "google cloud", "terraform", "cloud"],
    "ai_ml":       ["machine learning", "deep learning", "artificial intelligence", "nlp", "computer vision", "tensorflow", "pytorch", "keras", "scikit"],
    "database":    ["sql", "mysql", "postgresql", "oracle", "mongodb", "redis", "elasticsearch"],
    "devops":      ["docker", "kubernetes", "jenkins", "ci/cd", "github actions", "ansible"],
    "framework":   ["react", "angular", "vue", "next.js", "node.js", "django", "flask", "fastapi", "spring", ".net"],
    "data_engineering": ["apache spark", "hadoop", "kafka", "airflow", "etl"],
    "security":    ["cybersecurity", "penetration testing", "vulnerability", "encryption", "firewall"],
    "soft_skills": ["communication", "leadership", "teamwork", "agile", "scrum"],
}

SALARY_RE = re.compile(r'\$([0-9]{2,3})[kK]\b|\$([0-9]{2,3}),[0-9]{3}\b', re.IGNORECASE)


def ns(loc):
    if not loc: return "Unknown"
    loc = str(loc).strip()
    if len(loc) <= 3 and loc.isalpha() and loc.isupper(): return loc
    tail = loc.split(",")[-1].strip().lower()
    if tail in _STATE_MAP: return _STATE_MAP[tail]
    if tail in _ABBREV_MAP: return _ABBREV_MAP[tail]
    if loc.lower() in _STATE_MAP: return _STATE_MAP[loc.lower()]
    return "Remote"


def nsen(level, title):
    # Extract seniority from job_title first (more reliable than Kaggle's job_level)
    t = str(title).lower() if title else ""
    if any(k in t for k in ("intern", "internship", "entry level", "entry-level", "junior", "jr", "graduate", "trainee", "apprentice")): return "Junior"
    if any(k in t for k in ("manager", "director", "head of", "vp ")): return "Manager"
    if any(k in t for k in ("senior", "sr ", "lead", "principal", "staff", "architect", "expert", "vp of", "head", "fellow")): return "Senior"
    # Fallback to job_level if title has no seniority signal
    v = str(level).lower().strip() if level else ""
    if any(k in v for k in ("intern", "entry", "junior", "graduate")): return "Junior"
    if "associate" in v: return "Junior"
    if any(k in v for k in ("manager", "director", "head", "vp")): return "Manager"
    if "mid" in v: return "Mid"
    if any(k in v for k in ("senior", "lead", "principal", "staff", "executive")): return "Senior"
    return "Mid"


def md(title):
    t = str(title).lower()
    if any(k in t for k in ["data", "ai", "machine learning", "nlp"]): return "Data Science"
    if any(k in t for k in ["cloud", "devops", "sre", "system", "network", "infrastructure"]): return "DevOps/SRE"
    if any(k in t for k in ["security", "cyber"]): return "Cybersecurity"
    if any(k in t for k in ["test", "qa", "quality"]): return "QA/Testing"
    return "Software Engineering"


def njt(v):
    v = str(v).lower().strip() if v else ""
    if "remote" in v: return "Remote"
    if "hybrid" in v: return "Hybrid"
    if "full" in v: return "Full-time"
    if "part" in v: return "Part-time"
    if "contract" in v: return "Contract"
    return "Onsite" if v else "Unknown"


def count_skill(text, kws):
    if not text: return 0
    t = str(text).lower()
    return sum(1 for kw in kws if re.search(r'\b' + re.escape(kw.strip()) + r'\b', t))


def extract_salary(text):
    if not text: return np.nan
    m = SALARY_RE.findall(str(text))
    if m:
        vals = [float(a)*1000 if a else float(b)*1000 for a, b in m if a or b]
        return np.mean(vals) if vals else np.nan
    return np.nan


def process():
    print("=" * 60, flush=True)
    print("KAGGLE PREPROCESSOR (csv.DictReader)", flush=True)
    print("=" * 60, flush=True)

    # --- 1. Postings ---
    print("\n[1/4] Loading postings...", flush=True)
    df_post = pd.read_csv(POSTINGS, usecols=["job_link", "job_title", "company", "job_location", "job_level", "job_type"], low_memory=False)
    print(f"  {len(df_post)} rows", flush=True)

    title_lower = df_post["job_title"].str.lower()
    it_mask = title_lower.apply(lambda t: any(k in t for k in IT_KW) if pd.notna(t) else False)
    df_it = df_post[it_mask].copy()
    del df_post; gc.collect()
    it_links = set(df_it["job_link"].values)
    print(f"  IT: {len(df_it)} ({len(df_it)/max(1,len(title_lower))*100:.0f}%)", flush=True)

    # --- 2. Skills (pandas chunks) ---
    print("\n[2/4] Loading skills...", flush=True)
    skill_map = {}
    rows_read = 0
    for chunk in pd.read_csv(SKILLS, usecols=["job_link", "job_skills"], chunksize=CHUNK, low_memory=False):
        rows_read += len(chunk)
        sub = chunk[chunk["job_link"].isin(it_links)]
        for _, r in sub.iterrows():
            skill_map[r["job_link"]] = r.get("job_skills", "")
        if rows_read % (CHUNK * 5) == 0:
            print(f"  ...{rows_read/1e6:.1f}M scanned, {len(skill_map)} matched", flush=True)
    print(f"  Skills: {len(skill_map)}", flush=True)
    del chunk; gc.collect()

    # --- 3. Summary (csv.DictReader — fast C loop) ---
    print("\n[3/4] Loading summary (csv.DictReader)...", flush=True)
    sum_map = {}
    rows_read = 0
    with open(SUMMARY, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_read += 1
            jlink = row.get("job_link", "")
            if jlink in it_links:
                sum_map[jlink] = row.get("job_summary", "")
            if rows_read % 2_000_000 == 0:
                print(f"  ...{rows_read/1e6:.1f}M scanned, {len(sum_map)} matched", flush=True)
    print(f"  Summaries: {len(sum_map)}", flush=True)

    # --- 4. Build ---
    print("\n[4/4] Building dataset...", flush=True)
    rows = []
    for _, post in df_it.iterrows():
        link = post["job_link"]
        title = post.get("job_title", "")
        level = post.get("job_level", "")
        jtype = post.get("job_type", "")
        loc = post.get("job_location", "")
        company = post.get("company", "")
        sk = skill_map.get(link, "")
        sm = sum_map.get(link, "")
        combined = f"{sm} {sk}"

        sen = nsen(level, title)
        st = ns(loc)
        dom = md(title)
        jt = njt(jtype)
        sal = extract_salary(sm)

        feats = {f"skill_{c}": count_skill(combined, kws) for c, kws in SKILL_KW.items()}
        feats["num_skills"] = sum(feats.values())
        feats["skill_diversity"] = sum(1 for v in feats.values() if v > 0)

        rows.append({
            "job_link": link, "job_title": title, "company": company,
            "state": st, "it_domain": dom, "seniority_level": sen, "job_type": jt,
            "salary_annual": sal, **feats,
        })

        if len(rows) % 50_000 == 0:
            print(f"  ...{len(rows)} rows built", flush=True)

    df = pd.DataFrame(rows)
    print(f"  Total: {len(df)}", flush=True)

    # Outlier
    sal = df["salary_annual"].dropna()
    print(f"  Salaries: {len(sal)}", flush=True)
    if len(sal) >= 50:
        q1, q3 = sal.quantile(0.25), sal.quantile(0.75)
        iqr = q3 - q1
        lo, hi = max(q1 - 1.5 * iqr, 15000), min(q3 + 1.5 * iqr, 500000)
        n_before = len(df)
        df = df[(df["salary_annual"].isna()) | ((df["salary_annual"] >= lo) & (df["salary_annual"] <= hi))]
        print(f"  Outliers removed: {n_before - len(df)}", flush=True)

    # Save
    cols = ["job_link", "job_title", "company", "state", "it_domain", "seniority_level", "job_type",
            "salary_annual", "num_skills", "skill_diversity", "skill_programming", "skill_cloud",
            "skill_ai_ml", "skill_database", "skill_devops", "skill_framework", "skill_data_engineering",
            "skill_security", "skill_soft_skills"]
    df = df[[c for c in cols if c in df.columns]]
    sc = [c for c in df.columns if c.startswith("skill_") or c in ("num_skills", "skill_diversity")]
    df[sc] = df[sc].fillna(0).astype(int)
    df.to_csv(OUTPUT, index=False)

    print(f"\n{'='*60}", flush=True)
    print(f"  Output: {OUTPUT}", flush=True)
    print(f"  Rows:   {len(df)}", flush=True)
    print(f"  Seniority: {dict(df['seniority_level'].value_counts())}", flush=True)
    print(f"  State (top10): {dict(df['state'].value_counts().head(10))}", flush=True)
    print(f"  Job type: {dict(df['job_type'].value_counts())}", flush=True)
    print(f"  Domain: {dict(df['it_domain'].value_counts())}", flush=True)
    print(f"  Salaries: {df['salary_annual'].notna().sum()}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    process()
