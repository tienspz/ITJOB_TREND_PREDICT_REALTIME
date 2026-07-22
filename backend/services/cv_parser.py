"""
CV Parser Service
=================

Extracts text from PDF/Images and extracts skills to predict salary.
Uses pdfplumber for PDFs and a multi-engine OCR pipeline for images:
  1. EasyOCR (primary) — works without Tesseract binary
  2. PyTesseract (fallback) — requires system Tesseract installation
  3. Both engines run on the original + contrast-enhanced image for best results
"""

import os
import re
import logging
import joblib
import pandas as pd
import numpy as np

from backend import config

# ---------------------------------------------------------------------------
# Lazy / optional imports — gracefully degrade when a library is missing
# ---------------------------------------------------------------------------
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    Image = None

try:
    import easyocr
    _easyocr_reader = easyocr.Reader(['en', 'vi'], gpu=False)
except Exception:
    easyocr = None
    _easyocr_reader = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# "5+ years of experience", "8 yrs software experience"
YOE_RE = re.compile(
    r'(\d{1,2})\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:\w+\s+){0,2}?experience', re.IGNORECASE)

# Load metadata for categories
try:
    meta = joblib.load(os.path.join(config.MODELS_DIR, "salary_model_meta.joblib"))
except Exception as e:
    logger.warning(f"Could not load metadata: {e}")
    meta = {"it_domain": ["Software Engineering"], "state": ["CA"], "job_type": ["Remote"], "seniority_level": ["Mid"]}

# Full list of IT skills to search for in CVs
# (A real app would use a more exhaustive dict, we use our categories)
SKILL_CATEGORIES = {
    "programming": ["python", "java", "javascript", "typescript", "c++", "c#", "ruby", "golang", "go ", "rust", "sql", "html", "css"],
    "cloud": ["aws", "azure", "gcp", "google cloud", "terraform", "kubernetes"],
    "ai_ml": ["machine learning", "deep learning", "artificial intelligence", "nlp", "computer vision", "tensorflow", "pytorch", "keras", "scikit", "xgboost"],
    "database": ["sql", "mysql", "postgresql", "oracle", "mongodb", "redis", "elasticsearch"],
    "devops": ["docker", "kubernetes", "jenkins", "ci/cd", "github actions", "gitlab", "ansible", "terraform"],
    "framework": ["react", "angular", "vue", "next.js", "node.js", "express", "django", "flask", "fastapi", "spring", "bootstrap", "tailwind"],
    "data_engineering": ["apache spark", "hadoop", "kafka", "airflow", "etl", "dbt", "talend"],
    "security": ["cybersecurity", "penetration testing", "vulnerability", "encryption", "firewall"],
    "soft_skills": ["communication", "leadership", "teamwork", "problem solving", "agile", "scrum"]
}


# ---------------------------------------------------------------------------
# Image reading helpers — handle Unicode paths and PIL/cv2 interop
# ---------------------------------------------------------------------------
def _read_image_pil(filepath):
    """Read image as PIL.Image, handling Unicode file paths safely."""
    if Image is None:
        return None
    try:
        img = Image.open(filepath)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.error(f"PIL could not open image: {e}")
        return None


def _pil_to_cv2(pil_img):
    """Convert PIL Image to OpenCV BGR numpy array."""
    if cv2 is None or pil_img is None:
        return None
    import numpy as _np
    arr = _np.array(pil_img)
    # RGB → BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _enhance_for_ocr(pil_img):
    """Return a list of preprocessed PIL images to try OCR on."""
    variants = [pil_img]
    if Image is None:
        return variants
    try:
        # High contrast grayscale
        gray = pil_img.convert("L")
        enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
        variants.append(enhanced)

        # Adaptive-like: sharpen + slight resize up for small text
        w, h = pil_img.size
        if w < 1500:
            scale = 1500 / w
            big = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            variants.append(big)
    except Exception as e:
        logger.debug(f"Image enhancement variant failed: {e}")
    return variants


# ---------------------------------------------------------------------------
# OCR engines
# ---------------------------------------------------------------------------
def _ocr_easyocr(pil_img):
    """Run EasyOCR on a PIL image and return extracted text."""
    if _easyocr_reader is None or pil_img is None:
        return ""
    try:
        import numpy as _np
        arr = _np.array(pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img)
        results = _easyocr_reader.readtext(arr)
        return " ".join(t for _, t, prob in results if prob > 0.15)
    except Exception as e:
        logger.error(f"EasyOCR failed: {e}")
        return ""


def _ocr_pytesseract(pil_img):
    """Run PyTesseract on a PIL image and return extracted text."""
    if pytesseract is None or pil_img is None:
        return ""
    try:
        # Try Vietnamese + English first, fall back to English only
        try:
            text = pytesseract.image_to_string(pil_img, lang="vie+eng")
        except Exception:
            text = pytesseract.image_to_string(pil_img)
        return text
    except Exception as e:
        logger.error(f"PyTesseract failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Text extraction — PDF and images
# ---------------------------------------------------------------------------
def extract_text_from_pdf(filepath):
    """Extract text from a PDF using pdfplumber, with OCR fallback for
    scanned/image-only PDFs."""
    text = ""
    if pdfplumber is None:
        logger.warning("pdfplumber not installed, cannot extract PDF text.")
        return text
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    # Scanned page — try to OCR its rendered image
                    try:
                        page_img = page.to_image(resolution=300).original
                        ocr_text = _ocr_easyocr(page_img) or _ocr_pytesseract(page_img)
                        if ocr_text:
                            text += ocr_text + "\n"
                    except Exception as e:
                        logger.debug(f"PDF page OCR fallback failed: {e}")
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
    return text


def extract_text_from_image(filepath):
    """Multi-engine OCR: tries EasyOCR then PyTesseract on the original and
    contrast-enhanced versions of the image. Returns the best (longest) result."""
    pil_img = _read_image_pil(filepath)
    if pil_img is None:
        logger.warning(f"Could not open image: {filepath}")
        return ""

    variants = _enhance_for_ocr(pil_img)
    best_text = ""

    for variant in variants:
        # EasyOCR
        t = _ocr_easyocr(variant)
        if len(t.strip()) > len(best_text.strip()):
            best_text = t

        # PyTesseract
        t = _ocr_pytesseract(variant)
        if len(t.strip()) > len(best_text.strip()):
            best_text = t

        # If we already have a decent amount of text, stop early
        if len(best_text.strip()) > 100:
            break

    return best_text


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------
def extract_skills(text):
    text_lower = text.lower()
    found_skills = []
    features = {
        "num_skills": 0,
        "skill_programming": 0,
        "skill_cloud": 0,
        "skill_ai_ml": 0,
        "skill_database": 0,
        "skill_devops": 0,
        "skill_framework": 0,
        "skill_data_engineering": 0,
        "skill_security": 0,
        "skill_soft_skills": 0,
    }
    
    for category, keywords in SKILL_CATEGORIES.items():
        cat_count = 0
        for kw in keywords:
            # Need word boundaries to avoid partial matches like 'go' in 'good'
            if re.search(r'\b' + re.escape(kw.strip()) + r'\b', text_lower):
                found_skills.append(kw.strip().title())
                cat_count += 1
                features["num_skills"] += 1

        # Keyword counts per category — same semantics as the training data
        features[f"skill_{category}"] = cat_count
        
    features["skill_diversity"] = sum(1 for k, v in features.items() if k.startswith("skill_") and k != "skill_diversity" and v > 0)
    
    return found_skills, features


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def _build_prediction(text, model=None):
    """Shared prediction logic used by both file-upload and text-based parsing."""
    found_skills, features = extract_skills(text)
    
    # Try to extract seniority level
    seniority = "Mid"
    text_lower = text.lower()
    if any(x in text_lower for x in ["senior", "lead", "principal"]):
        seniority = "Senior"
    elif any(x in text_lower for x in ["manager", "director", "head"]):
        seniority = "Manager"
    elif any(x in text_lower for x in ["junior", "entry", "intern"]):
        seniority = "Junior"
        
    # Years of experience: regex from CV text, else infer from seniority
    years_experience = None
    m = YOE_RE.search(text)
    if m:
        y = int(m.group(1))
        if 0 <= y <= 40:
            years_experience = y
    if years_experience is None:
        years_experience = {"Junior": 1, "Mid": 3, "Senior": 8, "Manager": 12}.get(seniority, 3)
    features["years_experience"] = years_experience

    # We need defaults for the model since a CV doesn't have a specific job location/type
    known_seniority = meta.get("seniority_level", [])
    features["seniority_level"] = seniority if seniority in known_seniority else (known_seniority[0] if known_seniority else "Mid")
    features["job_type"] = "Remote"
    features["state"] = "CA" if "CA" in meta.get("state", []) else meta["state"][0]
    features["it_domain"] = "Software Engineering" if "Software Engineering" in meta.get("it_domain", []) else meta["it_domain"][0]
    # Engineered interaction features expected by the trained pipeline
    features["domain_seniority"] = features["it_domain"] + "_" + features["seniority_level"]
    features["state_seniority"] = features["state"] + "_" + features["seniority_level"]

    df_features = pd.DataFrame([features])
    
    try:
        if model is None:
            model = joblib.load(os.path.join(config.MODELS_DIR, "best_salary_model.joblib"))
        salary_pred = float(model.predict(df_features)[0])
    except Exception as e:
        logger.error(f"Error predicting salary: {e}")
        salary_pred = 0
        
    return {
        "extracted_text_length": len(text),
        "skills_found": found_skills,
        "inferred_seniority": seniority,
        "years_experience": years_experience,
        "predicted_salary": round(salary_pred, 2)
    }


def parse_cv_and_predict(filepath, ext, model=None):
    """Extract text from a CV file (PDF or image) and predict salary."""
    if ext == ".pdf":
        text = extract_text_from_pdf(filepath)
    else:
        text = extract_text_from_image(filepath)
        
    if not text.strip():
        return {"error": "Could not extract text from the file. It might be empty, password-protected, or unreadable."}
        
    return _build_prediction(text, model)


def parse_cv_text_and_predict(text, model=None):
    """Predict salary from pre-extracted CV text (e.g. from client-side OCR)."""
    if not text or not text.strip():
        return {"error": "No text provided."}
    return _build_prediction(text, model)
