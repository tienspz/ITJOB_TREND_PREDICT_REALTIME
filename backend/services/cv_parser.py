"""
CV Parser Service
=================

Extracts text from PDF/Images and extracts skills to predict salary.
Uses pdfplumber for PDFs and OpenCV + easyocr/pytesseract for images.
"""

import os
import re
import logging
import joblib
import pandas as pd
import numpy as np
import pdfplumber

from backend import config

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
except ImportError:
    easyocr = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

def extract_text_from_pdf(filepath):
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
    return text

def extract_text_from_image(filepath):
    text = ""
    if cv2 is None or easyocr is None:
        logger.warning("cv2 or easyocr not installed, skipping image OCR.")
        return text
        
    try:
        img = cv2.imread(filepath)
        # Preprocessing to improve OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Thresholding
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        results = reader.readtext(thresh)
        for bbox, t, prob in results:
            text += t + " "
    except Exception as e:
        logger.error(f"Error running OCR: {e}")
    return text

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

def parse_cv_and_predict(filepath, ext, model=None):
    if ext == ".pdf":
        text = extract_text_from_pdf(filepath)
    else:
        text = extract_text_from_image(filepath)
        
    if not text.strip():
        return {"error": "Could not extract text from the file. It might be empty, password-protected, or unreadable."}
        
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
        "predicted_salary": round(salary_pred, 2)
    }
