# talentlib.py
import io, re, json
from typing import List
from PyPDF2 import PdfReader
import docx
from cryptography.fernet import Fernet

# Fernet key generation / wrapper (for prototype)
# In production: replace with KMS-managed key
_FERNET = Fernet(Fernet.generate_key())

def encrypt_bytes(b: bytes) -> bytes:
    return _FERNET.encrypt(b)

def decrypt_bytes(b: bytes) -> bytes:
    return _FERNET.decrypt(b)

def extract_text_from_pdf_bytes(b: bytes) -> str:
    bio = io.BytesIO(b)
    reader = PdfReader(bio)
    pages = []
    for p in reader.pages:
        try:
            t = p.extract_text()
        except Exception:
            t = None
        if t:
            pages.append(t)
    return " ".join(pages)

def extract_text_from_docx_bytes(b: bytes) -> str:
    bio = io.BytesIO(b)
    doc = docx.Document(bio)
    return " ".join([para.text for para in doc.paragraphs])

def extract_text_from_bytes(b: bytes, filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(b)
    if name.endswith(".docx") or name.endswith(".doc"):
        return extract_text_from_docx_bytes(b)
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""

email_regex = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
def extract_email(text: str) -> str:
    if not text:
        return ""
    m = email_regex.search(text)
    return m.group(0) if m else ""

def redact_pii(text: str) -> str:
    if not text:
        return text
    text = re.sub(email_regex, "[EMAIL]", text)
    text = re.sub(r"\b\d{10}\b", "[PHONE]", text)
    return text

# --- OpenAI wrappers: the app uses HTTP endpoints, keep it lightweight ---
import os, requests
API_URL = "https://api.openai.com/v1"
def call_openai_chat(prompt: str, api_key: str, model: str="gpt-4o-mini", temperature: float=0.0, max_tokens:int=800) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages":[{"role":"user","content":prompt}], "temperature":temperature, "max_tokens": max_tokens}
    r = requests.post(f"{API_URL}/chat/completions", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    choice = j.get("choices", [])[0]
    if isinstance(choice.get("message"), dict):
        return choice["message"].get("content","")
    return choice.get("text","")

def call_openai_embeddings(texts, api_key: str, model: str="text-embedding-3-small"):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if isinstance(texts, str):
        inputs = [texts]
    else:
        inputs = texts
    payload = {"model": model, "input": inputs}
    r = requests.post(f"{API_URL}/embeddings", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    return [d["embedding"] for d in j["data"]]

# Example skill extractor — you may replace with your notebook implementation
def extract_skills_with_openai(text: str, role_hint: str, api_key: str, model: str="gpt-4o-mini") -> List[str]:
    if not text or not api_key:
        return []
    prompt = f"Extract distinct technical and domain skills from the following {role_hint} text. Return ONLY a JSON array of skill strings.\n\nText:\n\"\"\"{text}\"\"\""
    try:
        out = call_openai_chat(prompt, api_key=api_key, model=model, temperature=0)
    except Exception:
        return []
    # parse JSON array
    start = out.find('['); end = out.rfind(']')+1
    if start != -1 and end != -1:
        try:
            arr = json.loads(out[start:end])
            skills = [s.strip() for s in arr if isinstance(s, str)]
            return [s for i,s in enumerate(dict.fromkeys(skills))]  # dedupe preserve order
        except Exception:
            return []
    return []

# Simple canonicalization; swap or expand with your notebook knowledge
def canonicalize(skills: List[str]) -> List[str]:
    canon_map = {"py":"Python","python":"Python","ml":"Machine Learning","sql":"SQL"}
    out=[]
    seen=set()
    for s in skills:
        key=s.lower().strip()
        mapped = canon_map.get(key, s.title() if s.islower() else s)
        if mapped.lower() not in seen:
            out.append(mapped); seen.add(mapped.lower())
    return out

# Basic semantic matching driver — you may paste your notebook's smarter algorithm here
import numpy as np
def cosine_similarity(a,b):
    a=np.array(a); b=np.array(b)
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na==0 or nb==0: return 0.0
    return float(np.dot(a,b)/(na*nb))

# Simple hierarchical maps for domain synonyms (expand from your notebook)
DOMAIN_MAP = {
    "data_warehousing": ["data warehouse","redshift","snowflake","bigquery","dbt","data lake","data pipelines"],
    "etl_tools": ["etl","informatica","talend","aws glue","glue","nifi"],
    "orchestration": ["airflow","prefect","luigi","scheduler","cron", "orchestration"],
    "optimization": ["optimization","optimise","performance tuning","cost optimization", "performance"]
}
DOMAIN_MAP_LOWER = {k:[s.lower() for s in v] for k,v in DOMAIN_MAP.items()}

def semantic_match_all(jd_skills, cv_skills, api_key=None, embed_model="text-embedding-3-small", threshold=0.74):
    matched=set()
    if not jd_skills: return []
    cv_lower=[c.lower() for c in cv_skills]

    # exact / substring
    for js in jd_skills:
        jsl=js.lower()
        if any(jsl==c for c in cv_lower): matched.add(js); continue
        if any(jsl in c or c in jsl for c in cv_lower): matched.add(js); continue

    # domain mapping
    for js in jd_skills:
        jsl=js.lower()
        for domain,members in DOMAIN_MAP_LOWER.items():
            if any(dm in jsl for dm in members) or domain.replace("_"," ") in jsl:
                if any(any(m in c for m in members) for c in cv_lower):
                    matched.add(js); break

    # optimization stem
    if any("optimiz" in s.lower() or "optimis" in s.lower() or "tuning" in s.lower() for s in jd_skills):
        if any("optimiz" in c or "performance" in c or "tuning" in c for c in cv_lower):
            for s in jd_skills:
                if "optimiz" in s.lower() or "optimis" in s.lower() or "tuning" in s.lower():
                    matched.add(s)

    # embeddings semantic fallback
    remaining = [s for s in jd_skills if s not in matched]
    if remaining and cv_skills and api_key:
        try:
            texts = remaining + cv_skills
            embs = call_openai_embeddings(texts, api_key=api_key, model=embed_model)
            jd_embs = embs[:len(remaining)]
            cv_embs = embs[len(remaining):]
            for i,j in enumerate(remaining):
                best=0; bestcv=None
                for k,c in enumerate(cv_skills):
                    sim = cosine_similarity(jd_embs[i], cv_embs[k])
                    if sim>best:
                        best=sim; bestcv=c
                if best>=threshold:
                    matched.add(j)
        except Exception:
            pass

    return [s for s in jd_skills if s in matched]

