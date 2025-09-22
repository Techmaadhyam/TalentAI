# app.py
import os, io, json, time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from talentlib import (
    encrypt_bytes, decrypt_bytes, extract_text_from_bytes,
    redact_pii, extract_skills_with_openai, canonicalize,
    semantic_match_all, extract_email
)
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font
from email.message import EmailMessage
import smtplib

st.set_page_config(page_title="TalentMatch AI", layout="wide")

try:
    _ = st.session_state
except Exception:
    st.error("Run with `streamlit run app.py`. Deploy to Streamlit Cloud or run locally via streamlit run.")
    st.stop()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL") or st.secrets.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL") or st.secrets.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

SMTP_SERVER = os.environ.get("SMTP_SERVER") or st.secrets.get("SMTP_SERVER")
SMTP_PORT = int(os.environ.get("SMTP_PORT") or st.secrets.get("SMTP_PORT") or 0)
SMTP_USER = os.environ.get("SMTP_USER") or st.secrets.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD") or st.secrets.get("SMTP_PASSWORD")
FROM_EMAIL = os.environ.get("FROM_EMAIL") or st.secrets.get("FROM_EMAIL")

if "encrypted_files" not in st.session_state:
    st.session_state["encrypted_files"] = []

st.title("💼 TalentMatch AI — Streamlit")

st.sidebar.header("Uploader / Controls")
jd_text = st.sidebar.text_area("Paste Job Description (JD)", height=220)
uploaded = st.sidebar.file_uploader("Upload up to 25 CVs (pdf/docx/txt)", accept_multiple_files=True)
min_score = st.sidebar.slider("Minimum score to display", 0, 100, 0)
skill_filter = st.sidebar.text_input("Skill filter (optional)")
use_weighted = st.sidebar.checkbox("Use weighted scoring (core/optional)", value=True)

if uploaded:
    st.session_state["encrypted_files"] = []
    for f in uploaded[:25]:
        raw = f.read()
        enc = encrypt_bytes(raw)
        st.session_state["encrypted_files"].append({"name": f.name, "bytes": enc})
    st.sidebar.success(f"{len(st.session_state['encrypted_files'])} files encrypted in session.")

def send_email_smtp(to_email: str, subject: str, body: str):
    if not (SMTP_SERVER and SMTP_PORT and SMTP_USER and SMTP_PASSWORD and FROM_EMAIL):
        raise RuntimeError("SMTP not configured (set in Streamlit secrets).")
    msg = EmailMessage()
    msg["Subject"]=subject; msg["From"]=FROM_EMAIL; msg["To"]=to_email
    msg.set_content(body)
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
        s.starttls(); s.login(SMTP_USER, SMTP_PASSWORD); s.send_message(msg)

st.markdown("## Analyze CVs")
if st.button("Analyze CVs"):
    if not jd_text:
        st.error("Provide a Job Description (JD) in the sidebar.")
        st.stop()
    if not st.session_state.get("encrypted_files"):
        st.error("Upload CVs in sidebar first.")
        st.stop()

    jd_redacted = redact_pii(jd_text)
    jd_skills_raw = extract_skills_with_openai(jd_redacted, "JD", api_key=OPENAI_API_KEY, model=OPENAI_CHAT_MODEL)
    jd_skills = canonicalize(jd_skills_raw)
    st.info(f"Extracted JD skills ({len(jd_skills)}): " + ", ".join(jd_skills))

    files = st.session_state["encrypted_files"]
    results=[]
    progress = st.progress(0)
    for i,item in enumerate(files):
        name=item["name"]; enc=item["bytes"]
        try:
            raw = decrypt_bytes(enc)
        except Exception as e:
            st.error(f"Failed to decrypt {name}: {e}"); continue
        text = extract_text_from_bytes(raw, name)
        email = extract_email(text)
        text_red = redact_pii(text)
        cv_skills_raw = extract_skills_with_openai(text_red, "CV", api_key=OPENAI_API_KEY, model=OPENAI_CHAT_MODEL)
        cv_skills = canonicalize(cv_skills_raw)
        matched = semantic_match_all(jd_skills, cv_skills, api_key=OPENAI_API_KEY, embed_model=OPENAI_EMBED_MODEL)
        # simple scoring: matched / total_jd_skills *100
        score = round(len(matched)/max(1,len(jd_skills))*100,2)
        results.append({"CV File": name, "Email": email or "", "Score": score, "Matched Skills": matched, "Missing Skills":[s for s in jd_skills if s not in matched], "All CV Skills": cv_skills})
        progress.progress(int((i+1)/len(files)*100))
    progress.empty()

    # apply filters
    filtered=[r for r in results if r["Score"]>=min_score and (skill_filter=="" or any(skill_filter.lower()==s.lower() for s in r["Matched Skills"]))]
    st.markdown(f"### Showing {len(filtered)} of {len(results)}")

    for r in sorted(filtered, key=lambda x: x["Score"], reverse=True):
        with st.expander(f"{r['CV File']} — Score: {r['Score']} — {r['Email'] or 'No email'}"):
            st.write("**Matched:**", ", ".join(r["Matched Skills"]))
            st.write("**Missing:**", ", ".join(r["Missing Skills"]))
            st.write("**All CV Skills:**", ", ".join(r["All CV Skills"]))
            c1,c2,c3 = st.columns([1,1,2])
            with c1:
                if r["Email"]:
                    if st.button(f"Send Approve to {r['CV File']}", key=f"approve_{r['CV File']}"):
                        try:
                            send_email_smtp(r["Email"], "Application Approved", f"Your application {r['CV File']} is approved.")
                            st.success("Approval email sent.")
                        except Exception as e:
                            st.error(f"Email failed: {e}")
            with c2:
                if r["Email"]:
                    if st.button(f"Send Reject to {r['CV File']}", key=f"reject_{r['CV File']}"):
                        try:
                            send_email_smtp(r["Email"], "Application Update", f"Your application {r['CV File']} was not selected.")
                            st.success("Rejection email sent.")
                        except Exception as e:
                            st.error(f"Email failed: {e}")
            with c3:
                st.download_button(f"Download metadata for {r['CV File']}", data=json.dumps(r, indent=2), file_name=f"{r['CV File']}_meta.json", mime="application/json")

    # Top 5 plot
    top5 = sorted(filtered, key=lambda x: x["Score"], reverse=True)[:5]
    if top5:
        names=[r["CV File"] for r in top5]; scores=[r["Score"] for r in top5]
        fig,ax=plt.subplots(); ax.barh(names, scores); ax.invert_yaxis(); ax.set_xlabel("Score"); ax.set_title("Top 5")
        st.pyplot(fig)

    # Export CSV / Excel
    if filtered:
        df = pd.DataFrame(filtered)
        st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="talentmatch_results.csv", mime="text/csv")
        wb = Workbook(); ws = wb.active; ws.title = "Results"; headers=["CV File","Email","Score","Matched Skills","Missing Skills"]; ws.append(headers)
        for idx,r in enumerate(sorted(filtered, key=lambda x:x["Score"], reverse=True), start=1):
            ws.append([r["CV File"], r["Email"], r["Score"], ", ".join(r["Matched Skills"]), ", ".join(r["Missing Skills"])])
        out = io.BytesIO(); wb.save(out); out.seek(0)
        st.download_button("Download Excel", data=out, file_name="talentmatch_ranked.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
