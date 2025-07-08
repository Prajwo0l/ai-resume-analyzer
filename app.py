import streamlit as st
import fitz
import os
import re
import tempfile
import requests
from sentence_transformers import SentenceTransformer, util
from utils import extract_entities
from streamlit_extras.colored_header import colored_header
from streamlit_extras.mention import mention
from streamlit_extras.card import card

# Load model
model = SentenceTransformer('stsb-roberta-large')

# Session state to manage run control
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False

# Local AI Suggestion Function via Ollama
def get_ai_suggestions(resume_text, job_desc):
    prompt = f"""You are an expert resume reviewer.\n\nHere is a job description:\n{job_desc}\n\nAnd here is a resume:\n{resume_text}\n\nGive 3 personalized, actionable suggestions to improve the resume to better match the job. Return each suggestion as a short sentence, separated by new lines."""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False}
        )
        output = response.json().get('response', '')
        return [s.strip('-• ') for s in output.split('\n') if s.strip()]
    except Exception as e:
        return ["⚠️ Could not fetch AI suggestions. Is Ollama running?"]

st.set_page_config(page_title="AI Resume Analyzer", layout="centered", page_icon="📄")

colored_header(label="📄 AI Resume Analyzer", description="Optimize your resume by matching it with a job description.", color_name="blue-70")

with st.container():
    st.markdown("### 📤 Upload Resume")
    resume_file = st.file_uploader("Upload Resume (PDF):", type=["pdf"], help="Your resume must be in PDF format.")
    job_desc = st.text_area("Paste Job Description:", height=200, key="jd_text")

ai_toggle = st.checkbox("💡 Use AI-generated suggestions (via Ollama)", value=True)

# Run analysis only after button is clicked
if st.button("🔍 Analyze Match"):
    st.session_state.analyze_clicked = True

if resume_file and job_desc and st.session_state.analyze_clicked:
    with st.spinner("Analyzing resume and job description..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(resume_file.read())
            tmp_path = tmp_file.name

        text = ""
        doc = fitz.open(tmp_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        os.remove(tmp_path)
        resume_text = text.strip()

        embeddings = model.encode([resume_text, job_desc], convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        match_score = round(similarity_score * 100, 2)

        resume_entities = extract_entities(resume_text)
        resume_skills = set(w.lower() for w in resume_entities.get('SKILL', []))
        jd_words = set(re.findall(r'\b\w{4,}\b', job_desc.lower()))

        matched_skills = sorted(jd_words & resume_skills)
        missing_skills = sorted(jd_words - resume_skills)[:10]

        if ai_toggle:
            suggestions = get_ai_suggestions(resume_text, job_desc)
        else:
            suggestions = []
            if "flask" in jd_words and "flask" not in resume_text.lower():
                suggestions.append("Mention your experience working with Flask or similar web frameworks.")
            if "sql" in jd_words and "sql" not in resume_text.lower():
                suggestions.append("Include your experience with SQL databases.")
            if "api" in jd_words and "api" not in resume_text.lower():
                suggestions.append("Describe any projects where you built or worked with APIs.")
            if len(resume_text.split()) < 150:
                suggestions.append("Your resume seems short. Add more details about your experience, skills, or projects.")
            if not any(heading in resume_text.lower() for heading in ["experience", "work", "project"]):
                suggestions.append("Consider adding an 'Experience' or 'Projects' section to highlight your background.")

        st.markdown("---")
        colored_header("✅ Match Results", description="Here’s how your resume aligns with the job description.", color_name="green-70")

        col1, col2 = st.columns(2)
        col1.metric("📊 Match Score", f"{match_score}%", delta_color="normal")
        col2.markdown("<br>", unsafe_allow_html=True)
        col2.progress(min(1.0, similarity_score))

        st.markdown("### 🛠️ Matched Skills")
        if matched_skills:
            st.success(", ".join(matched_skills))
        else:
            st.info("No matched skills were found between the resume and job description.")

        st.markdown("### ❌ Missing Skills (Top 10)")
        if missing_skills:
            st.warning(", ".join(missing_skills))
        else:
            st.info("No missing skills detected.")

        st.markdown("### 💡 Suggestions to Improve Resume")
        if suggestions:
            for i, s in enumerate(suggestions, 1):
                card(title=f"Suggestion {i}", text=s, styles={"card": {"padding": "0.5rem"}})
        else:
            st.success("Your resume is well-aligned with the job description! 🥳")

        st.markdown("---")
        mention(label="Built with Streamlit & Transformers", icon="🤖", url="https://streamlit.io")
