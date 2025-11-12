# main.py
# AI Career & Study Companion - single-file version
# COPY-PASTE this entire file into main.py
#
# Requirements (install once in your venv):
# pip install streamlit pdfplumber transformers sentence-transformers spacy nltk matplotlib fpdf
# python -m spacy download en_core_web_sm
# (If memory is low, you can remove sentence-transformers usage; code will still run for basic features.)

import streamlit as st
import pdfplumber
import sqlite3
import re
import json
import os
from datetime import datetime, timedelta
from random import choice, sample, shuffle
import nltk
nltk.download('punkt', quiet=True)

# Try importing heavy libraries; give friendly message if missing.
have_transformers = True
have_sbert = True
have_spacy = True
try:
    from transformers import pipeline
except Exception as e:
    have_transformers = False

try:
    from sentence_transformers import SentenceTransformer, util
except Exception as e:
    have_sbert = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    have_spacy = False
    nlp = None

# --------------------------
# Simple in-file job roles (sample)
# You can expand these later or replace with a JSON file.
# --------------------------
JOB_ROLES = {
    "Data Scientist": ["Python", "Machine Learning", "Pandas", "NumPy", "Statistics", "SQL"],
    "Frontend Developer": ["HTML", "CSS", "JavaScript", "React", "Bootstrap"],
    "Backend Developer": ["Python", "Django", "Flask", "SQL", "APIs"],
    "ML Engineer": ["Python", "TensorFlow", "PyTorch", "Model Deployment", "Docker"],
    "QA Engineer": ["Testing", "Selenium", "Automation", "Python", "API Testing"]
}

# --------------------------
# Database utilities (sqlite)
# --------------------------
DB_PATH = "data/app.db"
os.makedirs("data", exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions
        (id INTEGER PRIMARY KEY, mode TEXT, title TEXT, content TEXT, created TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    ''')
    conn.commit()
    conn.close()

def save_session(mode, title, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO sessions (mode, title, content) VALUES (?, ?, ?)', (mode, title, content))
    conn.commit()
    conn.close()

init_db()

# --------------------------
# PDF / Text extraction utility
# --------------------------
def extract_text_from_pdf_filelike(filelike):
    try:
        with pdfplumber.open(filelike) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
    except Exception as e:
        return ""

# Simple text cleanup
def clean_text(t):
    return re.sub(r'\s+', ' ', t).strip()

# --------------------------
# Student module: summarizer, quiz generator, revision schedule
# --------------------------
# Initialize summarizer lazily to avoid long startup
_summarizer = None
def get_summarizer():
    global _summarizer
    if _summarizer is None:
        if not have_transformers:
            _summarizer = None
            return None
        # use a smallish summarization model that is good on CPU
        try:
            _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception:
            try:
                _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            except Exception:
                _summarizer = None
    return _summarizer

def summarize_text(text, max_length=200):
    text = clean_text(text)
    if len(text.split()) < 50:
        return text  # too short to summarize
    summ = get_summarizer()
    if summ is None:
        # fallback naive summary: first 3 sentences
        sents = nltk.tokenize.sent_tokenize(text)
        return " ".join(sents[:3])
    # chunk if very long
    words = text.split()
    if len(words) > 1000:
        chunks = [" ".join(words[i:i+1000]) for i in range(0, len(words), 1000)]
        parts = []
        for ch in chunks:
            out = summ(ch, max_length=max_length, min_length=40, do_sample=False)
            parts.append(out[0]['summary_text'])
        return "\n\n".join(parts)
    else:
        out = summ(text, max_length=max_length, min_length=40, do_sample=False)
        return out[0]['summary_text']

# Simple quiz generator (rule-based)
def generate_simple_mcqs(text, num_q=5):
    sents = nltk.tokenize.sent_tokenize(text)
    candidates = [s for s in sents if len(s) > 40]
    if not candidates:
        candidates = sents
    chosen = candidates[:num_q] if len(candidates) >= num_q else (candidates + sents)[:num_q]
    mcqs = []
    for sent in chosen:
        words = [w for w in nltk.word_tokenize(sent) if w.isalpha()]
        if len(words) < 5:
            continue
        key = choice(words[1:-1]) if len(words) > 2 else words[0]
        question = sent.replace(key, "_")
        options = [key]
        # pick other words from sentence as distractors
        extras = [w for w in words if w.lower() != key.lower()]
        shuffle(extras)
        for e in extras[:3]:
            if e not in options:
                options.append(e)
        # if not enough options, add common words
        while len(options) < 4:
            options.append(choice(["data","model","system","study","learn"]))
        shuffle(options)
        mcqs.append({'question': question, 'options': options, 'answer': key})
    return mcqs

# Revision schedule generator
def generate_revision_schedule(exam_date_str, days_before=14, frequency_per_week=3):
    try:
        exam = datetime.fromisoformat(exam_date_str)
    except Exception:
        # try other formats
        exam = datetime.strptime(exam_date_str, "%Y-%m-%d")
    start = exam - timedelta(days=days_before)
    schedule = []
    total_days = (exam - start).days
    if total_days <= 0:
        return [exam.strftime("%Y-%m-%d")]
    step_days = max(1, 7 // frequency_per_week)
    curr = start
    while curr < exam:
        schedule.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=step_days)
    return schedule

# --------------------------
# Career module: resume parsing, scoring, skill-gap detection, learning plan
# --------------------------
# Simple contact extractor
def extract_contact_info(text):
    email = re.search(r'[\w\.-]+@[\w\.-]+', text)
    phone = re.search(r'(\+?\d[\d\-\s]{7,}\d)', text)
    return {'email': email.group(0) if email else None,
            'phone': phone.group(0).strip() if phone else None}

# Extract skills by matching against a skills pool (from JOB_ROLES)
ALL_SKILLS_POOL = sorted(list({s for skills in JOB_ROLES.values() for s in skills}), key=lambda x: x.lower())

def extract_skills_from_text(text, skills_pool=None):
    t = text.lower()
    found = []
    pool = skills_pool if skills_pool else ALL_SKILLS_POOL
    for s in pool:
        if s.lower() in t:
            found.append(s)
    # small fallback using noun chunks if spacy available
    if not found and have_spacy and nlp:
        doc = nlp(text)
        for nc in doc.noun_chunks:
            if len(nc.text.split()) <= 3:
                found.append(nc.text.strip())
    return list(dict.fromkeys(found))  # unique preserve order

# Resume scoring (simple heuristic)
def score_resume(skills_found, text):
    score = 30
    score += min(len(skills_found), 10) * 5
    if 'project' in text.lower():
        score += 10
    if 'experience' in text.lower() or 'intern' in text.lower():
        score += 10
    # bonus for contact
    contact = extract_contact_info(text)
    if contact.get('email') or contact.get('phone'):
        score += 5
    return min(score, 100)

# Detect skill gaps comparing to a target role
def detect_skill_gaps(skills_found, target_role):
    required = JOB_ROLES.get(target_role, [])
    missing = [s for s in required if s.lower() not in [x.lower() for x in skills_found]]
    present = [s for s in required if s.lower() in [x.lower() for x in skills_found]]
    return {'required': required, 'present': present, 'missing': missing}

# Optional: simple semantic job recommender using sentence-transformers if available
_sbert = None
def get_sbert():
    global _sbert
    if _sbert is None and have_sbert:
        try:
            _sbert = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            _sbert = None
    return _sbert

def recommend_jobs_by_resume(skills_found, job_list=None, top_k=5):
    # job_list is a list of dicts with 'title' and 'description'
    # For demo, build job_list from JOB_ROLES
    jobs = []
    for role, skills in JOB_ROLES.items():
        jobs.append({'title': role, 'description': " ".join(skills)})
    if not have_sbert:
        # fallback: match by overlapping skills count
        scores = []
        for job in jobs:
            req = [w.lower() for w in job['description'].split()]
            overlap = sum(1 for s in skills_found if s.lower() in req)
            scores.append((overlap, job))
        scores.sort(reverse=True, key=lambda x:x[0])
        return [j for _, j in scores[:top_k]]
    sbert = get_sbert()
    if sbert is None:
        return jobs[:top_k]
    resume_emb = sbert.encode(" ".join(skills_found) if skills_found else "", convert_to_tensor=True)
    scored = []
    for job in jobs:
        emb = sbert.encode(job['description'], convert_to_tensor=True)
        sim = util.pytorch_cos_sim(resume_emb, emb).item()
        scored.append((sim, job))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [j for _, j in scored[:top_k]]

# Build a simple learning plan for missing skills
def build_learning_plan(missing_skills):
    plan = []
    for skill in missing_skills:
        plan.append({
            'skill': skill,
            'duration_days': 7,
            'resources': [
                f"https://www.google.com/search?q={skill.replace(' ', '+')}+tutorial",
                f"https://www.youtube.com/results?search_query={skill.replace(' ', '+')}+course"
            ]
        })
    return plan

# --------------------------
# Streamlit UI (single-file)
# --------------------------
st.set_page_config(page_title="AI Career & Study Companion", layout="wide")
st.title("AI Career & Study Companion")
st.markdown("*A dual-mode Python app:* Student Study Assistant (summarize + quiz + revision) and Career Coach (resume review + skill gaps + learning plan).")

# Sidebar navigation
mode = st.sidebar.selectbox("Choose Mode", ["Home", "Student Study Assistant", "Career Coach", "Saved Sessions"])

# Home / Info
if mode == "Home":
    st.header("👋 Welcome")
    st.write("""
    This app has two main sections:
    1. *Student Study Assistant* — upload notes (PDF/TXT) to get a summary, practice quiz, and a revision schedule.
    2. *Career Coach* — upload your resume (PDF/TXT) to get a resume score, detected skills, missing skill suggestions for a chosen role, and a simple learning plan.
    """)
    st.info("Tip: If transformer models are slow on your machine, the app will fallback to simple rules for many features.")
    st.write("Sample job roles available for matching:")
    st.write(JOB_ROLES)

# ---------------- Student Section ----------------
if mode == "Student Study Assistant":
    st.header("📘 Student Study Assistant")
    st.write("Upload class notes or textbook PDF (or paste text). The assistant will summarize, create a short quiz, and propose a revision schedule.")

    uploaded_file = st.file_uploader("Upload a PDF (notes) or a .txt file", type=["pdf","txt"])
    text_input = st.text_area("Or paste the text here (optional)", height=120)

    text = ""
    if uploaded_file:
        # streamlit gives UploadedFile which behaves like file object
        if uploaded_file.type == "application/pdf":
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf_filelike(uploaded_file)
        else:
            # txt
            text = uploaded_file.getvalue().decode("utf-8")
    if text_input and not text:
        text = text_input

    if text:
        st.subheader("Preview (first 800 chars)")
        st.write(clean_text(text)[:800] + ("..." if len(clean_text(text))>800 else ""))
        # Save session
        if st.button("Save this session (study)"):
            save_session("student", "notes", text[:1000])
            st.success("Saved session to database.")

        # Summarize
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                summary = summarize_text(text)
            st.subheader("📝 Summary")
            st.write(summary)
            st.download_button("Download Summary", summary, file_name="summary.txt")
        # Quiz
        if st.button("Generate Quiz (5 questions)"):
            with st.spinner("Generating quiz..."):
                mcqs = generate_simple_mcqs(text, num_q=5)
            st.subheader("🧠 Quiz")
            for i,q in enumerate(mcqs):
                st.markdown(f"*Q{i+1}:* {q['question']}")
                for opt in q['options']:
                    st.write("-", opt)
                st.write(f"Answer: {q['answer']}")
        # Revision schedule
        st.subheader("⏰ Revision Planner")
        exam_date = st.date_input("If you have an exam date, choose it (optional)", value=None)
        days_before = st.number_input("Start revisions how many days before exam?", min_value=1, max_value=180, value=14)
        freq = st.slider("Sessions per week", 1, 7, 3)
        if st.button("Create Revision Schedule") and exam_date:
            ed = exam_date.isoformat()
            schedule = generate_revision_schedule(ed, days_before, freq)
            st.write("Suggested revision dates:")
            for d in schedule:
                st.write("-", d)
            # save schedule summary
            save_session("student", f"revision_{ed}", json.dumps(schedule))
    else:
        st.info("Upload a PDF or paste text to begin.")

# ---------------- Career Section ----------------
if mode == "Career Coach":
    st.header("💼 Career Coach")
    st.write("Upload your resume PDF (or paste text). The coach will extract skills, give a resume score, suggest jobs, and propose a learning plan for missing skills.")

    uploaded_resume = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf","txt"])
    resume_text_area = st.text_area("Or paste resume text here", height=120)

    resume_text = ""
    if uploaded_resume:
        if uploaded_resume.type == "application/pdf":
            with st.spinner("Extracting text from resume..."):
                resume_text = extract_text_from_pdf_filelike(uploaded_resume)
        else:
            resume_text = uploaded_resume.getvalue().decode("utf-8")
    if resume_text_area and not resume_text:
        resume_text = resume_text_area

    if resume_text:
        st.subheader("Preview (first 800 chars)")
        st.write(clean_text(resume_text)[:800] + ("..." if len(clean_text(resume_text))>800 else ""))
        # Save raw resume session
        if st.button("Save this resume session"):
            save_session("career", "resume", resume_text[:1000])
            st.success("Saved resume session to database.")

        # Contact info
        contact = extract_contact_info(resume_text)
        st.write("*Contact found:*", contact)

        # Skills detection
        st.write("---")
        st.subheader("🔎 Detected Skills")
        skills_found = extract_skills_from_text(resume_text, skills_pool=ALL_SKILLS_POOL)
        if not skills_found:
            st.warning("No clear skills detected by simple matcher. Try adding 'Python', 'SQL', etc. into resume.")
        else:
            st.write(skills_found)

        # Resume scoring
        score = score_resume(skills_found, resume_text)
        st.metric("Resume Score", f"{score}/100")

        # Choose target role to detect gaps
        roles_list = ["-- None --"] + list(JOB_ROLES.keys())
        selected_role = st.selectbox("Select target role to compare with", roles_list)
        if selected_role != "-- None --":
            gaps = detect_skill_gaps(skills_found, selected_role)
            st.write("*Required skills for role:*", gaps['required'])
            st.write("*Present (from resume):*", gaps['present'])
            st.write("*Missing skills:*", gaps['missing'])
            # Learning plan
            if st.button("Build learning plan for missing skills"):
                plan = build_learning_plan(gaps['missing'])
                st.subheader("📚 Learning Plan")
                for p in plan:
                    st.write(f"- *{p['skill']}* (approx {p['duration_days']} days). Resources:")
                    for r in p['resources']:
                        st.write("  -", r)
                # Save plan
                save_session("career", f"plan_{selected_role}", json.dumps(plan))

        # Job recommendations
        st.write("---")
        if st.button("Recommend matching jobs (based on skills)"):
            with st.spinner("Finding matching jobs..."):
                recs = recommend_jobs_by_resume(skills_found)
            st.subheader("✨ Suggested Roles")
            for j in recs:
                st.write("-", j['title'])
    else:
        st.info("Upload a resume or paste text to begin.")

# ---------------- Saved Sessions ----------------
if mode == "Saved Sessions":
    st.header("💾 Saved Sessions")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, mode, title, substr(content,1,200), created FROM sessions ORDER BY created DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    if rows:
        for r in rows:
            st.write(f"ID: {r[0]} | Mode: {r[1]} | Title: {r[2]} | Created: {r[4]}")
            st.write(r[3] + ("..." if len(r[3])>=200 else ""))
            st.write("---")
    else:
        st.info("No saved sessions yet. Use Save buttons in other sections.")

# ---------------- Footer / tips ----------------
st.markdown("---")
st.markdown("*Tips:*\n\n- If transformer models are slow or unavailable, features fall back to simpler methods. \n- For reliable resume-job matching you can expand JOB_ROLES in the file or add a JSON. \n- For a demo, prepare one study PDF and one resume PDF in data/ and show both flows.\n\nGood luck with your evaluation! 🚀")


