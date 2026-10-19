import os
import re
import math
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import docx
import pypdf

app = Flask(__name__)
app.secret_key = "jobfit-ai-enterprise-secret-key-production"

# Upload directory configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# ----------------- Helper Functions -----------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + " "
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {e}")
    return text

def parse_years_experience(text):
    year_matches = re.findall(r'\b(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|Present|Current)\b', text, re.IGNORECASE)
    total_years = 0
    if year_matches:
        for start, end in year_matches:
            try:
                s_year = int(start)
                e_year = 2026 if end.lower() in ['present', 'current'] else int(end)
                if e_year >= s_year:
                    total_years += (e_year - s_year)
            except:
                pass
        return max(1, min(total_years, 20))
    
    explicit_match = re.search(r'(\d+)\+?\s*years?', text, re.IGNORECASE)
    if explicit_match:
        return int(explicit_match.group(1))
    
    return 1
def validate_resume_content(text):
    # Reject empty or unreadable documents
    if not text or not text.strip():
        return False, "The uploaded document is empty or contains no readable text."

    text_lower = text.lower()

    # Common resume-related sections
    resume_sections = [
        "education", "experience", "employment",
        "skills", "projects", "internship",
        "qualifications", "certifications",
        "professional summary", "profile"
    ]

    # Check for an email address
    has_email = bool(re.search(
        r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b', text
    ))

    # Count resume-related sections
    section_count = sum(
        1 for section in resume_sections
        if re.search(
            r'\b' + re.escape(section) + r'\b',
            text_lower
        )
    )

    # Require contact information and resume structure
    if has_email and section_count >= 2:
        return True, ""

    return False, (
        "This document does not appear to be a candidate resume. "
        "Please upload a resume containing contact information "
        "and relevant sections such as education, skills, "
        "projects, or work experience."
    )

def compute_jobfit_score(cv_text, jd_text):
    tech_skills = [
        "python", "react", "node.js", "javascript", "sql", "mysql", "mongodb", 
        "c#", "java", "docker", "kubernetes", "aws", "gcp", "machine learning", 
        "html", "css", "git", "github", "figma", "rest api", "express", "pytorch"
    ]
    
    cv_lower = cv_text.lower()
    jd_lower = jd_text.lower()
    
    # Gatekeeper check: zero out score for profiles lacking relevant domain skills
    found_skills = [skill for skill in tech_skills if skill in cv_lower]
    if not found_skills:
        return 0.0, "Low Match", 0, ["Non-technical profile"]

    req_skills = [skill for skill in tech_skills if skill in jd_lower]
    matched_skills = [s for s in req_skills if s in found_skills]
    
    skill_ratio = len(matched_skills) / max(len(req_skills), 1)
    
    cv_words = set(re.findall(r'\w+', cv_lower))
    jd_words = set(re.findall(r'\w+', jd_lower))
    jaccard = len(cv_words.intersection(jd_words)) / max(len(cv_words.union(jd_words)), 1)
    
    # Baseline deduction calibration (-0.22)
    blended_raw = (0.50 * skill_ratio) + (0.50 * jaccard * 3.5)
    blended_raw = min(1.0, blended_raw)
    calibrated = max(0.0, (blended_raw - 0.22) / (1.0 - 0.22)) * 100.0
    
    # Experience scaling bonus
    tenure_years = parse_years_experience(cv_text)
    tenure_bonus = min(15.0, math.sqrt(calibrated / 100.0) * (tenure_years * 2.5))
    
    final_score = round(min(96.5, calibrated + tenure_bonus), 1)
    
    if final_score >= 65.0:
        status = "Highly Suitable"
    elif final_score >= 45.0:
        status = "Suitable"
    else:
        status = "Low Match"
        
    return final_score, status, tenure_years, found_skills

# ----------------- Flask Routes -----------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/screen', methods=['GET', 'POST'])
def screen():
    if request.method == 'POST':
        jd_text = request.form.get('job_description', '').strip()
        files = request.files.getlist('resumes')
        
        # Validate screening inputs before processing
        if not jd_text:
            return "Please enter a job description before screening.", 400

        valid_files = [
            file for file in files

            if file and file.filename and allowed_file(file.filename)
        ]

        if not valid_files:
            return render_template(
                "screen.html",
                error="Please upload at least one valid PDF or DOCX resume.",
                job_description=jd_text
            ), 400

        if len(valid_files) != len([
            file for file in files if file and file.filename
        ]):
            return "Unsupported file format. Only PDF and DOCX files are allowed.", 400

        
        # Retrieve existing evaluation history from session
        evaluation_results = session.get('screening_results', [])
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                if filename.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                else:
                    text = extract_text_from_docx(file_path)
                
                
                # Skip resumes with no readable text
                # Validate document content before candidate scoring
                is_valid_resume, validation_message = validate_resume_content(text)

                if not is_valid_resume:
                        return render_template(
                           "screen.html",
                            error=f"{filename}: {validation_message}",
                            job_description=jd_text
                        ), 400

                score, status, tenure, skills = compute_jobfit_score(text, jd_text)
                
                candidate_name = filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
                
                # Update existing entry if file re-uploaded; otherwise append new candidate
                existing_entry = next((item for item in evaluation_results if item['filename'] == filename), None)
                if existing_entry:
                    existing_entry['score'] = score
                    existing_entry['status'] = status
                    existing_entry['tenure'] = tenure
                    existing_entry['skills'] = skills[:5]
                else:
                    evaluation_results.append({
                        'name': candidate_name,
                        'filename': filename,
                        'score': score,
                        'status': status,
                        'tenure': tenure,
                        'skills': skills[:5]
                    })
        
        # Sort combined results descending by score
        evaluation_results.sort(key=lambda x: x['score'], reverse=True)
        session['screening_results'] = evaluation_results
        
        return render_template('results.html', candidates=evaluation_results)
        
    return render_template('screen.html')

@app.route('/results')
def results():
    candidates = session.get('screening_results', [])
    return render_template('results.html', candidates=candidates)

@app.route('/clear_results')
def clear_results():
    session.pop('screening_results', None)
    return redirect(url_for('results'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)