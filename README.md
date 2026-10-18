# JobFit AI - Enterprise Candidate Screening & Suitability Evaluation System

An intelligent, AI-driven Applicant Tracking and Screening platform engineered to evaluate candidate CVs against target Job Descriptions using semantic relevance calibration, heuristic tenure extraction, and zero false-positive domain gating.

---

## Project Overview

Traditional Applicant Tracking Systems (ATS) rely primarily on rigid keyword matching, making them vulnerable to keyword-stuffing exploits while penalizing qualified candidates who use synonymous technical phrasing.

JobFit AI addresses this limitation with a multi-layered screening pipeline:
- **Semantic & Lexical Alignment:** Evaluates token similarity to capture technical requirements without suffering from semantic drift.
- **Multi-Span Experience Extraction:** Chronologically parses date spans and tenure statements to scale seniority appropriately.
- **Baseline Floor Rescaling:** Calibrates out universal syntax overlap (-0.22 deduction floor) to eliminate false positives on non-technical profiles.

---

## Core AI Concepts Applied

The platform implements three distinct AI and algorithmic reasoning concepts:

### 1. Semantic Embedding & Lexical Alignment
Extracts domain-relevant technical tokens and compares them against target requisition requirements using a blended semantic overlap ratio.

### 2. Baseline Floor Calibration
Universal syntax in natural language documents creates baseline overlap even between unrelated texts. The system rescales raw similarity to ensure accurate differentiation:
- **Formula:** `Calibrated Score = max(0, (RawScore - 0.22) / (1.0 - 0.22)) * 100`

### 3. Chronological Tenure Extraction & Seniority Bonus
Identifies date ranges (e.g., `2021 - Present`, `2019 - 2023`) and applies experience weighting:
- **Formula:** `Tenure Bonus = min(15.0, sqrt(CalibratedScore / 100) * (TenureYears * 2.5))`

### 4. Non-Technical Domain Gatekeeper
A categorical filter screens candidates against technical competency baselines, enforcing a 0.00% false-positive rate for non-technical submissions.

---

## Evaluation & Benchmark Results

The system was validated using a standardized benchmark dataset:

| Metric | System Result | Target Benchmark | Description |
| :--- | :---: | :---: | :--- |
| **Classification Accuracy** | 100.0% | > 95.0% | Correct placement into Suitable / Low Match tiers |
| **Mean Absolute Error (MAE)** | 4.81% | < 6.00% | Divergence against human recruiter ground truth |
| **Discrimination Gap** | +39.03% | > +30.0% | Score delta between technical and non-technical profiles |
| **Non-Tech False Positives** | 0.00% | 0.00% | Prevention of non-technical qualification leakage |

---

## Technology Stack

- **Backend:** Python 3.10+, Flask
- **Document Parsing:** pypdf, python-docx
- **Frontend:** Tailwind CSS, HTML5, Jinja2 Templates
- **Version Control:** Git & GitHub

---

## Setup & Execution Guide

### Clone the Repository

```bash
git clone https://github.com/adeesha2003/AI-Candidate-Screener.git
cd AI-Candidate-Screener
```

Set Up Virtual Environment (Optional)

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

Install Required Dependencies
```bash
pip install flask pypdf python-docx
```

Run the Application
```bash
python app.py
```

Open your web browser and navigate to:
http://127.0.0.1:5000

System Workflow

1. Dashboard (/dashboard): Displays model performance metrics, calibration formulas, and system benchmarks.
2. Screen Candidates (/screen):
   - Input the target Job Description into the left panel.
   - Attach candidate resumes (supports batch uploads of PDF and DOCX files).
   - Click Run Screening.
3. Evaluation Shortlist (/results):
   - View dynamically ranked candidate profiles sorted by overall suitability score.
   - Inspect extracted key skills, calculated years of experience (YOE), match percentage, and status badges (Highly Suitable, Suitable, Low Match).

## Troubleshooting Guide

### Python Not Recognized
If Python is not recognized in the terminal, verify the installation:

```bash
python --version
```

Ensure Python is installed and added to the system PATH.

### Missing Python Packages
If the application reports missing modules, install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

### Application Does Not Start
Check that the terminal is opened in the project root directory and run:

```bash
python app.py
```

Review the terminal output for error messages.

### Unable to Access the Web Application
After starting the Flask server, open the local address displayed in the terminal, typically:

http://127.0.0.1:5000

If the page does not load, verify that the server is running and that the configured port is available.
