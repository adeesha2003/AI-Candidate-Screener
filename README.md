# AI-Based Candidate Screening System (v3.0)

An end-to-end, multi-stage hybrid Natural Language Processing (NLP) screening engine designed to evaluate unstructured candidate resumes (PDF/DOCX) against complex technical job descriptions.

## Overview
Automated resume screening often suffers from semantic drift, token truncation on multi-page profiles, and high false-positive rates when handling non-technical applicants due to universal syntax overlap. This project addresses these constraints through a calibrated hybrid architecture combining dense semantic representations, sparse lexical retrieval, bounded competency ontologies, and multi-span tenure extraction.

## System Architecture

The pipeline consists of five decoupled layers:

1. **Document Ingestion & Multi-Span Tenure Parsing:**
   - Text extraction across multi-page PDF and DOCX files.
   - Comprehensive regex parser recovering explicit statements (e.g., `5+ years`) and compound multi-interval date ranges (`YYYY - YYYY`, `YYYY - Present`).
   - Domain-specific HR stopword filtering and synonym normalization.

2. **Document Chunking & Aggregation:**
   - 140-word sliding-window chunker with a 35-word overlap to bypass transformer 256-token truncation barriers.
   - Top-2 max-pooled chunk cosine similarity aggregation.

3. **Hybrid Ensemble Matching Engine:**
   - **Domain Sentence-BERT (45%):** Contrastive fine-tuned transformer (`all-MiniLM-L6-v2`) trained on domain job-resume pairs.
   - **Domain TF-IDF (35%):** Sublinear frequency vectorizer pre-fitted on 2,076 domain documents (48,393 n-grams).
   - **Bounded Skill Taxonomy (15%):** 143+ technical competencies structured across 4 engineering clusters with cross-domain transferability metrics.
   - **N-Gram Jaccard Index (5%):** Exact lexical intersection over union (IoU).

4. **Mathematical Calibration & Filtering:**
   - **Linear Floor Rescaling:** Corrects baseline background syntax overlap:
     $$\text{Score}_{\text{cal}} = \max\left(0.0, \frac{\text{Raw Blended} - 0.22}{1.0 - 0.22}\right) \times 100$$
   - **Dynamic Tenure Scaling:** Bounded square-root tenure bonus scaled by candidate relevance.
   - **Technical Gatekeeper:** Hard 0.0% cut-off for applicants lacking matching technical competencies, eliminating non-technical false positives.

5. **Recruiter Dashboard:**
   - Flask-based web interface providing ranked shortlists, category suitability badges, and skill match breakdowns.

## Empirical Performance & Evaluation

Evaluated against a 12-candidate target benchmark and 120 real-world Kaggle resume profiles across 15 engineering and non-technical disciplines:

| Metric | Baseline (v1.0) | Intermediate (v2.2) | Engine v3.0 (Current) |
| :--- | :---: | :---: | :---: |
| **Classification Accuracy** | 66.67% | 100.00% | **100.00%** |
| **Top-1 Precision** | 100.00% | 100.00% | **100.00%** |
| **Mean Absolute Error (MAE)** | 14.04% | 5.76% | **4.81%** |
| **Pearson Correlation ($r$)** | 0.9312 | 0.9847 | **0.9900** |
| **Statistical Discrimination Gap** | +24.10% | +34.55% | **+39.03%** |
| **Non-Technical False Positive Rate** | 25.00% | 0.00% | **0.00%** |

## Repository Structure

```text
├── app.py                       # Flask web application entrypoint
├── nlp_engine.py                # Core v3.0 hybrid matching and calibration engine
├── benchmark_test.py            # Automated benchmark evaluation suite
├── domain_tfidf_model.pkl       # Pre-fitted domain TF-IDF model
├── requirements.txt             # Production dependency specifications
├── final-hr-model/              # Fine-tuned SBERT transformer weights
├── static/                      # CSS styling and static assets
├── templates/                   # Jinja2 HTML templates for the recruiter UI
├── training/                    # Model fine-tuning and taxonomy generation scripts
└── evaluation/                  # Benchmark datasets and metric generation scripts

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/adeesha2003/AI-Candidate-Screener.git](https://github.com/adeesha2003/AI-Candidate-Screener.git)
   cd AI-Candidate-Screener

Create and activate a virtual environment:

Bash
python -m venv venv
On Windows:

Bash
venv\Scripts\activate
On macOS/Linux:

Bash
source venv/bin/activate
Install required packages:

Bash
pip install -r requirements.txt
Running the Benchmark Suite
To execute the automated evaluation test across benchmark profiles:

Bash
python benchmark_test.py
Launching the Web Application
Start the Flask development server:

Bash
python app.py
Access the dashboard by navigating to http://localhost:5000 in your browser.