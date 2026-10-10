import os
from flask import Flask, render_template, request, jsonify
from nlp_engine import calculate_match_scores
import PyPDF2
from docx import Document

app = Flask(__name__)
app.secret_key = 'talentpulse-ai-candidate-screener'

last_results = {
    'results': [],
    'stats': {'highly_suitable': 0, 'suitable': 0, 'low_match': 0},
    'labels': [],
    'scores': []
}

def extract_text_from_pdf(stream):
    reader = PyPDF2.PdfReader(stream)
    pages = [p.extract_text() for p in reader.pages if p.extract_text()]
    return '\n'.join(pages)

def extract_text_from_docx(stream):
    doc = Document(stream)
    text = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text.append(cell.text)
    return '\n'.join(text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/screen')
def screen():
    return render_template('screen.html')

@app.route('/results')
def results():
    return render_template('results.html',
                           results=last_results['results'],
                           stats=last_results['stats'],
                           labels=last_results['labels'],
                           scores=last_results['scores'])

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename.lower()
    try:
        if filename.endswith('.pdf'):
            extracted = extract_text_from_pdf(file.stream)
        elif filename.endswith('.docx'):
            extracted = extract_text_from_docx(file.stream)
        else:
            return jsonify({'status': 'error', 'error': 'Unsupported file format'}), 400

        if not extracted.strip():
            return jsonify({'status': 'error', 'error': 'File contains no extractable text'}), 400

        return jsonify({'status': 'success', 'text': extracted, 'filename': file.filename})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/score', methods=['POST'])
def score():
    global last_results
    data = request.get_json() or {}
    job_desc = data.get('job_desc', '')
    resumes = data.get('resumes', [])

    if not job_desc or not resumes:
        return jsonify({'error': 'Job description and resumes required'}), 400

    evaluated = calculate_match_scores(job_desc, resumes)

    stats = {
        'highly_suitable': sum(1 for r in evaluated if r['label'] == 'Highly Suitable'),
        'suitable': sum(1 for r in evaluated if r['label'] == 'Suitable'),
        'low_match': sum(1 for r in evaluated if r['label'] == 'Low Match'),
    }

    last_results = {
        'results': evaluated,
        'stats': stats,
        'labels': [r['candidate_id'] for r in evaluated],
        'scores': [r['score'] for r in evaluated]
    }

    if request.headers.get('HX-Request'):
        return render_template('partials/results_table.html',
                               results=evaluated,
                               stats=stats,
                               labels=last_results['labels'],
                               scores=last_results['scores'])
    return jsonify(evaluated)

if __name__ == '__main__':
    app.run(debug=True, port=5000)