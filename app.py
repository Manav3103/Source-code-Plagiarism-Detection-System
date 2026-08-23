import os
import json
import ast
from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DATA_FILE = 'submissions.json'

# --- Levenshtein Distance (Normalized) ---
def levenshtein_distance(a, b):
    if len(a) < len(b):
        return levenshtein_distance(b, a)
    if len(b) == 0:
        return len(a)
    previous_row = range(len(b) + 1)
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def normalized_levenshtein(a, b):
    dist = levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1 - dist / max_len

# --- Winnowing Algorithm for Fingerprinting ---
def winnowing(text, k=5, window_size=4):
    if len(text) < k:
        return set()
    
    hashes = []
    base = 257
    mod = 10**9 + 7
    
    current_hash = 0
    base_k = pow(base, k-1, mod)
    
    for i in range(len(text)):
        c = ord(text[i])
        if i < k:
            current_hash = (current_hash * base + c) % mod
            if i == k - 1:
                hashes.append(current_hash)
        else:
            left_char = ord(text[i-k])
            current_hash = (current_hash - left_char * base_k) % mod
            current_hash = (current_hash * base + c) % mod
            hashes.append(current_hash)
    
    fingerprints = set()
    if len(hashes) < window_size:
        fingerprints.update(hashes)
    else:
        for i in range(len(hashes) - window_size + 1):
            window = hashes[i:i+window_size]
            min_hash = min(window)
            fingerprints.add(min_hash)
    return fingerprints

def winnowing_similarity(code1, code2):
    f1 = winnowing(code1)
    f2 = winnowing(code2)
    if not f1 or not f2:
        return 0.0
    intersection = len(f1.intersection(f2))
    union = len(f1.union(f2))
    return intersection / union if union > 0 else 0.0

# --- AST Features Extraction ---
def extract_ast_features(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return "empty_tree"

    features = []

    class FeatureVisitor(ast.NodeVisitor):
        def generic_visit(self, node):
            features.append(type(node).__name__)
            if hasattr(node, 'name'):
                features.append(str(node.name))
            ast.NodeVisitor.generic_visit(self, node)

    FeatureVisitor().visit(tree)
    res = ' '.join(features)
    return res if res.strip() else "empty_features"

def compute_plagiarism(submissions):
    if not submissions:
        return []

    # Agar sirf 1 submission hai to compare karne ki zarurat nahi
    if len(submissions) == 1:
        entry = submissions[0]
        return [{
            'student_name': entry.get('student_name', 'Unknown'),
            'student_id': entry.get('student_id', 'N/A'),
            'plagiarism_percent': 0.0,
            'match_found': 'None',
            'feedback': entry.get('feedback', ''),
            'code': entry.get('code', '')
        }]

    codes = [entry.get('code', '') for entry in submissions]
    features = [extract_ast_features(code) for code in codes]

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(features)
        sim_matrix = cosine_similarity(tfidf_matrix)
    except Exception:
        sim_matrix = np.zeros((len(submissions), len(submissions)))

    n = len(submissions)
    results = []
    for i in range(n):
        max_combined = 0.0
        match_index = None
        for j in range(n):
            if i == j:
                continue

            cos_sim = sim_matrix[i][j]
            win_sim = winnowing_similarity(codes[i], codes[j])
            lev_sim = normalized_levenshtein(codes[i], codes[j])

            combined_score = (0.5 * cos_sim) + (0.3 * win_sim) + (0.2 * lev_sim)

            if combined_score > max_combined:
                max_combined = combined_score
                match_index = j

        percent = round(max_combined * 100, 2)
        matched_with = submissions[match_index]['student_name'] if match_index is not None else 'None'

        results.append({
            'student_name': submissions[i].get('student_name', 'Unknown'),
            'student_id': submissions[i].get('student_id', 'N/A'),
            'plagiarism_percent': percent,
            'match_found': matched_with,
            'feedback': submissions[i].get('feedback', ''),
            'code': submissions[i].get('code', '')
        })
    return results

@app.route('/')
def dashboard():
    submissions = load_submissions()
    submissions_with_plagiarism = compute_plagiarism(submissions)
    return render_template('dashboard.html', submissions=submissions_with_plagiarism)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        student_name = request.form.get('student_name', '').strip()
        student_id = request.form.get('student_id', '').strip()
        file = request.files.get('code_file')

        if not student_name or not student_id or not file:
            return "Missing data", 400

        filename = f"{student_id}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()

        submissions = load_submissions()
        submissions.append({
            'student_name': student_name,
            'student_id': student_id,
            'code': code,
            'feedback': ''
        })
        save_submissions(submissions)

        return redirect(url_for('dashboard'))

    return render_template('upload.html')

@app.route('/feedback_form/<student_id>', methods=['GET', 'POST'])
def feedback_form(student_id):
    submissions = load_submissions()

    if request.method == 'POST':
        feedback = request.form.get('feedback', '').strip()
        for entry in submissions:
            if str(entry.get('student_id')) == str(student_id):
                entry['feedback'] = feedback
                break
        save_submissions(submissions)
        return redirect(url_for('dashboard'))

    existing_feedback = ''
    for entry in submissions:
        if str(entry.get('student_id')) == str(student_id):
            existing_feedback = entry.get('feedback', '')
            break

    return render_template('feedback_form.html', student_id=student_id, feedback=existing_feedback)

def load_submissions():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_submissions(submissions):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(submissions, f, indent=4)

if __name__ == '__main__':
    app.run(debug=True)