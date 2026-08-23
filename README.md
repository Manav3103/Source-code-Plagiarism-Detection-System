# Source-code-Plagiarism-Detection-System

An automated, hybrid source code similarity detection platform built with Flask, scikit-learn, and Python AST analysis. The system evaluates submitted code across syntactic, structural, and character-level dimensions to compute accurate pairwise plagiarism scores in real time.

---

## Overview

Traditional string-matching methods fail when students rename identifiers, alter whitespace, or restructure control flow. This system deploys a multi-layered similarity pipeline combining **Abstract Syntax Tree (AST) analysis**, **Winnowing fingerprinting**, and **Normalized Levenshtein Distance** to identify both direct copies and logic-preserving transformations.

Instructors can view real-time similarity metrics on a centralized dashboard, identify matching peers, and submit targeted feedback on individual submissions.

---

## Key Features

* **Hybrid Similarity Engine**:
  * **AST Syntactic Analysis (50% Weight)**: Uses Python's native `ast` module to construct syntax trees, then applies TF-IDF vectorization and Cosine Similarity to detect semantic logic copies regardless of variable renaming.
  * **Winnowing Fingerprinting (30% Weight)**: Applies rolling hash k-grams to generate invariant code fingerprints, capturing structural code reuse.
  * **Normalized Levenshtein Distance (20% Weight)**: Evaluates dynamic-programming edit distance at the character sequence level to capture direct lexical duplication.
* **Real-Time Instructor Dashboard**: Displays student names, IDs, similarity percentages, matched source files, and review notes.
* **Severity Badging**: Color-coded indicators categorize plagiarism risk into Low, Medium, and High tiers.
* **Submission Feedback System**: Allows instructors to attach custom notes directly to student records.
* **Zero External DB Setup**: Employs lightweight JSON-based file persistence for fast, localized evaluation.

---

## Detection Formula

The final similarity score between any two code files is computed as a weighted sum:

$$\text{Plagiarism \%} = \Big( (0.5 \times \text{AST}_{\text{Cosine}}) + (0.3 \times \text{Winnowing}_{\text{Jaccard}}) + (0.2 \times \text{Levenshtein}_{\text{Normalized}}) \Big) \times 100$$

| Component | Target Domain | Technique |
| :--- | :--- | :--- |
| **AST Analysis** | Logic & Syntax Hierarchy | Node Visitor + TF-IDF Vectorizer + Cosine Similarity |
| **Winnowing** | Structural Block Match | K-gram Rolling Hash Fingerprints + Jaccard Index |
| **Levenshtein** | Lexical Text Overlap | Normalized Dynamic Edit Distance |

---

## Project Structure

```text
Source-code-Plagiarism-Detection-System/
│
├── app.py                  # Core Flask application and detection algorithms
├── requirements.txt        # Package dependencies
├── README.md               # Project documentation
│
├── data/
│   └── submissions.json    # Persistent JSON storage for submissions
│
├── uploads/                # Directory storing uploaded source scripts
│
├── examples/               # Sample test files
│   ├── 1.py
│   ├── 2.py
│   └── 3.py
│
├── templates/              # Jinja2 HTML templates
│   ├── dashboard.html      # Instructor dashboard view
│   ├── upload.html         # Code submission form
│   └── feedback_form.html  # Feedback submission interface
│
└── static/
    └── style.css           # UI styling and dashboard theme

