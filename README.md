# 🚀 AI-Powered Career Recommendation System

An intelligent, full-stack web application designed to parse resumes, categorize technical skill sets using Natural Language Processing (NLP), and leverage Deep Learning models to provide personalized career paths and job title recommendations.

🎯 **Live Demo:** [View Live App on Hugging Face Spaces](https://huggingface.co/spaces/D-nikhil/career-recommendation-system)

---

## 🌟 Key Features

* **Multi-Format Resume Parsing:** Supports smooth text extraction from `.pdf`, `.docx`, and `.txt` resume uploads.
* **Skill Extraction & Categorization:** Automatically identifies and groups core industry technical tags (e.g., Data Science, Programming, Web Development).
* **Zero-Shot Career Classification:** Powered by advanced Transformer pipelines to calculate profile match scores across multiple diverse domains.
* **Automated Resume Auditing & Rating:** Runs algorithmic heuristics to evaluate structure length, flag missing critical sections (Experience, Education), and assign a 5-star quality rating with actionable feedback.
* **Tailored Job Matching:** Dynamically suggests practical, real-world job titles mapping directly to highly scoring career trajectories.

---

## 🛠️ Tech Stack & Architecture

### Frontend
* **HTML5 / CSS3:** Custom, modern single-page dashboard featuring reactive UI micro-animations and loading states.
* **JavaScript (ES6):** Async network requests (`Fetch API`) handling multipart form document processing streams dynamically.

### Backend & Core AI Pipeline
* **Flask (Python):** Lightweight micro-framework serving RESTful endpoints and rendering template injection roots.
* **spaCy (`en_core_web_sm`):** Named Entity Recognition (NER) and syntactic analysis engine isolating industry keywords.
* **Hugging Face Transformers:** Zero-shot pipeline utilization backing the deep-learning semantic classification framework (`facebook/bart-large-mnli`).
* **PyPDF2 & python-docx:** Binary parsing layers extracting unformatted document matrices.

---

## 📦 Local Installation & Setup

To replicate this environment locally, make sure you have Python 3.10+ installed, then execute the following steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/dantulurinikhil30-debug/Career-Recommendation-System.git
   cd Career-Recommendation-System
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python app.py
