from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import spacy
from transformers import pipeline
import PyPDF2
import docx

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Career categories
career_categories = [
    "Software Development", "Data Science", "Digital Marketing",
    "Graphic Design", "Financial Analysis", "Human Resources",
    "Healthcare", "Education", "Business Management",
    "Artificial Intelligence", "Cybersecurity", "Content Writing"
]

def extract_text_from_file(filepath):
    if filepath.endswith('.pdf'):
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = " ".join([page.extract_text() for page in reader.pages])
    elif filepath.endswith('.docx'):
        doc = docx.Document(filepath)
        text = " ".join([para.text for para in doc.paragraphs])
    else:  # assuming txt file
        with open(filepath, 'r') as f:
            text = f.read()
    return text

def analyze_resume(text):
    doc = nlp(text)

    # Extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Skills detection
    skill_keywords = {
        "Programming": ["python", "java", "c++", "javascript", "sql", "html", "css"],
        "Data Science": ["machine learning", "deep learning", "tensorflow", "pytorch", "numpy", "pandas"],
        "Web Development": ["django", "flask", "react", "angular", "node.js"],
        "Design": ["photoshop", "illustrator", "figma", "ui/ux"],
        "Business": ["marketing", "finance", "accounting", "management"]
    }

    skills = {}
    for category, keywords in skill_keywords.items():
        found_skills = [kw.title() for kw in keywords if kw.lower() in text.lower()]
        if found_skills:
            skills[category] = found_skills

    # Resume rating
    rating = sum([
        len(text) > 500,
        "experience" in text.lower(),
        "education" in text.lower(),
        "skills" in text.lower(),
        len(skills) > 3
    ])

    # Mistakes
    mistakes = []
    if len(text) < 300:
        mistakes.append("Resume seems too short. Consider adding more details.")
    if "experience" not in text.lower():
        mistakes.append("Consider adding an experience section.")
    if "education" not in text.lower():
        mistakes.append("Consider adding an education section.")
    if len(skills) < 2:
        mistakes.append("Consider adding more skills to your resume.")

    return {
        "entities": entities,
        "skills": skills,
        "mistakes": mistakes,
        "rating": rating
    }

def recommend_career(text):
    result = classifier(text, career_categories)
    return {
        "recommendations": result["labels"][:3],
        "scores": result["scores"][:3]
    }

def recommend_jobs(text):
    job_titles = {
        "Software Development": ["Software Engineer", "Backend Developer", "Full Stack Developer"],
        "Data Science": ["Data Scientist", "Data Analyst", "Machine Learning Engineer"],
        "Digital Marketing": ["Digital Marketer", "SEO Specialist", "Social Media Manager"],
        "Graphic Design": ["Graphic Designer", "UI/UX Designer", "Art Director"],
        "Financial Analysis": ["Financial Analyst", "Investment Banker", "Nuclear Power Plant", "Accountant"],
        "Human Resources": ["HR Manager", "Recruiter", "Talent Acquisition Specialist"],
        "Healthcare": ["Doctor", "Nurse", "Healthcare Administrator"],
        "Education": ["Lecturer", "Professor", "Education Consultant"],
        "Business Management": ["Business Analyst", "Project Manager", "Operations Manager"],
        "Artificial Intelligence": ["AI Engineer", "NLP Specialist", "Computer Vision Engineer"],
        "Cybersecurity": ["Security Analyst", "Ethical Hacker", "Information Security Manager"],
        "Content Writing": ["Content Writer", "Technical Writer", "Copywriter"]
    }

    career_rec = recommend_career(text)
    recommended_jobs = []

    for category in career_rec["recommendations"]:
        if category in job_titles:
            recommended_jobs.extend(job_titles[category][:2])

    return {
        "jobs": list(set(recommended_jobs))[:5],
        "career_recommendations": career_rec
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            text = extract_text_from_file(filepath)
            analysis = analyze_resume(text)
            job_recommendations = recommend_jobs(text)

            return jsonify({
                "analysis": analysis,
                "recommendations": job_recommendations["career_recommendations"],
                "jobs": job_recommendations["jobs"]
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    return jsonify({"error": "File upload failed"}), 400

if __name__ == '__main__':
    app.run(debug=True)
