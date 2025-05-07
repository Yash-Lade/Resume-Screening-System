import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import pickle
import re
import nltk
import io
import PyPDF2
import docx
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# Initialize session state variables
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'login'

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# File paths
USERS_FILE = 'data/users.json'
JOBS_FILE = 'data/jobs.json'
APPLICATIONS_FILE = 'data/applications.json'
RESUME_DATASET_FILE = 'UpdatedResumeDataSet.csv'  # Use the dataset in the current directory
MODEL_FILE = 'models/resume_classifier.pkl'
VECTORIZER_FILE = 'models/vectorizer.pkl'
METRICS_FILE = 'models/model_metrics.json'
UPLOADS_DIR = 'uploads'

# Initialize files if they don't exist
def initialize_files():
    # Initialize users file
    if not os.path.exists(USERS_FILE):
        default_users = {
            'admin': {
                'name': 'Admin User',
                'password': hashlib.sha256('admin'.encode()).hexdigest(),
                'role': 'recruiter'
            }
        }
        with open(USERS_FILE, 'w') as f:
            json.dump(default_users, f)

    # Initialize jobs file
    if not os.path.exists(JOBS_FILE):
        with open(JOBS_FILE, 'w') as f:
            json.dump([], f)

    # Initialize applications file
    if not os.path.exists(APPLICATIONS_FILE):
        with open(APPLICATIONS_FILE, 'w') as f:
            json.dump([], f)

# Authentication functions
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def authenticate(username, password):
    users = load_users()
    if username in users and users[username]['password'] == hashlib.sha256(password.encode()).hexdigest():
        return True, users[username]['role']
    return False, None

def register_user(username, name, password, role):
    users = load_users()
    if username in users:
        return False, "Username already exists"

    users[username] = {
        'name': name,
        'password': hashlib.sha256(password.encode()).hexdigest(),
        'role': role
    }

    save_users(users)
    return True, "Registration successful"

# Job management functions
def load_jobs():
    if os.path.exists(JOBS_FILE):
        with open(JOBS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_jobs(jobs):
    with open(JOBS_FILE, 'w') as f:
        json.dump(jobs, f)

def add_job(title, company, description, skills, recruiter):
    jobs = load_jobs()
    job_id = str(len(jobs) + 1)

    job = {
        'id': job_id,
        'title': title,
        'company': company,
        'description': description,
        'skills': skills,
        'recruiter': recruiter,
        'posted_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    jobs.append(job)
    save_jobs(jobs)
    return job_id

# Application management functions
def load_applications():
    if os.path.exists(APPLICATIONS_FILE):
        with open(APPLICATIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_applications(applications):
    with open(APPLICATIONS_FILE, 'w') as f:
        json.dump(applications, f)

# Function to extract text from PDF file
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Function to extract text from DOCX file
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def add_application(job_id, candidate, resume_file=None, resume_text=None):
    applications = load_applications()
    jobs = load_jobs()
    users = load_users()

    job = next((j for j in jobs if j['id'] == job_id), None)
    if not job:
        return False, "Job not found"

    # Check if already applied
    if any(app['job_id'] == job_id and app['candidate'] == candidate for app in applications):
        return False, "You have already applied for this job"

    application_id = str(len(applications) + 1)

    # Process resume file if provided
    file_path = None
    extracted_text = ""

    if resume_file is not None:
        # Save the uploaded file
        file_path = save_uploaded_file(resume_file)

        # Extract text based on file type
        if resume_file.name.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(resume_file)
        elif resume_file.name.lower().endswith('.docx'):
            extracted_text = extract_text_from_docx(resume_file)
        else:
            return False, "Unsupported file format. Please upload a PDF or DOCX file."

    # Use provided text if no file or extraction failed
    final_resume_text = extracted_text if extracted_text else resume_text

    if not final_resume_text:
        return False, "Could not extract text from resume or no text provided."

    application = {
        'id': application_id,
        'job_id': job_id,
        'job_title': job['title'],
        'job_company': job['company'],
        'job_recruiter': job['recruiter'],
        'candidate': candidate,
        'candidate_name': users[candidate]['name'],
        'resume_text': final_resume_text,
        'resume_file': file_path,
        'applied_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'Pending'
    }

    applications.append(application)
    save_applications(applications)
    return True, "Application submitted successfully"

# Text preprocessing functions
def download_nltk_resources():
    # Download required NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove phone numbers
    text = re.sub(r'\+?[0-9][0-9\s-]{7,}[0-9]', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_text(text):
    # Ensure NLTK resources are downloaded
    download_nltk_resources()

    # Clean the text
    text = clean_text(text)

    # Tokenize
    try:
        tokens = nltk.word_tokenize(text)
    except Exception as e:
        # Fallback to simple tokenization if NLTK tokenizer fails
        tokens = text.split()

    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    except Exception as e:
        # Fallback if stopwords fail
        common_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with'}
        tokens = [token for token in tokens if token not in common_stopwords]

    # Lemmatize
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except Exception as e:
        # Skip lemmatization if it fails
        pass

    return ' '.join(tokens)

# Extract skills from text
def extract_skills(text, skills_list):
    skills_found = []
    text = text.lower()

    for skill in skills_list:
        if skill.lower() in text:
            skills_found.append(skill)

    return skills_found

# Common skills list
COMMON_SKILLS = [
    'Python', 'Java', 'JavaScript', 'C++', 'C#', 'SQL', 'HTML', 'CSS',
    'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'Spring',
    'Machine Learning', 'Deep Learning', 'Data Analysis', 'Data Science',
    'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Git',
    'Agile', 'Scrum', 'Project Management', 'Leadership', 'Communication',
    'Problem Solving', 'Critical Thinking', 'Teamwork', 'Creativity'
]

# ML model functions
def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='linear', probability=True, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report
        }

    # Select the best model based on F1 score (more balanced metric)
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    best_model = results[best_model_name]['model']
    best_metrics = {
        'accuracy': results[best_model_name]['accuracy'],
        'precision': results[best_model_name]['precision'],
        'recall': results[best_model_name]['recall'],
        'f1': results[best_model_name]['f1'],
        'report': results[best_model_name]['report']
    }

    # Save metrics to file
    with open(METRICS_FILE, 'w') as f:
        # Convert report to serializable format
        serializable_metrics = best_metrics.copy()
        serializable_metrics['report'] = str(serializable_metrics['report'])
        json.dump(serializable_metrics, f)

    return best_model, best_metrics, best_model_name, results

# This function is replaced by the auto_train_model function

def load_model_and_vectorizer():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)

            with open(VECTORIZER_FILE, 'rb') as f:
                vectorizer = pickle.load(f)

            return model, vectorizer
        except Exception as e:
            st.error(f"Error loading model: {e}")

    return None, None

# Resume ranking functions
def calculate_skills_match(resume_skills, job_skills):
    if not job_skills:
        return 0.0

    matched_skills = set(resume_skills).intersection(set(job_skills))
    score = len(matched_skills) / len(job_skills)

    return score

def calculate_text_similarity(resume_text, job_description, vectorizer):
    # Transform texts using the vectorizer
    resume_vector = vectorizer.transform([resume_text])
    job_vector = vectorizer.transform([job_description])

    # Calculate cosine similarity
    similarity = cosine_similarity(resume_vector, job_vector)[0][0]

    return similarity

def rank_resumes(job_id):
    applications = load_applications()
    jobs = load_jobs()

    # Get the job
    job = next((j for j in jobs if j['id'] == job_id), None)
    if not job:
        return []

    # Get applications for the job
    job_applications = [app for app in applications if app['job_id'] == job_id]
    if not job_applications:
        return []

    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    if not model or not vectorizer:
        st.warning("Model not trained yet. Please train the model first.")
        return job_applications

    # Process job description
    processed_job_desc = preprocess_text(job['description'])
    job_skills = job['skills']

    # Rank applications
    for app in job_applications:
        # Process resume text
        processed_resume = preprocess_text(app['resume_text'])

        # Extract skills from resume
        resume_skills = extract_skills(app['resume_text'], COMMON_SKILLS + job_skills)

        # Calculate skills match score (40% weight)
        skills_score = calculate_skills_match(resume_skills, job_skills)

        # Calculate text similarity score (60% weight)
        similarity_score = calculate_text_similarity(processed_resume, processed_job_desc, vectorizer)

        # Calculate overall score
        overall_score = (0.4 * skills_score) + (0.6 * similarity_score)

        # Add scores to application
        app['skills_score'] = skills_score
        app['similarity_score'] = similarity_score
        app['overall_score'] = overall_score
        app['skills'] = resume_skills

    # Sort applications by overall score in descending order
    job_applications.sort(key=lambda x: x.get('overall_score', 0), reverse=True)

    return job_applications

# UI Components
def login_page():
    st.header("Login")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if username and password:
                authenticated, role = authenticate(username, password)
                if authenticated:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_role = role
                    st.session_state.current_page = 'dashboard'
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.error("Please enter both username and password")

    with col2:
        st.subheader("Register")
        reg_username = st.text_input("Username", key="reg_username")
        reg_name = st.text_input("Full Name", key="reg_name")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_role = st.selectbox("Role", ["candidate", "recruiter"], key="reg_role")

        if st.button("Register"):
            if reg_username and reg_name and reg_password:
                success, message = register_user(reg_username, reg_name, reg_password, reg_role)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.error("Please fill in all fields")

def dashboard_page():
    st.header("Dashboard")

    if st.session_state.user_role == 'recruiter':
        recruiter_dashboard()
    else:
        candidate_dashboard()

def recruiter_dashboard():
    st.subheader("Recruiter Dashboard")

    # Get recruiter's jobs
    jobs = load_jobs()
    recruiter_jobs = [job for job in jobs if job['recruiter'] == st.session_state.username]

    # Get applications for recruiter's jobs
    applications = load_applications()
    recruiter_applications = [app for app in applications if app['job_recruiter'] == st.session_state.username]

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Total Jobs Posted:** {len(recruiter_jobs)}")
        st.write(f"**Total Applications Received:** {len(recruiter_applications)}")

    with col2:
        # Recent applications
        st.write("**Recent Applications:**")
        recent_apps = sorted(recruiter_applications, key=lambda x: x['applied_date'], reverse=True)[:5]

        if recent_apps:
            for app in recent_apps:
                st.write(f"- {app['candidate_name']} applied for {app['job_title']} ({app['applied_date']})")
        else:
            st.info("No applications received yet")

    # Job listings
    st.subheader("Your Job Listings")

    if recruiter_jobs:
        for job in recruiter_jobs:
            with st.expander(f"{job['title']} - {job['company']}"):
                st.write(f"**Description:** {job['description']}")
                st.write(f"**Skills Required:** {', '.join(job['skills'])}")
                st.write(f"**Posted on:** {job['posted_date']}")

                # View applications button
                if st.button(f"View Applications for {job['title']}", key=f"view_apps_{job['id']}"):
                    st.session_state.selected_job_id = job['id']
                    st.session_state.current_page = 'view_applications'
                    st.rerun()
    else:
        st.info("You haven't posted any jobs yet")

def candidate_dashboard():
    st.subheader("Candidate Dashboard")

    # Get available jobs
    jobs = load_jobs()

    # Get candidate's applications
    applications = load_applications()
    candidate_applications = [app for app in applications if app['candidate'] == st.session_state.username]

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Total Jobs Available:** {len(jobs)}")
        st.write(f"**Your Applications:** {len(candidate_applications)}")

    with col2:
        # Recent applications
        st.write("**Your Recent Applications:**")
        recent_apps = sorted(candidate_applications, key=lambda x: x['applied_date'], reverse=True)[:5]

        if recent_apps:
            for app in recent_apps:
                st.write(f"- Applied for {app['job_title']} at {app['job_company']} ({app['applied_date']})")
        else:
            st.info("You haven't applied to any jobs yet")

    # Job listings
    st.subheader("Available Jobs")

    if jobs:
        for job in jobs:
            with st.expander(f"{job['title']} - {job['company']}"):
                st.write(f"**Description:** {job['description']}")
                st.write(f"**Skills Required:** {', '.join(job['skills'])}")
                st.write(f"**Posted by:** {job['recruiter']}")
                st.write(f"**Posted on:** {job['posted_date']}")

                # Check if already applied
                already_applied = any(app['job_id'] == job['id'] and app['candidate'] == st.session_state.username for app in applications)

                if already_applied:
                    st.info("You have already applied for this job")
                else:
                    # Apply button
                    if st.button(f"Apply for {job['title']}", key=f"apply_{job['id']}"):
                        st.session_state.selected_job_id = job['id']
                        st.session_state.current_page = 'apply_job'
                        st.rerun()
    else:
        st.info("No jobs available at the moment")

def post_job_page():
    st.header("Post a New Job")

    with st.form("post_job_form"):
        job_title = st.text_input("Job Title")
        company = st.text_input("Company Name")
        job_description = st.text_area("Job Description")

        # Skills selection
        selected_skills = st.multiselect("Required Skills", COMMON_SKILLS)

        # Custom skills
        custom_skills = st.text_input("Add Custom Skills (comma-separated)")

        submitted = st.form_submit_button("Post Job")

        if submitted:
            if not job_title or not company or not job_description:
                st.error("Please fill in all required fields")
            else:
                # Process custom skills
                if custom_skills:
                    custom_skills_list = [skill.strip() for skill in custom_skills.split(',')]
                    all_skills = selected_skills + custom_skills_list
                else:
                    all_skills = selected_skills

                # Add job
                job_id = add_job(job_title, company, job_description, all_skills, st.session_state.username)

                st.success("Job posted successfully!")
                st.session_state.current_page = 'dashboard'
                st.rerun()

def view_applications_page():
    st.header("View Applications")

    if 'selected_job_id' not in st.session_state:
        st.error("No job selected")
        return

    job_id = st.session_state.selected_job_id

    # Get job details
    jobs = load_jobs()
    job = next((j for j in jobs if j['id'] == job_id), None)

    if not job:
        st.error("Job not found")
        return

    st.subheader(f"Applications for: {job['title']} - {job['company']}")

    # Rank applications
    ranked_applications = rank_resumes(job_id)

    if not ranked_applications:
        st.info("No applications received for this job yet")
        return

    # Display applications
    for i, app in enumerate(ranked_applications):
        with st.expander(f"{i+1}. {app['candidate_name']} - Match Score: {app.get('overall_score', 0):.2f}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Applied on:** {app['applied_date']}")
                st.write(f"**Skills Match:** {app.get('skills_score', 0):.2f}")
                st.write(f"**Content Similarity:** {app.get('similarity_score', 0):.2f}")

                if 'skills' in app:
                    st.write(f"**Skills Found:** {', '.join(app['skills'])}")

                # Show resume file link if available
                if 'resume_file' in app and app['resume_file']:
                    file_name = os.path.basename(app['resume_file'])
                    st.write(f"**Resume File:** {file_name}")

            with col2:
                st.write("**Resume Text:**")
                st.text_area("", value=app['resume_text'], height=200, key=f"resume_{app['id']}")

def apply_job_page():
    st.header("Apply for Job")

    if 'selected_job_id' not in st.session_state:
        st.error("No job selected")
        return

    job_id = st.session_state.selected_job_id

    # Get job details
    jobs = load_jobs()
    job = next((j for j in jobs if j['id'] == job_id), None)

    if not job:
        st.error("Job not found")
        return

    st.subheader(f"Apply for: {job['title']} - {job['company']}")

    with st.form("apply_job_form"):
        st.write("**Upload your resume (PDF or DOCX)**")
        resume_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

        st.write("**OR**")

        resume_text = st.text_area("Paste your resume text here", height=200)

        submitted = st.form_submit_button("Submit Application")

        if submitted:
            if not resume_file and not resume_text:
                st.error("Please either upload a resume file or provide resume text")
            else:
                # Add application
                success, message = add_application(
                    job_id,
                    st.session_state.username,
                    resume_file=resume_file,
                    resume_text=resume_text
                )

                if success:
                    st.success(message)
                    st.session_state.current_page = 'dashboard'
                    st.rerun()
                else:
                    st.error(message)

def my_applications_page():
    st.header("My Applications")

    # Get candidate's applications
    applications = load_applications()
    candidate_applications = [app for app in applications if app['candidate'] == st.session_state.username]

    if not candidate_applications:
        st.info("You haven't applied to any jobs yet")
        return

    # Sort by date (newest first)
    candidate_applications.sort(key=lambda x: x['applied_date'], reverse=True)

    # Display applications
    for app in candidate_applications:
        with st.expander(f"{app['job_title']} - {app['job_company']}"):
            st.write(f"**Applied on:** {app['applied_date']}")
            st.write(f"**Status:** {app['status']}")

            # Show resume file if available
            if 'resume_file' in app and app['resume_file']:
                file_name = os.path.basename(app['resume_file'])
                st.write(f"**Resume File:** {file_name}")

            st.write(f"**Resume Text:**")
            st.text_area("", value=app['resume_text'], height=200, key=f"my_resume_{app['id']}")

# This function is no longer needed as model training is automatic

# Function to automatically train the model on startup
def auto_train_model():
    # Check if model already exists
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        model, vectorizer = load_model_and_vectorizer()
        if model and vectorizer:
            st.success("Using pre-trained model")
            return model, vectorizer

    # Ensure NLTK resources are downloaded before training
    download_nltk_resources()

    # Train the model
    st.info("Training the model on the resume dataset. This may take a moment...")

    try:
        # Check if dataset exists
        if not os.path.exists(RESUME_DATASET_FILE):
            st.error(f"Resume dataset not found at {RESUME_DATASET_FILE}")
            return None, None

        # Load and process the dataset
        df = pd.read_csv(RESUME_DATASET_FILE)

        # Check if the required columns exist
        required_columns = ['Resume', 'Category']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Dataset must contain the following columns: {required_columns}")
            return None, None

        # Display dataset info
        st.write(f"Dataset loaded: {df.shape[0]} resumes, {df.shape[1]} columns")
        st.write(f"Categories: {df['Category'].unique()}")

        # Preprocess the resume text
        with st.spinner("Preprocessing resumes..."):
            df['processed_resume'] = df['Resume'].apply(preprocess_text)

        # Extract features
        with st.spinner("Extracting features..."):
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
            X = vectorizer.fit_transform(df['processed_resume'])
            y = df['Category']

        # Train and evaluate models
        with st.spinner("Training models..."):
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train multiple models
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'SVM': SVC(kernel='linear', probability=True, random_state=42)
            }

            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

                # Store results
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

            # Select the best model based on accuracy
            best_model_name = max(results, key=lambda k: results[k]['accuracy'])
            best_model = results[best_model_name]['model']

            # Save the best model and vectorizer
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(best_model, f)

            with open(VECTORIZER_FILE, 'wb') as f:
                pickle.dump(vectorizer, f)

            # Display model metrics
            st.success(f"Model trained successfully! Using {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")

            st.subheader("Model Performance Metrics")
            st.write(f"**Best Model:** {best_model_name}")
            st.write(f"**Accuracy:** {results[best_model_name]['accuracy']:.4f}")
            st.write(f"**Precision:** {results[best_model_name]['precision']:.4f}")
            st.write(f"**Recall:** {results[best_model_name]['recall']:.4f}")
            st.write(f"**F1 Score:** {results[best_model_name]['f1']:.4f}")

            # Compare models
            st.subheader("Model Comparison")
            model_names = list(results.keys())
            accuracies = [results[name]['accuracy'] for name in model_names]
            precisions = [results[name]['precision'] for name in model_names]
            recalls = [results[name]['recall'] for name in model_names]
            f1_scores = [results[name]['f1'] for name in model_names]

            comparison_df = pd.DataFrame({
                'Model': model_names,
                'Accuracy': accuracies,
                'Precision': precisions,
                'Recall': recalls,
                'F1 Score': f1_scores
            })

            st.dataframe(comparison_df)

            return best_model, vectorizer

    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

# Main application
def main():
    # Initialize files
    initialize_files()

    # Sidebar
    with st.sidebar:
        st.title("Resume Screening System")

        if st.session_state.authenticated:
            st.write(f"Logged in as: **{st.session_state.username}** ({st.session_state.user_role})")

            st.subheader("Navigation")

            if st.button("Dashboard"):
                st.session_state.current_page = 'dashboard'
                st.rerun()

            if st.session_state.user_role == 'recruiter':
                if st.button("Post Job"):
                    st.session_state.current_page = 'post_job'
                    st.rerun()

            elif st.session_state.user_role == 'candidate':
                if st.button("My Applications"):
                    st.session_state.current_page = 'my_applications'
                    st.rerun()

            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.user_role = None
                st.session_state.current_page = 'login'
                st.rerun()

    # Automatically train the model on startup
    if 'model_trained' not in st.session_state:
        model, vectorizer = auto_train_model()
        if model and vectorizer:
            st.session_state.model_trained = True

    # Main content
    if not st.session_state.authenticated:
        login_page()
    else:
        if st.session_state.current_page == 'dashboard':
            dashboard_page()
        elif st.session_state.current_page == 'post_job':
            post_job_page()
        elif st.session_state.current_page == 'view_applications':
            view_applications_page()
        elif st.session_state.current_page == 'apply_job':
            apply_job_page()
        elif st.session_state.current_page == 'my_applications':
            my_applications_page()

if __name__ == "__main__":
    main()
