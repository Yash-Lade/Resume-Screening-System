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
if 'view_resume_id' not in st.session_state:
    st.session_state.view_resume_id = None

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

# Resume evaluation functions for the 7 parameters

# 1. Extract and evaluate education
def extract_education(text):
    text = text.lower()
    education_score = 0
    education_details = []

    # Education level indicators with their weights
    education_levels = {
        'phd': 1.0,
        'doctorate': 1.0,
        'master': 0.8,
        'mba': 0.8,
        'bachelor': 0.6,
        'undergraduate': 0.6,
        'associate': 0.4,
        'diploma': 0.3,
        'certificate': 0.2,
        'high school': 0.1
    }

    # Check for education levels
    for level, weight in education_levels.items():
        if level in text:
            education_score = max(education_score, weight)

            # Extract the degree and field if possible
            pattern = f"(?:{level})\s+(?:degree|in|of)?\s+([\w\s&]+)"
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    match = match.strip()
                    if match and len(match) > 2:
                        education_details.append(f"{level.title()} in {match.title()}")

    # Look for common degree abbreviations
    degree_patterns = [
        r'\b(b\.?s\.?|b\.?a\.?|b\.?e\.?|b\.?tech\.?)\b',
        r'\b(m\.?s\.?|m\.?a\.?|m\.?e\.?|m\.?tech\.?|m\.?b\.?a\.?)\b',
        r'\b(ph\.?d\.?)\b'
    ]

    for pattern in degree_patterns:
        if re.search(pattern, text):
            # Assign appropriate score based on degree level
            if pattern.startswith(r'\b(b'):
                education_score = max(education_score, 0.6)
            elif pattern.startswith(r'\b(m'):
                education_score = max(education_score, 0.8)
            elif pattern.startswith(r'\b(ph'):
                education_score = max(education_score, 1.0)

    # Look for university/college names
    university_indicators = ['university', 'college', 'institute', 'school of']
    for indicator in university_indicators:
        pattern = f"([\w\s&]+)\s+{indicator}"
        matches = re.findall(pattern, text)
        for match in matches:
            match = match.strip()
            if match and len(match) > 2:
                education_details.append(f"{match.title()} {indicator.title()}")

    return education_score, education_details

# 2. Extract and evaluate relevant experience
def extract_relevant_experience(text, job_description):
    text = text.lower()
    job_description = job_description.lower()

    # Extract work experience sections
    experience_sections = extract_experience_sections(text)

    # Extract years of experience
    experience_years = extract_years_of_experience(text)

    # Extract job titles and roles
    job_titles = extract_job_titles(text)

    # Extract companies
    companies = extract_companies(text)

    # Calculate relevance by comparing job titles and experience with job description
    relevance_score = 0

    # Score based on job title matches
    for title in job_titles:
        title_lower = title.lower()
        # Check for exact or partial matches in job description
        if title_lower in job_description:
            relevance_score += 0.3  # Bonus for exact title match
        else:
            # Check for partial matches (e.g., "developer" in "software developer")
            title_words = title_lower.split()
            for word in title_words:
                if len(word) > 3 and word in job_description:  # Only consider significant words
                    relevance_score += 0.1
                    break

    # Score based on experience sections relevance
    if experience_sections:
        # Calculate similarity between experience sections and job description
        for section in experience_sections:
            # Count matching keywords
            job_keywords = set(job_description.split())
            section_keywords = set(section.split())
            matching_keywords = job_keywords.intersection(section_keywords)

            # Add score based on keyword matches
            if len(matching_keywords) > 5:
                relevance_score += 0.2
            elif len(matching_keywords) > 2:
                relevance_score += 0.1

    # Base score from years of experience (capped at 10 years)
    experience_score = min(experience_years / 10, 1.0)

    # Bonus for having multiple relevant positions
    if len(job_titles) >= 3:
        relevance_score += 0.1
    elif len(job_titles) >= 2:
        relevance_score += 0.05

    # Combine scores with more weight on relevance
    combined_score = (0.7 * min(relevance_score, 1.0)) + (0.3 * experience_score)
    combined_score = min(combined_score, 1.0)  # Cap at 1.0

    return combined_score, job_titles, experience_years, companies, experience_sections

# Helper function to extract experience sections
def extract_experience_sections(text):
    text = text.lower()
    experience_sections = []

    # Look for experience section headers
    section_patterns = [
        r'(?:work|professional|employment)\s+(?:experience|history)\s*:([^#]*?)(?:\n\s*\w+\s*:|$)',
        r'(?:experience)\s*:([^#]*?)(?:\n\s*\w+\s*:|$)',
        r'(?:career\s+summary|summary\s+of\s+experience)\s*:([^#]*?)(?:\n\s*\w+\s*:|$)'
    ]

    for pattern in section_patterns:
        section_matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        for section in section_matches:
            if len(section.strip()) > 20:  # Only consider substantial sections
                experience_sections.append(section.strip())

    # If no sections found, try to extract experience from bullet points
    if not experience_sections:
        bullet_pattern = r'[-•*]\s*([^\n]*(?:experience|work|position|role|job|career)[^\n]*)'
        bullet_matches = re.findall(bullet_pattern, text, re.IGNORECASE)
        for match in bullet_matches:
            if len(match.strip()) > 20:  # Only consider substantial points
                experience_sections.append(match.strip())

    return experience_sections

# Helper function to extract years of experience
def extract_years_of_experience(text):
    text = text.lower()
    experience_years = 0

    # Common patterns for years of experience
    patterns = [
        r'(\d+)\+?\s*(?:years|yrs|yr)\s*(?:of)?\s*experience',
        r'experience\s*(?:of)?\s*(\d+)\+?\s*(?:years|yrs|yr)',
        r'(\d+)\+?\s*(?:years|yrs|yr)\s*(?:in|at|with)',
        r'worked\s*(?:for)?\s*(\d+)\+?\s*(?:years|yrs|yr)',
        r'(?:over|more\s+than)\s+(\d+)\s*(?:years|yrs|yr)'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Take the highest number of years mentioned
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        for m in match:
                            if m.isdigit() or (m.endswith('+') and m[:-1].isdigit()):
                                years = int(m.replace('+', ''))
                                experience_years = max(experience_years, years)
                    else:
                        years = int(match.replace('+', ''))
                        experience_years = max(experience_years, years)
                except ValueError:
                    continue

    # If no explicit years mentioned, try to calculate from employment dates
    if experience_years == 0:
        experience_years = calculate_experience_from_dates(text)

    return experience_years

# Helper function to calculate experience from employment dates
def calculate_experience_from_dates(text):
    # Look for date ranges in format like "2015-2020" or "Jan 2015 - Dec 2020"
    date_patterns = [
        r'(\d{4})\s*[-–—to]\s*(\d{4}|present|current|now)',
        r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\s*[-–—to]\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4}|present|current|now)'
    ]

    total_years = 0
    current_year = datetime.now().year

    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                start_year = int(match[0])

                # Handle end year
                if match[1].isdigit():
                    end_year = int(match[1])
                else:  # 'present', 'current', 'now'
                    end_year = current_year

                # Calculate duration (cap at current year)
                if start_year <= current_year and end_year <= current_year:
                    duration = end_year - start_year
                    if 0 <= duration <= 50:  # Sanity check
                        total_years += duration
            except (ValueError, IndexError):
                continue

    return total_years

# Helper function to extract job titles
def extract_job_titles(text):
    text = text.lower()
    job_titles = []

    # Common job title patterns
    title_patterns = [
        r'(?:senior|lead|principal|junior|associate)\s+([\w\s]+)(?:developer|engineer|designer|manager|analyst|consultant)',
        r'([\w\s]+)(?:developer|engineer|designer|manager|director|analyst|consultant|architect)',
        r'(?:worked as|position as|role as|title of|job title|position title)\s*(?:a|an)?\s*([\w\s]+)',
        r'(?:^|\n|\.)\s*([\w\s]+\s+(?:developer|engineer|designer|manager|director|analyst|consultant|architect))',
        r'(?:^|\n|\.)\s*(?:senior|lead|principal|junior|associate)\s+([\w\s]+)'
    ]

    for pattern in title_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]  # Extract from tuple if needed

            match = match.strip()
            if match and len(match) > 2:
                # Filter out common false positives
                if not any(fp in match for fp in ['university', 'college', 'school', 'degree']):
                    job_titles.append(match.title())

    # Remove duplicates while preserving order
    unique_titles = []
    for title in job_titles:
        if title not in unique_titles:
            unique_titles.append(title)

    return unique_titles

# Helper function to extract companies
def extract_companies(text):
    text = text.lower()
    companies = []

    # Common company patterns
    company_patterns = [
        r'(?:at|with|for)\s+([\w\s]+)\s*(?:inc|llc|ltd|corporation|corp|company|co)',
        r'(?:employed by|work for|worked for)\s+([\w\s]+)',
        r'(?:^|\n)\s*([\w\s]+)\s*(?:inc|llc|ltd|corporation|corp|company|co)',
        r'(?:^|\n)\s*([\w\s]+)\s*\|\s*(?:senior|lead|principal|junior|associate|developer|engineer|designer|manager|director|analyst|consultant|architect)'
    ]

    for pattern in company_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]  # Extract from tuple if needed

            match = match.strip()
            if match and len(match) > 2:
                # Filter out common false positives
                if not any(fp in match for fp in ['university', 'college', 'school', 'degree']):
                    companies.append(match.title())

    # Remove duplicates while preserving order
    unique_companies = []
    for company in companies:
        if company not in unique_companies:
            unique_companies.append(company)

    return unique_companies

# 3. Extract and evaluate achievements & impact
def extract_achievements(text):
    text = text.lower()
    achievements = []
    achievement_score = 0

    # Achievement indicators
    achievement_indicators = [
        'achieved', 'accomplished', 'awarded', 'earned', 'won', 'recognized',
        'improved', 'increased', 'decreased', 'reduced', 'saved', 'generated',
        'delivered', 'launched', 'implemented', 'developed', 'created', 'designed',
        'led', 'managed', 'supervised', 'mentored', 'trained'
    ]

    # Quantitative indicators
    quantitative_patterns = [
        r'(increased|improved|enhanced|boosted)\s+[\w\s]+\s+by\s+(\d+)%',
        r'(reduced|decreased|cut|minimized)\s+[\w\s]+\s+by\s+(\d+)%',
        r'(saved|generated|earned|produced)\s+\$(\d+[\d,]*)',
        r'(managed|led|supervised)\s+(?:a|an)?\s+team\s+of\s+(\d+)'
    ]

    # Check for achievement indicators
    for indicator in achievement_indicators:
        pattern = f"{indicator}\s+([^.;:!?]*)"
        matches = re.findall(pattern, text)
        for match in matches:
            match = match.strip()
            if match and len(match) > 10:  # Ensure it's a substantial phrase
                achievements.append(f"{indicator.title()} {match}")
                achievement_score += 0.1  # Small increment for each achievement

    # Check for quantitative achievements (higher value)
    for pattern in quantitative_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            achievements.append(f"{match[0].title()} by {match[1]}")
            achievement_score += 0.2  # Higher increment for quantitative achievements

    # Cap the score at 1.0
    achievement_score = min(achievement_score, 1.0)

    return achievement_score, achievements

# 4. Extract and evaluate certifications & courses
def extract_certifications_and_courses(text, certification_list):
    text = text.lower()
    certifications_found = []
    cert_score = 0

    # Common certification indicators
    cert_indicators = ['certified', 'certification', 'certificate', 'credential', 'course', 'training']

    # Check for specific certifications (higher value)
    for cert in certification_list:
        if cert.lower() in text:
            certifications_found.append(cert)
            cert_score += 0.2  # Higher value for recognized certifications

    # Look for certification patterns
    for indicator in cert_indicators:
        pattern = f"{indicator}\s+(?:in|as|on)?\s+([\w\s]+)"
        matches = re.findall(pattern, text)
        for match in matches:
            match = match.strip()
            if match and match not in [c.lower() for c in certifications_found]:
                certifications_found.append(match.title())
                cert_score += 0.1  # Lower value for general certifications/courses

    # Cap the score at 1.0
    cert_score = min(cert_score, 1.0)

    return cert_score, certifications_found

# 5. Extract and evaluate projects
def extract_projects(text):
    text = text.lower()
    projects = []
    project_score = 0

    # Project indicators
    project_indicators = ['project', 'developed', 'created', 'built', 'implemented', 'designed']

    # Look for project descriptions
    for indicator in project_indicators:
        pattern = f"{indicator}\s+(?:a|an)?\s+([^.;:!?]*)"
        matches = re.findall(pattern, text)
        for match in matches:
            match = match.strip()
            if match and len(match) > 5:
                projects.append(f"{indicator.title()} {match}")
                project_score += 0.15  # Increment for each project

    # Look for project sections
    project_section_pattern = r'(?:projects?|portfolio)\s*:([^#]*?)(?:\n\s*\w+\s*:|$)'
    section_matches = re.findall(project_section_pattern, text, re.IGNORECASE | re.DOTALL)

    if section_matches:
        for section in section_matches:
            # Extract individual projects from the section
            project_items = re.findall(r'[-•*]\s*([^\n]*)', section)
            for item in project_items:
                if item.strip() and len(item) > 10:
                    projects.append(item.strip().title())
                    project_score += 0.15

    # Cap the score at 1.0
    project_score = min(project_score, 1.0)

    return project_score, projects

# 6. Extract and evaluate tools & technologies
def extract_tools_and_technologies(text, skills_list):
    text = text.lower()
    tools_found = []

    # Extract skills from text
    for skill in skills_list:
        if skill.lower() in text:
            tools_found.append(skill)

    # Calculate score based on number of tools found (cap at 20 for full score)
    tools_score = min(len(tools_found) / 20, 1.0)

    return tools_score, tools_found

# 7. Extract and evaluate soft skills & leadership roles
def extract_soft_skills_and_leadership(text):
    text = text.lower()
    soft_skills_found = []
    leadership_roles = []
    combined_score = 0

    # Common soft skills
    soft_skills = [
        'communication', 'teamwork', 'problem solving', 'critical thinking',
        'adaptability', 'flexibility', 'creativity', 'time management',
        'organization', 'collaboration', 'interpersonal', 'presentation',
        'negotiation', 'conflict resolution', 'emotional intelligence'
    ]

    # Leadership indicators
    leadership_indicators = [
        'led', 'managed', 'supervised', 'directed', 'coordinated', 'oversaw',
        'spearheaded', 'headed', 'chaired', 'guided', 'mentored', 'trained'
    ]

    # Check for soft skills
    for skill in soft_skills:
        if skill in text:
            soft_skills_found.append(skill.title())
            combined_score += 0.05  # Small increment for each soft skill

    # Check for leadership roles
    for indicator in leadership_indicators:
        pattern = f"{indicator}\s+([^.;:!?]*)"
        matches = re.findall(pattern, text)
        for match in matches:
            match = match.strip()
            if match and len(match) > 5:
                leadership_roles.append(f"{indicator.title()} {match}")
                combined_score += 0.1  # Higher increment for leadership roles

    # Cap the score at 1.0
    combined_score = min(combined_score, 1.0)

    return combined_score, soft_skills_found, leadership_roles

# Common skills list
COMMON_SKILLS = [
    'Python', 'Java', 'JavaScript', 'C++', 'C#', 'SQL', 'HTML', 'CSS',
    'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'Spring',
    'Machine Learning', 'Deep Learning', 'Data Analysis', 'Data Science',
    'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Git',
    'Agile', 'Scrum', 'Project Management', 'Leadership', 'Communication',
    'Problem Solving', 'Critical Thinking', 'Teamwork', 'Creativity'
]

# Common certifications list
COMMON_CERTIFICATIONS = [
    'AWS Certified', 'Azure Certified', 'Google Cloud Certified',
    'Certified ScrumMaster', 'PMP', 'CISSP', 'CompTIA', 'CCNA', 'CCNP',
    'Oracle Certified', 'Microsoft Certified', 'Certified Ethical Hacker',
    'ITIL', 'Six Sigma', 'CISA', 'CISM', 'CRISC', 'CGEIT', 'CAPM',
    'Agile Certified', 'Scrum Certified', 'PMI-ACP', 'CSM', 'CSPO',
    'CKA', 'CKAD', 'CKS', 'RHCSA', 'RHCE', 'MCSA', 'MCSE', 'MCTS',
    'Salesforce Certified', 'Tableau Certified', 'Hadoop Certified'
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

    # Get job skills and description
    job_skills = job['skills']
    job_description = job['description']

    # Rank applications using the 7 parameters
    for app in job_applications:
        resume_text = app['resume_text']

        # 1. Education evaluation (15% weight)
        education_score, education_details = extract_education(resume_text)

        # 2. Relevant Experience evaluation (20% weight)
        experience_score, job_titles, experience_years, companies, experience_sections = extract_relevant_experience(resume_text, job_description)

        # 3. Achievements & Impact evaluation (15% weight)
        achievements_score, achievements = extract_achievements(resume_text)

        # 4. Certifications & Courses evaluation (10% weight)
        certifications_score, certifications = extract_certifications_and_courses(resume_text, COMMON_CERTIFICATIONS)

        # 5. Projects evaluation (15% weight)
        projects_score, projects = extract_projects(resume_text)

        # 6. Tools & Technologies evaluation (15% weight)
        tools_score, tools = extract_tools_and_technologies(resume_text, COMMON_SKILLS + job_skills)

        # 7. Soft Skills & Leadership evaluation (10% weight)
        soft_skills_score, soft_skills, leadership_roles = extract_soft_skills_and_leadership(resume_text)

        # Calculate overall score with weights for each parameter
        overall_score = (
            0.15 * education_score +
            0.20 * experience_score +
            0.15 * achievements_score +
            0.10 * certifications_score +
            0.15 * projects_score +
            0.15 * tools_score +
            0.10 * soft_skills_score
        )

        # Scale scores from 0-1 to 1-100
        education_score_scaled = max(1, min(100, round(education_score * 100)))
        experience_score_scaled = max(1, min(100, round(experience_score * 100)))
        achievements_score_scaled = max(1, min(100, round(achievements_score * 100)))
        certifications_score_scaled = max(1, min(100, round(certifications_score * 100)))
        projects_score_scaled = max(1, min(100, round(projects_score * 100)))
        tools_score_scaled = max(1, min(100, round(tools_score * 100)))
        soft_skills_score_scaled = max(1, min(100, round(soft_skills_score * 100)))
        overall_score_scaled = max(1, min(100, round(overall_score * 100)))

        # Add scores and extracted information to application
        app['education_score'] = education_score_scaled
        app['experience_score'] = experience_score_scaled
        app['achievements_score'] = achievements_score_scaled
        app['certifications_score'] = certifications_score_scaled
        app['projects_score'] = projects_score_scaled
        app['tools_score'] = tools_score_scaled
        app['soft_skills_score'] = soft_skills_score_scaled
        app['overall_score'] = overall_score_scaled

        # Add extracted details
        app['education_details'] = education_details
        app['job_titles'] = job_titles
        app['experience_years'] = experience_years
        app['companies'] = companies
        app['experience_sections'] = experience_sections
        app['achievements'] = achievements
        app['certifications'] = certifications
        app['projects'] = projects
        app['tools'] = tools
        app['soft_skills'] = soft_skills
        app['leadership_roles'] = leadership_roles

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

    # Check if we're viewing a specific resume in detail
    if 'view_resume_id' in st.session_state and st.session_state.view_resume_id:
        view_resume_detail(ranked_applications, job)
        return

    # Display applications list
    for i, app in enumerate(ranked_applications):
        with st.expander(f"{i+1}. {app['candidate_name']} - Match Score: {app.get('overall_score', 0)}/100"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Applied on:** {app['applied_date']}")

                # Display overall score prominently
                st.markdown(f"### Overall Score: {app.get('overall_score', 0)}/100")

                # Create tabs for different evaluation categories
                eval_tab1, eval_tab2 = st.tabs(["Scores", "Details"])

                with eval_tab1:
                    # Display all 7 parameter scores
                    st.write(f"**1. Education:** {app.get('education_score', 0)}/100")
                    st.write(f"**2. Relevant Experience:** {app.get('experience_score', 0)}/100")
                    st.write(f"**3. Achievements & Impact:** {app.get('achievements_score', 0)}/100")
                    st.write(f"**4. Certifications & Courses:** {app.get('certifications_score', 0)}/100")
                    st.write(f"**5. Projects:** {app.get('projects_score', 0)}/100")
                    st.write(f"**6. Tools & Technologies:** {app.get('tools_score', 0)}/100")
                    st.write(f"**7. Soft Skills & Leadership:** {app.get('soft_skills_score', 0)}/100")

                with eval_tab2:
                    # Display extracted details for each parameter
                    if 'education_details' in app and app['education_details']:
                        st.write(f"**Education:** {', '.join(app['education_details'][:3])}")

                    if 'experience_years' in app:
                        st.write(f"**Years of Experience:** {app.get('experience_years', 0)}")

                    if 'job_titles' in app and app['job_titles']:
                        st.write(f"**Job Titles:** {', '.join(app['job_titles'][:3])}")

                    if 'companies' in app and app['companies']:
                        st.write(f"**Companies:** {', '.join(app['companies'][:3])}")

                    if 'achievements' in app and app['achievements']:
                        st.write(f"**Key Achievements:** {', '.join(app['achievements'][:2])}")

                    if 'certifications' in app and app['certifications']:
                        st.write(f"**Certifications:** {', '.join(app['certifications'][:3])}")

                    if 'projects' in app and app['projects']:
                        st.write(f"**Projects:** {', '.join(app['projects'][:2])}")

                    if 'tools' in app and app['tools']:
                        st.write(f"**Tools & Technologies:** {', '.join(app['tools'][:5])}")

                    if 'soft_skills' in app and app['soft_skills']:
                        st.write(f"**Soft Skills:** {', '.join(app['soft_skills'][:3])}")

                    if 'leadership_roles' in app and app['leadership_roles']:
                        st.write(f"**Leadership:** {', '.join(app['leadership_roles'][:2])}")

                # Show resume file link if available
                if 'resume_file' in app and app['resume_file']:
                    file_name = os.path.basename(app['resume_file'])
                    st.write(f"**Resume File:** {file_name}")

                # Add button to view full resume details
                if st.button(f"View Full Resume", key=f"view_{app['id']}"):
                    st.session_state.view_resume_id = app['id']
                    st.rerun()

            with col2:
                st.write("**Resume Preview:**")
                # Show a preview of the resume (first 500 characters)
                preview_text = app['resume_text'][:500] + "..." if len(app['resume_text']) > 500 else app['resume_text']
                st.text_area("", value=preview_text, height=200, key=f"preview_{app['id']}")

# Function to view a resume in detail
def view_resume_detail(applications, job):
    # Find the selected application
    app = next((a for a in applications if a['id'] == st.session_state.view_resume_id), None)

    if not app:
        st.error("Resume not found")
        if st.button("Back to Applications"):
            st.session_state.view_resume_id = None
            st.rerun()
        return

    # Add back button
    if st.button("← Back to Applications List"):
        st.session_state.view_resume_id = None
        st.rerun()

    # Display resume details
    st.title(f"Resume: {app['candidate_name']}")
    st.subheader(f"Applied for: {job['title']} - {job['company']}")
    st.write(f"**Applied on:** {app['applied_date']}")
    st.write(f"**Overall Match Score:** {app.get('overall_score', 0)}/100")

    # Create tabs for different sections
    resume_tab1, resume_tab2, resume_tab3 = st.tabs(["Evaluation", "Extracted Information", "Full Resume"])

    with resume_tab1:
        # Display evaluation scores with progress bars
        st.subheader("Evaluation Scores")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Education", f"{app.get('education_score', 0)}/100")
            st.progress(app.get('education_score', 0)/100)

            st.metric("Relevant Experience", f"{app.get('experience_score', 0)}/100")
            st.progress(app.get('experience_score', 0)/100)

            st.metric("Achievements & Impact", f"{app.get('achievements_score', 0)}/100")
            st.progress(app.get('achievements_score', 0)/100)

            st.metric("Certifications & Courses", f"{app.get('certifications_score', 0)}/100")
            st.progress(app.get('certifications_score', 0)/100)

        with col2:
            st.metric("Projects", f"{app.get('projects_score', 0)}/100")
            st.progress(app.get('projects_score', 0)/100)

            st.metric("Tools & Technologies", f"{app.get('tools_score', 0)}/100")
            st.progress(app.get('tools_score', 0)/100)

            st.metric("Soft Skills & Leadership", f"{app.get('soft_skills_score', 0)}/100")
            st.progress(app.get('soft_skills_score', 0)/100)

            st.metric("Overall Score", f"{app.get('overall_score', 0)}/100", delta="Match")
            st.progress(app.get('overall_score', 0)/100)

    with resume_tab2:
        # Display all extracted information
        st.subheader("Extracted Information")

        # Education
        if 'education_details' in app and app['education_details']:
            st.write("**🎓 Education:**")
            for edu in app['education_details']:
                st.write(f"- {edu}")

        # Experience
        st.write("**💼 Professional Experience:**")
        if 'experience_years' in app:
            st.write(f"- Years of Experience: {app.get('experience_years', 0)}")

        if 'job_titles' in app and app['job_titles']:
            st.write("**Job Titles:**")
            for title in app['job_titles']:
                st.write(f"- {title}")

        if 'companies' in app and app['companies']:
            st.write("**Companies:**")
            for company in app['companies']:
                st.write(f"- {company}")

        # Achievements
        if 'achievements' in app and app['achievements']:
            st.write("**🏆 Key Achievements:**")
            for achievement in app['achievements']:
                st.write(f"- {achievement}")

        # Certifications
        if 'certifications' in app and app['certifications']:
            st.write("**📜 Certifications & Courses:**")
            for cert in app['certifications']:
                st.write(f"- {cert}")

        # Projects
        if 'projects' in app and app['projects']:
            st.write("**🚀 Projects:**")
            for project in app['projects']:
                st.write(f"- {project}")

        # Tools & Technologies
        if 'tools' in app and app['tools']:
            st.write("**🔧 Tools & Technologies:**")
            for i, tool in enumerate(app['tools']):
                st.write(f"- {tool}")
                if i >= 9:  # Limit to 10 tools
                    st.write(f"- ... and {len(app['tools']) - 10} more")
                    break

        # Soft Skills
        if 'soft_skills' in app and app['soft_skills']:
            st.write("**🤝 Soft Skills:**")
            for skill in app['soft_skills']:
                st.write(f"- {skill}")

        # Leadership
        if 'leadership_roles' in app and app['leadership_roles']:
            st.write("**👑 Leadership Experience:**")
            for role in app['leadership_roles']:
                st.write(f"- {role}")

    with resume_tab3:
        # Display the full resume text
        st.subheader("Full Resume Text")

        # Show resume file link if available
        if 'resume_file' in app and app['resume_file']:
            file_name = os.path.basename(app['resume_file'])
            st.write(f"**Resume File:** {file_name}")

        # Display the full text
        st.text_area("", value=app['resume_text'], height=400)

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
