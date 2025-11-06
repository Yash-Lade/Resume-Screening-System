# Resume Screening System

A streamlined AI-powered resume screening system developed by **Yash Lade**.  
Built as a beginner-friendly Python app to automate the process of screening resumes for job applications.  
The system allows uploading a candidate’s resume, applies preprocessing and machine-learning logic to evaluate suitability for roles, and produces actionable screening/feedback.

---

## 🚀 Features

- Upload a resume file (PDF/DOCX) and extract key information (skills, experience, education).  
- Pre-trained model (based on the dataset included) to classify/resume screening outcome (e.g., “Suitable”, “Needs improvement”, “Not suitable”).  
- Web-based front-end using `Streamlit` (or simple Flask) for intuitive user interaction.  
- Easy to use single file deployment (`app.py`) for starter implementation.  
- Dataset and model included (`UpdatedResumeDataSet.csv`, `models/`) so you can retrain or extend as a beginner.  
- Clear code structure to support understanding of ML pipeline end-to-end (data → feature extraction → model → inference).

---

## 🧑‍💻 Getting Started

### Prerequisites  
- Python 3.7+ installed.  
- (Optional but recommended) A virtual environment (venv/conda) to isolate dependencies.

### Installation  
1. Clone this repository:
   ```bash
   git clone https://github.com/Yash-Lade/Resume-Screening-System.git
   cd Resume-Screening-System
Create and activate a virtual environment (optional but recommended):

bash
Copy code
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
Install required packages:

bash
Copy code
pip install -r requirements.txt
Running the App
bash
Copy code
streamlit run app.py
(or python app.py if Flask/other UI is used)
Then open the browser at http://localhost:8501 (or as shown in the terminal) and upload a resume.

🔍 How It Works

Data ingestion – the dataset UpdatedResumeDataSet.csv contains labelled resume entries for training/validation.

Feature extraction – the code extracts features like number of years of experience, key skills matched to roles, education level, etc.

Model training/inference – a classification model stored under models/ is used to predict screening outcome.

Front-end interface – user uploads a resume file, the back-end runs extraction + model + displays result and key feedback.

📁 Repository Structure
/
│  app.py                  ← main application script
│  app2.py                 ← alternate version / prototype
│  requirements.txt        ← Python dependencies
│  UpdatedResumeDataSet.csv← labeled training dataset
│
├── data/                  ← (optional) raw or processed data files
├── models/                ← serialized ML model(s) + feature extraction artifacts
├── uploads/               ← folder where uploaded resumes are stored (for demo)
└── .vscode/               ← workspace settings (editor config)

🎯 Why This Project?

Traditional resume-screening systems often face limitations: they rely on rigid keyword matching, generic feedback, or expensive commercial solutions. This project aims to democratize resume screening by offering a lightweight, beginner-friendly platform you can extend and customize for different job roles, sectors or languages—without requiring deep ML expertise.

🧠 Use Cases & Extensions

Customizing the model for specific job-roles (e.g., software engineer, data scientist, UX designer).

Adding more sophisticated NLP for resume parsing (e.g., named-entity recognition, semantic similarity).

Building a dashboard to compare candidates, track hiring funnel metrics, visualise screening results.

Integrating with a recruitment platform or website (via API) to automate end-to-end hiring workflows.
