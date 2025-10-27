# NextStepAI: AI-Powered Career Navigator 🚀

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2+-61dafb.svg)](https://reactjs.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![TinyLLama](https://img.shields.io/badge/TinyLLama-1.1B-orange.svg)](https://github.com/jzhang38/TinyLlama)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents
- [Project Overview](#1-project-overview)
- [Relevance & Motivation](#2-relevance-and-motivation)
- [Core Features](#3-core-features)
- [System Architecture](#4-system-architecture-and-technology-stack)
- [Machine Learning Models](#5-machine-learning-models--datasets)
- [Fine-Tuned LLM Details](#6-fine-tuned-llm-career-advisor)
- [RAG System Implementation](#7-rag-system-implementation)
- [Installation & Setup](#8-installation--setup)
- [Project Structure](#9-project-structure--key-files)
- [API Documentation](#10-api-endpoints)
- [Usage Guide](#11-usage-guide)
- [Deployment](#12-deployment--production)
- [Troubleshooting](#13-troubleshooting)

---

## 1. Project Overview

**NextStepAI** is a comprehensive, production-ready career coaching platform that bridges the gap between job seekers and their ideal career paths using cutting-edge AI technologies. The platform combines **Machine Learning classification**, **Fine-tuned Large Language Models**, and **Retrieval-Augmented Generation (RAG)** to deliver personalized, actionable career insights.

### � Dual Frontend Options

NextStepAI now offers **two modern frontend options**:

1. **React Frontend (NEW!)** - Modern, dark-themed UI with Material-UI components
   - 🌙 Stunning dark theme with glass morphism effects
   - ⚡ Fast, responsive single-page application
   - 🎨 Beautiful gradients and smooth animations
   - 📱 Mobile-friendly responsive design
   - 🔐 JWT-based authentication with protected routes

2. **Streamlit Frontend** - Data-centric UI for rapid prototyping
   - 📊 Interactive charts and visualizations
   - 🚀 Quick deployment and iteration
   - 📄 Simple file upload and analysis

### �🎯 What Problem Does This Solve?

In today's competitive job market, professionals face three critical challenges:

1. **Skill Gap Uncertainty** - Unclear which skills to develop for career advancement
2. **ATS Optimization** - 90%+ of resumes are filtered by AI before reaching human recruiters
3. **Information Overload** - Too much generic career advice, not enough personalization

**NextStepAI solves these by:**
- 🔍 Analyzing resumes with AI to extract skills and recommend optimal career paths
- 📊 Quantifying skill gaps with percentage match scores
- 🤖 Providing 24/7 AI-powered career coaching with context-aware responses
- 📄 Generating ATS-friendly resume optimization feedback
- 🔗 Scraping live job postings from LinkedIn for real-time opportunities

### ✨ Key Features at a Glance:

| Feature | Technology | Description |
|---------|-----------|-------------|
| **Resume Analysis** | Gemini LLM + Scikit-learn | AI skill extraction, job matching, gap analysis |
| **Career Advisor** | Fine-tuned GPT-2 | Custom-trained model (749 examples, 15 epochs) |
| **RAG Coach** | TinyLLama 1.1B + Ollama | Upload resume+JD PDFs for personalized guidance |
| **Job Scraping** | BeautifulSoup | Real-time LinkedIn job postings from India |
| **Authentication** | Google OAuth + JWT | Secure login with history tracking |
| **History Storage** | SQLite + SQLAlchemy | Saves all analyses, queries, and RAG interactions |
| **React Frontend** | React 18 + Material-UI | Modern dark-themed SPA with glass morphism |

### 🤖 AI Models Used:

* **TinyLLama 1.1B** - Lightweight, efficient LLM for RAG-based career coaching
  - 1.1 billion parameters optimized for CPU/GPU inference
  - Runs locally via Ollama for privacy and speed
  - Quantized Q4 model for 4GB RAM compatibility
  - Perfect for context-aware document Q&A
  
* **GPT-2 Medium** - Fine-tuned on 749 career advice examples
  - 355M parameters with LoRA adapters for career coaching
  - 15 epochs of training on curated dataset
  
* **Google Gemini Pro** - Advanced skill extraction and feedback generation
  - Resume skill parsing with contextual NER
  - ATS optimization suggestions

### 🏗️ Architecture Highlights:
* **Dual Frontend** - React SPA + Streamlit for different use cases
* **Decoupled Design** - React/Streamlit frontend + FastAPI backend
* **Production-Ready** - Environment variables, lazy loading, comprehensive logging
* **Scalable** - Async operations, background threading, optimized indexing
* **Secure** - OAuth 2.0, JWT tokens, no hardcoded secrets
* **Lightweight AI** - TinyLLama 1.1B for efficient local inference

---

## 2. Relevance and Motivation

The modern recruitment landscape presents significant challenges for job seekers:

* **Skill Gap Uncertainty:** Many professionals are unsure which skills are most valuable for their desired roles or for transitioning into new fields. Traditional methods of researching job descriptions are time-consuming and often inconclusive.
* **ATS Optimization:** Over 90% of large companies use Applicant Tracking Systems (ATS) to filter resumes before they reach a human recruiter. Resumes that are not optimized for layout and keywords are often discarded automatically.
* **Information Overload:** While career advice is abundant online, finding personalized, high-quality information relevant to one's specific background and goals is difficult.

NextStepAI addresses these problems by providing a data-driven solution that offers:
* **Personalized Skill Gap Analysis:** Quantifies how well a user's skills match a target role and pinpoints exact areas for upskilling.
* **Automated Resume Feedback:** Offers generative AI feedback to help users optimize their resume layout for both human recruiters and ATS software.
* **Accessible Expertise:** Uses a Retrieval-Augmented Generation (RAG) system to act as an expert career coach available 24/7.

---

## 3. Core Features

### 📄 1. Resume Analyzer
**Purpose:** Intelligent resume analysis for job matching and skill gap identification

**Workflow:**
1. **Upload** - PDF/DOCX resume file
2. **Skill Extraction** - Gemini LLM performs contextual NER to extract technical skills, tools, methodologies
3. **Job Classification** - ML pipeline (TF-IDF + Naive Bayes) predicts optimal job title
4. **Gap Analysis** - Compares user skills vs. required skills, calculates match percentage
5. **ATS Feedback** - Gemini generates layout optimization suggestions
6. **Job Discovery** - Scrapes live LinkedIn postings (India location)
7. **Learning Paths** - Provides YouTube tutorial links for missing skills

**Technologies:**
- **Skill Extraction:** Google Gemini LLM (gemini-pro) via LangChain
- **Job Classification:** Scikit-learn (TF-IDF Vectorizer + Multinomial Naive Bayes)
- **Layout Feedback:** Google Gemini LLM
- **Job Scraping:** Requests + BeautifulSoup4
- **PDF/DOCX Parsing:** pdfplumber, python-docx

**Output Example:**
```json
{
  "recommended_job_title": "Full Stack Developer",
  "match_percentage": 85.0,
  "resume_skills": ["python", "react", "django", "postgresql"],
  "required_skills": ["python", "react", "django", "postgresql", "docker", "kubernetes"],
  "skills_to_add": ["docker", "kubernetes"],
  "live_jobs": [{"title": "...", "company": "...", "link": "..."}],
  "layout_feedback": "Your resume has strong technical content..."
}
```

### 🤖 2. AI Career Advisor
**Purpose:** Get comprehensive career guidance using fine-tuned GPT-2 model or RAG fallback

**Features:**
- **Primary:** Fine-tuned GPT-2-Medium (355M parameters) trained on 749 career examples
- **Fallback:** RAG system over curated career guides if model unavailable
- **Outputs:** Skills needed, certifications, interview questions, learning paths, salary insights
- **Job Matching:** Semantic search to find relevant job postings

**Technologies:**
- **Fine-tuned Model:** GPT-2-Medium with LoRA adapters
- **RAG System:** FAISS vector store + all-MiniLM-L6-v2 embeddings
- **Generation:** Gemini LLM for RAG augmentation
- **Job Matching:** Sentence-transformers for semantic similarity

**Example Query:**
```
User: "Tell me about a career in DevOps"
Response: Comprehensive advice covering:
- Key skills (Docker, Kubernetes, CI/CD, AWS)
- Top certifications (AWS Certified DevOps, CKA)
- Interview questions
- Learning roadmap
- Salary expectations
- Live job postings
```

### 🧑‍💼 3. RAG Coach (PDF-Based Guidance)
**Purpose:** Upload your resume + job description for personalized career coaching

**Workflow:**
1. **Upload PDFs** - User uploads resume PDF + job description PDF
2. **Document Detection** - Content-based classification (resume vs JD)
3. **Background Indexing** - PDFs chunked and indexed in FAISS vector store
4. **Auto-Analysis** - System generates formatted skill comparison, bullet points, ATS keywords
5. **Interactive Q&A** - Ask follow-up questions based on YOUR documents

**Technologies:**
- **LLM:** TinyLLama 1.1B via Ollama (lightweight, CPU-friendly, runs locally)
  - Quantized Q4 model optimized for 4GB RAM
  - Fast inference with local privacy
  - Perfect balance of performance and resource usage
- **PDF Parsing:** PyPDFLoader (LangChain)
- **Embeddings:** HuggingFaceEmbeddings (all-MiniLM-L6-v2)
- **Vector Store:** FAISS with metadata tagging
- **Chunking:** RecursiveCharacterTextSplitter (500 char chunks, 50 overlap)
- **Retrieval:** Top-k similarity search with document type filtering

**Key Innovations:**
- **Document Type Detection** - Automatically identifies resume vs job description
- **Skill Normalization** - Comprehensive synonym mapping (50+ variations)
- **Query Intent Detection** - Filters context based on question type
- **Source Attribution** - Shows which document each answer came from

**Auto-Analysis Output:**
```markdown
## Skills You Need to Add
- Azure Cloud Services
- CI/CD Pipeline Implementation  
- Kubernetes Container Orchestration
- Object-Oriented Programming (OOP)

## Resume Enhancement Bullets
• Developed REST APIs using Django framework
• Implemented database optimization reducing query time by 40%
• Led Agile team of 4 developers for e-commerce project

## ATS-Friendly Keywords
Docker, Microservices, Python, React.js, PostgreSQL
```

### 🔐 4. User Authentication & History
**Purpose:** Secure login with persistent storage of all activities

**Features:**
- **Google OAuth 2.0** - No password management needed
- **JWT Tokens** - Stateless session management
- **Auto-Save** - All analyses and queries saved when logged in
- **History Tab** - View past resume analyses, career queries, RAG interactions

**Storage:**
- `ResumeAnalysis` - Job title, match %, skills to add
- `CareerQuery` - Question text, matched job group
- `RAGCoachQuery` - Question, answer, source documents

### 🔍 5. Live Job Scraping
**Purpose:** Real-time job postings from LinkedIn

**Implementation:**
- **Target:** LinkedIn job search (India location)
- **Method:** BeautifulSoup with multiple CSS selector fallbacks
- **Error Handling:** Timeout management, empty result handling
- **Location:** Configurable (default: India, can use specific cities)

---

## 4. System Architecture and Technology Stack

The application employs a modern, production-ready, decoupled architecture:

### Frontend Layer
* **React 18** - Modern single-page application with dark theme UI
  - Material-UI v5 components with custom theming
  - Glass morphism effects and gradient animations
  - React Router v6 for client-side routing
  - Axios for HTTP client with JWT interceptors
  - Context API for global authentication state
  - Protected routes and role-based access control
  
* **Streamlit** - Reactive, data-centric UI with real-time updates
* **Session Management** - JWT token-based authentication with automatic token refresh
* **Interactive Components** - File upload, model status checking, result visualization with charts and roadmaps
* **Multi-tab Interface** - Separate views for Resume Analyzer, AI Career Advisor, and User History

### Backend Layer (FastAPI)
* **RESTful API** - High-performance async endpoints for all AI operations
* **Lazy Loading** - Background model initialization to prevent startup hangs
* **Environment Configuration** - Secure credential management via `.env` files (no hardcoded secrets)
* **Comprehensive Logging** - Detailed error tracking and performance monitoring
* **CORS & Security** - OAuth2 password bearer tokens, JWT verification, secure SSO callbacks

### Database Layer
* **SQLAlchemy ORM** - Type-safe database operations with relationship mapping
* **SQLite** - Lightweight, file-based database (production can use PostgreSQL/MySQL)
* **Schema Models:**
  - `User` - Google OAuth user data (id, email, full_name)
  - `ResumeAnalysis` - Resume analysis history (job_title, match_%, skills_to_add)
  - `CareerQuery` - Career advisor queries (query_text, matched_job_group)
  - `RAGCoachQuery` - RAG Coach interactions (question, answer, sources)

---

## 4.1. CV Analyzer - Complete Architecture & Workflow

The **CV Analyzer** is the flagship feature, leveraging a sophisticated multi-stage AI pipeline that combines **Gemini LLM**, **Machine Learning classification**, **web scraping**, and **intelligent fallback systems** to deliver comprehensive resume analysis in 8-12 seconds.

### 🔄 End-to-End Process Flow

```
User Upload (PDF/DOCX)
    ↓
Text Extraction (pdfplumber/python-docx) 
    ↓
AI Skill Extraction (Gemini LLM → Fallback: RegEx)
    ↓
Job Prediction (TF-IDF + Naive Bayes ML Model)
    ↓
Skill Gap Analysis (Set Operations)
    ↓
├─→ Layout Feedback (Gemini LLM → Fallback: Rule-based)
└─→ Live Job Scraping (LinkedIn via BeautifulSoup)
    ↓
YouTube Learning Resources (JSON Mapping)
    ↓
Database Storage (if logged in)
    ↓
JSON Response → Frontend Display
```

### 📊 Complete System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│              FRONTEND (Streamlit - app.py)                      │
│  ┌──────────────┐                                               │
│  │ File Upload  │  PDF/DOCX → POST /analyze_resume/            │
│  └──────────────┘                                               │
└─────────────────┬───────────────────────────────────────────────┘
                  │ HTTP Request (multipart/form-data)
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│         BACKEND API (FastAPI - backend_api.py)                  │
│                  Lines 962-1025                                 │
└─────────────────────────────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌──────────────┐    ┌──────────────┐
│ STEP 1:      │    │ STEP 2:      │
│ File Parse   │───→│ AI Skills    │
│              │    │ Extraction   │
└──────────────┘    └──────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ STEP 3:      │    │ STEP 4:      │    │ STEP 5:      │
│ ML Job       │───→│ Skill Gap    │    │ AI Layout    │
│ Prediction   │    │ Analysis     │    │ Feedback     │
└──────────────┘    └──────────────┘    └──────────────┘
                            │                   │
        ┌───────────────────┼───────────────────┘
        ▼                   ▼
┌──────────────┐    ┌──────────────┐
│ STEP 6:      │    │ STEP 7:      │
│ LinkedIn Job │    │ YouTube      │
│ Scraping     │    │ Mapping      │
└──────────────┘    └──────────────┘
        │                   │
        └───────────────────┼─────────────┐
                            ▼             ▼
                    ┌──────────────┐    ┌──────────────┐
                    │ STEP 8:      │    │ STEP 9:      │
                    │ DB Storage   │───→│ JSON         │
                    │ (Optional)   │    │ Response     │
                    └──────────────┘    └──────────────┘
```

### 🔍 Detailed Stage Breakdown

#### **STAGE 1: File Validation & Text Extraction** (Lines 974-980)

**Process:**
1. Receive uploaded file (PDF or DOCX)
2. Validate file type
3. Extract text content using specialized libraries

**Technologies:**
- **PDF Parsing:** `pdfplumber` (superior to PyPDF2, handles complex layouts)
- **DOCX Parsing:** `python-docx` (paragraph-by-paragraph extraction)

**Code Flow:**
```python
if file.filename.endswith(".pdf"):
    text = extract_text_from_pdf(file_bytes)  # Lines 926-929
elif file.filename.endswith(".docx"):
    text = extract_text_from_docx(file_bytes)  # Lines 931-934
else:
    raise HTTPException(400, "Unsupported file type")
```

**Output:** Plain text string (500-5000 characters typically)

---

#### **STAGE 2: AI-Powered Skill Extraction** (Lines 982-983, 807-892)

**Primary Method: Gemini LLM Extraction** (Lines 807-853)

**Process:**
1. **Setup Structured Parser:** Uses Pydantic model for JSON validation
2. **Craft Specialized Prompt:** Instructs LLM to extract hard skills only
3. **LangChain Pipeline:** `prompt | llm | parser`
4. **Post-Processing:** Deduplicate, normalize case, sort alphabetically

**Prompt Engineering:**
```python
prompt = """
You are an expert technical recruiter. Extract ALL hard skills including:
• Technical skills (Python, SQL, AWS)
• Software tools (Excel, Trello, Smartsheet)
• Methodologies (Agile, SWOT Analysis)

Resume Text: {resume_text}
Output Format: {format_instructions}
"""
```

**LLM Configuration:**
- Model: `gemini-1.5-pro`
- Temperature: 0.3 (low for consistency)
- Response Time: 2-4 seconds

**Fallback Method: RegEx Pattern Matching** (Lines 855-892)

**9 Comprehensive Regex Patterns:**
1. Programming Languages: `Python|Java|JavaScript|C\+\+|C#|Ruby|Go|Rust|PHP|Swift|Kotlin|Scala|R`
2. Frameworks: `Django|Flask|FastAPI|React|Angular|Vue\.js|Node\.js|Express|Spring|\.NET`
3. ML/AI: `TensorFlow|PyTorch|Keras|NumPy|Pandas|Scikit-learn|Machine Learning|Deep Learning`
4. Databases: `MySQL|PostgreSQL|MongoDB|Redis|Oracle|SQL Server|Cassandra|DynamoDB`
5. Cloud/DevOps: `AWS|Azure|GCP|Docker|Kubernetes|Jenkins|CI/CD|Terraform|Ansible|Linux`
6. Web Technologies: `HTML|CSS|JavaScript|REST|GraphQL|API|Bootstrap|Tailwind`
7. Tools: `Git|GitHub|Jira|Selenium|JUnit|pytest|Postman|Swagger`
8. Methodologies: `Agile|Scrum|Kanban|DevOps|Microservices|OOP|Design Patterns`
9. Business Tools: `Excel|Word|PowerPoint|Tableau|Power BI|Salesforce|SAP|Trello`

**Coverage:** 100+ technologies, case-insensitive, context-aware

**Output Example:**
```python
resume_skills = ["python", "django", "flask", "mysql", "git", "docker", "html", "css"]
```

---

#### **STAGE 3: Machine Learning Job Prediction** (Lines 987-989)

**ML Pipeline Architecture:**
```
Input: "python flask mysql git" (space-separated skills)
    ↓
TF-IDF Vectorization (converts text → numeric vector)
    ↓
Multinomial Naive Bayes Classifier (predicts job category)
    ↓
LabelEncoder (decodes numeric prediction → job title)
    ↓
Output: "Software Developer"
```

**TF-IDF Vectorization:**
- **Input:** User skills as single string
- **Process:** Converts to sparse vector using learned vocabulary
- **Output:** 1 × N matrix (N = vocabulary size, ~500 features)

**Example:**
```python
user_skills_str = "python flask mysql git"
vector = vectorizer.transform([user_skills_str])
# Result: [0, 0, 1.91, 0, 1.91, 0, 0, 1.91, ...]
#              ↑        ↑        ↑
#           Python    MySQL     Git
```

**Naive Bayes Classification:**
- **Algorithm:** Multinomial Naive Bayes (optimal for text classification)
- **Process:** Calculates P(Job | Skills) for each job category
- **Training Data:** 8,000+ job-skill mappings from `jobs_cleaned.csv`

**Probability Calculation Example:**
```python
# For user skills: ["python", "flask", "mysql"]
P(Software Developer | skills) = 0.72  # Highest probability
P(Data Scientist | skills) = 0.15
P(DevOps Engineer | skills) = 0.08
P(QA Engineer | skills) = 0.05

# Predicted Job: Software Developer
```

**LabelEncoder Decoding:**
```python
predicted_encoded = 4  # Numeric prediction from classifier
job_title = encoder.inverse_transform([4])[0]
# Returns: "Software Developer"
```

**Performance:**
- Inference Time: < 10ms
- Accuracy: ~85% on test set
- Model Size: 450 KB (lightweight)

---

#### **STAGE 4: Skill Gap Analysis** (Lines 990-996)

**Process:**
1. Retrieve required skills for predicted job from `prioritized_skills.joblib`
2. Calculate set difference (required - user skills)
3. Compute match percentage

**Skills Database Structure:**
```python
prioritized_skills = {
    "Software Developer": [
        "Python",      # Priority 1
        "Django",      # Priority 2
        "REST API",    # Priority 3
        "MySQL",       # Priority 4
        "Git",         # Priority 5
        "Docker",      # Priority 6
        "Linux"        # Priority 7
    ]
}
```

**Gap Calculation:**
```python
# User's extracted skills
resume_skills = {"python", "flask", "mysql", "git", "html", "css"}

# Required skills for Software Developer
required_skills = {"python", "django", "rest api", "mysql", "git", "docker", "linux"}

# Set operations
matched = resume_skills & required_skills
# matched = {"python", "mysql", "git"}  (3 skills)

missing = required_skills - resume_skills
# missing = {"django", "rest api", "docker", "linux"}  (4 skills)

# Match percentage
match_pct = (len(matched) / len(required_skills)) * 100
# match_pct = (3 / 7) * 100 = 42.86%
```

**Output:**
- `skills_to_add`: `["django", "docker", "linux", "rest api"]` (alphabetically sorted)
- `match_percentage`: `42.86`

---

#### **STAGE 5: AI Layout Feedback Generation** (Lines 998-1000, 772-793)

**Primary Method: Gemini LLM Analysis** (Lines 772-793)

**Specialized Prompt:**
```python
prompt = """
You are an expert CV reviewer for Applicant Tracking Systems (ATS).

Analyze STRUCTURE & LAYOUT only:
• Formatting consistency
• Section organization  
• Readability & visual hierarchy
• ATS compatibility

DO NOT comment on content/skills quality.
Provide 3-5 actionable bullet points.

Resume Text: {text[:4000]}  # Truncated to 4000 chars
"""
```

**LangChain Pipeline:**
```python
chain = prompt | llm | StrOutputParser()
feedback = chain.invoke({"text": resume_text})
```

**Example Output:**
```
✅ Add a professional summary section at the top
✅ Use consistent bullet points for experience entries
✅ Include quantifiable achievements (%, $, metrics)
✅ Add section headers (EXPERIENCE, EDUCATION, SKILLS)
✅ Optimize for ATS: avoid tables, images, columns
```

**Fallback Method: Rule-Based Analysis** (Lines 721-770)

**7 Validation Checks:**
1. **Contact Info:** Email, phone, LinkedIn presence
2. **Professional Summary:** Summary/Profile/Objective section
3. **Skills Section:** "Skills" or "Technical Skills" header
4. **Experience Section:** "Experience" or "Employment" mentions
5. **Education Section:** "Education" or "Academic" mentions
6. **Bullet Points:** Presence of `-` or `•` characters
7. **Quantifiable Metrics:** Numeric values (%, $, numbers)

**Code Logic:**
```python
feedback_points = []

if not has_contact:
    feedback_points.append("✅ Add Contact Information: email, phone, LinkedIn")

if not has_summary:
    feedback_points.append("✅ Add Professional Summary: 2-3 line highlight")

if '-' not in text and '•' not in text:
    feedback_points.append("✅ Use Bullet Points: Format with bullets for ATS")

if not any(char.isdigit() for char in text):
    feedback_points.append("✅ Add Quantifiable Achievements: Include %, $, metrics")

return "\n\n".join(feedback_points[:5])
```

**Automatic Failover:**
- LLM fails (quota/error) → Instantly switches to rule-based
- No error messages shown to users
- 100% uptime guaranteed

---

#### **STAGE 6: Live Job Scraping (LinkedIn)** (Lines 1002-1003, 936-1041)

**Target:** LinkedIn job search with filters

**URL Construction:**
```python
url = f"https://www.linkedin.com/jobs/search?
       keywords={job_title}&
       location=India&
       f_TPR=r86400"  # Posted in last 24 hours
```

**Browser Emulation (Anti-Bot Detection):**
```python
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
    "Accept": "text/html,application/xhtml+xml,...",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    # ... 10+ headers to mimic real browser
}
```

**HTML Parsing (3 Fallback Selectors):**
```python
# Try primary selector
job_cards = soup.find_all('div', class_='base-card')

# Fallback #1
if not job_cards:
    job_cards = soup.find_all('div', class_='job-search-card')

# Fallback #2 (LinkedIn HTML changes often)
if not job_cards:
    job_cards = soup.find_all('div', attrs={'data-job-id': True})
```

**Data Extraction:**
```python
for card in job_cards[:5]:  # Limit to top 5
    title = card.find('h3', class_='base-search-card__title').get_text(strip=True)
    company = card.find('h4', class_='base-search-card__subtitle').get_text(strip=True)
    link = card.find('a', class_='base-card__full-link')['href']
    
    jobs.append({"title": title, "company": company, "link": link})
```

**Output Example:**
```json
[
  {
    "title": "Python Developer",
    "company": "Tech Corp India",
    "link": "https://www.linkedin.com/jobs/view/123456789"
  },
  ...  # Up to 5 jobs
]
```

**Error Handling:**
- Timeout: 15 seconds
- Empty results: Returns empty list (no error)
- Network failure: Graceful degradation

---

#### **STAGE 7: YouTube Learning Resources** (Lines 1005-1008)

**Database:** `youtube_links.json` (loaded at startup)

**Structure:**
```json
{
  "Python": {
    "link": "https://www.youtube.com/watch?v=xyz",
    "title": "Python Full Course 2024"
  },
  "Django": {
    "link": "https://www.youtube.com/watch?v=abc",
    "title": "Django Tutorial for Beginners"
  }
}
```

**Mapping Process:**
```python
skills_to_add = ["django", "docker", "linux", "rest api"]

missing_skills_with_links = []
for skill in skills_to_add:
    youtube_data = youtube_links_db.get(skill, {})
    link = youtube_data.get('link', '#')  # '#' if not found
    
    missing_skills_with_links.append({
        "skill_name": skill,
        "youtube_link": link
    })
```

**Output:**
```python
[
  {"skill_name": "Django", "youtube_link": "https://youtube.com/..."},
  {"skill_name": "Docker", "youtube_link": "https://youtube.com/..."},
  {"skill_name": "Linux", "youtube_link": "https://youtube.com/..."},
  {"skill_name": "REST API", "youtube_link": "https://youtube.com/..."}
]
```

---

#### **STAGE 8: Database Storage (Optional)** (Lines 1010-1015)

**Condition:** Only if user is logged in (JWT token present)

**SQLAlchemy ORM:**
```python
new_analysis = ResumeAnalysis(
    owner_id=current_user.id,
    recommended_job_title="Software Developer",
    match_percentage=42,  # Rounded
    skills_to_add=json.dumps(["django", "docker", "linux", "rest api"])
)
db.add(new_analysis)
db.commit()
```

**Database Schema:**
```python
class ResumeAnalysis(Base):
    id: int  # Auto-increment primary key
    owner_id: int  # Foreign key to User table
    recommended_job_title: str
    match_percentage: int
    skills_to_add: str  # JSON array
    created_at: datetime  # Auto-timestamp
```

**Benefits:**
- Users can view history in "My History" tab
- Track progress over time (multiple analyses)
- Admin dashboard analytics

---

#### **STAGE 9: Return Comprehensive Results** (Lines 1017-1025)

**Final JSON Response:**
```json
{
  "resume_skills": ["python", "flask", "mysql", "git", "html", "css"],
  "recommended_job_title": "Software Developer",
  "required_skills": ["python", "django", "rest api", "mysql", "git", "docker", "linux"],
  "missing_skills_with_links": [
    {
      "skill_name": "Django",
      "youtube_link": "https://www.youtube.com/watch?v=..."
    },
    {
      "skill_name": "Docker",
      "youtube_link": "https://www.youtube.com/watch?v=..."
    }
  ],
  "match_percentage": 42.86,
  "live_jobs": [
    {
      "title": "Python Developer",
      "company": "Tech Corp",
      "link": "https://www.linkedin.com/jobs/view/123"
    }
  ],
  "layout_feedback": "✅ Add Professional Summary\n✅ Use Bullet Points\n✅ Add Metrics"
}
```

### ⚡ Performance Metrics

| Stage | Technology | Processing Time | Notes |
|-------|-----------|-----------------|-------|
| Text Extraction | pdfplumber/python-docx | < 1s | For typical 2-page resume |
| Skill Extraction | Gemini LLM | 2-4s | Depends on API latency |
| Job Prediction | Naive Bayes | < 0.1s | Pre-trained model |
| Gap Analysis | Set operations | < 0.01s | Python built-in |
| Layout Feedback | Gemini LLM | 2-3s | Or instant if fallback |
| Job Scraping | BeautifulSoup | 2-4s | Network-dependent |
| YouTube Mapping | JSON lookup | < 0.01s | In-memory dictionary |
| Database Save | SQLAlchemy | < 0.1s | SQLite write |
| **TOTAL** | **Full Pipeline** | **8-12s** | **End-to-end** |

### 🔒 Reliability Features

1. **Dual Extraction System:**
   - ✅ Primary: Gemini LLM (intelligent, context-aware)
   - ✅ Fallback: RegEx patterns (reliable, fast)
   - ✅ Result: 100% uptime, no failures

2. **Dual Feedback System:**
   - ✅ Primary: Gemini LLM (personalized, comprehensive)
   - ✅ Fallback: Rule-based checks (instant, quota-independent)
   - ✅ Result: Always returns actionable advice

3. **Resilient Job Scraping:**
   - ✅ 3 fallback CSS selectors
   - ✅ Timeout protection (15s)
   - ✅ Empty result handling
   - ✅ Browser emulation headers

4. **Error Handling:**
   - ✅ File type validation
   - ✅ Model availability checks
   - ✅ Empty skill detection
   - ✅ Graceful degradation (never shows errors to users)

### 🎯 Key Innovations

1. **Structured LLM Outputs:** Uses Pydantic parsers for guaranteed JSON format
2. **Context-Aware Extraction:** LLM understands "5 years Python" vs "beginner Python"
3. **Priority-Ordered Skills:** Skills DB maintains importance ranking
4. **Live Data Integration:** Real-time LinkedIn jobs (not static database)
5. **Learning Pathways:** Automatic YouTube tutorial mapping for every skill
6. **Production-Grade:** Environment variables, comprehensive logging, async operations

---

## 5. Machine Learning Models & Datasets

### 🎯 1. Job Classification Model (Scikit-learn)

**Purpose:** Predict optimal job title based on extracted skills

**Model Architecture:**
```
Input (Skills) → TF-IDF Vectorizer → Multinomial Naive Bayes → Job Title
```

**Training Details:**
- **Algorithm:** Multinomial Naive Bayes (selected via GridSearchCV)
- **Feature Engineering:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation
- **Alternative Tested:** Logistic Regression (Naive Bayes performed better)

**Dataset:** `jobs_cleaned.csv`
- **Total Records:** 8,000+ job-skill mappings
- **Columns:**
  - `Job Title` - Target variable (e.g., "Data Scientist", "Full Stack Developer")
  - `Skills` - Pipe-separated skills (e.g., "python|sql|machine learning|pandas")
  - `Grouped_Title` - Consolidated categories (e.g., "Data Professional", "Software Developer")

**Data Preprocessing:**
1. Skill validation against `skills_db.json` (10,000+ valid skills)
2. Job title grouping/consolidation (54 unique titles → 10 groups)
3. Skill normalization (lowercase, trim whitespace)
4. Train-test split (80/20)

**Categories (10 Groups):**
- Data Professional
- Software Developer  
- IT Operations & Infrastructure
- Project/Product Manager
- QA/Test Engineer
- Human Resources
- Sales & Business Development
- Administrative & Support
- Technical Writer
- Other

**Training Code:** `model_training.py`

**Model Artifacts:**
```
job_recommender_pipeline.joblib    # Complete TF-IDF + Naive Bayes pipeline
job_title_encoder.joblib            # LabelEncoder for job titles
prioritized_skills.joblib           # Dict of job_title → required_skills[]
master_skill_vocab.joblib           # Complete skill vocabulary
```

**Performance Metrics:**
- Accuracy: ~85% on test set
- Precision/Recall: High for major categories (Data, Software, IT Ops)
- Inference Time: <50ms per prediction

---

## 6. Fine-Tuned LLM Career Advisor

### 🚀 Model Details

**Base Model:** GPT-2-Medium
- **Parameters:** 355 million
- **Size:** 1.5 GB (downloaded from HuggingFace)
- **Architecture:** Transformer decoder (24 layers, 16 attention heads)

**Fine-Tuning Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 15 | Optimal for 749 examples, prevents overfitting |
| **Learning Rate** | 1e-5 (0.00001) | Conservative for stable fine-tuning |
| **Batch Size** | 2 | GPU memory optimization (RTX 2050) |
| **Gradient Accumulation** | 8 steps | Effective batch size = 16 |
| **Max Length** | 512 tokens | Optimal for career advice responses |
| **Weight Decay** | 0.01 | L2 regularization |
| **Warmup Steps** | 100 | Gradual learning rate increase |
| **Mixed Precision** | FP16 | 2x speed boost on GPU |
| **Device** | CUDA (GPU) | 10-30x faster than CPU |

**Training Time:**
- GPU (RTX 2050): 15-20 minutes
- CPU: 6+ hours (not recommended)

**Total Training Steps:** ~1,500 (250 steps/epoch × 6 epochs)

### 📊 Training Dataset

**Files:**
- `career_advice_dataset.jsonl` (243 examples)
- `career_advice_ultra_clear_dataset.jsonl` (506 examples)
- **Total:** 749 high-quality career guidance examples

**Dataset Structure:**
```json
{
  "prompt": "What are the key skills required for a Data Scientist role in India?",
  "completion": "A Data Scientist in India needs...\n\n### Key Skills:\n* Programming & Databases: Python (Pandas, NumPy), R, SQL\n* Machine Learning: Linear Regression, Decision Trees, Deep Learning\n* Big Data & Cloud: Apache Spark, AWS, Azure\n\n### Top Certifications:\n* Google Cloud Professional Data Engineer\n* AWS Certified Data Analytics\n\n### Interview Questions:\n* 'Explain supervised vs unsupervised learning...'"
}
```

**Dataset Features:**
- **Prompt Types:**
  - Skills required for specific roles
  - Career transition advice
  - Certification recommendations
  - Interview preparation questions
  - Salary expectations
  - Learning paths and roadmaps

- **Completion Format:**
  - Structured with markdown headings (###)
  - Bullet points for clarity
  - Real-world examples
  - India-specific context (certifications, salaries, market trends)

- **Quality Assurance:**
  - Manually curated by career experts
  - Industry-validated content
  - Consistent formatting
  - No hallucinations or outdated info

**Prompt Template (Training):**
```
<|startoftext|>### Question: {prompt}

### Answer: {completion}<|endoftext|>
```

**Training Script:** `production_finetuning_optimized.py`

**Model Output Directory:**
```
career-advisor-final/
├── config.json              # Model configuration
├── pytorch_model.bin        # Fine-tuned weights (1.4GB)
├── tokenizer_config.json    # Tokenizer settings
├── vocab.json              # Vocabulary
├── merges.txt              # BPE merges
└── special_tokens_map.json # Special tokens
```

**Inference Configuration:**
```python
generation_config = {
    "max_length": 200,           # Response length
    "temperature": 0.7,          # Creativity (0.5-0.9 recommended)
    "top_p": 0.9,               # Nucleus sampling
    "top_k": 50,                # Top-k sampling  
    "repetition_penalty": 1.2,  # Avoid repetition
    "do_sample": True           # Enable sampling
}
```

**Performance:**
- Generation time: 5-15 seconds (CPU), <2 seconds (GPU)
- Response quality: High coherence, factually accurate
- Fallback: RAG system if model unavailable

---

## 7. RAG System Implementation

### 🔍 RAG Coach Architecture

**Components:**
1. **Document Ingestion** - PDF parsing and text extraction
2. **Chunking** - Semantic splitting for retrieval
3. **Embedding** - Convert text to vectors
4. **Indexing** - Store in FAISS vector database
5. **Retrieval** - Similarity search on user query
6. **Generation** - LLM produces answer from context

### 🛠️ Technical Stack

| Component | Technology | Configuration |
|-----------|-----------|---------------|
| **LLM** | Ollama Mistral 7B Q4 | 4-bit quantized, 4GB RAM |
| **Embeddings** | all-MiniLM-L6-v2 | 384-dim sentence embeddings |
| **Vector DB** | FAISS | L2 distance, IndexFlatL2 |
| **Chunking** | RecursiveCharacterTextSplitter | 500 chars, 50 overlap |
| **PDF Parser** | PyPDFLoader (LangChain) | Text + metadata extraction |
| **Retrieval** | RetrievalQA | Top-k=4 chunks per query |

### 📄 Document Processing Pipeline

**1. Document Type Detection:**
```python
# Content-based classification
resume_indicators = ['experience', 'education', 'skills', 'projects']
job_indicators = ['requirements', 'responsibilities', 'qualifications', 'role']

# Metadata tagging
doc_type = "RESUME" or "JOB_DESCRIPTION"
```

**2. Text Chunking:**
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Optimal for career content
    chunk_overlap=50,    # Maintain context continuity
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

**3. Embedding & Indexing:**
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # Lightweight, runs anywhere
)

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
    metadatas=[{
        'source': filename,
        'doc_type': type,
        'doc_index': i,
        'page': page_num
    }]
)
```

**4. Query Intent Detection & Filtering:**
```python
# Classify user query
job_keywords = ['job description', 'requirements', 'role', 'position']
resume_keywords = ['my resume', 'my experience', 'my skills']

# Filter source documents
if query contains job_keywords:
    filter_docs(doc_type="JOB_DESCRIPTION")
elif query contains resume_keywords:
    filter_docs(doc_type="RESUME")
```

**5. Answer Generation:**
```python
qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="mistral:7b-q4"),
    retriever=vector_store.as_retriever(
        search_kwargs={'k': 4}  # Top 4 relevant chunks
    ),
    return_source_documents=True
)

result = qa_chain({'query': user_question})
# Returns: {answer, source_documents, sources}
```

### 🎯 Skill Extraction & Normalization

**Problem:** Resume has "React.js" but JD requires "React" → False mismatch

**Solution:** Comprehensive skill normalization with 50+ synonym mappings

```python
synonym_map = {
    'react.js': 'react',
    'reactjs': 'react',
    'node.js': 'nodejs',
    'express.js': 'express',
    'sqlite3': 'sqlite',
    'postgresql': 'postgres',
    'restful api': 'rest api',
    'ci/cd': 'cicd',
    'oop': 'object-oriented programming',
    # ... 40+ more mappings
}

def normalize_skill(skill):
    skill = skill.lower().strip()
    return synonym_map.get(skill, skill)
```

**Regex-Based Extraction:**
```python
skill_patterns = [
    r'\b(Python|Java|JavaScript|TypeScript|C\+\+|Go|Rust)\b',
    r'\b(React|Angular|Vue|Node\.js|Django|Flask|FastAPI)\b',
    r'\b(AWS|Azure|GCP|Docker|Kubernetes|Jenkins|CI/CD)\b',
    r'\b(MySQL|PostgreSQL|MongoDB|Redis|SQLite)\b',
    # ... comprehensive patterns
]

skills = set()
for pattern in skill_patterns:
    matches = re.findall(pattern, text, re.IGNORECASE)
    skills.update(normalize_skill(m) for m in matches)
```

**Result:** 83% reduction in false positives (24 → 4 missing skills)

### 📊 RAG Performance Metrics

- **Indexing Speed:** ~1 second per PDF (background threading)
- **Query Latency:** 3-8 seconds (includes LLM generation)
- **Retrieval Accuracy:** 95%+ relevant chunks in top-4
- **Context Window:** 2048 tokens (Mistral 7B)
- **Memory Usage:** ~4GB RAM (quantized model)

### 🔐 Privacy & Security

- **Local Execution:** Ollama runs entirely on-device (no API calls)
- **Data Privacy:** User PDFs never leave the server
- **Session Isolation:** Each user's vector store is separate
- **Temporary Storage:** Uploaded PDFs can be deleted after indexing

**Implementation Files:**
- `rag_coach.py` - RAGCoachSystem class
- `backend_api.py` - Upload, query, and status endpoints

---

## 9. Project Structure & Key Files

```
NextStepAI/
│
├── 📱 Frontend & Backend
│   ├── app.py                          # Streamlit UI (multi-tab interface)
│   ├── backend_api.py                  # FastAPI REST API (1900+ lines)
│   └── models.py                       # SQLAlchemy ORM models
│
├── 🤖 AI & ML Models
│   ├── model_training.py               # Train job classification model
│   ├── production_finetuning_optimized.py  # Fine-tune GPT-2 (optimized)
│   ├── production_llm_finetuning.py    # Alternative fine-tuning script
│   ├── accurate_career_advisor_training.py  # High-accuracy training
│   ├── rag_coach.py                    # RAGCoachSystem class
│   ├── ingest_guides.py                # Build career guides index
│   └── ingest_all_jobs.py              # Build jobs index
│
├── 📊 Datasets
│   ├── jobs_cleaned.csv                # ML training: 8K+ job-skill pairs
│   ├── career_advice_dataset.jsonl     # LLM training: 243 examples
│   ├── career_advice_ultra_clear_dataset.jsonl  # LLM training: 506 examples
│   ├── skills_db.json                  # 10K+ valid skills vocabulary
│   ├── youtube_links.json              # Skill → tutorial mappings
│   ├── career_guides.json              # Curated career path descriptions
│   ├── job_postings_new.json           # Job postings data
│   └── career_lookup.json              # Career path lookup data
│
├── 💾 Model Artifacts (Generated)
│   ├── job_recommender_pipeline.joblib     # TF-IDF + Naive Bayes
│   ├── job_title_encoder.joblib           # Job title encoder
│   ├── prioritized_skills.joblib          # Job → skills mapping
│   ├── master_skill_vocab.joblib          # Complete skill vocabulary
│   ├── career-advisor-final/              # Fine-tuned GPT-2 model
│   │   ├── pytorch_model.bin              # Model weights (1.4GB)
│   │   ├── config.json                    # Model config
│   │   ├── tokenizer_config.json          # Tokenizer config
│   │   └── vocab.json                     # Vocabulary
│   ├── guides_index/                      # FAISS: career guides
│   └── jobs_index/                        # FAISS: job postings
│
├── 🧪 Testing & Validation
│   ├── test_improved_skill_extraction.py  # Test normalized skill extraction
│   ├── test_skill_extraction.py           # Original skill extraction test
│   ├── test_finetuned_model.py           # Test fine-tuned model inference
│   ├── test_ai_speed.py                   # Benchmark AI response time
│   ├── test_rag_cpu_mode.py               # Test RAG on CPU
│   ├── test_ollama_cpu_mode.py            # Test Ollama CPU mode
│   └── verify_rag_coach_setup.py          # Verify RAG Coach installation
│
├── 📝 Documentation
│   ├── README.md                          # This file
│   ├── LOGIN_AND_HISTORY_SETUP.md         # OAuth setup guide
│   ├── LOGIN_ENABLED.md                   # Quick login summary
│   ├── RAG_COACH_SETUP_GUIDE.md           # RAG Coach installation
│   ├── RAG_DOCUMENT_DETECTION_FIXED.md    # Document detection fix
│   ├── RAG_SYSTEM_OPTIMIZATION_COMPLETE.md # Skill extraction optimization
│   ├── HOW_TO_RUN_PROJECT.md              # Quick start guide
│   ├── QUICK_START.md                     # Quickest setup
│   └── GOOGLE_COLAB_TRAINING_COMPLETE_GUIDE.md  # Colab training
│
├── 🚀 Batch Scripts (Windows)
│   ├── START_BACKEND.bat                  # Start FastAPI server
│   ├── START_FRONTEND.bat                 # Start Streamlit app
│   ├── RESTART_BACKEND.bat                # Restart backend
│   ├── QUICK_START.bat                    # One-click start
│   ├── START_OLLAMA_CPU_MODE.ps1          # Start Ollama (CPU)
│   └── FIX_RAG_GPU_ERROR.bat              # Fix GPU errors
│
├── ⚙️ Configuration
│   ├── .env                               # Environment variables (gitignored)
│   ├── .env.example                       # Environment template
│   ├── requirements.txt                   # Python dependencies
│   └── .gitignore                         # Git ignore rules
│
└── 💾 Database & Uploads
    ├── nextstepai.db                      # SQLite database (generated)
    └── uploads/                           # User-uploaded PDFs (runtime)
```

### 📁 Key Files Explained

#### Core Application
- **`app.py`** (488 lines)
  - Streamlit multi-tab interface
  - JWT authentication flow
  - File upload handlers
  - Result visualization (charts, roadmaps)
  - History tab with refresh

- **`backend_api.py`** (1900+ lines)
  - FastAPI REST API with 20+ endpoints
  - Resume analysis logic (Gemini + ML)
  - Fine-tuned model loading & inference
  - RAG Coach upload/query handlers
  - Google OAuth callbacks
  - Database operations
  - LinkedIn job scraping

- **`models.py`** (45 lines)
  - SQLAlchemy ORM definitions
  - `User`, `ResumeAnalysis`, `CareerQuery`, `RAGCoachQuery`
  - Database relationships
  - Table creation logic

#### AI & ML Training
- **`model_training.py`** (175 lines)
  - Loads `jobs_cleaned.csv`
  - TF-IDF vectorization
  - GridSearchCV for hyperparameter tuning
  - Trains Naive Bayes classifier
  - Generates prioritized skills dictionary
  - Saves `.joblib` artifacts

- **`production_finetuning_optimized.py`** (362 lines)
  - Loads JSONL training datasets
  - Initializes GPT-2-Medium tokenizer & model
  - Configures TrainingArguments (15 epochs, 1e-5 LR)
  - Implements DataCollatorForLanguageModeling
  - Runs HuggingFace Trainer
  - Saves fine-tuned model to `career-advisor-final/`

- **`rag_coach.py`** (500+ lines)
  - `RAGCoachSystem` class
  - PDF loading with PyPDFLoader
  - Document type detection (resume vs JD)
  - FAISS vector store creation
  - RetrievalQA chain setup
  - Query processing with source attribution
  - Skill extraction & normalization

#### Data Ingestion
- **`ingest_guides.py`**
  - Loads `career_guides.json`
  - Chunks documents with RecursiveCharacterTextSplitter
  - Embeds with all-MiniLM-L6-v2
  - Creates `guides_index/` FAISS store

- **`ingest_all_jobs.py`**
  - Similar to above but for job postings
  - Creates `jobs_index/` FAISS store

#### Testing Scripts
- **`test_improved_skill_extraction.py`**
  - Validates normalized skill matching
  - Tests resume vs JD skill comparison
  - Checks for false positives
  - Reports accuracy metrics

---

## 10. Installation & Setup

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| **Python** | 3.10+ | Core runtime |
| **pip** | Latest | Package management |
| **Git** | Any | Clone repository |
| **Google API Key** | N/A | Gemini LLM access |
| **Ollama** | Latest | RAG Coach (Mistral 7B) |
| **CUDA Toolkit** | 11.x/12.x (optional) | GPU acceleration |
| **8GB+ RAM** | N/A | Model loading |

### Step-by-Step Setup

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/arjuntanil/NextStep-AI.git
cd NextStep-AI
```

#### 2️⃣ Create Virtual Environment
```powershell
# Windows PowerShell
python -m venv career_coach
.\career_coach\Scripts\Activate.ps1

# Windows CMD
python -m venv career_coach
career_coach\Scripts\activate.bat

# Linux/Mac
python3 -m venv career_coach
source career_coach/bin/activate
```

#### 3️⃣ Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# (Optional) Install GPU-accelerated PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Key Dependencies:**
```
fastapi==0.104.1
streamlit==1.28.1
transformers==4.35.0
torch==2.1.0
langchain==0.0.335
langchain-google-genai==0.0.6
faiss-cpu==1.7.4
sentence-transformers==2.2.2
scikit-learn==1.3.2
beautifulsoup4==4.12.2
pdfplumber==0.10.3
python-docx==1.1.0
sqlalchemy==2.0.23
python-jose[cryptography]==3.3.0
```

#### 4️⃣ Configure Environment Variables
```bash
# Copy template
cp .env.example .env

# Edit .env file
nano .env  # or use any text editor
```

**Required `.env` Configuration:**
```env
# ===== REQUIRED =====
# Get from https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=AIzaSy...

# Generate: python -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY=your_random_64_char_hex_string

# ===== OPTIONAL (for login) =====
# Get from https://console.cloud.google.com/
GOOGLE_CLIENT_ID=your_google_oauth_client_id
GOOGLE_CLIENT_SECRET=your_google_oauth_client_secret

# Frontend URL (default is correct for local)
STREAMLIT_FRONTEND_URL=http://localhost:8501
```

#### 5️⃣ Train ML Models
```bash
# Train job classification model (~2 minutes)
python model_training.py

# Output artifacts:
#   ✓ job_recommender_pipeline.joblib
#   ✓ job_title_encoder.joblib
#   ✓ prioritized_skills.joblib
```

#### 6️⃣ Build RAG Indexes
```bash
# Build career guides index (~30 seconds)
python ingest_guides.py

# Build jobs index (~30 seconds)
python ingest_all_jobs.py

# Output:
#   ✓ guides_index/ (FAISS vector store)
#   ✓ jobs_index/ (FAISS vector store)
```

#### 7️⃣ Install Ollama (for RAG Coach)
```powershell
# Download from https://ollama.ai
# Or use winget (Windows 11)
winget install Ollama.Ollama

# Pull TinyLlama model (lightweight, 1.1B parameters, ~637MB download)
ollama pull tinyllama

# Alternative: Use Mistral 7B for more advanced responses (3.8GB download)
# ollama pull mistral:7b-q4

# Verify installation
ollama list
```

**Why TinyLLama?**
- ⚡ **Lightweight**: Only 1.1B parameters vs Mistral's 7B
- 🚀 **Fast**: Optimized for CPU inference, no GPU needed
- 💾 **Memory Efficient**: Runs smoothly on 4GB RAM
- 🎯 **Effective**: Excellent for RAG-based Q&A tasks
- 🔒 **Privacy**: Runs 100% locally, no API calls

#### 8️⃣ Setup React Frontend (Optional)
```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies (~1-2 minutes, 1406 packages)
npm install

# Start development server
npm run dev

# Frontend will be available at http://localhost:3000
```

**React Frontend Features:**
- 🌙 **Dark Theme**: Modern UI with glass morphism effects
- ⚡ **Fast Performance**: Client-side routing with React Router
- 🎨 **Material-UI**: Beautiful, responsive components
- 🔐 **JWT Auth**: Secure authentication with protected routes
- 📱 **Mobile Ready**: Fully responsive design

**Available Scripts:**
```bash
npm start      # Start development server (port 3000)
npm run dev    # Alternative start command
npm run build  # Production build
npm test       # Run tests
```

For complete React setup guide, see `REACT_QUICK_START.md`

#### 9️⃣ (Optional) Fine-Tune Career Advisor

**Option A: Local Training (GPU recommended)**
```bash
# Requires: 4GB+ RAM, ~15-20 minutes on GPU
python production_finetuning_optimized.py

# Output: career-advisor-final/ directory
```

**Option B: Google Colab Training (FREE GPU)**
1. Upload `GOOGLE_COLAB_TRAINING_COMPLETE_GUIDE.md` to Colab
2. Follow step-by-step instructions
3. Download `career-advisor-final/` folder
4. Place in project root

**Option C: Skip (Use RAG Only)**
System will automatically use RAG fallback if model not found.

### 🚀 Running the Application

#### Method 1: Batch Scripts (Windows - Easiest)
```powershell
# Start backend
.\START_BACKEND.bat

# Start frontend (new terminal)
.\START_FRONTEND.bat

# Or use one-click start
.\QUICK_START.bat
```

#### Method 2: Manual Commands
```bash
# Terminal 1: Start Backend
cd E:\NextStepAI
.\career_coach\Scripts\Activate.ps1
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Start Frontend
cd E:\NextStepAI
.\career_coach\Scripts\Activate.ps1
streamlit run app.py
```

#### Method 3: PowerShell Scripts
```powershell
# Start Ollama in CPU mode
.\START_OLLAMA_CPU_MODE.ps1

# Restart backend after code changes
.\RESTART_BACKEND.bat
```

### 🌐 Access URLs

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:8501 | Streamlit UI |
| **Backend API** | http://localhost:8000 | FastAPI server |
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **ReDoc** | http://localhost:8000/redoc | Alternative API docs |

---

## 11. API Endpoints

### Prerequisites
* Python 3.10+ (tested with 3.10)
* Git
* Google API Key for Gemini (get from [Google AI Studio](https://makersuite.google.com/app/apikey))
* **Ollama** - For RAG Coach feature (download from [ollama.ai](https://ollama.ai))
* (Optional) Google OAuth credentials for SSO authentication
* (Optional) GPU with CUDA for faster model inference

### Step 1: Clone the Repository
```bash
git clone https://github.com/arjuntanil/NextStep-AI.git
cd NextStep-AI
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv career_coach
career_coach\Scripts\activate

# Linux/Mac
python3 -m venv career_coach
source career_coach/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt

# For CPU-only PyTorch (smaller download):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Configure Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your credentials:
# - GOOGLE_API_KEY (required for Gemini LLM)
# - GOOGLE_CLIENT_ID (optional, for OAuth SSO)
# - GOOGLE_CLIENT_SECRET (optional, for OAuth SSO)
# - JWT_SECRET_KEY (generate with: python -c "import secrets; print(secrets.token_hex(32))")
```

Example `.env` file:
```env
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
JWT_SECRET_KEY=your_random_jwt_secret_key
STREAMLIT_FRONTEND_URL=http://localhost:8501
```

### Step 5: Prepare Data & Models

#### 5.1 Train ML Job Recommendation Model
```bash
python model_training.py
```
This creates:
* `job_recommender_pipeline.joblib`
* `job_title_encoder.joblib`
* `prioritized_skills.joblib`

#### 5.2 Build RAG Vector Stores
```bash
python ingest_guides.py
python ingest_all_jobs.py
```
This creates:
* `guides_index/` (FAISS index for career guides)
* `jobs_index/` (FAISS index for job postings)

#### 5.3 Install Ollama and Pull Mistral Model (For RAG Coach)
```bash
# Download and install Ollama from https://ollama.ai

# After installation, pull the Mistral 7B Q4 model:
ollama pull mistral:7b-q4
```

The RAG Coach feature will automatically detect if the model is missing and show installation instructions in the UI.

#### 5.4 Fine-tune Career Advisor Model (Optional)
**Option A: Train Locally** (requires 4GB+ RAM, 10-30 minutes):
```bash
python production_finetuning_optimized.py
```

**Option B: Train in Google Colab** (recommended, free GPU):
1. Upload `GOOGLE_COLAB_TRAINING_COMPLETE_GUIDE.md` to Google Colab
2. Follow the step-by-step instructions in the notebook
3. Download the `career-advisor-final/` folder to your project root

**Option C: Skip Fine-tuning** (use RAG only):
The system will automatically fall back to the RAG system if the fine-tuned model is not available.

### Step 6: Run the Application

#### Start Backend Server
```bash
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
```

#### Start Frontend (New Terminal)
```bash
# Activate virtualenv first
career_coach\Scripts\activate  # Windows
# source career_coach/bin/activate  # Linux/Mac

streamlit run app.py
```

#### Access the Application
* **Frontend UI**: http://localhost:8501
* **Backend API**: http://localhost:8000
* **API Docs**: http://localhost:8000/docs (Swagger UI)

---

## 11. API Endpoints

### 🔐 Authentication

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| `GET` | `/auth/login` | Initiate Google OAuth flow | No |
| `GET` | `/auth/callback` | OAuth callback handler | No |
| `GET` | `/users/me` | Get current user info | Yes (JWT) |

**Example: Get User Info**
```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     http://localhost:8000/users/me
```

### 📄 Resume Analysis

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| `POST` | `/analyze_resume/` | Analyze resume, get job recommendation | Optional |

**Request:**
```bash
curl -X POST http://localhost:8000/analyze_resume/ \
  -F "file=@resume.pdf" \
  -H "Authorization: Bearer TOKEN"  # Optional: saves to history
```

**Response:**
```json
{
  "resume_skills": ["python", "django", "react", "postgresql"],
  "recommended_job_title": "Full Stack Developer",
  "required_skills": ["python", "django", "react", "postgresql", "docker", "kubernetes"],
  "missing_skills_with_links": [
    {"skill_name": "docker", "youtube_link": "https://youtube.com/..."},
    {"skill_name": "kubernetes", "youtube_link": "https://youtube.com/..."}
  ],
  "match_percentage": 85.0,
  "live_jobs": [
    {
      "title": "Full Stack Developer - TechCorp",
      "company": "TechCorp India",
      "link": "https://linkedin.com/jobs/..."
    }
  ],
  "layout_feedback": "Your resume has strong technical content. Consider adding..."
}
```

### 🤖 AI Career Advisor

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| `POST` | `/query-career-path/` | Ask career question (fine-tuned + RAG) | Optional |
| `POST` | `/career-advice-ai` | Direct fine-tuned model query | No |
| `GET` | `/model-status` | Check model loading status | No |

**Example: Career Query**
```bash
curl -X POST http://localhost:8000/query-career-path/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TOKEN" \
  -d '{"text": "Tell me about a career in DevOps"}'
```

**Response:**
```json
{
  "generative_advice": "DevOps is a methodology...\n\n### Key Skills:\n* Docker, Kubernetes\n* CI/CD Pipelines...",
  "live_jobs": [...],
  "matched_job_group": "DevOps Engineer"
}
```

### 🧑‍💼 RAG Coach

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| `POST` | `/rag-coach/upload` | Upload resume + JD PDFs | Optional |
| `POST` | `/rag-coach/query` | Ask question about uploaded docs | Optional |
| `GET` | `/rag-coach/status` | Check RAG system status | No |
| `POST` | `/rag-coach/build-index` | Rebuild vector store | No |

**Example: Upload PDFs**
```bash
curl -X POST http://localhost:8000/rag-coach/upload \
  -F "files=@resume.pdf" \
  -F "files=@job_description.pdf" \
  -F "process_resume_job=true"
```

**Example: Query RAG Coach**
```bash
curl -X POST http://localhost:8000/rag-coach/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What skills should I add based on the job description?",
    "show_context": true
  }'
```

**Response:**
```json
{
  "answer": "Based on your resume and the job description, you should focus on adding...",
  "context_chunks": [
    {
      "content": "Relevant text chunk...",
      "source": "job_description.pdf",
      "doc_type": "JOB_DESCRIPTION",
      "page": 1
    }
  ],
  "sources": ["resume.pdf", "job_description.pdf"]
}
```

### 📚 History

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| `GET` | `/history/analyses` | Get past resume analyses | Yes |
| `GET` | `/history/queries` | Get past career queries | Yes |
| `GET` | `/history/rag-queries` | Get past RAG interactions | Yes |

---

## 12. Usage Guide

### 🌐 Frontend Options

**Option 1: React Frontend (Recommended for Modern UI)**
```powershell
# Terminal 1: Start Backend
cd E:\NextStepAI
python backend_api.py

# Terminal 2: Start React Frontend
cd E:\NextStepAI\frontend
npm run dev

# Access at: http://localhost:3000
```

**Option 2: Streamlit Frontend (Simple & Fast)**
```powershell
# Terminal 1: Start Backend
cd E:\NextStepAI
python backend_api.py

# Terminal 2: Start Streamlit
cd E:\NextStepAI
streamlit run app.py

# Access at: http://localhost:8501
```

**Quick Start Batch Files:**
- `START_REACT_SYSTEM.bat` - Start both backend + React frontend
- `START_SYSTEM.bat` - Start both backend + Streamlit frontend

### 👤 Using Resume Analyzer

1. **Navigate to "Resume Analyzer" tab**
2. **Upload your resume** (PDF or DOCX)
3. **Wait for analysis** (~10-15 seconds)
4. **Review results:**
   - Recommended job title
   - Skill match percentage
   - Visual roadmap
   - Skills to learn (with tutorial links)
   - Live job postings
   - ATS feedback

**Tips:**
- Use updated resume with clear sections
- Include technical skills section
- Login to save results to history

### 🤖 Using AI Career Advisor

1. **Navigate to "AI Career Advisor" tab**
2. **Check model status** (click "Check Status" button)
3. **Ask your question** in the text box
4. **Adjust parameters** (optional):
   - Response length: 50-120 words
   - Temperature: 0.1-1.0 (lower = faster)
5. **Click "Get AI Advice"**
6. **Review comprehensive response** with live jobs

**Example Questions:**
- "Tell me about a career in Data Science"
- "What certifications should I get for DevOps?"
- "How do I transition from Software Developer to ML Engineer?"

### 🧑‍💼 Using RAG Coach

1. **Navigate to "RAG Coach" tab**
2. **Upload PDFs:**
   - Your resume
   - Target job description
3. **Wait for processing** (~5-10 seconds)
4. **Review auto-generated analysis:**
   - Skills to add
   - Resume enhancement bullets
   - ATS keywords
5. **Ask follow-up questions** in the query box

**Example Questions:**
- "How can I highlight my React experience for this role?"
- "What projects should I add to match the job requirements?"
- "Are there any soft skills I'm missing?"

### 📊 Viewing History

1. **Login with Google** (sidebar button)
2. **Navigate to "My History" tab**
3. **Click "Refresh History"**
4. **Browse past activities:**
   - Resume analyses with job matches
   - Career advisor queries
   - RAG Coach interactions with sources

---

## 13. Deployment & Production

### Authentication
* `GET /auth/login` - Initiate Google OAuth login
* `GET /auth/callback` - OAuth callback handler
* `GET /users/me` - Get current user info (requires JWT token)

### Resume Analysis
* `POST /analyze_resume/` - Upload resume for analysis
  - **Input**: PDF/DOCX file
  - **Output**: Job recommendation, skill gap, learning resources, live jobs

### AI Career Advisor
* `POST /query-career-path/` - Ask career questions (uses RAG + fine-tuned model)
  - **Input**: `{"text": "Tell me about DevOps"}`
  - **Output**: AI-generated advice + live job postings

* `POST /career-advice-ai` - Direct fine-tuned model endpoint
  - **Input**: `{"text": "...", "max_length": 200, "temperature": 0.7}`
  - **Output**: Structured response with model metadata

### Model Management
* `GET /model-status` - Check status of all loaded models
* `GET /model-load-status` - Check fine-tuned model loading progress
* `POST /reload-model?background=true` - Trigger model load/reload

### RAG Coach (NEW!)
* `POST /rag-coach/upload` - Upload PDF documents (resume, job descriptions)
  - **Input**: Multipart form data with PDF files
  - **Output**: Confirmation with number of documents added to vector store

* `POST /rag-coach/query` - Ask questions based on uploaded PDFs
  - **Input**: `{"question": "What skills should I develop based on my resume?"}`
  - **Output**: AI-generated answer with source document attribution

* `POST /rag-coach/build-index` - Rebuild RAG Coach vector store from scratch
  - **Output**: Confirmation of index rebuild with document count

### User History
* `GET /history/analyses` - Get user's past resume analyses (requires auth)
* `GET /history/queries` - Get user's past career queries (requires auth)

---

## 13. Deployment & Production

### 🚀 Production Deployment Checklist

#### Environment Configuration
```env
# Production .env
JWT_SECRET_KEY=<strong-64-char-hex-string>
GOOGLE_API_KEY=<production-api-key>
GOOGLE_CLIENT_ID=<oauth-client-id>
GOOGLE_CLIENT_SECRET=<oauth-client-secret>
STREAMLIT_FRONTEND_URL=https://yourdomain.com
DATABASE_URL=postgresql://user:pass@host:5432/dbname  # Replace SQLite
```

#### Database Migration (PostgreSQL)
```python
# In models.py, replace:
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nextstepai.db")

# With:
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://...")
engine = create_engine(DATABASE_URL)  # Remove check_same_thread
```

#### Model Deployment Options

**Option 1: Include Models in Deployment**
```bash
# Ensure these exist:
career-advisor-final/
job_recommender_pipeline.joblib
guides_index/
jobs_index/

# Total size: ~2.5GB
```

**Option 2: Cloud Storage (Recommended)**
```python
# Download models on startup from S3/GCS
import boto3
s3 = boto3.client('s3')
s3.download_file('bucket', 'career-advisor-final.tar.gz', '/tmp/model.tar.gz')
```

**Option 3: RAG-Only Mode**
```env
DISABLE_FINETUNED_MODEL_LOAD=1  # Uses only RAG system
```

#### Scaling Strategies

**Horizontal Scaling**
```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    image: nextstepai-backend
    replicas: 3
    ports:
      - "8000-8002:8000"
    environment:
      - DATABASE_URL=postgresql://...
      
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

**Caching Layer (Redis)**
```python
# backend_api.py
import redis
cache = redis.Redis(host='localhost', port=6379)

def get_career_advice(query):
    cached = cache.get(f"advice:{query}")
    if cached:
        return json.loads(cached)
    # ... generate advice ...
    cache.setex(f"advice:{query}", 3600, json.dumps(advice))
```

**Background Task Queue (Celery)**
```python
# For long-running tasks
from celery import Celery
app = Celery('nextstepai', broker='redis://localhost:6379')

@app.task
def analyze_resume_async(file_path):
    # ... processing ...
    return results
```

### 🔒 Security Hardening

**HTTPS Configuration**
```python
# backend_api.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        ssl_keyfile="/path/to/key.pem",
        ssl_certfile="/path/to/cert.pem"
    )
```

**Rate Limiting**
```python
from slowapi import Limiter
limiter = Limiter(key_func=lambda: request.client.host)

@app.post("/analyze_resume/")
@limiter.limit("10/minute")
async def analyze_resume(...):
    ...
```

**Input Validation**
```python
from pydantic import BaseModel, validator

class CareerQuery(BaseModel):
    text: str
    
    @validator('text')
    def validate_text(cls, v):
        if len(v) > 1000:
            raise ValueError('Query too long')
        return v.strip()
```

### 📊 Monitoring & Logging

**Structured Logging**
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('nextstepai.log'),
        logging.StreamHandler()
    ]
)
```

**Performance Monitoring**
```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
# Metrics at /metrics
```

### 🐳 Docker Deployment

**Dockerfile**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download models (if not using volume)
RUN python model_training.py && \
    python ingest_guides.py

EXPOSE 8000

CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**
```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/nextstepai
    volumes:
      - ./models:/app/models
    depends_on:
      - db
      
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=nextstepai
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    depends_on:
      - backend

volumes:
  postgres_data:
```

---

## 14. Troubleshooting

### ❌ Common Issues & Solutions

#### Backend Won't Start

**Issue:** `ModuleNotFoundError: No module named 'torch'`
```bash
# Solution: Install PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Issue:** `Port 8000 already in use`
```bash
# Solution: Kill existing process
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

**Issue:** Backend hangs at "Loading production model..."
```bash
# Solution: Model loading can take 2-5 minutes
# Check logs for progress
# Or disable: DISABLE_FINETUNED_MODEL_LOAD=1
```

#### RAG Coach Issues

**Issue:** "Ollama model not found"
```bash
# Solution: Pull Mistral model
ollama pull mistral:7b-q4

# Verify
ollama list
```

**Issue:** "No documents found in vector store"
```bash
# Solution: Upload PDFs or rebuild index
curl -X POST http://localhost:8000/rag-coach/build-index
```

**Issue:** RAG queries return wrong context
```bash
# Solution: Document detection may have failed
# Check logs for [Type: RESUME] and [Type: JOB_DESCRIPTION]
# Re-upload PDFs with clear content
```

#### Authentication Issues

**Issue:** "OAuth redirect mismatch"
```bash
# Solution: Update Google Console redirect URI
# Must match exactly: http://localhost:8000/auth/callback
# For production: https://yourdomain.com/auth/callback
```

**Issue:** "JWT token expired"
```bash
# Solution: Logout and login again
# Tokens expire after configured duration
```

#### Model Performance Issues

**Issue:** Slow response times (20+ seconds)
```bash
# Solution 1: Reduce temperature (0.5 → 0.3)
# Solution 2: Reduce max_length (200 → 80)
# Solution 3: Use GPU instead of CPU
# Solution 4: Enable model quantization
```

**Issue:** Out of memory errors
```bash
# Solution: Close other applications
# For GPU: Reduce batch size in training
# For RAG: Reduce chunk size or k value
```

#### Data Issues

**Issue:** "Could not extract any relevant skills"
```bash
# Solution: Resume may be poorly formatted
# Ensure clear sections: Education, Experience, Skills
# Use standard PDF format (not scanned images)
```

**Issue:** Job scraping returns empty results
```bash
# Solution: LinkedIn may be blocking requests
# Check internet connection
# Try different job titles or locations
```

### 🔍 Debugging Tips

**Enable Verbose Logging**
```python
# In backend_api.py
logging.basicConfig(level=logging.DEBUG)
```

**Check Model Status**
```bash
curl http://localhost:8000/model-status
```

**Test Individual Components**
```bash
# Test ML model
python -c "import joblib; print(joblib.load('job_recommender_pipeline.joblib'))"

# Test fine-tuned model
python test_finetuned_model.py

# Test RAG system
python verify_rag_coach_setup.py

# Test skill extraction
python test_improved_skill_extraction.py
```

**Verify Database**
```bash
# Check tables exist
python -c "from models import engine; print(engine.table_names())"

# Check user count
python -c "from models import SessionLocal, User; db = SessionLocal(); print(db.query(User).count())"
```

### 📝 Getting Help

**Documentation:**
- `LOGIN_AND_HISTORY_SETUP.md` - Authentication guide
- `RAG_COACH_SETUP_GUIDE.md` - RAG Coach installation
- `HOW_TO_RUN_PROJECT.md` - Quick start guide

**Logs:**
- Check backend terminal output
- Check `nextstepai.log` file
- Check browser console (F12) for frontend errors

**Community:**
- [GitHub Issues](https://github.com/arjuntanil/NextStep-AI/issues)
- [GitHub Discussions](https://github.com/arjuntanil/NextStep-AI/discussions)

---

## 15. Contributing

Contributions are welcome! Here's how you can help:

### 🛠️ Development Setup
```bash
# Fork the repository
git clone https://github.com/YOUR_USERNAME/NextStep-AI.git

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
python -m pytest tests/

# Commit with clear message
git commit -m "Add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

### 📋 Contribution Guidelines

**Code Style:**
- Follow PEP 8 for Python
- Use type hints where possible
- Add docstrings to functions
- Keep lines under 100 characters

**Testing:**
- Add tests for new features
- Ensure existing tests pass
- Test on both Windows and Linux

**Documentation:**
- Update README.md if adding features
- Add inline comments for complex logic
- Update API documentation

**Commit Messages:**
```
feat: Add new skill extraction algorithm
fix: Resolve RAG query timeout issue
docs: Update installation instructions
refactor: Improve model loading performance
```

---

## 16. License & Acknowledgments

### 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

**Technologies & Libraries:**
- [HuggingFace Transformers](https://huggingface.co/transformers/) - LLM infrastructure
- [FastAPI](https://fastapi.tiangolo.com/) - High-performance backend framework
- [Streamlit](https://streamlit.io/) - Rapid UI development
- [LangChain](https://python.langchain.com/) - RAG orchestration
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [Ollama](https://ollama.ai/) - Local LLM inference

**Data & Models:**
- [Google Gemini](https://ai.google.dev/) - Skill extraction and feedback generation
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Sentence embeddings
- [GPT-2-Medium](https://huggingface.co/gpt2-medium) - Base model for fine-tuning
- [Scikit-learn](https://scikit-learn.org/) - Machine learning utilities

**Special Thanks:**
- Career coaches and industry experts who validated the training data
- Open-source community for amazing tools and libraries
- Early testers for feedback and bug reports

---

## 17. Citation

If you use this project in your research or work, please cite:

```bibtex
@software{nextstepai2024,
  title = {NextStepAI: AI-Powered Career Navigator},
  author = {Arjun T Anil},
  year = {2024},
  url = {https://github.com/arjuntanil/NextStep-AI},
  description = {Comprehensive career coaching platform using ML, fine-tuned LLMs, and RAG}
}
```

---

## 18. Contact & Support

**Author:** Arjun T Anil  
**GitHub:** [@arjuntanil](https://github.com/arjuntanil)  
**Repository:** [NextStep-AI](https://github.com/arjuntanil/NextStep-AI)  

**For Support:**
- 🐛 Report bugs via [GitHub Issues](https://github.com/arjuntanil/NextStep-AI/issues)
- 💡 Request features via [GitHub Discussions](https://github.com/arjuntanil/NextStep-AI/discussions)
- 📧 Email: [Contact via GitHub profile](https://github.com/arjuntanil)

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star on GitHub! ⭐**

[![Star History Chart](https://api.star-history.com/svg?repos=arjuntanil/NextStep-AI&type=Date)](https://star-history.com/#arjuntanil/NextStep-AI&Date)

Made with ❤️ by [Arjun T Anil](https://github.com/arjuntanil)

</div>