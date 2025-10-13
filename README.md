# NextStepAI: AI-Powered Career Navigator

## 1. Project Overview

**NextStepAI** is a comprehensive, production-ready career coaching platform designed to bridge the gap between job seekers and their ideal career paths. In today's competitive job market, candidates often struggle with understanding Applicant Tracking Systems (ATS), identifying critical skill gaps, and navigating complex career transitions. This project leverages a sophisticated combination of **Machine Learning**, **Fine-tuned Large Language Models (LLMs)**, and **Retrieval-Augmented Generation (RAG)** to provide personalized, actionable insights.

### Key Features:
* **AI-Powered Resume Analysis** - Upload PDF/DOCX resumes for intelligent skill extraction, job matching, and ATS optimization feedback
* **Fine-tuned Career Advisor (Ai_career_Advisor)** - Custom-trained GPT-2 model specifically fine-tuned on career guidance data for accurate, context-aware career advice
* **Live Job Postings** - Real-time job scraping from LinkedIn with intelligent fallback mechanisms
* **Skill Gap Analysis** - Quantified skill matching with personalized learning paths and YouTube tutorial links
* **RAG-Powered Q&A** - Semantic search over curated career guides using FAISS vector stores
* **Secure Authentication** - Google OAuth SSO with JWT tokens for personalized history tracking
* **Production-Ready Architecture** - Environment-based configuration, lazy model loading, and comprehensive error handling

The application features a modern, decoupled architecture with a **Streamlit** frontend for user interaction and a **FastAPI** backend for processing, AI inference, and database management, ensuring scalability, security, and maintainability.

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

## 3. Core Functionalities

### 3.1 Resume Analysis & Job Recommendation
* **AI Skill Extraction:** Uses Google Gemini LLM to extract technical skills, tools, and methodologies from resumes with contextual understanding
* **ML Job Classification:** Pre-trained Scikit-learn pipeline (TF-IDF + Multinomial Naive Bayes) recommends optimal job titles based on skill patterns
* **Skill Gap Analysis:** Calculates match percentage and identifies missing skills required for target roles
* **ATS Optimization:** Generative AI feedback on resume layout, formatting, and structure for improved ATS compatibility

### 3.2 Fine-tuned AI Career Advisor (Ai_career_Advisor)
* **Custom GPT-2 Model:** Fine-tuned on 749+ career guidance examples with 15 epochs for maximum accuracy
* **Contextual Responses:** Generates comprehensive career advice including skills, interview questions, learning paths, and salary insights
* **Background Loading:** Lazy model initialization to ensure fast server startup while supporting on-demand model loading
* **Multi-Strategy Fallback:** Automatically falls back to RAG system if fine-tuned model is unavailable

### 3.3 Live Job Discovery
* **LinkedIn Job Scraping:** Real-time scraping using BeautifulSoup with multiple CSS selector fallbacks to handle LinkedIn's dynamic HTML
* **Intelligent Retry Logic:** Comprehensive error handling with detailed logging for debugging scraping issues
* **Contextual Job Matching:** Semantic similarity matching to find relevant job postings for user queries

### 3.4 RAG-Powered Career Guidance
* **FAISS Vector Search:** Semantic similarity search over career guides using all-MiniLM-L6-v2 embeddings
* **Two Knowledge Bases:** Separate indices for career guides (`guides_index`) and job postings (`jobs_index`)
* **Hybrid Generation:** Combines retrieved context with LLM generation for factually grounded, comprehensive answers

### 3.5 User Management & Security
* **Google OAuth SSO:** Secure authentication using environment-based credentials (no hardcoded secrets)
* **JWT Tokens:** Stateless session management with configurable secret keys
* **History Tracking:** Persistent storage of resume analyses and career queries per user
* **SQLite Database:** Lightweight database with SQLAlchemy ORM for user data and analysis history

---

## 4. System Architecture and Technology Stack

The application employs a modern, production-ready, decoupled architecture:

### Frontend Layer
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
* **Schema Models** - `User`, `ResumeAnalysis`, `CareerQuery` with foreign key relationships

### AI & ML Stack

#### 1. Fine-tuned LLM (Primary Career Advisor)
* **Model:** Custom GPT-2-Medium (355M parameters) fine-tuned on career guidance dataset
* **Training:** 15 epochs with 1e-5 learning rate, gradient accumulation, early stopping
* **Format:** HuggingFace Transformers with saved tokenizer and model weights
* **Device Support:** Automatic GPU/CPU detection with torch device management
* **Generation:** Temperature sampling, top-p/top-k filtering, repetition penalty for coherent responses

#### 2. Traditional ML (Job Classification)
* **Model:** TF-IDF Vectorizer + Multinomial Naive Bayes (selected via GridSearchCV)
* **Training Data:** `jobs_cleaned.csv` with preprocessed job-skill mappings
* **Artifacts:** Joblib-serialized pipeline, label encoder, and prioritized skills dictionary

#### 3. Generative AI (Google Gemini)
* **Primary Use:** Skill extraction from resumes using structured output parsing
* **Secondary Use:** Resume layout feedback generation
* **API:** Google Generative AI via Langchain with PydanticOutputParser for type safety

#### 4. RAG System (Retrieval-Augmented Generation)
* **Embeddings:** HuggingFaceEmbeddings with `all-MiniLM-L6-v2` sentence transformer
* **Vector Store:** FAISS indices for fast similarity search (L2 distance)
* **Chunking:** RecursiveCharacterTextSplitter for semantic document splitting
* **Retrieval:** Top-k similarity search with configurable k parameter
* **Generation:** Langchain chains combining retrieval + LLM generation

#### 5. Web Scraping
* **Library:** Requests + BeautifulSoup4 for HTML parsing
* **Target:** LinkedIn job search pages with fallback CSS selectors
* **Headers:** Comprehensive browser headers to mimic real user agents
* **Error Handling:** Timeout management, request exception catching, empty result handling

---

## 5. NLP Techniques and Model Rationale

This project uses three distinct Natural Language Processing (NLP) techniques, each chosen for its specific strengths in solving a particular part of the problem.

### Technique 1: Generative Skill Extraction (LLM NER)

* **Purpose:** To extract skills from the unstructured text of a user's resume.
* **Technology Used:** Google Gemini LLM via Langchain.
* **Why this approach?** Traditional skill extraction relies on static keyword lists (like `skills_db.json`) and rule-based matchers (like SpaCy's `PhraseMatcher`). This approach fails to identify 
    a) new or niche skills not present in the list, and 
    b) business process skills (e.g., "SWOT Analysis," "Agile Methodologies") that are often described in prose.
 By using a generative LLM, the system can perform contextual **Named Entity Recognition (NER)**, accurately identifying skills based on their context in the resume, leading to much higher quality extraction results.

### Technique 2: Text Classification for Job Recommendation (TF-IDF + Classifier)

* **Purpose:** To recommend a job title based on a list of extracted skills.
* **Technology Used:** **TF-IDF Vectorizer** and **Scikit-learn** (Multinomial Naive Bayes).
* **Why this approach?** Once skills are extracted, the task becomes classifying a "bag of words" (the skill list) into a category (the job title). TF-IDF (Term Frequency-Inverse Document Frequency) is highly effective for this because it converts the list of skills into a numerical vector, giving higher importance to skills that are distinctive for a particular job category and lower importance to generic skills found everywhere. This numerical representation is then efficiently processed by a fast and interpretable classifier like Logistic Regression.

### Technique 3: Semantic Search for RAG (Sentence Transformers)

* **Purpose:** To power the "AI Career Advisor" by finding relevant documents to answer user questions.
* **Technology Used:** **all-MiniLM-L6-v2** embedding model and **FAISS** vector store.
* **Note on `all-MiniLM-L6-v2`:** This model is a **Sentence Transformer**, a lightweight and highly efficient variant of larger transformer models like BERT. It excels at generating "sentence embeddings"—numerical representations where sentences with similar meanings have similar vectors.
* **Why this approach?** For the RAG system to work, we need to retrieve relevant context for a user's query (e.g., find documents about "Data Science salary" when the user asks about pay). `all-MiniLM-L6-v2` is chosen because it offers state-of-the-art performance for semantic similarity search while being small enough to run quickly and cost-effectively, making it ideal for real-time retrieval in a web application.

---

## 6. Detailed Functionality Workflow

### Functionality 1: Resume Analysis

This is the core feature of the application. The workflow involves a hybrid approach, combining LLMs for interpretation and ML models for classification.

**Workflow:**

1.  **Upload and Parsing (`app.py` -> `backend_api.py`):**
    * The user uploads a PDF or DOCX file via the Streamlit interface.
    * FastAPI receives the file and uses libraries like `pdfplumber` and `python-docx` to extract the raw text content.

2.  **Skill Extraction (`backend_api.py`):**
    * **Technology:** Generative LLM (Gemini) using Langchain.
    * **Process:** The raw resume text is passed to the `extract_skills_with_llm` function. The LLM analyzes the context of the entire resume and extracts a list of technical skills, software tools (e.g., "Trello," "SEMrush"), and methodologies (e.g., "SWOT Analysis").

3.  **Job Recommendation (`backend_api.py`):**
    * **Algorithm:** Logistic Regression or Multinomial Naive Bayes (selected by `GridSearchCV` during training in `model_training.py`).
    * **Process:** The list of extracted skills from Step 2 is used as input for the pre-trained ML model (`job_recommender_pipeline.joblib`). The model predicts the most probable job title based on the skill patterns learned from the training data (`jobs_cleaned.csv`).

4.  **Skill Gap Analysis (`backend_api.py`):**
    * The system retrieves the set of required skills for the recommended job title from `prioritized_skills.joblib`.
    * It compares the user's skills with the required skills to generate a list of `skills_to_add` and calculates a `match_percentage`.

5.  **Layout Feedback and Data Enrichment (`backend_api.py`):**
    * **Layout Feedback:** A separate call is made to the Gemini LLM (`generate_layout_feedback`) to analyze the resume's structure and provide formatting advice.
    * **Job Scraping:** The `scrape_live_jobs` function uses **Requests** and **BeautifulSoup** to scrape live job postings from LinkedIn based on the recommended job title.
    * **Learning Resources:** Links for missing skills are retrieved from the static `youtube_links.json` file.

6.  **Response (`backend_api.py` -> `app.py`):** All generated data points are aggregated into a JSON response and displayed on the Streamlit frontend.

### Functionality 2: AI Career Advisor (RAG System)

This feature provides expert-level answers to user questions about career paths using Retrieval-Augmented Generation (RAG).

**Phase A: Offline Data Ingestion (`ingest_guides.py`)**

1.  **Load Data:** Read source material from `career_guides.json`, which contains detailed descriptions of various career paths.
2.  **Chunking:** Split large documents into smaller, semantically meaningful chunks using `RecursiveCharacterTextSplitter`.
3.  **Embedding:** Convert each text chunk into a numerical vector using `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`).
4.  **Index:** Store these vectors in a **FAISS** vector store, creating the `guides_index`.

**Phase B: Online Retrieval and Generation (`backend_api.py`)**

1.  **User Query:** The user asks a question like, "Tell me about a career in Data Science."
2.  **Retrieve:** The system embeds the user query into a vector and searches the `guides_index` for the most relevant text chunks from the career guides.
3.  **Augment:** The retrieved text chunks are inserted into a prompt template alongside the user's original query.
4.  **Generate:** The augmented prompt is sent to the Gemini LLM, which generates a comprehensive answer based on the factual context provided.

---

## 8. Installation & Setup

### Prerequisites
* Python 3.10+ (tested with 3.10)
* Git
* Google API Key for Gemini (get from [Google AI Studio](https://makersuite.google.com/app/apikey))
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

#### 5.3 Fine-tune Career Advisor Model (Optional)
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

## 9. API Endpoints

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

### User History
* `GET /history/analyses` - Get user's past resume analyses (requires auth)
* `GET /history/queries` - Get user's past career queries (requires auth)

---

## 10. Deployment & Production Considerations

### Environment Variables (Production)
```env
# Use strong, randomly generated secrets
JWT_SECRET_KEY=<generate-with-secrets-module>

# Use environment-specific URLs
STREAMLIT_FRONTEND_URL=https://your-domain.com

# Optional: Use a production API key with rate limits
GOOGLE_API_KEY=<production-api-key>
```

### Database Migration (Production)
Replace SQLite with PostgreSQL for production:
```python
# In backend_api.py, update:
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/nextstepai")
```

### Model Deployment Options

#### Option 1: Include Model in Deployment
* Add `career-advisor-final/` to deployment package
* Ensure sufficient RAM (4GB+ for GPT-2-Medium)
* Use GPU instances for faster inference

#### Option 2: Model on Cloud Storage
* Store model in AWS S3 / Google Cloud Storage
* Download on first server startup
* Cache locally for subsequent requests

#### Option 3: RAG-Only Deployment
* Set environment variable: `DISABLE_FINETUNED_MODEL_LOAD=1`
* System will use RAG chain exclusively
* Faster startup, lower memory requirements

### Scaling Considerations
* **Horizontal Scaling**: Deploy multiple FastAPI instances behind load balancer
* **Caching**: Implement Redis for frequently accessed RAG results
* **Async Processing**: Use Celery for background model loading and long-running tasks
* **CDN**: Serve static Streamlit assets via CDN

### Security Checklist
- ✅ All secrets in environment variables (not in code)
- ✅ `.env` file in `.gitignore`
- ✅ JWT tokens with expiration
- ✅ HTTPS only in production
- ✅ Rate limiting on API endpoints
- ✅ Input validation and sanitization
- ✅ CORS configured for frontend domain only

---

## 11. Troubleshooting

### Issue: Backend startup hangs at "Loading production model..."
**Solution**: Model loading can take 2-5 minutes. Check terminal logs. If it persists >10 minutes, the model may be too large for your RAM. Use RAG-only mode or deploy with more RAM.

### Issue: "No module named torch"
**Solution**: Install PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "No live job postings found"
**Solution**: LinkedIn scraping may be blocked. The function will return empty list. Consider using the Adzuna API (add credentials to `.env`) or the function will work intermittently.

### Issue: "GOOGLE_API_KEY not found"
**Solution**: Create `.env` file in project root with your Google API key:
```bash
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### Issue: Fine-tuned model not loading
**Solution**: 
1. Check if `career-advisor-final/` folder exists in project root
2. Verify folder contains `config.json`, `pytorch_model.bin`, tokenizer files
3. Check backend logs for specific error messages
4. System will automatically fall back to RAG if model unavailable

---

## 12. Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow code style**: PEP 8 for Python, use type hints
3. **Add tests** for new features
4. **Update documentation** including README and docstrings
5. **Create pull request** with clear description of changes

### Development Setup
```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black .

# Check linting
flake8 backend_api.py app.py
```

---

## 13. License & Acknowledgments

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
* **Hugging Face** - Transformers library and pre-trained models
* **Google** - Gemini LLM API and OAuth services
* **Streamlit** - Reactive frontend framework
* **FastAPI** - High-performance backend framework
* **Langchain** - RAG orchestration framework
* **FAISS** - Efficient similarity search library

### Built With
* Python 3.10+
* Transformers (Hugging Face)
* FastAPI
* Streamlit
* SQLAlchemy
* Scikit-learn
* Langchain
* FAISS
* BeautifulSoup4
* PyTorch

---

## 14. Contact & Support

* **Author**: Arjun T Anil
* **GitHub**: [@arjuntanil](https://github.com/arjuntanil)
* **Repository**: [NextStep-AI](https://github.com/arjuntanil/NextStep-AI)
* **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/arjuntanil/NextStep-AI/issues)

---

**⭐ If you find this project helpful, please consider giving it a star on GitHub!**

## 7. Project Structure & Key Files

### Core Application Files
* **`app.py`** - Streamlit frontend with multi-tab UI, session management, API integration, and result visualization
* **`backend_api.py`** - FastAPI backend with all REST endpoints, AI model orchestration, authentication, and database operations
* **`models.py`** - SQLAlchemy ORM models (`User`, `ResumeAnalysis`, `CareerQuery`) defining database schema and relationships

### AI Model Files
* **`model_training.py`** - Scikit-learn pipeline training script for job recommendation classifier
* **`production_llm_finetuning.py`** - Fine-tuning script for GPT-2 career advisor model
* **`production_finetuning_optimized.py`** - Optimized training with gradient accumulation and early stopping
* **`accurate_career_advisor_training.py`** - High-accuracy training configuration (15 epochs, 1e-5 LR)

### Data Ingestion Scripts
* **`ingest_guides.py`** - Builds `guides_index` FAISS vector store from `career_guides.json`
* **`ingest_all_jobs.py`** - Creates `jobs_index` from job posting data for semantic search
* **`build_training_data.py`** - Prepares and preprocesses career advice datasets for LLM fine-tuning

### Configuration & Data Files
* **`.env`** - Environment variables for API keys and secrets (gitignored, use `.env.example` template)
* **`.env.example`** - Template showing required environment variables
* **`.gitignore`** - Excludes virtualenvs, model files, secrets, and large data files
* **`requirements.txt`** - Python dependencies for the entire project
* **`skills_db.json`** - Comprehensive skills database used for training data validation
* **`youtube_links.json`** - Skill-to-tutorial mapping for learning resource recommendations
* **`career_guides.json`** - Curated career path descriptions for RAG system
* **`jobs_cleaned.csv`** - Preprocessed job-skill dataset for ML training

### Model Artifacts (Generated, Not in Git)
* **`job_recommender_pipeline.joblib`** - Trained TF-IDF + Naive Bayes pipeline
* **`job_title_encoder.joblib`** - Label encoder for job title categories
* **`prioritized_skills.joblib`** - Dictionary mapping job titles to required skills
* **`career-advisor-final/`** - Fine-tuned GPT-2 model directory (tokenizer + model weights)
* **`guides_index/`** - FAISS vector store for career guides
* **`jobs_index/`** - FAISS vector store for job postings

### Testing & Validation Files
* **`test_accurate_model.py`** - Tests fine-tuned model accuracy and response quality
* **`test_api_endpoint.py`** - FastAPI endpoint integration tests
* **`test_inference.py`** - Model inference performance benchmarks
* **`COLAB_MODEL_TESTER.py`** - Google Colab testing script with automatic validation

### Documentation
* **`README.md`** - This comprehensive project documentation
* **`GOOGLE_COLAB_TRAINING_COMPLETE_GUIDE.md`** - Step-by-step guide for training models in Google Colab
* **`HOW_TO_TEST_MODEL.md`** - Instructions for testing fine-tuned models locally and in Colab
* **`PRODUCTION_LLM_DEPLOYMENT.md`** - Deployment guide for production environments
* **`GPU_OPTIMIZED_TRAINING_GUIDE.md`** - GPU-specific training optimizations and configurations