# NextStepAI: AI-Powered Career Navigator

## 1. Project Overview

**NextStepAI** is a comprehensive career coaching platform designed to bridge the gap between job seekers and their ideal career paths. In today's competitive job market, candidates often struggle with understanding Applicant Tracking Systems (ATS), identifying critical skill gaps, and navigating complex career transitions. This project leverages a sophisticated combination of **Machine Learning** and **Large Language Models (LLMs)** to provide personalized, actionable insights.

Users can upload their resumes to receive an in-depth analysis that includes a recommended job title, a direct comparison of their existing skills against market requirements, and AI-generated feedback on resume layout. Furthermore, an interactive AI Career Advisor provides detailed guidance on various career paths, helping users make informed decisions about their professional development.

The application features a decoupled architecture with a **Streamlit** frontend for user interaction and a **FastAPI** backend for processing, AI inference, and database management, ensuring scalability and maintainability.

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

* **Resume Analysis and Job Recommendation:** Analyzes user resumes to extract skills, recommends the most suitable job title, and calculates a skill match percentage.
* **Skill Gap Identification and Learning Path:** Generates a detailed list of missing skills required for the recommended job and provides direct links to relevant learning resources (YouTube tutorials).
* **Generative Resume Layout Feedback:** Provides AI-driven feedback on resume formatting, structure, and ATS compatibility.
* **Live Job Scraping:** Fetches current job openings from LinkedIn relevant to the recommended job title.
* **AI Career Advisor:** Allows users to ask open-ended questions about career paths, responsibilities, and industry trends, receiving detailed answers from an AI coach.
* **User Authentication and History:** Secure user login via Google SSO to save and review past analysis results and career queries.

---

## 4. System Architecture and Technology Stack

The application employs a modern, decoupled architecture:

* **Frontend:** **Streamlit** is used to create a reactive, data-centric user interface. It handles file uploads, user inputs, and rendering of analysis results.
* **Backend:** **FastAPI** provides a high-performance REST API to handle all business logic and AI processing. This separation allows the frontend and backend to be scaled independently.
* **Database:** **SQLAlchemy** manages the database schema, using **SQLite** for development. It stores user data, analysis history, and query history.
* **AI & Machine Learning:**
    * **Job Recommendation:** A traditional machine learning classifier (**Naive Bayes**) trained using **Scikit-learn**.
    * **Skill Extraction & Generative AI:** **Google Gemini** via **Langchain** for dynamic skill extraction from resumes and generative feedback.
    * **Information Retrieval (RAG):** **FAISS** vector store with **Hugging Face sentence-transformers** for efficient similarity search in the AI Career Advisor.

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

## 7. Role of Project Files

* **`app.py`**: Frontend application logic. Creates the Streamlit UI, manages user sessions, and makes API calls to the backend.
* **`backend_api.py`**: Backend application logic. Defines all FastAPI endpoints, orchestrates the analysis workflow, interacts with AI models, and handles database operations.
* **`models.py`**: Database schema definition. Contains the SQLAlchemy models (`User`, `ResumeAnalysis`, `CareerQuery`) that define the structure of the SQLite database tables.
* **`model_training.py`**: Offline training script. Responsible for loading `jobs_cleaned.csv`, processing skills, training the job recommendation classifier using Scikit-learn, and saving the final model artifacts (`job_recommender_pipeline.joblib`, `prioritized_skills.joblib`).
* **`ingest_guides.py`**: Offline RAG ingestion script. Reads `career_guides.json`, processes the text, and builds the `guides_index` FAISS vector store for the AI Career Advisor.
* **`ingest_all_jobs.py`**: Offline data ingestion script. Reads a separate job data source (`monster_india.json`) and builds the `jobs_index` vector store, likely for a semantic search feature.
* **`skills_db.json`**: Static data file. Contains a predefined list of valid skills used to filter data during the model training process in `model_training.py`.
* **`youtube_links.json`**: Static data file. A dictionary mapping skills to YouTube links, used to provide learning resources to users.