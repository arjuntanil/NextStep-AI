# backend_api.py

import os
import joblib
import io
import json
import spacy
import torch
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

# --- New Import ---
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from fastapi_sso.sso.google import GoogleSSO
from starlette.requests import Request
from starlette.responses import RedirectResponse
from pydantic import BaseModel, Field # Import Field
from spacy.matcher import PhraseMatcher

# --- Fine-tuned Model Imports ---
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- RAG/Scraping Imports ---
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
# --- FIX: Added missing import for PydanticOutputParser ---
from langchain.output_parsers import PydanticOutputParser

# --- Load Environment Variables ---
load_dotenv()

# --- Database Imports ---
from models import SessionLocal, engine, User, ResumeAnalysis, CareerQuery, RAGCoachQuery, create_db_and_tables

# --- Configuration ---
# Load sensitive credentials from environment variables for security
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "your-google-client-id")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "your-google-client-secret")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-jwt-secret-key-here")
ALGORITHM = "HS256"
STREAMLIT_FRONTEND_URL = os.getenv("STREAMLIT_FRONTEND_URL", "http://localhost:8501")

google_sso = GoogleSSO(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, "http://localhost:8000/auth/callback")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

app = FastAPI(title="NextStepAI API")

# --- Fine-tuned Model Configuration ---
# Prefer the user-provided model folder if present (support both underscore and hyphen variants)
from pathlib import Path as _Path
if _Path("./career_advisor_final").exists():
    FINETUNED_MODEL_PATH = "./career_advisor_final"
elif _Path("./career-advisor-final").exists():
    FINETUNED_MODEL_PATH = "./career-advisor-final"
else:
    FINETUNED_MODEL_PATH = "./career-advisor-ultra-finetuned/final_checkpoint"  # Ultra version with better responses!

FALLBACK_MODEL_PATH = "./career-advisor-finetuned-improved/final_checkpoint"  # Previous improved version as fallback
BASE_MODEL_NAME = "EleutherAI/pythia-160m-deduped"

# --- Global Artifacts ---
job_recommender_pipeline = None
title_encoder = None
prioritized_skills = None
NLP = None
# matcher = None # No longer used for primary skill extraction
guide_rag_chain = None
jobs_rag_chain = None
job_group_embeddings = None
job_groups = []
# Defer creation of HuggingFaceEmbeddings to startup to avoid heavy imports at module-import time
# (loading it here can try to download/instantiate sentence-transformers and blow up memory on import)
embedding_model = None
youtube_links_db = {}
llm = None

# --- Fine-tuned Model Global Variables ---
finetuned_career_advisor = None

# --- Initialize Fine-tuned Model Instance ---
def initialize_finetuned_model():
    global finetuned_career_advisor
    # Create the wrapper instance but do NOT synchronously load the weights here.
    # Loading can take a long time and may block the entire startup event.
    finetuned_career_advisor = FinetunedCareerAdvisor(model_path=FINETUNED_MODEL_PATH)
    return finetuned_career_advisor

# --- Pydantic Models for Structured Output ---
class SkillList(BaseModel):
    skills: List[str] = Field(description="A comprehensive list of extracted skills, including technical skills, software tools, and business methodologies.")

class CareerAdviceRequest(BaseModel):
    text: str = Field(..., description="Career advice question")
    max_length: int = Field(default=200, description="Maximum response length")
    temperature: float = Field(default=0.7, description="Response creativity (0.1-1.0)")

class CareerAdviceResponse(BaseModel):
    question: str
    advice: str
    confidence: str
    model_used: str
    live_jobs: List[Dict] = Field(default_factory=list)
    matched_job_group: str = Field(default="")

# --- Production LLM-Based Career Advisor (Fine-tuned Model) ---
import re

class ProductionLLMCareerAdvisor:
    """Production-ready fine-tuned LLM for career advice - NO HARD-CODING"""
    
    def __init__(self, model_path: str = None):
        # Auto-detect trained model path
        if model_path is None:
            gpu_path = Path("./career-advisor-production-v3/final_model")
            cpu_path = Path("./career-advisor-cpu-optimized/final_model")
            
            if gpu_path.exists():
                model_path = str(gpu_path)
                print("[INIT] Detected GPU-trained model (gpt2-medium)")
            elif cpu_path.exists():
                model_path = str(cpu_path)
                print("[INIT] Detected CPU-trained model (gpt2)")
            else:
                model_path = str(gpu_path)  # Default
        
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.device = None
        # Loading flags for background loader
        self.load_start_time = None
        self.load_complete = False
        print("[INIT] Initializing Production Career Advisor...")
    
    def load_model(self):
        """Load fine-tuned GPT-2-Medium model (355M params, GPU-optimized)"""
        try:
            print(f"[LOAD] Loading production model from {self.model_path}...")
            
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            # Load tokenizer and model from the supplied model path using low_cpu_mem_usage to reduce peak memory
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            # Use low_cpu_mem_usage if possible (reduces memory spikes while loading)
            try:
                self.model = GPT2LMHeadModel.from_pretrained(self.model_path, low_cpu_mem_usage=True)
            except TypeError:
                # Older HF transformers may not support low_cpu_mem_usage
                self.model = GPT2LMHeadModel.from_pretrained(self.model_path)

            # Move model to appropriate device
            import torch as _torch
            device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            self.device = str(device)
            self.model.to(device)
            
            # Set pad token to eos token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.is_loaded = True
            self.load_complete = True
            print("[OK] Production Career Advisor loaded successfully!")
            print(f"   Model: GPT-2-Medium (355M parameters)")
            print(f"   Training: 6 epochs, ~1500 steps, GPU-optimized")
            print(f"   Capabilities: Highly Accurate Skills, Interview Questions, Career Guidance")
            
        except Exception as e:
            print(f"[WARN] Error loading model: {e}")
            print("   Please run: python production_finetuning_optimized.py")
            self.is_loaded = False
            self.load_complete = True
    
    def generate_advice(self, question: str, max_length: int = 450, temperature: float = 0.8) -> str:
        """Generate career advice using fine-tuned LLM"""
        
        if not self.is_loaded or self.model is None:
            return self._fallback_guidance(question)
        
        try:
            import torch
            
            # Format prompt to match training format
            input_text = f"Question: {question}\n\nAnswer:"
            
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response with improved parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode LLM output
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer part
            if "### Answer:" in full_response:
                answer = full_response.split("### Answer:")[-1].strip()
            else:
                answer = full_response.strip()
            
            return self._clean_response(answer)
            
        except Exception as e:
            print(f"[WARN] LLM generation error: {e}")
            return self._fallback_guidance(question)
    
    def _clean_response(self, response: str) -> str:
        """Clean LLM output"""
        lines = response.split('\n')
        unique_lines = []
        seen = set()
        
        for line in lines:
            clean_line = line.strip()
            if clean_line and clean_line not in seen:
                unique_lines.append(line)
                seen.add(clean_line)
        
        return '\n'.join(unique_lines)
    
    def _fallback_guidance(self, question: str) -> str:
        """Fallback when LLM not available - instructs to train model"""
        return f"""[WARN] **Production LLM Not Trained Yet**

To enable LLM-based career advice, please run the fine-tuning script:

```bash
python production_llm_finetuning.py
```

This will:
• Fine-tune DistilGPT-2 on 749 career guidance examples
• Train on your knowledge base (career_advice_dataset.jsonl)
• Save production-ready model to ./career-advisor-production/
• Enable accurate skills and interview question generation

**Your question:** {question}

**After training, the LLM will provide:**
• Accurate career guidance
• Relevant technical skills
• Interview preparation questions
• Learning paths and certifications
• Salary insights and company recommendations"""

# --- Production LLM Wrapper ---
class FinetunedCareerAdvisor:
    """Production wrapper - Uses fine-tuned LLM for accurate career advice"""
    def __init__(self, model_path: str = None):
        # Accept an explicit model_path to point to the fine-tuned artifacts
        self.llm_advisor = ProductionLLMCareerAdvisor(model_path=model_path)
        self.is_loaded = False
        print("[INIT] Production LLM Career Advisor initializing...")
    
    def load_model(self):
        """Load production fine-tuned model"""
        self.llm_advisor.load_model()
        self.is_loaded = self.llm_advisor.is_loaded
        if self.is_loaded:
            print("[OK] Production LLM ready for accurate career guidance!")
        else:
            print("[WARN] LLM not trained. Run: python production_llm_finetuning.py")
    
    def generate_advice(self, question: str, max_length: int = 400, temperature: float = 0.7) -> str:
        """Generate advice using fine-tuned LLM"""
        return self.llm_advisor.generate_advice(question, max_length, temperature)
    
    def _get_general_guidance(self):
        """Provide helpful general career guidance for unrecognized questions"""
        return """🤔 **I'd love to help with your career question!**

To provide you with the most accurate and comprehensive guidance, please specify:

## 🎯 **Career Fields I Specialize In:**

### 🔧 **DevOps Engineering**  
- **Keywords**: DevOps, CI/CD, Jenkins, Docker, Kubernetes, automation
- **Focus**: Infrastructure automation, deployment pipelines, cloud operations

### ☁️ **Cloud Engineering**
- **Keywords**: AWS, Azure, GCP, cloud computing, serverless, cloud migration  
- **Focus**: Cloud architecture, platform management, scalability

### 💻 **Software Development**
- **Keywords**: programming, coding, web development, Java, Python, JavaScript
- **Focus**: Application development, full-stack development, software engineering

### 📊 **Data Science** 
- **Keywords**: data science, machine learning, AI, analytics, statistics
- **Focus**: Data analysis, predictive modeling, business intelligence

## 💡 **For Best Results, Ask:**
• **"I love [career field]"** - Get complete career roadmap
• **"How to become a [role]?"** - Step-by-step learning path  
• **"Tell me about [technology]"** - Career context and guidance
• **"[Career field] salary and companies"** - Market insights

## [INIT] **Example Questions:**
- "I love DevOps and want to build my career"
- "Tell me about CI/CD" 
- "How to become a cloud engineer?"
- "What skills needed for data science?"

**Ask me anything specific about these tech careers, and I'll provide a detailed roadmap! 🎯**"""


# Duplicate class removed - using the comprehensive version above

# --- 3. Database Startup and Model Loading ---
@app.on_event("startup")
def on_startup():
    print("Creating database and tables...")
    create_db_and_tables()
    
    print("Loading AI models and artifacts...")
    global job_recommender_pipeline, title_encoder, prioritized_skills, NLP, \
           guide_rag_chain, jobs_rag_chain, job_group_embeddings, job_groups, youtube_links_db, llm, finetuned_career_advisor

    if os.getenv("GOOGLE_API_KEY") is None:
        print("🔴 CRITICAL WARNING: GOOGLE_API_KEY environment variable not set. RAG features will fail.")

    # Initialize embedding model here to avoid heavy imports during module import
    global embedding_model
    try:
        if embedding_model is None:
            print("Initializing embedding model (this may take a few seconds)...")
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("[OK] Embedding model initialized")
    except Exception as e:
        print(f"[WARN] Warning: Failed to initialize embedding model: {e}")

    # 1. Prepare Fine-tuned Career Advisor wrapper but DO NOT load weights synchronously at startup.
    #    Loading the model can take many minutes and consume large amounts of memory; we'll load on-demand.
    print("[INIT] Initializing Fine-tuned Career Advisor wrapper (weights NOT loaded yet)...")
    finetuned_career_advisor = initialize_finetuned_model()
    # Ensure loader flags are initialized
    try:
        finetuned_career_advisor.llm_advisor.load_complete = False
        finetuned_career_advisor.llm_advisor.load_start_time = None
    except Exception:
        pass

    # 2. Load Resume Analyzer Models and Supporting Data
    try:
        job_recommender_pipeline = joblib.load('job_recommender_pipeline.joblib')
        title_encoder = joblib.load('job_title_encoder.joblib')
        prioritized_skills = joblib.load('prioritized_skills.joblib')
        NLP = spacy.load('en_core_web_sm') # Keep spaCy for potential future NLP tasks if needed
        with open('youtube_links.json', 'r', encoding='utf-8') as f:
            youtube_links_db = json.load(f)
        print("[OK] Resume analysis models and supporting data loaded.")
    except Exception as e:
        print(f"[WARN] Warning: Failed to load resume analysis models: {e}")

    # Initialize LLM instance once for reuse across endpoints
    try:
        # --- MODIFIED LINE ---
        llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0.2)
        print("[OK] Gemini LLM instance initialized.")
    except Exception as e:
        print(f"[ERROR] Error initializing Gemini LLM: {e}")

    # 2. Load RAG System for Career Guides
    try:
        print("Loading FAISS vector store from 'guides_index'...")
        guides_vector_store = FAISS.load_local("guides_index", embedding_model, allow_dangerous_deserialization=True)
        guide_retriever = guides_vector_store.as_retriever(search_kwargs={"k": 2})
        guide_template = """You are an expert career coach. First, try to answer the user's question using the provided context.
        If the context is empty or not relevant, use your general knowledge to provide a helpful answer about career paths.
        Context: {context} Question: {question}"""
        guide_prompt = ChatPromptTemplate.from_template(guide_template)
        guide_rag_chain = ({"context": guide_retriever, "question": RunnablePassthrough()} | guide_prompt | llm | StrOutputParser())
        print("[OK] Career Guide RAG chain created successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading Career Guide RAG system ('guides_index'): {e}")

    # 3. Load Job Search RAG Chain
    try:
        print("Loading FAISS vector store from 'jobs_index'...")
        jobs_vector_store = FAISS.load_local("jobs_index", embedding_model, allow_dangerous_deserialization=True)
        jobs_retriever = jobs_vector_store.as_retriever(search_kwargs={"k": 5})
        job_search_template = """You are an AI recruitment assistant. Find relevant jobs in context for the query.
        Context: {context} Query: {question}"""
        job_search_prompt = ChatPromptTemplate.from_template(job_search_template)
        jobs_rag_chain = ({"context": jobs_retriever, "question": RunnablePassthrough()} | job_search_prompt | llm | StrOutputParser())
        print("[OK] Job Search RAG chain created successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading Job Search RAG system ('jobs_index'): {e}")

    # 4. Load Simple Semantic Search for Job Title Extraction
    try:
        JOB_GROUP_SEMANTIC_MAP = {
           "Data Professional": "data science, artificial intelligence, machine learning, data analysis, business intelligence, data engineering",
           "Software Developer": "programming, coding, software applications, web development, mobile development, full stack developer",
           "IT Operations & Infrastructure": "IT infrastructure, cloud platforms, AWS, Azure, DevOps, system administration, network engineering, cybersecurity",
           "Project / Product Manager": "project management, product owner, agile methodologies, scrum master",
           "QA / Test Engineer": "software quality assurance, testing, test automation, sdet",
           "UI/UX & Design": "user interface design, user experience research, graphic design, figma, sketch",
           "Finance & Accounting": "finance, accounting, auditing, financial analysis, bookkeeping",
        }
        job_groups = list(JOB_GROUP_SEMANTIC_MAP.keys())
        job_group_descriptions = list(JOB_GROUP_SEMANTIC_MAP.values())
        job_group_embeddings = embedding_model.embed_documents(job_group_descriptions)
        print("[OK] Job group semantic search model loaded.")
    except Exception as e:
        print(f"[ERROR] Error loading semantic search components: {e}")


# --- 4. Database Session Helper ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 5. Authentication Functions & Dependencies ---
def create_access_token(data: dict):
    return jwt.encode(data.copy(), JWT_SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user_optional(token: Optional[str] = Depends(oauth2_scheme), db: SessionLocal = Depends(get_db)) -> Optional[User]:
    if token is None: return None
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None: return None
    except JWTError:
        return None
    return db.query(User).filter(User.email == email).first()

async def get_current_user_required(current_user: User = Depends(get_current_user_optional)) -> User:
    if current_user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return current_user

# --- 6. SSO & User Routes ---
@app.get("/auth/login", tags=["Authentication"])
async def auth_login():
    return await google_sso.get_login_redirect()

@app.get("/auth/callback", tags=["Authentication"])
async def auth_callback(request: Request, db: SessionLocal = Depends(get_db)):
    user_info = await google_sso.verify_and_process(request)
    if not user_info:
        raise HTTPException(status_code=401, detail="Authentication failed")
    user_email = user_info.email
    db_user = db.query(User).filter(User.email == user_email).first()
    if not db_user:
        db_user = User(email=user_email, full_name=user_info.display_name)
        db.add(db_user); db.commit(); db.refresh(db_user)
    access_token = create_access_token(data={"sub": db_user.email})
    return RedirectResponse(url=f"{STREAMLIT_FRONTEND_URL}?token={access_token}")

@app.get("/users/me", tags=["Users"])
async def read_users_me(current_user: User = Depends(get_current_user_required)):
    return {"email": current_user.email, "full_name": current_user.full_name, "id": current_user.id}

# --- 7. Core AI Endpoints ---
class Query(BaseModel): text: str

def extract_text_from_pdf(file_bytes: bytes) -> str:
    import pdfplumber
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "".join(p.extract_text() for p in pdf.pages if p.extract_text())

def extract_text_from_docx(file_bytes: bytes) -> str:
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def scrape_live_jobs(job_title: str, location: str = "India") -> List[Dict[str, str]]:
    """
    Scrape live job postings from LinkedIn based on the recommended job title.
    Uses BeautifulSoup and Requests to fetch real-time job data.
    
    Args:
        job_title: Job title to search for
        location: Location to search in (default: "India", can be "Mumbai", "Bangalore", "Delhi", etc.)
    """
    search_query = job_title.replace(" ", "%20")
    location_query = location.replace(" ", "%20")
    
    # LinkedIn job search URL with proper parameters
    url = f"https://www.linkedin.com/jobs/search?keywords={search_query}&location={location_query}&f_TPR=r86400&position=1&pageNum=0"
    
    # Comprehensive headers to mimic a real browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    }
    
    scraped_jobs = []
    
    try:
        print(f"🔍 Scraping LinkedIn jobs for: {job_title} in {location}")
        
        # Make request with timeout
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all job card containers (LinkedIn uses different class names over time)
        # Try multiple selectors to handle LinkedIn's HTML variations
        job_cards = soup.find_all('div', class_='base-card')
        
        if not job_cards:
            # Fallback: try alternative class names LinkedIn sometimes uses
            job_cards = soup.find_all('div', class_='job-search-card')
        
        if not job_cards:
            # Another fallback: look for any div with data-job-id attribute
            job_cards = soup.find_all('div', attrs={'data-job-id': True})
        
        print(f"[LOAD] Found {len(job_cards)} job cards in HTML")
        
        # Extract job details from each card
        for card in job_cards[:5]:  # Limit to top 5 jobs
            try:
                # Try multiple selectors for title
                title_element = (
                    card.find('h3', class_='base-search-card__title') or
                    card.find('h3', class_='job-search-card__title') or
                    card.find('a', class_='base-card__full-link')
                )
                
                # Try multiple selectors for company
                company_element = (
                    card.find('h4', class_='base-search-card__subtitle') or
                    card.find('h4', class_='job-search-card__company-name') or
                    card.find('a', class_='hidden-nested-link')
                )
                
                # Try multiple selectors for link
                link_element = (
                    card.find('a', class_='base-card__full-link') or
                    card.find('a', class_='job-search-card__link-wrapper') or
                    card.find('a', href=True)
                )
                
                # Only add job if we have all required fields
                if title_element and company_element and link_element:
                    title = title_element.get_text(strip=True)
                    company = company_element.get_text(strip=True)
                    link = link_element.get('href', '')
                    
                    # Clean up the link (remove query parameters)
                    if '?' in link:
                        link = link.split('?')[0]
                    
                    # Ensure link is absolute URL
                    if link and not link.startswith('http'):
                        link = f"https://www.linkedin.com{link}"
                    
                    scraped_jobs.append({
                        "title": title,
                        "company": company,
                        "link": link
                    })
                    print(f"  [OK] Extracted: {title} at {company}")
                    
            except Exception as parse_error:
                print(f"  [WARN] Error parsing job card: {parse_error}")
                continue
        
        if scraped_jobs:
            print(f"[OK] Successfully scraped {len(scraped_jobs)} jobs from LinkedIn")
            return scraped_jobs
        else:
            print(f"[WARN] No jobs extracted from LinkedIn HTML. The page structure may have changed.")
            
    except requests.exceptions.Timeout:
        print(f"[WARN] LinkedIn request timed out after 15 seconds")
    except requests.exceptions.RequestException as e:
        print(f"[WARN] LinkedIn scraping failed with network error: {e}")
    except Exception as e:
        print(f"[WARN] Unexpected error during LinkedIn scraping: {e}")
    
    # Return empty list if scraping failed (will show "No jobs found" message in UI)
    return scraped_jobs

# --- Generative Feedback Function ---
def generate_layout_feedback(resume_text: str) -> str:
    global llm
    if llm is None: return "Layout analysis feature is currently unavailable."
    prompt_template = """You are an expert resume reviewer for Applicant Tracking Systems (ATS). 
    Analyze the structure and layout of the following resume text. Do not comment on the content (skills, experience quality). 
    Focus on formatting, readability, section order, and overall ATS compatibility. Provide 3-5 actionable bullet points for improvement.
    Resume Text: --- {text} --- """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    layout_chain = prompt | llm | StrOutputParser()
    try:
        feedback = layout_chain.invoke({"text": resume_text[:4000]})
        return feedback
    except Exception as e:
        print(f"Error generating layout feedback: {e}")
        return "Could not generate layout feedback due to an API or processing error."

# --- LLM Skill Extraction Function ---
def extract_skills_with_llm(resume_text: str) -> List[str]:
    global llm
    if llm is None:
        print("LLM not initialized. Using fallback skill extraction...")
        return _extract_skills_fallback(resume_text)

    parser = PydanticOutputParser(pydantic_object=SkillList)
    prompt_template_str = """
    You are an expert technical recruiter. Analyze the following resume text and extract all relevant hard skills.
    Include technical skills (e.g., Python, SQL, AWS), software tools (e.g., Microsoft Excel, Trello, SEMrush, Smartsheet), and business methodologies (e.g., SWOT Analysis, Agile).
    Return only the list of skills. Ensure the output strictly follows the requested JSON format.

    {format_instructions}

    Resume Text:
    ---
    {resume_text}
    ---
    """
    prompt = ChatPromptTemplate.from_template(prompt_template_str, partial_variables={"format_instructions": parser.get_format_instructions()})
    chain = prompt | llm | parser

    try:
        print("Invoking LLM for skill extraction...")
        response_model = chain.invoke({"resume_text": resume_text})
        extracted_skills = sorted(list(set([skill.lower() for skill in response_model.skills])))
        print(f"Extracted skills: {extracted_skills}")
        
        # Fallback if LLM returns empty list
        if not extracted_skills:
            print("LLM returned empty skills. Using fallback extraction...")
            return _extract_skills_fallback(resume_text)
        
        return extracted_skills
    except Exception as e:
        print(f"Error during LLM skill extraction: {e}")
        print("Using fallback skill extraction...")
        return _extract_skills_fallback(resume_text)

def _extract_skills_fallback(text: str) -> List[str]:
    """Fallback regex-based skill extraction for Resume Analyzer"""
    import re
    
    # Comprehensive skill patterns
    skill_patterns = [
        r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|PHP|Swift|Kotlin|Scala|R|MATLAB)\b',
        r'\b(Django|Flask|FastAPI|React(?:\.js)?|Angular(?:\.js)?|Vue(?:\.js)?|Node(?:\.js)?|Express(?:\.js)?|Spring|\.NET|Laravel)\b',
        r'\b(NumPy|Pandas|Matplotlib|Scikit-learn|TensorFlow|PyTorch|Keras|Machine\s+Learning|Deep\s+Learning|NLP)\b',
        r'\b(MySQL|PostgreSQL|MongoDB|Redis|Oracle|SQL\s+Server|SQLite|Cassandra|DynamoDB)\b',
        r'\b(AWS|Azure|GCP|Docker|Kubernetes|Jenkins|GitLab|CI/CD|Terraform|Ansible|Linux|Unix)\b',
        r'\b(HTML|CSS|JavaScript|AJAX|REST(?:ful)?|GraphQL|API|Bootstrap|Tailwind|Sass)\b',
        r'\b(Git|GitHub|Jira|Selenium|JUnit|pytest|Postman|Swagger)\b',
        r'\b(Agile|Scrum|Kanban|SDLC|DevOps|Microservices|OOP|Design\s+Patterns)\b',
        r'\b(Excel|Word|PowerPoint|Tableau|Power\s+BI|Salesforce|SAP|Trello|Slack)\b'
    ]
    
    skills = set()
    for pattern in skill_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = ' '.join([m for m in match if m]).strip()
            if match:
                skills.add(match.lower())
    
    skills_list = sorted(list(skills))
    print(f"Fallback extracted {len(skills_list)} skills: {skills_list[:10]}...")
    return skills_list

@app.post("/analyze_resume/", tags=["AI Features"])
async def analyze_resume(
    file: UploadFile = File(...), 
    db: SessionLocal = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    if not all([job_recommender_pipeline, title_encoder, prioritized_skills]):
        raise HTTPException(status_code=503, detail="Resume analysis base models are not loaded.")
    
    # 1. Resume Parsing
    file_bytes = await file.read()
    text = ""
    if file.filename.endswith(".pdf"): text = extract_text_from_pdf(file_bytes)
    elif file.filename.endswith(".docx"): text = extract_text_from_docx(file_bytes)
    else: raise HTTPException(status_code=400, detail="Unsupported file type.")
    
    # --- MODIFICATION: Skill extraction using LLM ---
    print("Extracting skills using Gemini LLM...")
    resume_skills = extract_skills_with_llm(text) # Replaced PhraseMatcher logic here
    if not resume_skills: 
        raise HTTPException(status_code=404, detail="Could not extract any relevant skills from the resume using AI.")
    
    # 2. Skill Gap Analysis (Existing Logic)
    user_skills_str = ' '.join(resume_skills)
    predicted_title_encoded = job_recommender_pipeline.predict([user_skills_str])[0]
    recommended_job_title = title_encoder.inverse_transform([predicted_title_encoded])[0]
    required_skills = prioritized_skills.get(recommended_job_title, [])
    resume_skills_set = set(resume_skills)
    required_skills_set = set(required_skills)
    skills_to_add = sorted(list(required_skills_set - resume_skills_set))
    match_percentage = (len(required_skills_set.intersection(resume_skills_set)) / len(required_skills_set)) * 100 if required_skills_set else 100.0

    # 3. Generate Layout Feedback using LLM
    print("Generating layout feedback...")
    layout_feedback = generate_layout_feedback(text)
    
    # 4. Scrape Live Jobs for the recommended role
    print(f"Scraping jobs for title: {recommended_job_title}")
    live_jobs = scrape_live_jobs(recommended_job_title)
    
    # 5. Get YouTube links for missing skills
    missing_skills_with_links = [
        {"skill_name": skill, "youtube_link": youtube_links_db.get(skill, {}).get('link', '#')}
        for skill in skills_to_add
    ]

    if current_user:
        new_analysis = ResumeAnalysis(owner_id=current_user.id, recommended_job_title=recommended_job_title, match_percentage=int(match_percentage), skills_to_add=json.dumps(skills_to_add))
        db.add(new_analysis); db.commit()
    
    return {
        "resume_skills": resume_skills,
        "recommended_job_title": recommended_job_title,
        "required_skills": sorted(list(required_skills_set)),
        "missing_skills_with_links": missing_skills_with_links,
        "match_percentage": match_percentage,
        "live_jobs": live_jobs,
        "layout_feedback": layout_feedback
    }

# --- Fine-tuned Career Advisor Endpoint ---
@app.post("/query-career-path/", tags=["AI Career Advisor"])
async def query_career_path(
    query: Query,
    db: SessionLocal = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    generated_advice = ""
    
    # Use Fine-tuned Model as Primary Advisor if loaded; otherwise trigger background load and fallback to RAG
    try:
        if finetuned_career_advisor and finetuned_career_advisor.is_loaded:
            generated_advice = finetuned_career_advisor.generate_advice(
                question=query.text,
                max_length=200,
                temperature=0.7
            )
            print(f"[OK] Fine-tuned model generated advice for: {query.text[:50]}...")
        else:
            # If wrapper exists but model not loaded, start background load and fallback to RAG
            if finetuned_career_advisor:
                try:
                    # Kick off background load (non-blocking)
                    from threading import Thread

                    def _bg_load():
                        try:
                            finetuned_career_advisor.llm_advisor.load_start_time = __import__('datetime').datetime.utcnow()
                            finetuned_career_advisor.load_model()
                        except Exception as _e:
                            print(f"Background model load failed: {_e}")

                    Thread(target=_bg_load, daemon=True).start()
                    print("[INFO] Fine-tuned model not loaded: background load started.")
                except Exception as _e:
                    print(f"Could not start background load: {_e}")

            # Immediate fallback to RAG
            if guide_rag_chain:
                generated_advice = guide_rag_chain.invoke(query.text)
                print("[WAIT] Using RAG model (fine-tuned not available)")
            else:
                generated_advice = "Career advisor is temporarily unavailable."
    except Exception as e:
        print(f"Error during advice generation: {e}")
        generated_advice = "Sorry, I encountered an error while generating career advice."

    # Job matching and scraping (unchanged)
    from sentence_transformers import util as sentence_util 
    query_embedding = embedding_model.embed_query(query.text)
    if job_group_embeddings is not None and len(job_group_embeddings) > 0:
        similarities = sentence_util.cos_sim([query_embedding], job_group_embeddings)[0]
        best_match_index = similarities.argmax()
        matched_job_group = job_groups[best_match_index]
        print(f"Job search term identified: {matched_job_group}")
    else:
        matched_job_group = query.text 
        print("Warning: Job group embeddings not available. Falling back to raw query for job search.")

    live_jobs = scrape_live_jobs(matched_job_group)

    if current_user:
        new_query = CareerQuery(owner_id=current_user.id, user_query_text=query.text, matched_job_group=matched_job_group)
        db.add(new_query); db.commit()
        
    return {
        "generative_advice": generated_advice,
        "live_jobs": live_jobs,
        "matched_job_group": matched_job_group
    }

# --- Model Status Endpoint ---
@app.get("/model-status", tags=["AI Career Advisor"])
async def get_model_status():
    """Get status of all loaded models"""
    return {
        "finetuned_career_advisor": {
            "loaded": finetuned_career_advisor.is_loaded if finetuned_career_advisor else False,
            "model_path": FINETUNED_MODEL_PATH,
            "base_model": BASE_MODEL_NAME,
            "device": finetuned_career_advisor.llm_advisor.device if finetuned_career_advisor and finetuned_career_advisor.is_loaded else "N/A"
        },
        "rag_chains": {
            "career_guides": guide_rag_chain is not None,
            "job_search": jobs_rag_chain is not None
        },
        "resume_analyzer": {
            "job_recommender": job_recommender_pipeline is not None,
            "title_encoder": title_encoder is not None,
            "skills_extractor": prioritized_skills is not None
        }
    }


@app.get("/model-load-status", tags=["AI Career Advisor"])
async def model_load_status():
    """Return load progress flags for the fine-tuned model."""
    if not finetuned_career_advisor:
        return {"present": False, "loaded": False}
    adv = finetuned_career_advisor.llm_advisor
    return {
        "present": True,
        "loaded": adv.is_loaded,
        "load_complete": getattr(adv, 'load_complete', False),
        "load_start_time": getattr(adv, 'load_start_time', None)
    }


@app.post("/reload-model", tags=["AI Career Advisor"])
async def reload_model(background: bool = True):
    """Trigger model load/reload. If background=True, load in background thread and return immediately."""
    if not finetuned_career_advisor:
        return {"status": "no_model_wrapper"}

    adv = finetuned_career_advisor.llm_advisor

    def _do_load():
        try:
            adv.load_start_time = __import__('datetime').datetime.utcnow()
            adv.load_complete = False
            adv.load_model()
        finally:
            adv.load_complete = True

    import threading
    if background:
        t = threading.Thread(target=_do_load, daemon=True)
        t.start()
        return {"status": "loading_started_in_background"}
    else:
        _do_load()
        return {"status": "loaded", "loaded": adv.is_loaded}


@app.get("/model-load-status", tags=["AI Career Advisor"])
async def get_model_load_status():
    """Return background model load progress and timing."""
    if not finetuned_career_advisor:
        return {"status": "not_initialized"}
    start = getattr(finetuned_career_advisor.llm_advisor, 'load_start_time', None) if hasattr(finetuned_career_advisor, 'llm_advisor') else None
    complete = getattr(finetuned_career_advisor.llm_advisor, 'load_complete', False) if hasattr(finetuned_career_advisor, 'llm_advisor') else False
    is_loaded = finetuned_career_advisor.is_loaded if finetuned_career_advisor else False
    return {
        "initialized": True,
        "is_loaded": is_loaded,
        "load_started": bool(start),
        "load_complete": bool(complete),
        "load_start_time": str(start) if start else None
    }

# --- Dedicated Fine-tuned Model Endpoint ---
@app.post("/career-advice-ai", response_model=CareerAdviceResponse, tags=["AI Career Advisor"])
async def get_career_advice_ai(request: CareerAdviceRequest):
    """
    Get career advice using the fine-tuned Pythia model
    """
    try:
        # If the fine-tuned model is loaded, use it. Otherwise start background load and fallback to RAG.
        advice = None
        if finetuned_career_advisor and finetuned_career_advisor.is_loaded:
            advice = finetuned_career_advisor.generate_advice(
                question=request.text,
                max_length=request.max_length,
                temperature=request.temperature
            )
        else:
            # Start background load if possible
            if finetuned_career_advisor:
                try:
                    from threading import Thread

                    def _bg_load():
                        try:
                            finetuned_career_advisor.llm_advisor.load_start_time = __import__('datetime').datetime.utcnow()
                            finetuned_career_advisor.load_model()
                        except Exception as _e:
                            print(f"Background model load failed: {_e}")

                    Thread(target=_bg_load, daemon=True).start()
                    print("[INFO] Fine-tuned model not loaded: background load started.")
                except Exception as _e:
                    print(f"Could not start background load: {_e}")

            # Fallback to RAG immediately
            if guide_rag_chain:
                try:
                    advice = guide_rag_chain.invoke(request.text)
                    print("[WAIT] Using RAG model (fine-tuned not available)")
                except Exception as e:
                    print(f"Error during RAG invocation: {e}")
                    advice = "Sorry, I encountered an error while generating career advice."
            else:
                advice = "Fine-tuned career advisor is temporarily unavailable."
        
        # Get job matching for completeness
        from sentence_transformers import util as sentence_util 
        query_embedding = embedding_model.embed_query(request.text)
        if job_group_embeddings is not None and len(job_group_embeddings) > 0:
            similarities = sentence_util.cos_sim([query_embedding], job_group_embeddings)[0]
            best_match_index = similarities.argmax()
            matched_job_group = job_groups[best_match_index]
        else:
            matched_job_group = request.text

        live_jobs = scrape_live_jobs(matched_job_group)
        
        return CareerAdviceResponse(
            question=request.text,
            advice=advice,
            confidence="high" if len(advice) > 50 else "medium",
            model_used="Ai_career_Advisor",
            live_jobs=live_jobs,
            matched_job_group=matched_job_group
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating career advice: {str(e)}")

# --- 8. History Endpoints ---
@app.get("/history/analyses", tags=["History"])
async def get_analyses_history(current_user: User = Depends(get_current_user_required), db: SessionLocal = Depends(get_db)):
    analyses = db.query(ResumeAnalysis).filter(ResumeAnalysis.owner_id == current_user.id).order_by(ResumeAnalysis.id.desc()).all()
    return analyses

@app.get("/history/queries", tags=["History"])
async def get_queries_history(current_user: User = Depends(get_current_user_required), db: SessionLocal = Depends(get_db)):
    queries = db.query(CareerQuery).filter(CareerQuery.owner_id == current_user.id).order_by(CareerQuery.id.desc()).all()
    return queries

@app.get("/history/rag-queries", tags=["History"])
async def get_rag_queries_history(current_user: User = Depends(get_current_user_required), db: SessionLocal = Depends(get_db)):
    rag_queries = db.query(RAGCoachQuery).filter(RAGCoachQuery.owner_id == current_user.id).order_by(RAGCoachQuery.id.desc()).all()
    return rag_queries


# ===========================
# RAG COACH ENDPOINTS  
# ===========================

# Global RAG Coach instance
rag_coach_instance = None

# RAG processing state used to track resume+JD auto-processing
rag_processing_state = {
    "processing": False,
    "ready": False,
    "result": None,
    "files": [],
}

class RAGCoachQueryRequest(BaseModel):
    question: str = Field(..., description="User's career coaching question")
    show_context: bool = Field(default=True, description="Whether to include retrieved context")

class RAGCoachResponse(BaseModel):
    answer: str
    context_chunks: List[Dict[str, Any]]
    sources: List[str]

class RAGCoachUploadResponse(BaseModel):
    message: str
    files_uploaded: List[str]

@app.post("/rag-coach/upload", response_model=RAGCoachUploadResponse, tags=["RAG Coach"])
async def upload_rag_documents(
    files: List[UploadFile] = File(...),
    process_resume_job: bool = Form(False),
    current_user: User = Depends(get_current_user_optional)
):
    """Upload PDF documents for RAG Coach and kick off background indexing."""
    import shutil
    from pathlib import Path
    import threading

    upload_dir = Path("./uploads")
    upload_dir.mkdir(exist_ok=True)

    uploaded_files = []

    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

        file_path = upload_dir / file.filename
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uploaded_files.append(file.filename)
        logging.info(f"[UPLOAD] {file.filename}")

    # Start background indexing so upload returns quickly
    def _background_index():
        try:
            from rag_coach import RAGCoachSystem

            global rag_coach_instance
            if rag_coach_instance is None:
                rag_coach_instance = RAGCoachSystem()

            # Collect PDFs to index
            pdf_files = []
            for folder in ["./career_guides", "./uploads"]:
                if os.path.exists(folder):
                    pdf_files.extend([
                        os.path.join(folder, f)
                        for f in os.listdir(folder)
                        if f.lower().endswith('.pdf')
                    ])

            if not pdf_files:
                logging.warning("[WARN] No PDFs found after upload to index.")
                return

            logging.info(f"[DATA] Background: building index from {len(pdf_files)} PDFs...")
            rag_coach_instance.build_vector_store(pdf_files, force_rebuild=False)
            rag_coach_instance.setup_qa_chain()
            logging.info("[OK] Background: RAG Coach indexing complete")
        except Exception as e:
            logging.error(f"[ERROR] Background indexing failed: {e}")

    threading.Thread(target=_background_index, daemon=True).start()

    # If caller requested resume+job processing, start a background task
    # This extracts resume + job-description text and generates formatted
    # suggestions (summary, skills, bullets, keywords) using the LLM.
    if process_resume_job:
        import time, json, datetime
        import re
        
        def _extract_keywords_fallback(resume_text, job_text):
            """Extract skills/keywords when LLM fails"""
            # Common skills keywords
            skills_patterns = [
                r'\b(Python|Java|JavaScript|C\+\+|SQL|React|Angular|Node\.js|Docker|Kubernetes|AWS|Azure|GCP)\b',
                r'\b(Machine Learning|Data Science|DevOps|Agile|Scrum|CI/CD|Git|REST|API)\b',
                r'\b(Leadership|Communication|Problem Solving|Teamwork|Project Management)\b'
            ]
            
            skills = set()
            for pattern in skills_patterns:
                skills.update(re.findall(pattern, job_text, re.IGNORECASE))
            
            skills_list = list(skills)[:8]
            
            return f"""### Skills to Add
{chr(10).join('- ' + skill for skill in skills_list)}

### Resume Tips
- Match your experience to job requirements
- Use specific metrics and achievements  
- Include technologies from job description
- Highlight relevant projects and outcomes"""
        
        def _format_skills_section(llm_response, job_text):
            """Extract and format skills in a clean bullet list"""
            # Try to extract skills from LLM response or job description
            import re
            
            # Extended skills patterns
            skills_patterns = [
                # Programming languages
                r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|PHP|Swift|Kotlin)\b',
                # Frameworks & Libraries
                r'\b(Django|Flask|FastAPI|React|Angular|Vue|Node\.js|Express|Spring|\.NET|Laravel)\b',
                r'\b(NumPy|Pandas|Matplotlib|Scikit-learn|TensorFlow|PyTorch|Keras)\b',
                # Databases
                r'\b(MySQL|PostgreSQL|MongoDB|Redis|Cassandra|Oracle|SQL Server|DynamoDB|SQLite)\b',
                # Cloud & DevOps
                r'\b(AWS|Azure|GCP|Docker|Kubernetes|Jenkins|GitLab|CI/CD|Terraform|Ansible)\b',
                # Web Technologies
                r'\b(HTML|CSS|JavaScript|AJAX|REST|RESTful|GraphQL|WebSocket|API)\b',
                # Methodologies & Tools
                r'\b(Agile|Scrum|Git|GitHub|Jira|Selenium|JUnit|pytest|Travis CI)\b',
                # General concepts
                r'\b(OOP|Object-Oriented Programming|Microservices|Database Design|Testing)\b'
            ]
            
            all_skills = set()
            for pattern in skills_patterns:
                all_skills.update(re.findall(pattern, job_text, re.IGNORECASE))
            
            # Create organized skill categories
            skills_dict = {
                'Programming & Scripting': [],
                'Web Frameworks': [],
                'Data & Analytics': [],
                'Databases': [],
                'DevOps & Tools': [],
                'Other Technologies': []
            }
            
            # Categorize skills
            for skill in all_skills:
                skill_lower = skill.lower()
                if skill_lower in ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go']:
                    skills_dict['Programming & Scripting'].append(skill)
                elif skill_lower in ['django', 'flask', 'fastapi', 'react', 'angular', 'vue', 'node.js', 'express']:
                    skills_dict['Web Frameworks'].append(skill)
                elif skill_lower in ['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'tensorflow', 'pytorch']:
                    skills_dict['Data & Analytics'].append(skill)
                elif skill_lower in ['mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'oracle', 'sqlite']:
                    skills_dict['Databases'].append(skill)
                elif skill_lower in ['docker', 'kubernetes', 'jenkins', 'gitlab', 'ci/cd', 'git', 'github', 'aws', 'azure', 'gcp']:
                    skills_dict['DevOps & Tools'].append(skill)
                else:
                    skills_dict['Other Technologies'].append(skill)
            
            # Build formatted output
            output_lines = []
            for category, skills in skills_dict.items():
                if skills:
                    for skill in sorted(set(skills)):
                        output_lines.append(f"• {skill}")
            
            # Add essential Python-related skills if this is a Python job
            if 'python' in job_text.lower():
                essential_python_skills = [
                    "Python Programming & Scripting",
                    "Object-Oriented Programming (OOP)",
                    "RESTful API Development",
                    "Database Design & Management",
                    "Software Testing (pytest, unittest)",
                    "Version Control (Git, GitHub, GitLab)",
                    "CI/CD Pipelines"
                ]
                output_lines.extend([f"• {skill}" for skill in essential_python_skills if skill not in str(output_lines)])
            
            if not output_lines:
                output_lines = [
                    "• Python Programming & Scripting",
                    "• Django / Flask",
                    "• RESTful API Development",
                    "• MySQL / PostgreSQL",
                    "• Git & GitHub",
                    "• CI/CD Pipelines"
                ]
            
            return '\n'.join(output_lines)
        
        def _format_resume_bullets(resume_text, job_text):
            """Generate sample resume bullet points"""
            bullets = [
                "• Developed and maintained Python applications using Django/Flask frameworks, improving system performance by 30%",
                "• Designed and implemented RESTful APIs serving 100K+ requests/day with 99.9% uptime",
                "• Collaborated with cross-functional teams to deliver features, reducing deployment time by 40%",
                "• Optimized database queries and schema design in MySQL/PostgreSQL, resulting in 45% faster data retrieval",
                "• Implemented CI/CD pipelines using Jenkins/GitLab, automating deployment process and reducing errors by 60%",
                "• Built data processing pipelines with NumPy and Pandas, handling 1M+ records daily",
                "• Wrote comprehensive unit and integration tests with pytest/Selenium, achieving 85%+ code coverage",
                "• Mentored junior developers on best practices for OOP, design patterns, and code quality"
            ]
            return '\n'.join(bullets)
        
        def _format_ats_keywords(job_text):
            """Extract ATS keywords from job description"""
            import re
            
            # Extract key technical terms and phrases
            keywords = set()
            
            # Common tech keywords
            tech_keywords = [
                'Python', 'Django', 'Flask', 'API', 'REST', 'Database', 
                'MySQL', 'PostgreSQL', 'MongoDB', 'Git', 'GitHub', 'GitLab',
                'CI/CD', 'Jenkins', 'Docker', 'Agile', 'Scrum', 'Testing',
                'NumPy', 'Pandas', 'OOP', 'Web Development', 'JavaScript',
                'HTML', 'CSS', 'AJAX', 'Problem Solving', 'Team Collaboration'
            ]
            
            # Find which keywords appear in the job description
            for keyword in tech_keywords:
                if keyword.lower() in job_text.lower():
                    keywords.add(keyword)
            
            # Ensure we have at least 10 keywords
            if len(keywords) < 10:
                keywords.update(['Python Programming', 'Web Frameworks', 'Database Management', 
                                'Version Control', 'Continuous Integration', 'Software Testing'])
            
            # Format as comma-separated list
            keyword_list = sorted(list(keywords))[:20]
            return ', '.join(keyword_list)

        def _normalize_skill(skill_text):
            """Normalize a single skill to its canonical form"""
            import re
            if not skill_text:
                return ""
            
            # Convert to lowercase
            normalized = skill_text.strip().lower()
            
            # Remove punctuation and special chars except spaces, +, #
            normalized = re.sub(r'[^\w\s+#-]', '', normalized)
            
            # Remove .js, .css, .html suffixes
            normalized = re.sub(r'\.(js|css|html|py)$', '', normalized)
            
            # Remove version numbers (e.g., "3", "5", "3.10")
            normalized = re.sub(r'\s*\d+(\.\d+)*\s*$', '', normalized)
            
            # Normalize spaces
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            # Comprehensive synonym mapping
            synonym_map = {
                # API & REST
                'restful api': 'rest api',
                'restful': 'rest api',
                'rest': 'rest api',
                'api': 'rest api',
                
                # Databases
                'sqlite3': 'sqlite',
                'sqlite': 'sqlite',
                'postgresql': 'postgres',
                'postgres': 'postgres',
                'mongodb': 'mongo',
                'mongo': 'mongo',
                'mysql': 'mysql',
                
                # Web
                'html5': 'html',
                'html': 'html',
                'css3': 'css',
                'css': 'css',
                'javascript': 'javascript',
                'js': 'javascript',
                
                # Cloud
                'amazon web services': 'aws',
                'aws': 'aws',
                'azure': 'azure',
                'microsoft azure': 'azure',
                'gcp': 'gcp',
                'google cloud platform': 'gcp',
                
                # Programming concepts
                'object-oriented programming': 'oop',
                'object oriented programming': 'oop',
                'oop': 'oop',
                
                # ML
                'machine learning': 'machine learning',
                'ml': 'machine learning',
                
                # DevOps
                'ci/cd': 'cicd',
                'cicd': 'cicd',
                'ci cd': 'cicd',
                'continuous integration': 'cicd',
                'continuous deployment': 'cicd',
                
                # Frameworks
                'express': 'express',
                'expressjs': 'express',
                'node': 'nodejs',
                'nodejs': 'nodejs',
                'react': 'react',
                'reactjs': 'react',
                'angular': 'angular',
                'angularjs': 'angular',
                'vue': 'vue',
                'vuejs': 'vue',
                
                # Python frameworks
                'django': 'django',
                'flask': 'flask',
                'fastapi': 'fastapi',
                
                # ML libraries
                'numpy': 'numpy',
                'pandas': 'pandas',
                'pytorch': 'pytorch',
                'tensorflow': 'tensorflow',
                'scikit-learn': 'sklearn',
                'scikit learn': 'sklearn',
                'sklearn': 'sklearn',
                
                # Tools
                'git': 'git',
                'github': 'github',
                'gitlab': 'gitlab',
                'docker': 'docker',
                'kubernetes': 'kubernetes',
                'k8s': 'kubernetes',
                'jenkins': 'jenkins',
                'linux': 'linux',
                
                # Other
                'bootstrap': 'bootstrap',
                'tailwind': 'tailwind',
                'tailwind css': 'tailwind',
                'langchain': 'langchain',
                'faiss': 'faiss',
                'spacy': 'spacy',
            }
            
            return synonym_map.get(normalized, normalized)
        
        def _extract_skill_tokens(text):
            """Return a set of normalized skill tokens found in the provided text."""
            import re
            if not text:
                return set()

            # Comprehensive regex patterns for skill extraction
            skills_patterns = [
                # Programming languages
                r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|PHP|Swift|Kotlin|Scala)\b',
                
                # Web frameworks
                r'\b(Django|Flask|FastAPI|React(?:\.js)?|Angular(?:\.js)?|Vue(?:\.js)?|Node(?:\.js)?|Express(?:\.js)?|Spring|\.NET|Laravel|Next\.js|Svelte)\b',
                
                # ML/AI libraries
                r'\b(NumPy|Pandas|Matplotlib|Scikit-learn|TensorFlow|PyTorch|Keras|Machine\s+Learning|ML|LangChain|FAISS|spaCy|Hugging\s+Face|Sentence-BERT)\b',
                
                # Databases
                r'\b(MySQL|PostgreSQL|MongoDB|Redis|Cassandra|Oracle|SQL\s+Server|DynamoDB|SQLite(?:3)?|Postgres)\b',
                
                # Cloud & DevOps
                r'\b(AWS|Amazon\s+Web\s+Services|Azure|Microsoft\s+Azure|GCP|Google\s+Cloud(?:\s+Platform)?|Docker|Kubernetes|K8s|Jenkins|GitLab|CI/CD|CI\s*/\s*CD|Terraform|Ansible|Linux)\b',
                
                # Web technologies
                r'\b(HTML5?|CSS3?|AJAX|REST(?:ful)?(?:\s+API)?|GraphQL|WebSocket|API|Bootstrap|Tailwind(?:\s+CSS)?)\b',
                
                # Methodologies & Tools
                r'\b(Agile|Scrum|Git|GitHub|Jira|Selenium|JUnit|pytest|Travis\s+CI)\b',
                
                # Concepts & Patterns
                r'\b(OOP|Object-Oriented\s+Programming|Microservices|Database(?:\s+(?:Design|Management))?|Testing|Problem[- ]Solving)\b'
            ]

            raw_matches = set()
            for p in skills_patterns:
                matches = re.findall(p, text, re.IGNORECASE)
                for m in matches:
                    if isinstance(m, tuple):
                        m = ' '.join([part for part in m if part]).strip()
                    if m:
                        raw_matches.add(m)
            
            # Normalize all extracted skills
            normalized_tokens = set()
            for skill in raw_matches:
                normalized = _normalize_skill(skill)
                if normalized and len(normalized) > 1:  # Ignore single chars
                    normalized_tokens.add(normalized)
            
            return normalized_tokens

        
        def _background_process():
            try:
                global rag_processing_state, rag_coach_instance
                rag_processing_state.update({"processing": True, "ready": False, "result": None, "files": uploaded_files})

                from rag_coach import RAGCoachSystem

                # Ensure we have an LLM instance available
                if rag_coach_instance is None:
                    rag_coach_instance = RAGCoachSystem()

                # Make sure LLM is initialized (CPU mode etc.)
                try:
                    rag_coach_instance._initialize_llm()
                except Exception:
                    # If LLM can't be initialized, record failure and exit
                    rag_processing_state.update({"processing": False, "ready": False, "result": {"error": "LLM not available"}})
                    return

                # Load uploaded documents (use existing loader to keep parsing consistent)
                upload_dir = "./uploads"
                pdf_paths = [os.path.join(upload_dir, f) for f in uploaded_files if f.lower().endswith('.pdf')]
                docs = rag_coach_instance.load_pdf_documents(pdf_paths)

                # Aggregate texts per source
                sources_text = {}
                for d in docs:
                    src = d.metadata.get('source', 'uploaded')
                    sources_text.setdefault(src, [])
                    sources_text[src].append(d.page_content)

                # Enhanced heuristic: detect document type by content analysis
                def _detect_document_type(text, filename):
                    """Detect if document is resume or job description based on content + filename"""
                    text_lower = text.lower()
                    fname_lower = filename.lower()
                    
                    # Filename-based detection (primary)
                    if 'resume' in fname_lower or 'cv' in fname_lower or 'profile' in fname_lower:
                        return 'resume'
                    elif 'job' in fname_lower or 'description' in fname_lower or 'jd' in fname_lower or 'position' in fname_lower:
                        return 'job'
                    
                    # Content-based detection (fallback)
                    resume_indicators = [
                        'technical skills', 'education', 'professional experience', 
                        'key projects', 'certifications', 'achievements',
                        'linkedin', 'github', 'email', 'mobile', 'phone'
                    ]
                    
                    job_indicators = [
                        'job title', 'job summary', 'key responsibilities', 
                        'required skills', 'qualifications', 'job type',
                        'we are looking for', 'ideal candidate', 'apply', 'salary range'
                    ]
                    
                    resume_score = sum(1 for ind in resume_indicators if ind in text_lower)
                    job_score = sum(1 for ind in job_indicators if ind in text_lower)
                    
                    if resume_score > job_score:
                        return 'resume'
                    elif job_score > resume_score:
                        return 'job'
                    else:
                        return 'unknown'
                
                # Detect and separate resume vs job description
                resume_text = ""
                job_text = ""
                resume_file = None
                job_file = None
                
                for fname in uploaded_files:
                    full_text = "\n".join(sources_text.get(fname, []))
                    doc_type = _detect_document_type(full_text, fname)
                    
                    if doc_type == 'resume':
                        resume_text = full_text
                        resume_file = fname
                        logging.info(f"[RAG] Detected RESUME: {fname}")
                    elif doc_type == 'job':
                        job_text = full_text
                        job_file = fname
                        logging.info(f"[RAG] Detected JOB DESCRIPTION: {fname}")
                
                # Fallback: if detection failed and we have exactly 2 files
                if not resume_text and not job_text and len(uploaded_files) >= 2:
                    # Assume first file is resume, second is job (common upload pattern)
                    resume_file, job_file = uploaded_files[0], uploaded_files[1]
                    resume_text = "\n".join(sources_text.get(resume_file, []))
                    job_text = "\n".join(sources_text.get(job_file, []))
                    logging.info(f"[RAG] Fallback: {resume_file} → RESUME, {job_file} → JOB")
                
                # If still empty, log error
                if not resume_text or not job_text:
                    logging.warning(f"[RAG] Could not identify both resume and JD. Resume: {len(resume_text)} chars, Job: {len(job_text)} chars")

                # ULTRA-SHORT text to prevent crashes (300 chars = ~75 tokens)
                resume_short = resume_text[:300] if resume_text else ""
                job_short = job_text[:300] if job_text else ""

                # MINIMAL prompt (total ~100 tokens to fit in 512 context)
                prompt = f"""Resume: {resume_short}

Job: {job_short}

List 5 skills."""

                llm_response = ""
                try:
                    logging.info(f"[RAG] Calling Ollama ({len(prompt)} chars)...")
                    llm_response = rag_coach_instance.llm.invoke(prompt)
                    logging.info(f"[RAG] Success: {len(llm_response)} chars")
                    
                    if not llm_response or len(llm_response.strip()) < 10:
                        raise Exception("Response too short")
                        
                except Exception as e:
                    logging.error(f"[RAG] Ollama failed: {e}")
                    # Simple keyword-based fallback
                    llm_response = _extract_keywords_fallback(resume_text, job_text)
                    logging.info("[RAG] Using keyword fallback")

                # Compute JD-only skills using the extractor
                job_skills = _extract_skill_tokens(job_text)
                resume_skills = _extract_skill_tokens(resume_text)
                jd_only_skills_normalized = sorted(list(job_skills.difference(resume_skills)))

                # Create display-friendly names for the skills
                display_name_map = {
                    'cicd': 'CI/CD Pipelines',
                    'oop': 'Object-Oriented Programming (OOP)',
                    'rest api': 'RESTful API Development',
                    'machine learning': 'Machine Learning',
                    'aws': 'Amazon Web Services (AWS)',
                    'azure': 'Microsoft Azure',
                    'gcp': 'Google Cloud Platform (GCP)',
                    'html': 'HTML',
                    'css': 'CSS',
                    'sqlite': 'SQLite',
                    'postgres': 'PostgreSQL',
                    'mongo': 'MongoDB',
                    'mysql': 'MySQL',
                    'javascript': 'JavaScript',
                    'typescript': 'TypeScript',
                    'nodejs': 'Node.js',
                    'react': 'React.js',
                    'express': 'Express.js',
                    'angular': 'Angular',
                    'vue': 'Vue.js',
                    'django': 'Django',
                    'flask': 'Flask',
                    'fastapi': 'FastAPI',
                    'docker': 'Docker',
                    'kubernetes': 'Kubernetes (K8s)',
                    'git': 'Git',
                    'github': 'GitHub',
                    'gitlab': 'GitLab',
                    'numpy': 'NumPy',
                    'pandas': 'Pandas',
                    'pytorch': 'PyTorch',
                    'tensorflow': 'TensorFlow',
                    'sklearn': 'Scikit-learn',
                    'langchain': 'LangChain',
                    'faiss': 'FAISS',
                    'spacy': 'spaCy',
                    'python': 'Python',
                    'java': 'Java',
                    'c++': 'C++',
                    'linux': 'Linux',
                    'jenkins': 'Jenkins',
                    'agile': 'Agile Methodology',
                    'scrum': 'Scrum',
                    'bootstrap': 'Bootstrap',
                    'tailwind': 'Tailwind CSS',
                    'testing': 'Software Testing',
                    'problem solving': 'Problem-Solving'
                }
                
                jd_only_skills_display = []
                for skill in jd_only_skills_normalized:
                    display_name = display_name_map.get(skill, skill.title())
                    jd_only_skills_display.append(display_name)

                # Friendly bullet lines for JD-only skills (vertical format with newlines)
                if jd_only_skills_display:
                    skills_section = '\n'.join([f'• {s}' for s in jd_only_skills_display])
                    # Create ATS-friendly keywords from JD-only skills
                    ats_keywords_from_jd = ', '.join(jd_only_skills_display)
                else:
                    # Fall back to LLM/formatted extraction if difference is empty
                    skills_section = "✅ Great! Your resume already covers most key skills from the job description.\n\nConsider emphasizing these skills more prominently in your experience bullets."
                    ats_keywords_from_jd = _format_ats_keywords(job_text)

                # Format response into structured sections (include JD-only skills prominently)
                formatted_output = f"""## ✅ Skills to Add (from Job Description, missing from your Resume)

{skills_section}

## 🔑 ATS-Friendly Keywords (Skills to Add)

**Copy these exact keywords into your resume:**

{ats_keywords_from_jd}

## ✍️ Suggested Resume Bullet Points

{_format_resume_bullets(resume_text, job_text)}

---

## 💡 How to Use These Suggestions

**Skills Section**: Add the skills listed above to your Technical Skills section. List each skill on a separate line or separate them with commas.

**ATS Keywords**: Use the exact keywords provided above throughout your resume - especially in your Skills section and experience bullets. ATS systems scan for these exact matches.

**Experience Section**: Adapt the resume bullets to reflect your actual achievements. Replace placeholder percentages with real metrics from your work.

**Profile/Summary**: Create a 2-3 sentence summary combining your experience level + key technologies + impact. Example: "Python Developer with 3+ years building web applications using Django and Flask. Expertise in RESTful API development, database optimization, and CI/CD automation."

**Formatting Tips**:
• Use bullet points for skills and experience
• Start experience bullets with action verbs (Developed, Implemented, Designed, Optimized)
• Include metrics wherever possible (%, $, time saved)
• Keep each bullet to 1-2 lines maximum
• Match the exact terminology from the job description"""

                result = {
                    "formatted": formatted_output,
                    "summary": None,
                    "skills": jd_only_skills_display,
                    "resume_bullets": None,
                    "keywords": ats_keywords_from_jd if jd_only_skills_display else _format_ats_keywords(job_text),
                    "generated_at": datetime.datetime.utcnow().isoformat() + 'Z'
                }

                # Save to file for quick retrieval
                processed_dir = Path("./uploads/processed")
                processed_dir.mkdir(exist_ok=True)
                out_path = processed_dir / f"processed_{int(time.time())}.json"
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(result, fh, ensure_ascii=False, indent=2)

                rag_processing_state.update({"processing": False, "ready": True, "result": result})

            except Exception as e:
                logging.error(f"[ERROR] Background processing failed: {e}")
                rag_processing_state.update({"processing": False, "ready": False, "result": {"error": str(e)}})

        threading.Thread(target=_background_process, daemon=True).start()

    return RAGCoachUploadResponse(
        message=f"Successfully uploaded {len(uploaded_files)} file(s). Indexing started in background.",
        files_uploaded=uploaded_files
    )

@app.post("/rag-coach/build-index", tags=["RAG Coach"])
async def build_rag_index(
    pdf_files: List[str] = None,
    force_rebuild: bool = False,
    background: bool = False,
    current_user: User = Depends(get_current_user_optional)
):
    """Build FAISS vector index from uploaded PDFs.

    If `background=True` this will start indexing in a daemon thread and return immediately.
    """
    import threading
    global rag_coach_instance

    try:
        from rag_coach import RAGCoachSystem

        if rag_coach_instance is None:
            rag_coach_instance = RAGCoachSystem()

        # Find PDFs
        if pdf_files is None:
            pdf_files = []
            for folder in ["./career_guides", "./uploads"]:
                if os.path.exists(folder):
                    pdf_files.extend([
                        os.path.join(folder, f)
                        for f in os.listdir(folder)
                        if f.lower().endswith('.pdf')
                    ])

        if not pdf_files:
            raise HTTPException(
                status_code=400,
                detail="No PDF files found. Upload PDFs first using /rag-coach/upload"
            )

        def _index_task():
            try:
                logging.info(f"[DATA] Building index from {len(pdf_files)} PDFs...")
                rag_coach_instance.build_vector_store(pdf_files, force_rebuild=force_rebuild)
                rag_coach_instance.setup_qa_chain()
                logging.info("[OK] RAG Coach index built successfully")
            except Exception as e:
                logging.error(f"[ERROR] RAG background indexing failed: {e}")

        if background:
            threading.Thread(target=_index_task, daemon=True).start()
            return {"message": "Indexing started in background", "indexed_files": len(pdf_files)}

        # Synchronous build (blocking) if background=False
        _index_task()
        return {
            "message": "RAG Coach index built successfully",
            "indexed_files": len(pdf_files),
            "files": [os.path.basename(f) for f in pdf_files]
        }

    except ConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Ollama not available",
                "message": str(e),
                "instructions": [
                    "1. Install Ollama: https://ollama.ai/download",
                    "2. Start Ollama service",
                    "3. Pull model: ollama pull mistral:7b-instruct",
                    "4. Verify: ollama list"
                ]
            }
        )
    except Exception as e:
        logging.error(f"[ERROR] Building RAG index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to build index: {str(e)}")

@app.post("/rag-coach/query", response_model=RAGCoachResponse, tags=["RAG Coach"])
async def query_rag_coach(
    query: RAGCoachQueryRequest,
    db: SessionLocal = Depends(get_db),
    current_user: User = Depends(get_current_user_optional)
):
    """Ask RAG Coach a question - AUTO-INITIALIZES if needed"""
    global rag_coach_instance
    
    # AUTO-INITIALIZE if not ready
    if rag_coach_instance is None or rag_coach_instance.qa_chain is None:
        logging.info("[INIT] RAG Coach not ready - attempting auto-initialization...")
        
        try:
            from rag_coach import RAGCoachSystem
            
            # Initialize RAG Coach if needed
            if rag_coach_instance is None:
                rag_coach_instance = RAGCoachSystem()
                logging.info("[OK] RAG Coach instance created")
            
            # Check for existing PDFs in uploads folder
            pdf_files = []
            upload_dir = "./uploads"
            if os.path.exists(upload_dir):
                pdf_files = [
                    os.path.join(upload_dir, f) 
                    for f in os.listdir(upload_dir) 
                    if f.endswith('.pdf')
                ]
            
            if pdf_files:
                # Build index from existing PDFs
                logging.info(f"[INIT] Found {len(pdf_files)} PDFs, building index...")
                rag_coach_instance.build_vector_store(pdf_files)
                rag_coach_instance.setup_qa_chain()
                logging.info("[OK] RAG Coach index built successfully!")
            else:
                # No PDFs - use direct LLM mode
                logging.info("[WARN] No PDFs found - initializing direct LLM mode")
                rag_coach_instance._initialize_llm()
                
        except Exception as e:
            logging.error(f"[ERROR] Auto-initialization failed: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"RAG Coach initialization failed: {str(e)}. Please upload PDFs first using the RAG Coach tab."
            )
    
    try:
        # Use RAG if available
        if rag_coach_instance.qa_chain is not None:
            result = rag_coach_instance.answer_query(
                question=query.question,
                show_context=query.show_context
            )
            
            sources = list(set([
                chunk['source'] for chunk in result['context_chunks']
            ]))
            
            # Save to history if user is logged in
            if current_user:
                new_rag_query = RAGCoachQuery(
                    owner_id=current_user.id,
                    question=query.question,
                    answer=result['answer'],
                    sources=json.dumps(sources)
                )
                db.add(new_rag_query)
                db.commit()
                logging.info(f"[HISTORY] Saved RAG query for user {current_user.email}")
            
            return RAGCoachResponse(
                answer=result['answer'],
                context_chunks=result['context_chunks'],
                sources=sources
            )
        else:
            # Direct LLM mode
            logging.info("[MSG] Using direct LLM (no RAG)")
            answer = rag_coach_instance.llm.invoke(query.question)
            
            # Save to history if user is logged in
            if current_user:
                new_rag_query = RAGCoachQuery(
                    owner_id=current_user.id,
                    question=query.question,
                    answer=answer,
                    sources=json.dumps(["Direct LLM (No PDFs uploaded)"])
                )
                db.add(new_rag_query)
                db.commit()
                logging.info(f"[HISTORY] Saved RAG query for user {current_user.email}")
            
            return RAGCoachResponse(
                answer=answer,
                context_chunks=[],
                sources=["Direct LLM (No PDFs uploaded)"]
            )
        
    except Exception as e:
        logging.error(f"[ERROR] RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/rag-coach/status", tags=["RAG Coach"])
async def get_rag_coach_status():
    """Get RAG Coach system status"""
    global rag_coach_instance
    # Merge system-level status with processing state
    system_status = {
        "initialized": False,
        "vector_store_ready": False,
        "qa_chain_ready": False,
        "llm_model": None,
        "message": "RAG Coach not initialized"
    }

    if rag_coach_instance is not None:
        system_status.update({
            "initialized": True,
            "vector_store_ready": rag_coach_instance.vector_store is not None,
            "qa_chain_ready": rag_coach_instance.qa_chain is not None,
            "llm_model": rag_coach_instance.llm_model_name if hasattr(rag_coach_instance, 'llm_model_name') else "Unknown",
            "message": "RAG Coach operational"
        })

    # Include processing state if available
    try:
        return {
            **system_status,
            "processing": rag_processing_state.get("processing", False),
            "processing_ready": rag_processing_state.get("ready", False),
            "processing_files": rag_processing_state.get("files", []),
        }
    except NameError:
        # If processing state missing, return only system status
        return system_status


@app.get("/rag-coach/processed-result", tags=["RAG Coach"])
async def get_rag_processed_result():
    """Return processed resume/JD suggestions (if ready)"""
    global rag_processing_state

    if not rag_processing_state.get("ready", False):
        raise HTTPException(status_code=404, detail="No processed result available yet")

    return {
        "files": rag_processing_state.get("files", []),
        "result": rag_processing_state.get("result", {})
    }