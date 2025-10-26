"""
RAG Coach - Retrieval-Augmented Generation Career Coaching System
Uses LangChain + FAISS + Ollama Gemma 2B for personalized career advice

This module is completely independent and won't affect existing Resume Analyzer 
or AI Career Advisor functionality.
"""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

try:
    # LangChain imports
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import Ollama
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    logger.error("Install required packages: pip install langchain langchain-community pypdf2 faiss-cpu")
    raise


class RAGCoachSystem:
    """
    Complete RAG-based Career Coaching System
    
    Features:
    - Load multiple PDF documents (career guides + user uploads)
    - Create FAISS vector embeddings for semantic search
    - Use Ollama Gemma 2B for intelligent response generation
    - Retrieve relevant context before answering queries
    """
    
    def __init__(
        self, 
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = None,
        vector_store_path: str = "./rag_coach_index"
    ):
        """
        Initialize RAG Coach System
        
        Args:
            embedding_model_name: HuggingFace embedding model for vector creation
            llm_model_name: Ollama model name for text generation (auto-detects available Mistral models)
            vector_store_path: Path to save/load FAISS vector store
        """
        self.embedding_model_name = embedding_model_name
        
        # Auto-detect Mistral model if not specified
        if llm_model_name is None:
            llm_model_name = self._detect_mistral_model()
        
        self.llm_model_name = llm_model_name
        self.vector_store_path = vector_store_path
        
        # Initialize components (set to None until loaded)
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        
        logger.info(f"🚀 RAG Coach System initialized with model: {self.llm_model_name}")
    
    def _detect_mistral_model(self) -> str:
        """
        Auto-detect available Mistral model from Ollama
        
        Returns:
            Model name string (prefers instruct variants)
        """
        try:
            import subprocess
            import os
            
            # Try to find ollama executable
            ollama_cmd = 'ollama'
            
            # Check common Windows locations
            possible_paths = [
                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Ollama', 'ollama.exe'),
                r'C:\Program Files\Ollama\ollama.exe',
                r'C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe'.format(os.environ.get('USERNAME', '')),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    ollama_cmd = path
                    logger.info(f"✅ Found Ollama at: {path}")
                    break
            
            result = subprocess.run([ollama_cmd, 'list'], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                models = result.stdout.lower()
                
                # Priority order for models (smallest/fastest first for low RAM systems)
                preferred_models = [
                    'tinyllama',                     # 637 MB RAM - BEST for very low memory
                    'mistral:7b-instruct-q2_k',      # 800 MB RAM - Good for low memory
                    'mistral:7b-instruct-q4_k_m',    # 1.6 GB RAM
                    'mistral:7b-instruct',           # 1.6 GB RAM
                    'mistral:latest',
                    'mistral',
                ]
                
                for model in preferred_models:
                    if model in models:
                        logger.info(f"✅ Auto-detected Ollama model: {model}")
                        return model
                
                # Check if any mistral variant exists
                if 'mistral' in models:
                    logger.info(f"✅ Found Mistral model in Ollama")
                    return 'mistral'
                    
                # Check for tinyllama
                if 'tinyllama' in models:
                    logger.info(f"✅ Found TinyLlama model in Ollama")
                    return 'tinyllama'
            
            logger.warning("⚠️  No model found in Ollama, using default: tinyllama")
            return 'tinyllama'
            
        except Exception as e:
            logger.warning(f"⚠️  Could not detect Ollama models: {e}")
            return 'tinyllama'
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings model"""
        if self.embeddings is None:
            logger.info(f"📦 Loading embedding model: {self.embedding_model_name}")
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("✅ Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                raise
    
    def _initialize_llm(self):
        """Initialize Ollama LLM with SPEED OPTIMIZATIONS + CPU-ONLY MODE"""
        if self.llm is None:
            logger.info(f"🤖 Connecting to Ollama with model: {self.llm_model_name}")
            try:
                # Force CPU-only mode to avoid GPU memory errors
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
                
                # ULTRA-CONSERVATIVE settings to prevent crashes
                self.llm = Ollama(
                    model=self.llm_model_name,
                    temperature=0.7,
                    num_ctx=512,      # MINIMAL context to prevent crashes
                    top_k=20,
                    top_p=0.9,
                    num_predict=256,  # Moderate output length
                    repeat_penalty=1.1,
                    # CPU-specific optimizations
                    num_thread=2,     # Reduced threads for stability
                    num_gpu=0         # Force CPU-only mode
                )
                # Quick connection test
                test_response = self.llm.invoke("Hi")
                logger.info("✅ Ollama LLM connected successfully (CPU MODE - STABLE)")
            except Exception as e:
                logger.error("❌ Failed to connect to Ollama")
                logger.error("\n🔧 To fix this issue:")
                logger.error("1. Install Ollama: https://ollama.ai/download")
                logger.error("2. Start Ollama service")
                logger.error(f"3. Pull the model: ollama pull {self.llm_model_name}")
                logger.error("4. Verify: ollama list")
                raise ConnectionError(f"Ollama not available: {e}")
    
    def load_pdf_documents(self, pdf_paths: List[str]) -> List[Document]:
        """
        Load multiple PDF files and extract text content with enhanced metadata
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of LangChain Document objects with text content and doc_type metadata
        """
        logger.info(f"📄 Loading {len(pdf_paths)} PDF documents...")
        all_documents = []
        
        def _detect_document_type(text, filename):
            """Detect if document is resume or job description"""
            text_lower = text.lower()
            fname_lower = filename.lower()
            
            # Filename-based detection (primary)
            if 'resume' in fname_lower or 'cv' in fname_lower or 'profile' in fname_lower:
                return 'RESUME'
            elif 'job' in fname_lower or 'description' in fname_lower or 'jd' in fname_lower or 'position' in fname_lower:
                return 'JOB_DESCRIPTION'
            
            # Content-based detection (fallback)
            resume_indicators = [
                'technical skills', 'education', 'professional experience', 
                'key projects', 'certifications', 'achievements',
                'linkedin', 'github', 'email', 'mobile', 'phone'
            ]
            
            job_indicators = [
                'job title', 'job summary', 'key responsibilities', 
                'required skills', 'qualifications', 'job type',
                'we are looking for', 'ideal candidate', 'apply', 'salary'
            ]
            
            resume_score = sum(1 for ind in resume_indicators if ind in text_lower)
            job_score = sum(1 for ind in job_indicators if ind in text_lower)
            
            if resume_score > job_score:
                return 'RESUME'
            elif job_score > resume_score:
                return 'JOB_DESCRIPTION'
            else:
                return 'UNKNOWN'
        
        for i, pdf_path in enumerate(pdf_paths):
            if not os.path.exists(pdf_path):
                logger.warning(f"⚠️ File not found: {pdf_path}")
                continue
            
            try:
                # Load PDF using LangChain's PyPDFLoader
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                # Get full text to detect document type
                full_text = "\n".join([doc.page_content for doc in documents])
                filename = os.path.basename(pdf_path)
                doc_type = _detect_document_type(full_text, filename)
                
                # Add enhanced metadata to identify source AND document type
                for doc in documents:
                    doc.metadata['source'] = filename
                    doc.metadata['doc_type'] = doc_type
                    doc.metadata['doc_index'] = i
                
                all_documents.extend(documents)
                logger.info(f"  ✅ Loaded: {filename} ({len(documents)} pages) [Type: {doc_type}]")
                
            except Exception as e:
                logger.error(f"  ❌ Error loading {pdf_path}: {e}")
                continue
        
        logger.info(f"✅ Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def split_documents(
        self, 
        documents: List[Document], 
        chunk_size: int = 600,  # REDUCED for faster processing (was 1000)
        chunk_overlap: int = 100  # REDUCED overlap (was 200)
    ) -> List[Document]:
        """
        Split documents into smaller chunks - OPTIMIZED FOR SPEED
        
        Args:
            documents: List of Document objects
            chunk_size: Maximum size of each chunk (reduced for speed)
            chunk_overlap: Overlap between consecutive chunks (reduced)
            
        Returns:
            List of split Document chunks
        """
        logger.info("✂️ Splitting documents into chunks (FAST MODE)...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        logger.info(f"✅ Created {len(splits)} text chunks (optimized)")
        
        return splits
    
    def build_vector_store(self, pdf_files: List[str], force_rebuild: bool = False):
        """
        Build FAISS vector store from PDF documents
        
        Args:
            pdf_files: List of paths to PDF files to index
            force_rebuild: If True, rebuild even if vector store exists
        """
        # Initialize embeddings
        self._initialize_embeddings()
        
        # Check if vector store already exists
        if os.path.exists(self.vector_store_path) and not force_rebuild:
            logger.info(f"📚 Loading existing vector store from {self.vector_store_path}")
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✅ Vector store loaded successfully")
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to load existing vector store: {e}")
                logger.info("🔄 Building new vector store...")
        
        # Load and process documents
        documents = self.load_pdf_documents(pdf_files)
        
        if not documents:
            raise ValueError("No documents loaded. Please check PDF file paths.")
        
        # Split into chunks
        splits = self.split_documents(documents)
        
        # Create FAISS vector store
        logger.info("🧠 Creating FAISS vector embeddings...")
        try:
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            
            # Save for future use
            self.vector_store.save_local(self.vector_store_path)
            logger.info(f"💾 Vector store saved to {self.vector_store_path}")
            logger.info("✅ Vector store built successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to build vector store: {e}")
            raise
    
    def setup_qa_chain(self):
        """
        Setup the Retrieval QA chain with LLM and vector store
        """
        # Initialize LLM
        self._initialize_llm()
        
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call build_vector_store() first.")
        
        # Create retriever - SPEED OPTIMIZED
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        )
        
        # IMPROVED prompt with clear instructions to use uploaded documents
        prompt_template = """You are an expert career advisor analyzing a user's RESUME and a JOB DESCRIPTION they uploaded.

CRITICAL INSTRUCTIONS:
1. Your answer MUST be based ONLY on the provided context from the uploaded documents
2. If asked about job requirements/title/skills - refer to the JOB DESCRIPTION document
3. If asked about the user's background/experience - refer to the RESUME document  
4. Extract specific information (job titles, skills, requirements) directly from the documents
5. DO NOT make up generic career advice - use the ACTUAL content from the uploaded files

Context from uploaded documents:
{context}

User Question: {question}

Answer (based strictly on the uploaded documents):"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("✅ QA chain configured successfully")
    
    def _post_process_answer(self, raw_answer: str, question: str, context_docs: List[Document]) -> str:
        """
        Post-process LLM answer to extract concise, relevant information
        
        Args:
            raw_answer: Raw answer from LLM
            question: Original user question
            context_docs: Retrieved context documents
            
        Returns:
            Cleaned, concise answer
        """
        import re
        
        question_lower = question.lower()
        
        # Check if this is a job title question
        job_title_keywords = [
            'job title', 'job role', 'position', 'role', 'title',
            'what job', 'which job', 'what position', 'which position',
            'what should i mention', 'what do i mention'
        ]
        
        # Check if this is a skills question - ENHANCED
        skills_keywords = [
            'skills required', 'required skills', 'what skills', 'which skills',
            'skills needed', 'technical skills', 'qualifications',
            'skills mentioned', 'skills in job', 'skills from job',
            'highlight skills', 'skills in the jd', 'jd skills'
        ]
        
        is_job_title_question = any(kw in question_lower for kw in job_title_keywords)
        is_skills_question = any(kw in question_lower for kw in skills_keywords)
        
        if is_job_title_question:
            # Extract job title from context documents (more reliable than LLM answer)
            job_titles = self._extract_job_titles_from_context(context_docs)
            
            if job_titles:
                # Return the most prominent job title
                primary_title = job_titles[0]
                
                if len(job_titles) == 1:
                    return f"**{primary_title}**\n\nThis is the job title mentioned in the job description you uploaded."
                else:
                    # Multiple titles found
                    other_titles = ", ".join(job_titles[1:])
                    return f"**{primary_title}**\n\nAlternatively, the job description also mentions: {other_titles}"
            else:
                # No pattern match - try to extract from LLM answer
                logger.warning("⚠️ No job titles from patterns, analyzing LLM response...")
                title_from_llm = self._extract_title_from_llm_answer(raw_answer)
                if title_from_llm:
                    return f"**{title_from_llm}**\n\nExtracted from the job description context."
                else:
                    # Last resort - ask user to check the JD directly
                    return "I couldn't identify a specific job title from the uploaded documents. Please check the job description PDF for the exact job title field. Common locations:\n• Top of the document labeled 'Job Title:' or 'Position:'\n• In the subject line\n• First paragraph mentioning the role\n\nIf you can point me to the specific section, I can help extract it."
        
        elif is_skills_question:
            # Extract skills from context documents
            skills = self._extract_skills_from_context(context_docs)
            
            if skills:
                skills_list = "\n".join([f"• {skill}" for skill in skills[:15]])  # Top 15 skills
                return f"**Required Skills (from Job Description):**\n\n{skills_list}\n\nThese are the key skills mentioned in the uploaded job description."
        
        # For other questions, return the LLM answer as-is
        return raw_answer
    
    def _extract_job_titles_from_context(self, context_docs: List[Document]) -> List[str]:
        """
        Extract job titles directly from context documents using comprehensive pattern matching
        
        Args:
            context_docs: List of document chunks
            
        Returns:
            List of extracted job titles
        """
        import re
        
        job_titles = []
        all_content = ""
        
        logger.info("\n🔍 Starting Job Title Extraction...")
        
        # Collect all JOB_DESCRIPTION content
        for doc in context_docs:
            if doc.metadata.get('doc_type') == 'JOB_DESCRIPTION':
                all_content += "\n" + doc.page_content
        
        if not all_content:
            logger.warning("⚠️ No JOB_DESCRIPTION documents found in context")
            return []
        
        logger.info(f"📄 Analyzing {len(all_content)} characters from JOB_DESCRIPTION")
        
        # ENHANCED PATTERNS - More comprehensive and flexible
        patterns = [
            # Explicit formats
            (r'Job Title[:\s]*[:\-]?\s*([^\n]{3,60})', 'Explicit Job Title'),
            (r'Position[:\s]*[:\-]?\s*([^\n]{3,60})', 'Position Field'),
            (r'Role[:\s]*[:\-]?\s*([^\n]{3,60})', 'Role Field'),
            
            # Hiring statements
            (r'(?:We are|We\'re)\s+(?:looking for|hiring|seeking)\s+(?:a|an)\s+([A-Z][^\n]{5,50})', 'Looking For'),
            (r'(?:Hiring|Seeking|Recruiting)\s+(?:for\s+)?(?:a|an)?\s*([A-Z][A-Za-z\s\-]+(?:Developer|Engineer|Architect))', 'Hiring Statement'),
            
            # Technology-specific (MOST IMPORTANT for Java Developer)
            (r'\b(Java\s+Developer)\b', 'Java Developer'),
            (r'\b(Java\s+Software\s+Engineer)\b', 'Java Engineer'),
            (r'\b(Senior\s+Java\s+Developer)\b', 'Senior Java Dev'),
            (r'\b(Junior\s+Java\s+Developer)\b', 'Junior Java Dev'),
            (r'\b(Full\s*Stack\s+Java\s+Developer)\b', 'Full Stack Java'),
            (r'\b(Backend\s+Java\s+Developer)\b', 'Backend Java'),
            
            # Other tech stacks
            (r'\b(Python\s+Developer)\b', 'Python Developer'),
            (r'\b(Full\s*Stack\s+Developer)\b', 'Full Stack'),
            (r'\b(Backend\s+Developer)\b', 'Backend Dev'),
            (r'\b(Frontend\s+Developer)\b', 'Frontend Dev'),
            (r'\b(Software\s+Engineer)\b', 'Software Engineer'),
            (r'\b(Software\s+Developer)\b', 'Software Developer'),
            (r'\b(Web\s+Developer)\b', 'Web Developer'),
            (r'\b(Data\s+Engineer)\b', 'Data Engineer'),
            (r'\b(DevOps\s+Engineer)\b', 'DevOps Engineer'),
            (r'\b(Cloud\s+Engineer)\b', 'Cloud Engineer'),
            
            # With seniority
            (r'\b((?:Senior|Junior|Lead|Principal|Staff|Mid-level|Entry-level)\s+[A-Z][A-Za-z\s]*(?:Developer|Engineer|Architect))\b', 'With Seniority'),
            
            # Generic (lower priority)
            (r'\b([A-Z][A-Za-z\s]{3,40}(?:Developer|Engineer|Architect|Manager|Analyst|Specialist))\b', 'Generic'),
        ]
        
        for pattern, pattern_name in patterns:
            matches = re.findall(pattern, all_content, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                logger.info(f"  ✓ Pattern '{pattern_name}' found {len(matches)} matches")
            
            for match in matches:
                # Clean up the title
                title = match.strip()
                
                # Remove common prefixes/suffixes
                title = re.sub(r'^(a|an|the)\s+', '', title, flags=re.IGNORECASE)
                title = re.sub(r'\s*[:\-\.]\s*$', '', title)
                title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
                
                # Filter out invalid matches
                if len(title) < 5 or len(title) > 60:
                    continue
                
                # Must contain key job terms
                job_terms = ['developer', 'engineer', 'architect', 'manager', 'analyst', 'specialist', 'scientist', 'designer', 'consultant', 'administrator']
                if not any(term in title.lower() for term in job_terms):
                    continue
                
                # Capitalize properly
                words = title.split()
                capitalized = []
                for word in words:
                    if word.lower() in ['and', 'or', 'of', 'the', 'a', 'an']:
                        capitalized.append(word.lower())
                    elif word.upper() in ['AI', 'ML', 'API', 'UI', 'UX', 'AWS', 'GCP']:
                        capitalized.append(word.upper())
                    else:
                        capitalized.append(word.capitalize())
                
                title = ' '.join(capitalized)
                
                # Add if unique
                if title not in job_titles and title.lower() not in [t.lower() for t in job_titles]:
                    job_titles.append(title)
                    logger.info(f"    → Extracted: '{title}' (from pattern: {pattern_name})")
        
        if job_titles:
            logger.info(f"✅ Total job titles extracted: {len(job_titles)}")
            for i, title in enumerate(job_titles[:5], 1):
                logger.info(f"   {i}. {title}")
        else:
            logger.warning("⚠️ No job titles extracted from patterns")
            # Debug: Show first 500 chars of content
            logger.info(f"📝 Content preview:\n{all_content[:500]}")
        
        return job_titles[:3]  # Return top 3 most relevant titles
    
    def _extract_title_from_llm_answer(self, llm_answer: str) -> str:
        """
        Extract job title from LLM's verbose answer as fallback
        
        Args:
            llm_answer: Raw LLM response
            
        Returns:
            Extracted job title or empty string
        """
        import re
        
        # Look for common patterns in LLM responses
        patterns = [
            r'(?:role as|position as|job as)\s+([A-Z][A-Za-z\s]+(?:Developer|Engineer|Architect))',
            r'(?:applying for|interested in)\s+(?:the|a|an)?\s*([A-Z][A-Za-z\s]+(?:Developer|Engineer|Architect))',
            r'\b(Java\s+Developer)\b',
            r'\b(Software\s+Engineer)\b',
            r'\b(Full\s*Stack\s+Developer)\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, llm_answer, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                # Capitalize properly
                words = title.split()
                capitalized = ' '.join([word.capitalize() for word in words])
                logger.info(f"  → Extracted from LLM answer: '{capitalized}'")
                return capitalized
        
        return ""
    
    def _extract_skills_from_context(self, context_docs: List[Document]) -> List[str]:
        """
        Extract skills directly from job description context
        
        Args:
            context_docs: List of document chunks
            
        Returns:
            List of extracted skills
        """
        import re
        
        skills = []
        
        # Common technology and skill patterns
        skill_patterns = [
            # Programming Languages
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|PHP|Swift|Kotlin|Scala|R|MATLAB)\b',
            # Web Frameworks
            r'\b(Django|Flask|FastAPI|Spring|Spring Boot|React|Angular|Vue\.js|Node\.js|Express|Laravel|ASP\.NET|Rails)\b',
            # Databases
            r'\b(MySQL|PostgreSQL|MongoDB|Redis|Cassandra|Oracle|SQL Server|SQLite|DynamoDB|Elasticsearch)\b',
            # Cloud & DevOps
            r'\b(AWS|Azure|GCP|Google Cloud|Docker|Kubernetes|Jenkins|GitLab CI|CI/CD|Terraform|Ansible|CircleCI)\b',
            # Data & ML
            r'\b(Machine Learning|Deep Learning|TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy|Spark|Hadoop)\b',
            # Methodologies
            r'\b(Agile|Scrum|Kanban|DevOps|Microservices|RESTful API|GraphQL|OOP|Test-Driven Development|TDD)\b',
            # Tools
            r'\b(Git|GitHub|GitLab|Jira|Confluence|VS Code|IntelliJ|Eclipse|Postman|Swagger)\b',
            # Soft Skills
            r'\b(Communication|Leadership|Team Collaboration|Problem Solving|Critical Thinking|Time Management)\b'
        ]
        
        for doc in context_docs:
            # Only extract from JOB_DESCRIPTION documents
            if doc.metadata.get('doc_type') != 'JOB_DESCRIPTION':
                continue
            
            content = doc.page_content
            
            # Extract using patterns
            for pattern in skill_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Clean and capitalize properly
                    skill = match.strip()
                    
                    # Handle special cases
                    if skill.lower() in ['c++', 'c#']:
                        skill = skill.upper()
                    elif skill.lower() == 'node.js':
                        skill = 'Node.js'
                    elif skill.lower() == 'vue.js':
                        skill = 'Vue.js'
                    elif '.' in skill:
                        skill = skill.title()
                    else:
                        skill = skill.capitalize()
                    
                    if skill not in skills:
                        skills.append(skill)
            
            # Also look for explicit "Required Skills:" or "Technical Skills:" sections
            skills_section_match = re.search(
                r'(?:Required Skills|Technical Skills|Key Skills|Skills Required)[:\s]+([^\n]+(?:\n[^\n]+)*)',
                content,
                re.IGNORECASE
            )
            
            if skills_section_match:
                section_text = skills_section_match.group(1)
                # Extract comma or bullet separated items
                items = re.split(r'[,•\n]', section_text)
                for item in items:
                    item = item.strip()
                    if len(item) > 2 and len(item) < 50 and item not in skills:
                        skills.append(item)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique_skills.append(skill)
        
        return unique_skills

    def _extract_skills_from_resume(self, context_docs: List[Document]) -> List[str]:
        """
        Extract skills from RESUME documents (uses similar patterns but tuned for resume phrasing)

        Args:
            context_docs: List of Document chunks

        Returns:
            List of extracted skills from resume
        """
        import re

        skills = []

        # Reuse many of the same patterns but allow shorter context and additional resume cues
        skill_patterns = [
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|PHP|Swift|Kotlin|Scala|R|MATLAB)\b',
            r'\b(Django|Flask|FastAPI|Spring|Spring Boot|React|Angular|Vue\.js|Node\.js|Express|Laravel|ASP\.NET|Rails)\b',
            r'\b(MySQL|PostgreSQL|MongoDB|Redis|Cassandra|Oracle|SQL Server|SQLite|DynamoDB|Elasticsearch)\b',
            r'\b(AWS|Azure|GCP|Google Cloud|Docker|Kubernetes|Jenkins|GitLab CI|CI/CD|Terraform|Ansible|CircleCI)\b',
            r'\b(Machine Learning|Deep Learning|TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy|Spark|Hadoop)\b',
            r'\b(Agile|Scrum|Kanban|DevOps|Microservices|RESTful API|GraphQL|OOP|Test-Driven Development|TDD)\b',
            r'\b(Git|GitHub|GitLab|Jira|Confluence|VS Code|IntelliJ|Eclipse|Postman|Swagger)\b',
            r'\b(Communication|Leadership|Team Collaboration|Problem Solving|Critical Thinking|Time Management)\b'
        ]

        for doc in context_docs:
            # Only extract from RESUME documents
            if doc.metadata.get('doc_type') != 'RESUME':
                continue

            content = doc.page_content

            for pattern in skill_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    skill = match.strip()
                    if skill.lower() in ['c++', 'c#']:
                        skill = skill.upper()
                    elif skill.lower() == 'node.js':
                        skill = 'Node.js'
                    elif skill.lower() == 'vue.js':
                        skill = 'Vue.js'
                    elif '.' in skill:
                        skill = skill.title()
                    else:
                        skill = skill.capitalize()

                    if skill not in skills:
                        skills.append(skill)

            # Also check resume-specific lines like 'Skills:' or 'Technical Skills'
            skills_section_match = re.search(r'(?:Skills|Technical Skills|Skills:|Key Skills)[:\s]+([^\n]+(?:\n[^\n]+)*)', content, re.IGNORECASE)
            if skills_section_match:
                section_text = skills_section_match.group(1)
                items = re.split(r'[,•\n]', section_text)
                for item in items:
                    item = item.strip()
                    if 2 < len(item) < 50 and item not in skills:
                        skills.append(item)

        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique_skills.append(skill)

        return unique_skills
    
    def answer_query(
        self, 
        question: str, 
        show_context: bool = True
    ) -> Dict[str, any]:
        """
        Answer a user query using RAG with intelligent document filtering
        
        Args:
            question: User's question
            show_context: If True, return retrieved context chunks
            
        Returns:
            Dictionary with 'answer', 'source_documents', and 'context' keys
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call setup_qa_chain() first.")
        
        logger.info(f"\n❓ Query: {question}")
        logger.info("🔍 Retrieving relevant context...")
        
        # Detect query intent to prioritize correct document type
        question_lower = question.lower()
        
        # Enhanced keyword detection for job-related queries
        job_keywords = [
            'job title', 'position', 'role', 'job description', 'required', 'qualification', 
            'responsibility', 'candidate', 'requirements', 'looking for',
            'job requires', 'position requires', 'what should i mention',
            'what title', 'which title', 'job posting', 'hiring for',
            'skills mentioned in', 'mentioned in the job', 'from the job description',
            'in the jd', 'skills in job', 'jd skills', 'job skills'
        ]
        
        # Keywords for resume-related queries  
        resume_keywords = [
            'my skills', 'my experience', 'my resume', 'my background', 
            'my projects', 'i worked', 'i developed', 'my education',
            'my qualifications', 'my achievements', 'i have experience in'
        ]
        
        # IMPORTANT: Check job keywords first (more specific)
        query_about_job = any(kw in question_lower for kw in job_keywords)
        query_about_resume = any(kw in question_lower for kw in resume_keywords) and not query_about_job
        
        try:
            # Retrieve documents using the base retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}  # Get more candidates for filtering
            )
            
            # Get candidate documents
            candidate_docs = retriever.get_relevant_documents(question)
            
            # Log retrieved documents for debugging
            logger.info(f"📥 Retrieved {len(candidate_docs)} candidate documents:")
            for i, doc in enumerate(candidate_docs, 1):
                doc_type = doc.metadata.get('doc_type', 'UNKNOWN')
                source = doc.metadata.get('source', 'Unknown')
                logger.info(f"   {i}. {source} [{doc_type}]")
            
            # Smart filtering based on query intent
            filtered_docs = candidate_docs
            
            if query_about_job and not query_about_resume:
                # CRITICAL: Prioritize JOB_DESCRIPTION documents
                job_docs = [doc for doc in candidate_docs if doc.metadata.get('doc_type') == 'JOB_DESCRIPTION']
                if job_docs:
                    filtered_docs = job_docs[:3]  # Top 3 job description chunks
                    logger.info(f"🎯 FILTERED to {len(filtered_docs)} JOB_DESCRIPTION chunks (query about job requirements)")
                else:
                    logger.warning("⚠️ No JOB_DESCRIPTION documents found, using all retrieved docs")
                    filtered_docs = candidate_docs[:3]
                    
            elif query_about_resume and not query_about_job:
                # Prioritize RESUME documents
                resume_docs = [doc for doc in candidate_docs if doc.metadata.get('doc_type') == 'RESUME']
                if resume_docs:
                    filtered_docs = resume_docs[:3]  # Top 3 resume chunks
                    logger.info(f"🎯 FILTERED to {len(filtered_docs)} RESUME chunks (query about user's background)")
                else:
                    logger.warning("⚠️ No RESUME documents found, using all retrieved docs")
                    filtered_docs = candidate_docs[:3]
            else:
                # Use both types but limit total
                filtered_docs = candidate_docs[:3]
                logger.info(f"📊 Using mixed document types ({len(filtered_docs)} chunks)")
            
            # Build context string from filtered documents
            context_str = "\n\n".join([
                f"[From {doc.metadata.get('source', 'Unknown')} - {doc.metadata.get('doc_type', 'UNKNOWN')}]\n{doc.page_content}"
                for doc in filtered_docs
            ])
            
            # Generate answer using filtered context
            from langchain.prompts import PromptTemplate
            
            prompt_template = """You are an expert career advisor analyzing a user's RESUME and a JOB DESCRIPTION they uploaded.

CRITICAL INSTRUCTIONS:
1. Answer MUST be based ONLY on the provided context from uploaded documents
2. For job title questions: Extract the EXACT job title/position from the JOB DESCRIPTION
3. For required skills: List ONLY the skills mentioned in the JOB DESCRIPTION
4. For user background: Extract from RESUME only
5. Be DIRECT and CONCISE - answer the specific question asked
6. DO NOT add generic career advice or explanations unless asked
7. If asked "what job title" or "which role" - respond with ONLY the job title from the document

Context from uploaded documents:
{context}

User Question: {question}

Direct Answer (extract exact information from documents):"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            formatted_prompt = prompt.format(context=context_str, question=question)
            
            # Get answer from LLM
            raw_answer = self.llm.invoke(formatted_prompt)
            
            # Post-process answer for job title queries to extract concise answer
            answer = self._post_process_answer(raw_answer, question, filtered_docs)
            
            # Extract context from filtered documents for display
            context_chunks = []
            for i, doc in enumerate(filtered_docs, 1):
                context_chunks.append({
                    'chunk_number': i,
                    'content': doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', 'N/A'),
                    'doc_type': doc.metadata.get('doc_type', 'UNKNOWN')
                })
            
            if show_context:
                logger.info("\n📚 Retrieved Context:")
                for chunk in context_chunks:
                    logger.info(f"\n  [{chunk['chunk_number']}] From: {chunk['source']} (Page {chunk['page']}) [Type: {chunk['doc_type']}]")
                    logger.info(f"  {chunk['content'][:200]}...")
            
            logger.info(f"\n💬 Final Answer:\n{answer}\n")
            
            # --- Similarity metrics between Job Description and Resume ---
            try:
                import difflib

                job_skills = self._extract_skills_from_context(filtered_docs)
                resume_skills = self._extract_skills_from_resume(filtered_docs)

                # Helper: best fuzzy match of a skill against job skills
                def _best_match(skill, candidates):
                    best = ("", 0.0)
                    for c in candidates:
                        score = difflib.SequenceMatcher(None, skill.lower(), c.lower()).ratio()
                        if score > best[1]:
                            best = (c, score)
                    return best

                per_skill_scores = []
                matched_job_skills = set()

                for r_skill in resume_skills:
                    best_job, score = _best_match(r_skill, job_skills) if job_skills else ("", 0.0)
                    per_skill_scores.append({
                        'resume_skill': r_skill,
                        'best_job_match': best_job,
                        'score': round(score, 3)
                    })
                    if score >= 0.75 and best_job:
                        matched_job_skills.add(best_job)

                # Skills from job description that are not matched
                missing_job_skills = [s for s in job_skills if s not in matched_job_skills]

                # Matched skills (as seen in resume and matched to job)
                matched_skills = [s for s in resume_skills if any(s.lower() == mj.lower() or any(ps['resume_skill']==s and ps['score']>=0.75 for ps in per_skill_scores) for mj in matched_job_skills)]

                # Overall score: Jaccard-like based on matched job skills vs union
                union_count = len(set([s.lower() for s in job_skills]) | set([s.lower() for s in resume_skills]))
                overall_score = 0.0
                if union_count > 0:
                    overall_score = (len(matched_job_skills) / union_count) * 100

                similarity_metrics = {
                    'overall_score_pct': round(overall_score, 1),
                    'num_job_skills': len(job_skills),
                    'num_resume_skills': len(resume_skills),
                    'num_matched_skills': len(matched_job_skills),
                    'matched_job_skills': list(matched_job_skills),
                    'matched_resume_skills': matched_skills,
                    'missing_job_skills': missing_job_skills,
                    'per_skill_scores': per_skill_scores
                }

            except Exception as e:
                logger.warning(f"⚠️ Could not compute similarity metrics: {e}")
                similarity_metrics = {}

            return {
                'answer': answer,
                'source_documents': filtered_docs,
                'context_chunks': context_chunks,
                'similarity_metrics': similarity_metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            raise


def main():
    """
    Demo script showing RAG Coach in action
    """
    print("="*70)
    print("🎓 RAG Coach - AI Career Coaching Assistant")
    print("="*70)
    
    # Initialize RAG system
    rag_coach = RAGCoachSystem()
    
    # Example PDF files (adjust paths as needed)
    # These are placeholder paths - replace with actual files
    career_guide_pdfs = [
        "./career_guides/data_analyst.pdf",
        "./career_guides/resume_tips.pdf",
        "./career_guides/interview_tips.pdf",
    ]
    
    # User uploads (example)
    user_uploads = [
        "./uploads/resume.pdf",
        "./uploads/job_description.pdf"
    ]
    
    # Combine all PDFs
    all_pdfs = career_guide_pdfs + user_uploads
    
    # Filter to only existing files for demo
    existing_pdfs = [pdf for pdf in all_pdfs if os.path.exists(pdf)]
    
    if not existing_pdfs:
        print("\n⚠️ No PDF files found. Using demo mode without actual files.")
        print("\n📝 To use RAG Coach:")
        print("1. Create a 'career_guides' folder")
        print("2. Add PDF files: data_analyst.pdf, resume_tips.pdf, interview_tips.pdf")
        print("3. Create an 'uploads' folder for user files")
        return
    
    try:
        # Step 1: Build vector store
        print("\n🔨 Step 1: Building vector store...")
        rag_coach.build_vector_store(existing_pdfs)
        
        # Step 2: Setup QA chain
        print("\n⚙️ Step 2: Setting up QA chain...")
        rag_coach.setup_qa_chain()
        
        # Step 3: Answer sample queries
        print("\n💡 Step 3: Answering career questions...")
        
        sample_questions = [
            "How can I tailor my resume to match a job description?",
            "What should I focus on during a technical interview?",
            "How can I highlight my data analysis skills effectively?"
        ]
        
        for question in sample_questions[:1]:  # Answer first question for demo
            result = rag_coach.answer_query(question, show_context=True)
            print("\n" + "-"*70)
        
        print("\n✅ RAG Coach demo completed successfully!")
        
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\n🔧 Setup Instructions:")
        print("1. Install Ollama: https://ollama.ai/download")
        print("2. Start Ollama (it runs as a service)")
        print("3. Pull Gemma model: ollama pull gemma:2b")
        print("4. Verify installation: ollama list")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
