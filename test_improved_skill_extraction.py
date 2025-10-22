"""
Test script to verify IMPROVED JD-only skill extraction with robust normalization
"""
import re

def _normalize_skill(skill_text):
    """Normalize a single skill to its canonical form"""
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


# Test with the user's actual data
resume_text = """Programming Languages: Python, Java, C++
• Frontend Development: JavaScript, React.js, HTML, CSS, Bootstrap, Tailwind CSS
• Backend Development: Django, Flask, Node.js, Express.js, FastAPI, RESTful API
• Artificial Intelligence and Machine Learning: PyTorch, TensorFlow, Scikit-learn, Pandas, NumPy, Machine Learning
Algorithms, LangChain, FAISS, spaCy, Sentence-BERT, Hugging Face, RegEx
• Database Management: MySQL, PostgreSQL, MongoDB, SQLite3
• Cloud & DevOps: Amazon Web Services (AWS), Linux, Docker, Git, GitHub"""

job_text = """Job Title: Python Developer
Job Summary:
We are looking for a skilled and motivated Python Developer to design, develop, and
maintain efficient, reusable, and reliable Python-based applications. The ideal candidate
should have a good understanding of Python programming, web frameworks (like Django
or Flask), database management, and API integration.
Key Responsibilities:
• Develop, test, and deploy Python applications and scripts.
• Build and maintain web applications using frameworks such as Django or Flask.
• Design and integrate RESTful APIs for frontend-backend communication.
• Write clean, scalable, and efficient code following best practices.
• Work with databases such as MySQL, PostgreSQL, or MongoDB.
• Debug and fix issues, ensuring high performance and responsiveness.
• Collaborate with frontend developers, designers, and other team members.
• Participate in code reviews and contribute to team knowledge sharing.
Required Skills and Qualifications:
• Strong knowledge of Python programming language.
• Experience with Django or Flask frameworks.
• Familiarity with HTML, CSS, JavaScript (for basic frontend understanding).
• Knowledge of databases (MySQL, SQLite, or MongoDB).
• Understanding of Object-Oriented Programming (OOP) concepts.
• Experience with version control systems like Git and GitHub.
• Problem-solving attitude and ability to learn new technologies quickly.
Preferred (Optional) Skills:
• Knowledge of REST API development.
• Experience with machine learning or data analysis (NumPy, Pandas).
• Familiarity with cloud platforms (AWS, Azure) or deployment tools.
• Understanding of Docker and CI/CD pipelines."""

print("=" * 80)
print("IMPROVED SKILL EXTRACTION TEST (Robust Normalization)")
print("=" * 80)

# Extract skills
resume_skills = _extract_skill_tokens(resume_text)
job_skills = _extract_skill_tokens(job_text)

print("\n📄 RESUME SKILLS FOUND (Normalized):")
for skill in sorted(resume_skills):
    print(f"  - {skill}")

print(f"\nTotal: {len(resume_skills)} skills")

print("\n📋 JOB DESCRIPTION SKILLS FOUND (Normalized):")
for skill in sorted(job_skills):
    print(f"  - {skill}")

print(f"\nTotal: {len(job_skills)} skills")

# Compute difference
jd_only_skills = sorted(list(job_skills.difference(resume_skills)))

print("\n✅ SKILLS TO ADD (in JD but NOT in Resume):")
if jd_only_skills:
    # Create display-friendly names
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
    for skill in jd_only_skills:
        display_name = display_name_map.get(skill, skill.title())
        jd_only_skills_display.append(display_name)
    
    for skill in jd_only_skills_display:
        print(f"• {skill}")
    
    print(f"\nTotal missing: {len(jd_only_skills_display)} skills")
    
    print("\n🔑 ATS-FRIENDLY KEYWORDS:")
    print(", ".join(jd_only_skills_display))
else:
    print("✅ Great! Your resume already covers all key skills from the job description!")

print("\n" + "=" * 80)
print("TEST COMPLETE - Now restart backend to apply changes!")
print("=" * 80)
