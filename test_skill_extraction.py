"""
Test script to verify JD-only skill extraction works correctly
"""
import re

def _extract_skill_tokens(text):
    """Return a set of normalized skill tokens found in the provided text."""
    if not text:
        return set()

    # Reuse patterns similar to those used for formatting
    skills_patterns = [
        r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|PHP|Swift|Kotlin)\b',
        r'\b(Django|Flask|FastAPI|React(?:\.js)?|Angular|Vue|Node(?:\.js)?|Express(?:\.js)?|Spring|\.NET|Laravel)\b',
        r'\b(NumPy|Pandas|Matplotlib|Scikit-learn|TensorFlow|PyTorch|Keras|Machine\s+Learning|ML)\b',
        r'\b(MySQL|PostgreSQL|MongoDB|Redis|Cassandra|Oracle|SQL\s+Server|DynamoDB|SQLite(?:3)?)\b',
        r'\b(AWS|Amazon\s+Web\s+Services|Azure|GCP|Docker|Kubernetes|Jenkins|GitLab|CI/CD|Terraform|Ansible|Linux)\b',
        r'\b(HTML5?|CSS3?|AJAX|REST(?:ful)?|GraphQL|WebSocket|API|Bootstrap|Tailwind)\b',
        r'\b(Agile|Scrum|Git|GitHub|Jira|Selenium|JUnit|pytest|Travis\s+CI)\b',
        r'\b(OOP|Object-Oriented\s+Programming|Microservices|Database|Testing|FAISS|LangChain|spaCy)\b'
    ]

    tokens = set()
    for p in skills_patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        for m in matches:
            # matches may be tuples if pattern contains groups; join if needed
            if isinstance(m, tuple):
                m = ' '.join([part for part in m if part])
            
            # Normalize the token
            normalized = m.strip().lower()
            # Remove .js suffix for consistency
            normalized = re.sub(r'\.js$', '', normalized)
            # Remove version numbers
            normalized = re.sub(r'[0-9]+$', '', normalized)
            # Normalize spaces
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            # Map synonyms to canonical forms
            synonym_map = {
                'restful api': 'rest',
                'restful': 'rest',
                'sqlite3': 'sqlite',
                'html5': 'html',
                'css3': 'css',
                'amazon web services': 'aws',
                'object-oriented programming': 'oop',
                'machine learning': 'ml',
                'ci/cd': 'cicd',
                'express.js': 'express',
                'node.js': 'node',
                'react.js': 'react'
            }
            
            normalized = synonym_map.get(normalized, normalized)
            
            if normalized:
                tokens.add(normalized)

    # Additional keyword list to catch common terms
    tech_keywords = [
        'python', 'django', 'flask', 'fastapi', 'api', 'rest', 'database', 'mysql', 
        'postgresql', 'mongodb', 'git', 'github', 'gitlab', 'cicd', 'jenkins', 
        'docker', 'agile', 'scrum', 'testing', 'numpy', 'pandas', 'oop', 
        'javascript', 'html', 'css', 'kubernetes', 'terraform', 'aws', 'azure', 'gcp',
        'react', 'angular', 'vue', 'node', 'express', 'sqlite', 'linux', 'pytorch',
        'tensorflow', 'scikit-learn', 'ml', 'langchain', 'faiss', 'spacy', 'bootstrap',
        'tailwind'
    ]

    lower_text = text.lower()
    for kw in tech_keywords:
        if kw in lower_text:
            tokens.add(kw)

    return tokens


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
print("SKILL EXTRACTION TEST")
print("=" * 80)

# Extract skills
resume_skills = _extract_skill_tokens(resume_text)
job_skills = _extract_skill_tokens(job_text)

print("\n📄 RESUME SKILLS FOUND:")
print(sorted(resume_skills))

print("\n📋 JOB DESCRIPTION SKILLS FOUND:")
print(sorted(job_skills))

# Compute difference
jd_only_skills = sorted(list(job_skills.difference(resume_skills)))

print("\n✅ SKILLS TO ADD (in JD but NOT in Resume):")
if jd_only_skills:
    # Create display-friendly names
    display_name_map = {
        'cicd': 'CI/CD',
        'oop': 'Object-Oriented Programming (OOP)',
        'rest': 'RESTful API',
        'ml': 'Machine Learning',
        'aws': 'Amazon Web Services (AWS)',
        'gcp': 'Google Cloud Platform (GCP)',
        'html': 'HTML/HTML5',
        'css': 'CSS/CSS3',
        'sqlite': 'SQLite',
        'postgresql': 'PostgreSQL',
        'mongodb': 'MongoDB',
        'mysql': 'MySQL',
        'javascript': 'JavaScript',
        'typescript': 'TypeScript',
        'node': 'Node.js',
        'react': 'React.js',
        'express': 'Express.js',
        'django': 'Django',
        'flask': 'Flask',
        'fastapi': 'FastAPI',
        'docker': 'Docker',
        'kubernetes': 'Kubernetes',
        'git': 'Git',
        'github': 'GitHub',
        'gitlab': 'GitLab',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'pytorch': 'PyTorch',
        'tensorflow': 'TensorFlow',
        'scikit-learn': 'Scikit-learn',
        'langchain': 'LangChain',
        'faiss': 'FAISS',
        'spacy': 'spaCy',
        'azure': 'Microsoft Azure'
    }
    
    jd_only_skills_display = []
    for skill in jd_only_skills:
        display_name = display_name_map.get(skill, skill.title())
        jd_only_skills_display.append(display_name)
    
    for skill in jd_only_skills_display:
        print(f"• {skill}")
    
    print("\n🔑 ATS-FRIENDLY KEYWORDS:")
    print(", ".join(jd_only_skills_display))
else:
    print("✅ Great! Your resume already covers all key skills from the job description!")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
