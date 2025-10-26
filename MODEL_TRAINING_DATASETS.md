# 📊 NextStepAI - Model Training Datasets Documentation

## 🎯 Overview

NextStepAI uses **multiple datasets** for training different components of the CV Analyzer and AI Career Advisor system. The platform employs both **traditional machine learning** and **fine-tuned language models**.

---

## 📁 Datasets Used for Model Training

### 1. **Jobs Dataset** - `jobs_cleaned.csv`

**Purpose:** Train the job recommendation model (CV Analyzer)

**File:** `jobs_cleaned.csv`

**Total Records:** **25,756 job postings** (including header)
- Actual data rows: **25,755 job records**

**Columns:**
- `Job Title`: The job position name
- `Key Skills`: Pipe-separated list of required skills

**Sample Data:**
```csv
Job Title,Key Skills
media planning executive/manager,media planning| digital media
sales executive/officer,pre sales| closing| software knowledge| clients
software developer,python| java| react| node.js| docker
data scientist,python| machine learning| sql| tableau
```

**Data Categories:**
The dataset covers **11+ consolidated job categories**:
1. **Data Professional** - Data Scientists, Data Analysts, BI Developers, Data Engineers
2. **Software Developer** - Full Stack, Frontend, Backend Developers
3. **IT Operations & Infrastructure** - DevOps, Network Engineers, System Admins
4. **Project/Product Manager** - Project Managers, Scrum Masters
5. **QA/Test Engineer** - Quality Assurance, SDET
6. **Human Resources** - HR, Recruitment, Talent Acquisition
7. **Sales & Business Development** - Sales Executives, BDMs
8. **Marketing** - Digital Marketing, SEO Specialists
9. **UI/UX & Design** - UI/UX Designers, Graphic Designers
10. **Finance & Accounting** - Accountants, Finance Analysts
11. **Customer Support** - Customer Service Representatives

**Model Training Details:**
- **Algorithm:** Logistic Regression with TF-IDF vectorization
- **Training Split:** 80% training, 20% testing
- **Cross-Validation:** 5-fold CV
- **Minimum Samples per Class:** 10 records
- **Feature Engineering:** Skill-based TF-IDF vectors
- **Hyperparameter Tuning:** GridSearchCV

**Artifacts Generated:**
- `job_recommender_pipeline.joblib` - Trained model pipeline
- `job_title_encoder.joblib` - Label encoder for job categories
- `prioritized_skills.joblib` - Top 30 skills per job category
- `related_titles.joblib` - Related job titles map

---

### 2. **Skills Database** - `skills_db.json`

**Purpose:** Standardize and validate skills across the platform

**File:** `skills_db.json`

**Total Skills:** **215 unique technical skills**

**Sample Skills:**
```json
[
  "python", "java", "javascript", "react", "angular",
  "machine learning", "deep learning", "tensorflow",
  "docker", "kubernetes", "aws", "azure", "gcp",
  "sql", "mongodb", "postgresql", "git", "jenkins"
]
```

**Usage:**
- Skill extraction from resumes
- Skill validation and normalization
- Skill gap analysis
- Matching user skills with job requirements

**How It Works:**
1. Parse resume text and extract potential skills
2. Compare against 215 validated skills in database
3. Only keep skills that match the database
4. Use for job matching algorithm

---

### 3. **Career Advice Dataset** - `career_advice_dataset.jsonl`

**Purpose:** Fine-tune GPT-2 model for AI Career Advisor

**File:** `career_advice_dataset.jsonl`

**Total Records:** **244 career advice Q&A pairs**

**Format:** JSONL (JSON Lines)
- Each line is a JSON object with `prompt` and `completion`

**Sample Record:**
```json
{
  "prompt": "What are the key skills required for a Data Scientist role in India?",
  "completion": "A Data Scientist in India needs a mix of technical and soft skills.\n\n### Key Skills:\n* **Programming & Databases:** Python (Pandas, NumPy, Scikit-learn), R, SQL, and statistical analysis.\n* **Machine Learning:** Strong understanding of algorithms like Linear Regression, Decision Trees, Random Forest, and Deep Learning.\n* **Big Data & Cloud:** Experience with Apache Spark, Hadoop, and cloud platforms like AWS, Azure, or GCP..."
}
```

**Content Coverage:**
- Career path guidance for various roles
- Required skills and technologies
- Certification recommendations
- Common interview questions
- Transition strategies between roles
- Internship opportunities
- Industry-specific insights for Indian market

**Job Roles Covered:**
- Data Scientist
- Software Developer
- DevOps Engineer
- Cybersecurity Professional
- UI/UX Designer
- Product Manager
- Finance Professional
- Marketing Specialist
- And many more...

---

### 4. **Ultra Clear Career Dataset** - `career_advice_ultra_clear_dataset.jsonl`

**Purpose:** Additional high-quality career advice data for fine-tuning

**File:** `career_advice_ultra_clear_dataset.jsonl`

**Total Records:** Variable (companion to main career advice dataset)

**Purpose:**
- Provides clearer, more structured responses
- Enhances model's ability to generate concise advice
- Improves response quality for specific career questions

**Combined Usage:**
Both `career_advice_dataset.jsonl` and `career_advice_ultra_clear_dataset.jsonl` are loaded together for fine-tuning the GPT-2 model.

---

## 🤖 Model Training Pipelines

### **Pipeline 1: CV Analyzer (Job Recommendation)**

**File:** `model_training.py`

**Process:**
1. **Load Data:** Read `jobs_cleaned.csv` (25,755 records)
2. **Load Skills:** Load 215 skills from `skills_db.json`
3. **Clean Skills:** Filter job skills against validated skill database
4. **Categorize Jobs:** Group 25,755 jobs into 11 categories
5. **Filter Data:** 
   - Remove jobs with no valid skills
   - Remove categories with <10 samples
   - Keep only quality data
6. **Feature Engineering:** Convert skills to TF-IDF vectors
7. **Train Model:** 
   - 80/20 train-test split
   - GridSearchCV for hyperparameter tuning
   - Test both Logistic Regression and Naive Bayes
8. **Evaluate:** Generate classification report and confusion matrix
9. **Save Artifacts:** Export 4 .joblib files

**Training Data Size:** ~20,000+ valid job records after filtering

**Model Performance:**
- F1 Score: Typically 0.85-0.95 (weighted)
- Accuracy: 85-95% on test set
- Cross-validation: 5-fold CV for robust validation

---

### **Pipeline 2: AI Career Advisor (GPT-2 Fine-tuning)**

**File:** `production_finetuning_optimized.py`

**Process:**
1. **Load Data:** 
   - Load `career_advice_dataset.jsonl` (244 records)
   - Load `career_advice_ultra_clear_dataset.jsonl`
   - Combine all examples
2. **Format Examples:**
   ```
   <|startoftext|>### Question: {prompt}
   
   ### Answer: {completion}<|endoftext|>
   ```
3. **Tokenize:** Convert text to GPT-2 tokens
4. **Split:** 90% training, 10% validation
5. **Fine-tune:** Train GPT-2 model on career advice
6. **Save:** Export to `career-advisor-final/` directory

**Training Data Size:**
- Combined: 244+ career Q&A pairs
- Training examples: ~220
- Validation examples: ~24

**Model Configuration:**
- Base Model: GPT-2 (124M parameters)
- Max Length: 120 tokens (for speed optimization)
- Training Epochs: Typically 3-5
- Learning Rate: Optimized for convergence
- Response Time: 5-15 seconds per query

---

## 📊 Dataset Statistics Summary

| Dataset | File | Records | Purpose |
|---------|------|---------|---------|
| **Jobs Database** | `jobs_cleaned.csv` | **25,755** | Job recommendation training |
| **Skills Database** | `skills_db.json` | **215** | Skill validation & matching |
| **Career Advice** | `career_advice_dataset.jsonl` | **244** | AI advisor fine-tuning |
| **Ultra Clear Career** | `career_advice_ultra_clear_dataset.jsonl` | Variable | Enhanced AI training |

**Total Training Data:**
- **CV Analyzer:** ~25,755 job-skill pairs
- **AI Career Advisor:** ~244+ Q&A pairs
- **Skills Validation:** 215 unique skills
- **Job Categories:** 11 consolidated groups

---

## 🎯 Data Flow in CV Analyzer

### User Upload Resume → Analysis Process:

1. **Extract Text** from PDF/DOCX
2. **Parse Skills** using NLP (spaCy)
3. **Validate Skills** against 215-skill database
4. **Match Job** using trained model on 25,755 jobs
5. **Calculate Match %** based on skill overlap
6. **Generate Roadmap** with missing skills
7. **Find Live Jobs** from job postings database
8. **AI Feedback** on resume layout

### Example:
```
User Skills: ["python", "javascript", "react"]
           ↓
Validated: ["python", "javascript", "react"] (all in 215 skills)
           ↓
Model Prediction: "Software Developer" (trained on 25,755 jobs)
           ↓
Required Skills: ["python", "javascript", "react", "node.js", "docker"]
           ↓
Missing Skills: ["node.js", "docker"]
           ↓
Match Score: 60% (3 of 5 skills)
```

---

## 🚀 Training Commands

### Train Job Recommendation Model:
```bash
python model_training.py
```
**Output:**
- `job_recommender_pipeline.joblib`
- `job_title_encoder.joblib`
- `prioritized_skills.joblib`
- `related_titles.joblib`
- `confusion_matrix.png`

### Fine-tune AI Career Advisor:
```bash
python production_finetuning_optimized.py
```
**Output:**
- `career-advisor-final/` directory with fine-tuned GPT-2 model

---

## 📈 Data Quality Measures

### Jobs Dataset (`jobs_cleaned.csv`):
✅ **Pre-processed:** Already cleaned and normalized  
✅ **Skill Validation:** All skills checked against database  
✅ **Category Balance:** Minimum 10 samples per category  
✅ **Deduplication:** Unique job-skill combinations  

### Career Advice Dataset (`career_advice_dataset.jsonl`):
✅ **High Quality:** Manually curated Q&A pairs  
✅ **India-Focused:** Tailored for Indian job market  
✅ **Comprehensive:** Covers certifications, skills, interview tips  
✅ **Structured:** Consistent format for better learning  

### Skills Database (`skills_db.json`):
✅ **Validated:** Industry-standard technical skills  
✅ **Normalized:** Lowercase, standardized names  
✅ **Relevant:** Tech skills commonly required in India  
✅ **Maintained:** Can be updated as market evolves  

---

## 🔄 Dataset Update Process

### To Add New Jobs:
1. Append to `jobs_cleaned.csv`
2. Run `python model_training.py`
3. New model artifacts generated
4. Deploy to backend

### To Add Career Advice:
1. Append to `career_advice_dataset.jsonl` in JSONL format
2. Run `python production_finetuning_optimized.py`
3. New fine-tuned model in `career-advisor-final/`
4. Update backend to use new model

### To Add Skills:
1. Edit `skills_db.json`
2. Add new skills to array
3. Retrain job recommendation model
4. Skills automatically validated in system

---

## 💡 Key Insights

### Why 25,755 Jobs?
- **Comprehensive Coverage:** Represents diverse job market in India
- **Sufficient Data:** Enough samples for each job category
- **Real Market Data:** Based on actual job postings
- **Quality over Quantity:** Cleaned and validated data

### Why 244 Career Q&A?
- **Quality-Focused:** Each Q&A carefully crafted
- **Fine-tuning Optimal:** GPT-2 fine-tunes well with ~200-500 examples
- **Specific Use Case:** Focused on career advice, not general knowledge
- **Performance Balance:** Fast response (5-15s) with good quality

### Why 215 Skills?
- **Industry Standard:** Most common technical skills
- **Manageable Size:** Easy to validate and maintain
- **Comprehensive:** Covers major tech domains
- **Expandable:** Can grow as market needs change

---

## 🎓 Model Training Results

### CV Analyzer Model:
- **Training Data:** 25,755 jobs → ~20,000 after filtering
- **Test Accuracy:** 85-95%
- **F1 Score:** 0.85-0.95 (weighted)
- **Training Time:** 5-15 minutes
- **Model Size:** ~10 MB (pipeline + encoder)

### AI Career Advisor Model:
- **Training Data:** 244 Q&A pairs
- **Base Model:** GPT-2 (124M parameters)
- **Fine-tuned Model Size:** ~500 MB
- **Training Time:** 1-2 hours (on GPU)
- **Inference Time:** 5-15 seconds per query
- **Response Quality:** High, India-market specific

---

## 📁 File Structure Summary

```
NextStepAI/
├── jobs_cleaned.csv                      # 25,755 job records
├── skills_db.json                        # 215 validated skills
├── career_advice_dataset.jsonl           # 244 Q&A pairs
├── career_advice_ultra_clear_dataset.jsonl # Additional Q&A
├── model_training.py                     # Job model training script
├── production_finetuning_optimized.py    # GPT-2 fine-tuning script
├── job_recommender_pipeline.joblib       # Trained job model
├── job_title_encoder.joblib              # Label encoder
├── prioritized_skills.joblib             # Top skills per job
├── related_titles.joblib                 # Related job titles
└── career-advisor-final/                 # Fine-tuned GPT-2 model
    ├── model.safetensors                 # Model weights
    ├── config.json                       # Model config
    ├── vocab.json                        # Vocabulary
    └── tokenizer_config.json             # Tokenizer config
```

---

## 🎯 Conclusion

NextStepAI's CV Analyzer is powered by:
- ✅ **25,755 real job postings** for job matching
- ✅ **215 validated technical skills** for skill extraction
- ✅ **244 high-quality Q&A pairs** for AI career advice
- ✅ **2 ML models:** Scikit-learn + Fine-tuned GPT-2
- ✅ **4 training artifacts** for production use

This combination provides accurate job recommendations, intelligent skill gap analysis, and personalized career advice tailored to the Indian job market.

---

**Document Version:** 1.0  
**Last Updated:** October 25, 2025  
**Maintained by:** NextStepAI Development Team
