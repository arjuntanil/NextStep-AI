# RAG Coach Auto-Analysis Feature

## Overview
The RAG Coach now automatically analyzes uploaded resumes and job descriptions, providing immediate suggestions before allowing follow-up questions.

## How It Works

### 1. Automatic Processing on Upload
When you upload your resume and job description PDFs:
- The backend automatically processes both documents
- Extracts text from PDFs
- Identifies which document is the resume and which is the job description
- Sends both to the LLM (Mistral 7B) for analysis

### 2. Real-Time Status Updates
While processing (typically 30-60 seconds):
- **"🔄 Preparing analysis..."** - Initial setup
- **"📚 Building knowledge base..."** - Creating vector embeddings
- **"⏳ Analyzing your documents..."** - LLM generating suggestions
- **"✅ Analysis Complete!"** - Results ready

### 3. Automatic Results Display
The system automatically shows:
- **Profile Summary**: A concise 2-3 sentence summary to add at the top of your resume
- **Key Skills**: 5-8 essential skills to highlight (tailored to the job description)
- **Resume Bullets**: 5 tailored bullet points for your experience section
- **ATS Keywords**: 8-12 keywords/phrases for Applicant Tracking System optimization

### 4. Follow-up Questions
After the automatic analysis is displayed:
- The query box appears below the results
- You can ask follow-up questions like:
  - "How can I better highlight my leadership experience?"
  - "What certifications should I pursue for this role?"
  - "Can you elaborate on the technical skills I should develop?"

## User Flow

```
┌─────────────────────────────────┐
│ Upload Resume + Job Description │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Click "Upload & Analyze"      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Status: Analyzing (30-60s)     │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Display Enhancement Suggestions │
│  - Profile Summary              │
│  - Skills to Add                │
│  - Resume Bullets               │
│  - ATS Keywords                 │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Ask Follow-up Questions        │
│  (Query box now visible)        │
└─────────────────────────────────┘
```

## Technical Details

### Backend Processing (`backend_api.py`)
- **Endpoint**: `/rag-coach/upload` with `process_resume_job=true` parameter
- **Background Thread**: Processes documents asynchronously
- **LLM Prompt**: Structured prompt requesting specific sections
- **Result Storage**: Saves to `./uploads/processed/` for retrieval

### Frontend Polling (`app.py`)
- **Status Check**: Polls `/rag-coach/status` every 2 seconds
- **Max Wait**: 120 seconds (2 minutes)
- **Result Fetch**: Retrieves from `/rag-coach/processed-result`
- **Session State**: Tracks `rag_documents_uploaded` to show query box

### File Detection Heuristics
The system automatically identifies documents by filename:
- **Resume**: Contains "resume", "cv", or "profile" in filename
- **Job Description**: Contains "job", "description", or "jd" in filename
- **Fallback**: If unclear, first file = resume, second file = job description

## Benefits

1. **Immediate Value**: Get actionable suggestions without asking
2. **Structured Output**: Organized sections easy to copy-paste
3. **ATS Optimization**: Keywords specifically for beating ATS filters
4. **Contextual Advice**: Based on YOUR resume and the SPECIFIC job
5. **Follow-up Flexibility**: Ask deeper questions after initial analysis

## Example Output

```
### 🎯 Resume Enhancement Suggestions

**Profile Summary**
Results-driven Data Scientist with 5+ years of experience in machine learning,
predictive analytics, and big data processing. Proven track record of delivering
business value through advanced statistical modeling and data-driven insights.

**Key Skills to Highlight**
• Python & R Programming
• Machine Learning (TensorFlow, PyTorch)
• SQL & NoSQL Databases
• Data Visualization (Tableau, Power BI)
• Statistical Analysis
• AWS/Azure Cloud Platforms
• A/B Testing
• Natural Language Processing

**Tailored Resume Bullets**
• Developed predictive models that improved customer retention by 25% using
  ensemble methods and feature engineering
• Led cross-functional teams in deploying ML pipelines that reduced processing
  time by 40%
• Analyzed large datasets (10M+ records) to identify revenue optimization
  opportunities worth $2M annually
• Implemented automated reporting dashboards using Python and Tableau, saving
  15 hours/week of manual work
• Mentored 5 junior data scientists in best practices for model validation
  and deployment

**ATS Keywords**
Machine Learning • Python • SQL • Data Analysis • Predictive Modeling •
TensorFlow • Cloud Computing • Statistical Analysis • Data Visualization •
Big Data • A/B Testing • Feature Engineering
```

## Troubleshooting

### Issue: Analysis taking too long
- **Cause**: LLM processing large documents or slow CPU
- **Solution**: Wait up to 2 minutes; follow-up questions will still work

### Issue: Generic suggestions
- **Cause**: File detection failed (resume/JD not distinguished)
- **Solution**: Rename files to include "resume" and "job_description"

### Issue: No automatic analysis shown
- **Cause**: Backend processing failed
- **Solution**: Check backend logs; Ollama Mistral model must be running

### Issue: Query box not appearing
- **Cause**: Session state not set
- **Solution**: Refresh page and upload again

## Configuration

### Modify LLM Prompt
Edit `backend_api.py` line ~1110 to customize the analysis format:
```python
prompt = (
    "You are an expert resume coach.\n\n"
    "Resume:\n" + resume_text[:20000] + "\n\n"
    "Job Description:\n" + job_text[:20000] + "\n\n"
    # Customize sections here
)
```

### Adjust Polling Settings
Edit `app.py` line ~342:
```python
max_wait = 120  # Maximum wait time in seconds
poll_interval = 2  # Check status every N seconds
```

## Future Enhancements

- [ ] PDF preview of uploaded documents
- [ ] Export suggestions to DOCX/PDF
- [ ] Save analysis history to user profile
- [ ] Compare multiple job descriptions side-by-side
- [ ] Industry-specific templates
- [ ] Skill gap percentage calculation
- [ ] LinkedIn profile optimization suggestions

## Related Documentation

- **RAG_AUTO_PROCESSING_COMPLETE.md**: Technical implementation details
- **RAG_AUTO_PROCESSING_QUICKSTART.txt**: Quick reference guide
- **HOW_TO_RUN_PROJECT.md**: Setup and deployment instructions
