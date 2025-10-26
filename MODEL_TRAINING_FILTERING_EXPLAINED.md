# 📊 Model Training Data Filtering Explained

## ❓ Your Questions Answered

### **Q1: Where is jobs_cleaned.csv used?**
**A:** In `model_training.py` at line 78:
```python
df = pd.read_csv('jobs_cleaned.csv', encoding='utf-8')
```

### **Q2: Why is support only 1,627 when we have 25,754 records?**
**A:** The data goes through **6 filtering steps** that reduce it from 25,754 to 8,133 records. The support value of ~1,627 represents the **test set (20% of 8,133)**.

---

## 🔄 Data Filtering Pipeline (Step-by-Step)

```
📊 STEP 1: Load Original Data
├─ Load jobs_cleaned.csv
└─ Result: 25,754 records
        ↓
🔧 STEP 2: Validate Skills Against Database
├─ Check each job's skills against 215-skill database
├─ Records with NO valid skills: 11,819 (45.9%) ❌
└─ Records WITH valid skills: 13,935 ✅
        ↓
🏷️ STEP 3: Categorize Jobs into Groups
├─ Assign each job to one of 11 categories
└─ Distribution:
    • Other: 13,428 jobs
    • Software Developer: 6,014
    • Sales & Business Development: 2,805
    • Human Resources: 1,024
    • Finance & Accounting: 654
    • Project/Product Manager: 521
    • IT Operations: 419
    • Marketing: 323
    • QA/Test Engineer: 275
    • UI/UX & Design: 204
    • Data Professional: 87
        ↓
❌ STEP 4: Remove 'Other' Category
├─ Remove jobs that don't fit 11 core categories
├─ Records in 'Other': 13,428 ❌
└─ Records remaining: 12,326
        ↓
🔍 STEP 5: Remove Empty Skill Records
├─ Remove jobs with no valid skills after filtering
├─ Empty skill records: 4,193 ❌
└─ Records remaining: 8,133 ✅
        ↓
📉 STEP 6: Remove Rare Categories
├─ Minimum 10 samples per category required
├─ All categories have >= 10 samples ✅
└─ Final records: 8,133
        ↓
🎓 STEP 7: Train/Test Split (80/20)
├─ Training set: 6,507 records (80%)
└─ Test set: 1,626 records (20%) 👈 THIS IS THE "SUPPORT"
```

---

## 📈 Data Reduction Summary

| Step | Records | Removed | % Retained |
|------|---------|---------|------------|
| **Original Data** | 25,754 | - | 100% |
| **After Skill Validation** | 13,935 | 11,819 | 54.1% |
| **After Categorization** | 12,326 | 1,609 | 47.8% |
| **After Removing 'Other'** | 12,326 | 13,428 | 47.8% |
| **After Removing Empty Skills** | 8,133 | 4,193 | 31.6% |
| **After Rare Class Filter** | 8,133 | 0 | 31.6% |
| **Training Set (80%)** | 6,507 | - | 25.3% |
| **Test Set (20%)** | **1,626** | - | **6.3%** |

---

## 🎯 Why the Massive Reduction?

### **1. Skills Not in Database (45.9% lost)**
**Problem:** Many job postings have skills that aren't in our 215-skill database

**Example:**
```csv
Job: "Business Development Manager"
Skills: "cold calling| lead generation| client meetings"
Result: NONE of these match the 215 technical skills ❌
```

**Why:** The 215-skill database contains **technical skills** like:
- python, java, react, docker, aws, kubernetes, etc.

But many jobs (especially non-tech) have **soft skills** or **domain-specific skills**:
- negotiation, sales, communication, cold calling, etc.

**Solution:** These jobs get filtered out because they have 0 valid skills.

---

### **2. 'Other' Category Jobs (52.1% of original data)**
**Problem:** 13,428 jobs (52%) don't fit into the 11 core categories

**Example Job Titles in 'Other':**
- Delivery Boy
- Security Guard  
- Content Writer
- Graphic Designer (non-UI/UX)
- Real Estate Agent
- Teacher
- Nurse

**Why:** The system focuses on **11 tech/business categories**:
1. Software Developer
2. Data Professional
3. IT Operations & Infrastructure
4. Project/Product Manager
5. QA/Test Engineer
6. Sales & Business Development
7. Marketing
8. Human Resources
9. Finance & Accounting
10. UI/UX & Design
11. Customer Support

**Solution:** All non-matching jobs categorized as 'Other' are removed for focused training.

---

### **3. Empty Skills After Filtering**
**Problem:** Some jobs, after skill validation, have no skills left

**Example:**
```csv
Original: "sales executive| negotiation| client handling"
After filtering: [] (empty - none match 215 technical skills)
```

---

## 💡 Final Dataset Quality

### **What We Keep (8,133 records):**
✅ Jobs with **valid technical skills** from 215-skill database  
✅ Jobs in **10 core categories** (removed 'Other')  
✅ Categories with **>= 10 samples** for reliable training  
✅ **High-quality data** suitable for machine learning  

### **Final Distribution:**
```
Software Developer              4,774 (58.7%)
Sales & Business Development    1,606 (19.7%)
Finance & Accounting              438 (5.4%)
Project / Product Manager         374 (4.6%)
IT Operations & Infrastructure    264 (3.2%)
Marketing                         231 (2.8%)
Human Resources                   151 (1.9%)
UI/UX & Design                    151 (1.9%)
QA / Test Engineer                 80 (1.0%)
Data Professional                  64 (0.8%)
```

---

## 🎓 Why Support = 1,626?

### **Understanding "Support" in Classification Report:**

**Support** = Number of actual samples in the **test set** for each class

```
Training Pipeline:
8,133 total records
    ↓
80/20 Split
    ↓
├─ Training: 6,507 (used to train model)
└─ Testing: 1,626 (used to evaluate model) 👈 THIS IS SUPPORT
```

**In the classification report:**
- Each class shows its support (number of test samples)
- Total support across all classes = **1,626**
- This is 20% of 8,133 final records
- This is only 6.3% of original 25,754 records

---

## 📊 Example Classification Report:

```
                                precision  recall  f1-score  support

Software Developer                  0.95    0.94      0.95      955
Sales & Business Development        0.88    0.89      0.89      321
Finance & Accounting                0.82    0.85      0.83       88
Project / Product Manager           0.80    0.78      0.79       75
IT Operations & Infrastructure      0.75    0.72      0.73       53
Marketing                           0.85    0.83      0.84       46
Human Resources                     0.78    0.80      0.79       30
UI/UX & Design                      0.82    0.80      0.81       30
QA / Test Engineer                  0.70    0.68      0.69       16
Data Professional                   0.65    0.62      0.64       12

                          accuracy                      0.89     1626
                         macro avg      0.80    0.79      0.80     1626
                      weighted avg      0.89    0.89      0.89     1626
```

**Notice:** Total support = 1,626 (sum of all class supports)

---

## 🔍 Code Location Summary

### **Primary Usage:**
**File:** `model_training.py`  
**Line:** 78  
```python
df = pd.read_csv('jobs_cleaned.csv', encoding='utf-8')
```

### **Filtering Logic:**
**File:** `model_training.py`  
**Lines:** 84-95  
```python
# Process skills (validate against 215-skill database)
df['Cleaned Skills'] = df['Key Skills'].apply(process_skills)

# Categorize jobs
df['Job Group'] = df['Job Title'].apply(group_job_titles)

# Remove 'Other' category
df = df[df['Job Group'] != 'Other']

# Remove empty skills
df = df[df['Cleaned Skills'].map(len) > 0]

# Remove rare classes (< 10 samples)
MINIMUM_SAMPLES_PER_CLASS = 10
group_counts = df['Job Group'].value_counts()
classes_to_keep = group_counts[group_counts >= MINIMUM_SAMPLES_PER_CLASS].index
df = df[df['Job Group'].isin(classes_to_keep)]
```

---

## 📖 Related Files

1. **`jobs_cleaned.csv`** - Original 25,754 job records
2. **`skills_db.json`** - 215 validated technical skills
3. **`model_training.py`** - Training pipeline with filtering
4. **`analyze_data_filtering.py`** - Diagnostic script showing filtering steps
5. **`job_recommender_pipeline.joblib`** - Trained model (on 6,507 samples)
6. **`job_title_encoder.joblib`** - Label encoder for 10 categories

---

## ✅ Key Takeaways

1. **Original data:** 25,754 records in `jobs_cleaned.csv`
2. **Filtered data:** 8,133 records after quality filtering (31.6% retained)
3. **Training data:** 6,507 records (80% of filtered)
4. **Test data:** 1,626 records (20% of filtered) 👈 **This is the "support"**
5. **Data loss reasons:**
   - 45.9% have no technical skills from 215-skill database
   - 52.1% are in 'Other' category (not in 11 core categories)
   - Quality over quantity approach
6. **Result:** High-quality model trained on focused, validated data

---

**The support value of ~1,626 is correct!**  
It represents the 20% test set from 8,133 quality-filtered records, not the original 25,754 records.

---

**Document Version:** 1.0  
**Last Updated:** October 25, 2025  
**Created by:** Data Analysis Pipeline
