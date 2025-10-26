"""
Diagnostic script to show data filtering in model training
Shows how 25,754 records get reduced to ~1,627 for training
"""

import pandas as pd
import json
from collections import Counter

# Load skills database
with open('skills_db.json', 'r') as f:
    SKILLS_DB = json.load(f)
SKILL_DB_SET = set(SKILLS_DB)

def process_skills(skills_str):
    """Cleans and filters skills based on the loaded SKILL_DB_SET."""
    if pd.isnull(skills_str):
        return []
    cleaned_skills = [s.strip().lower() for s in str(skills_str).split('|')]
    return sorted([s for s in cleaned_skills if s in SKILL_DB_SET])

def group_job_titles(title):
    """Categorize jobs into consolidated groups"""
    title = str(title).lower()
    if any(s in title for s in ['data scientist', 'data science', 'data analyst', 'business intelligence', 'data engineer', 'etl', 'bi developer']):
        return 'Data Professional'
    if any(s in title for s in ['software developer', 'software engineer', 'programmer', 'coding', 'full stack developer', 'backend developer', 'frontend developer']):
        return 'Software Developer'
    if any(s in title for s in ['network engineer', 'network administrator', 'system administrator', 'it support', 'devops engineer', 'sre', 'cloud engineer']):
        return 'IT Operations & Infrastructure'
    if any(s in title for s in ['project manager', 'product manager', 'program manager', 'scrum master']):
        return 'Project / Product Manager'
    if any(s in title for s in ['qa engineer', 'test engineer', 'quality assurance', 'sdet']):
        return 'QA / Test Engineer'
    if any(s in title for s in ['hr', 'human resources', 'recruitment', 'talent acquisition']):
        return 'Human Resources'
    if any(s in title for s in ['sales executive', 'business development manager', 'account manager']):
        return 'Sales & Business Development'
    if any(s in title for s in ['marketing manager', 'digital marketing', 'seo specialist', 'social media manager']):
        return 'Marketing'
    if any(s in title for s in ['ui designer', 'ux designer', 'graphic designer', 'product designer']):
        return 'UI/UX & Design'
    if any(s in title for s in ['accountant', 'finance analyst', 'financial reporting', 'auditor']):
        return 'Finance & Accounting'
    if any(s in title for s in ['customer service representative', 'customer support specialist']):
        return 'Customer Support'
    return 'Other'

print("\n" + "="*80)
print("DATA FILTERING ANALYSIS - MODEL TRAINING PIPELINE")
print("="*80)

# Step 1: Load original data
df = pd.read_csv('jobs_cleaned.csv', encoding='utf-8')
print(f"\n📊 STEP 1: Load jobs_cleaned.csv")
print(f"   Original records: {len(df)}")

# Step 2: Process skills
df['Cleaned Skills'] = df['Key Skills'].apply(process_skills)
print(f"\n🔧 STEP 2: Process and validate skills against 215-skill database")
print(f"   Records after processing: {len(df)}")

# Check how many records have no valid skills
no_skills = df[df['Cleaned Skills'].map(len) == 0]
print(f"   Records with NO valid skills: {len(no_skills)} ({len(no_skills)/len(df)*100:.1f}%)")
print(f"   Records WITH valid skills: {len(df) - len(no_skills)}")

# Step 3: Categorize jobs
df['Job Group'] = df['Job Title'].apply(group_job_titles)
print(f"\n🏷️  STEP 3: Categorize jobs into groups")
print(f"   Job groups distribution:")
job_counts = df['Job Group'].value_counts()
for group, count in job_counts.items():
    print(f"      {group}: {count}")

# Step 4: Remove 'Other' category
df_no_other = df[df['Job Group'] != 'Other']
print(f"\n❌ STEP 4: Remove 'Other' category jobs")
print(f"   Records in 'Other' category: {len(df[df['Job Group'] == 'Other'])}")
print(f"   Records after removing 'Other': {len(df_no_other)}")

# Step 5: Remove records with no skills
df_with_skills = df_no_other[df_no_other['Cleaned Skills'].map(len) > 0]
print(f"\n🔍 STEP 5: Remove records with no valid skills")
print(f"   Records with empty skills: {len(df_no_other) - len(df_with_skills)}")
print(f"   Records after filtering: {len(df_with_skills)}")

# Step 6: Remove rare classes (< 10 samples)
MINIMUM_SAMPLES_PER_CLASS = 10
group_counts = df_with_skills['Job Group'].value_counts()
classes_to_keep = group_counts[group_counts >= MINIMUM_SAMPLES_PER_CLASS].index.tolist()
df_final = df_with_skills[df_with_skills['Job Group'].isin(classes_to_keep)]

print(f"\n📉 STEP 6: Filter rare job categories (minimum 10 samples)")
print(f"   Categories with < 10 samples (removed):")
rare_classes = group_counts[group_counts < MINIMUM_SAMPLES_PER_CLASS]
for group, count in rare_classes.items():
    print(f"      {group}: {count} samples (REMOVED)")

print(f"\n   Categories kept (>= 10 samples):")
for group in classes_to_keep:
    print(f"      {group}: {group_counts[group]} samples")

print(f"\n" + "="*80)
print(f"FINAL TRAINING DATASET")
print(f"="*80)
print(f"✅ Final valid records: {len(df_final)}")
print(f"✅ Job categories: {df_final['Job Group'].nunique()}")
print(f"✅ Data reduction: {len(df)} → {len(df_final)} ({len(df_final)/len(df)*100:.1f}% retained)")
print(f"\n💡 WHY THE REDUCTION?")
print(f"   - Many jobs have skills NOT in the 215-skill database")
print(f"   - Many jobs categorized as 'Other' (not in 11 core categories)")
print(f"   - Some categories have too few samples for reliable training")
print(f"   - Quality over quantity: Only train on high-quality data")

print(f"\n📊 Final Job Group Distribution:")
print(df_final['Job Group'].value_counts())

# Calculate train/test split
test_size = int(len(df_final) * 0.2)
train_size = len(df_final) - test_size
print(f"\n🎓 Training/Testing Split (80/20):")
print(f"   Training records: {train_size}")
print(f"   Testing records: {test_size}")
print(f"   Total: {len(df_final)}")

print(f"\n" + "="*80)
print(f"This explains why 'support' in classification report is ~{test_size}")
print(f"(Support = number of test samples, which is 20% of {len(df_final)})")
print(f"="*80 + "\n")
