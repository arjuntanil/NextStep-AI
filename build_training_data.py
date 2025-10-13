# build_training_data.py

import pandas as pd
from transformers import pipeline
import joblib

NER_MODEL_NAME = "ianshulx/skill-ner-v2"

def group_job_titles(title):
    title = str(title).lower()
    if any(s in title for s in ['software', 'developer', 'programmer', 'coding']): return 'Software Developer'
    if any(s in title for s in ['test', 'qa', 'quality assurance']): return 'QA / Test Engineer'
    if any(s in title for s in ['data scientist', 'data science', 'data analyst', 'business intelligence', 'data engineer', 'etl']): return 'Data Professional'
    if any(s in title for s in ['product manager', 'project manager', 'program manager']): return 'Project / Product Manager'
    if any(s in title for s in ['hr', 'human resources', 'recruitment']): return 'HR / Recruiter'
    if any(s in title for s in ['sales', 'business development']): return 'Sales / Business Development'
    if any(s in title for s in ['marketing', 'digital media', 'seo']): return 'Marketing'
    if any(s in title for s in ['network engineer', 'system administrator', 'it support', 'devops', 'sre']): return 'IT Operations & DevOps'
    if any(s in title for s in ['ui', 'ux', 'web design']): return 'UI/UX Designer'
    if any(s in title for s in ['accounts', 'accountant', 'finance']): return 'Accounts / Finance'
    if any(s in title for s in ['customer support', 'customer service']): return 'Customer Support'
    return 'Other'

def main():
    print("🚀 Starting data building process with new JSON dataset...")

    try:
        # ⭐ FIX: Removed 'lines=True' to correctly read the standard JSON file
        df = pd.read_json('monster_india.json')
    except Exception as e:
        print(f"❌ Error reading 'monster_india.json': {e}")
        return

    print(f"Loading NER model '{NER_MODEL_NAME}'...")
    ner_pipeline = pipeline("ner", model=NER_MODEL_NAME, aggregation_strategy="simple", device=-1)

    print("Extracting skills from job descriptions... (This can take a long time)")
    df['job_description'] = df['job_description'].astype(str)

    descriptions = df['job_description'].tolist()
    all_skills = []
    batch_size = 16 
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i+batch_size]
        try:
            results = ner_pipeline(batch)
            for res in results:
                skills = sorted(list(set([entity['word'].lower() for entity in res if entity['entity_group'] == 'Skill'])))
                all_skills.append(skills)
        except Exception as e:
            print(f"  - Error processing batch {i // batch_size + 1}: {e}")
            # Add empty skill lists for the failed batch to keep lengths consistent
            all_skills.extend([[] for _ in batch])

        print(f"  Processed {min(i+batch_size, len(descriptions))}/{len(descriptions)} descriptions...")

    df['Cleaned Skills'] = all_skills
    df['Job Group'] = df['title'].apply(group_job_titles)

    final_df = df[df['Job Group'] != 'Other']
    final_df = final_df[final_df['Cleaned Skills'].map(len) > 0]
    final_df = final_df[['Job Group', 'title', 'Cleaned Skills']]

    final_df.to_json('training_data_clean.jsonl', orient='records', lines=True)
    print(f"\n✅ Successfully created 'training_data_clean.jsonl' with {len(final_df)} clean records.")

if __name__ == "__main__":
    main()