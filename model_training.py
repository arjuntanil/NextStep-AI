# model_training.py (Consolidated Categories + Enhancements)
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. Load Skills from JSON ---
try:
    with open('skills_db.json', 'r') as f:
        SKILLS_DB = json.load(f)
    SKILL_DB_SET = set(SKILLS_DB)
    print(f"✅ Successfully loaded {len(SKILL_DB_SET)} skills from skills_db.json")
except FileNotFoundError:
    print("❌ Error: skills_db.json not found.")
    SKILL_DB_SET = set()

def process_skills(skills_str):
    """Cleans and filters skills based on the loaded SKILL_DB_SET."""
    if pd.isnull(skills_str):
        return []
    cleaned_skills = [s.strip().lower() for s in str(skills_str).split('|')]
    return sorted([s for s in cleaned_skills if s in SKILL_DB_SET])

# --- 2. Job Categorization Engine (Consolidated Groups) ---
def group_job_titles(title):
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

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("✅ Confusion matrix saved to confusion_matrix.png")

def main():
    print("🚀 Starting training pipeline with CONSOLIDATED categories...")
    try:
        df = pd.read_csv('jobs_cleaned.csv', encoding='utf-8')
    except FileNotFoundError:
        print("❌ Error: jobs_cleaned.csv not found.")
        return

    # --- Data Preparation ---
    df['Cleaned Skills'] = df['Key Skills'].apply(process_skills)
    df['Job Group'] = df['Job Title'].apply(group_job_titles)

    df = df[df['Job Group'] != 'Other']
    df = df[df['Cleaned Skills'].map(len) > 0]

    # --- Filtering Rare Classes ---
    MINIMUM_SAMPLES_PER_CLASS = 10
    group_counts = df['Job Group'].value_counts()
    classes_to_keep = group_counts[group_counts >= MINIMUM_SAMPLES_PER_CLASS].index.tolist()
    df = df[df['Job Group'].isin(classes_to_keep)].reset_index(drop=True)

    print(f"📊 Training with {len(df)} valid records across {df['Job Group'].nunique()} standardized job groups.")
    print("\n--- Final Job Group Distribution ---")
    print(df['Job Group'].value_counts())
    print("----------------------------------\n")

    # --- Artifact Generation for Skills ---
    print("💾 Generating prioritized skill lists...")
    prioritized_skills = {}
    grouped_skills = df.groupby('Job Group')['Cleaned Skills'].agg(list)
    for group, skills_lists in grouped_skills.items():
        flat_list = [skill for sublist in skills_lists for skill in sublist]
        prioritized_skills[group] = [skill for skill, count in Counter(flat_list).most_common(30)]
    joblib.dump(prioritized_skills, 'prioritized_skills.joblib')

    # --- Model Training Pipeline ---
    df['Skills_Str'] = df['Cleaned Skills'].apply(' '.join)
    X = df['Skills_Str']
    title_encoder = LabelEncoder()
    y = title_encoder.fit_transform(df['Job Group'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define pipeline structure
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression()) # Placeholder, will be replaced by GridSearchCV
    ])

    # Define parameter grid to test both MultinomialNB and LogisticRegression
    param_grid = [
        {
            'clf': [MultinomialNB()],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__alpha': [0.1, 0.5, 1.0]
        },
        {
            'clf': [LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=42, max_iter=1000)],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__C': [1, 10, 20]
        }
    ]

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1_weighted')
    print("🔬 Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)

    print("\n✅ Tuning complete!")
    print(f"🏆 Best Model Parameters: {grid_search.best_params_}")
    print(f"🎯 Best Cross-Validation F1 Score: {grid_search.best_score_:.4f}\n")

    best_model = grid_search.best_estimator_

    # --- Model Evaluation ---
    accuracy = best_model.score(X_test, y_test)
    y_pred = best_model.predict(X_test)

    print("\n" + "="*50 + "\nFINAL MODEL EVALUATION REPORT\n" + "="*50)
    print(f"🎯 Test Set Accuracy: {accuracy:.4f}\n")
    print("📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=title_encoder.classes_, zero_division=0))
    print("="*50 + "\n")

    print("📈 Generating Confusion Matrix...")
    plot_confusion_matrix(y_test, y_pred, title_encoder.classes_)

    # --- Saving Model Artifacts ---
    print("💾 Saving final model and encoder artifacts...")
    joblib.dump(best_model, 'job_recommender_pipeline.joblib')
    joblib.dump(title_encoder, 'job_title_encoder.joblib')
    
    # Save related titles based on training data
    related_titles_map = {}
    for group in df['Job Group'].unique():
        titles_in_group = df[df['Job Group'] == group]['Job Title'].value_counts().index.tolist()
        related_titles_map[group] = titles_in_group[:10]
    joblib.dump(related_titles_map, 'related_titles.joblib')

    print("🎉 Training complete!")

if __name__ == '__main__':
    main()