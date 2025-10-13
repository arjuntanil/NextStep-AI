# clean.py

import pandas as pd
import re

def clean_jobs_csv(input_file='jobs.csv', output_file='jobs_cleaned.csv'):
    """
    Cleans the raw jobs CSV by automatically detecting columns, standardizing formats,
    and removing invalid or duplicate data.
    """
    print("🧹 Starting CSV cleanup process...")
    
    try:
        df = pd.read_csv(input_file, skipinitialspace=True)
        print(f"📊 Original dataset shape: {df.shape}")
        
        # Auto-detect 'Job Title' and 'Key Skills' columns
        cols = {col.lower().replace(' ', '').replace('_', ''): col for col in df.columns}
        title_col = cols.get('jobtitle', cols.get('title', None))
        skills_col = cols.get('keyskills', cols.get('skills', None))

        if not title_col or not skills_col:
            raise ValueError("Could not auto-detect 'Job Title' and 'Key Skills' columns. Please name them appropriately.")
            
        print(f"🗺️ Auto-detected columns: Job Title='{title_col}', Key Skills='{skills_col}'")
        
        # Create a new DataFrame with just the essential columns
        cleaned_df = df[[title_col, skills_col]].copy()
        cleaned_df.columns = ['Job Title', 'Key Skills']
        
        # Drop rows where essential data is missing
        cleaned_df.dropna(inplace=True)
        
        # Standardize and clean text data
        cleaned_df['Job Title'] = cleaned_df['Job Title'].astype(str).apply(lambda x: ' '.join(x.strip().lower().split()))
        cleaned_df['Key Skills'] = cleaned_df['Key Skills'].astype(str).apply(lambda x: '|'.join(sorted(list(set(s.strip().lower() for s in re.split(r'[,;/]+', x) if s.strip())))))

        # Remove rows that became empty after cleaning
        cleaned_df = cleaned_df[cleaned_df['Job Title'].str.len() > 2]
        cleaned_df = cleaned_df[cleaned_df['Key Skills'].str.len() > 2]
        
        # Drop duplicates
        cleaned_df.drop_duplicates(inplace=True)
        
        print(f"✅ Cleanup completed! Cleaned dataset shape: {cleaned_df.shape}")
        print(f"💾 Saving cleaned data to: {output_file}")
        cleaned_df.to_csv(output_file, index=False)
        print("\n📋 Sample of cleaned data:")
        print(cleaned_df.head())

    except Exception as e:
        print(f"❌ Error during CSV cleaning: {e}")

if __name__ == "__main__":
    clean_jobs_csv()