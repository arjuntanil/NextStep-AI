import json

SOURCE_FILE = 'monster_india.json'
OUTPUT_FILE = 'jobs_knowledge_base.jsonl'

try:
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_file:
        for entry in data:
            out_file.write(json.dumps(entry) + '\n')
            
    print(f"Successfully converted {len(data)} records from {SOURCE_FILE} to {OUTPUT_FILE}.")

except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {SOURCE_FILE}. Check file format.")
except Exception as e:
    print(f"An error occurred: {e}")