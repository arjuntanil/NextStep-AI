"""
Test script: upload two placeholder PDF files to /rag-coach/upload with process_resume_job=True
and poll for processed result.
"""
import requests
import time
import json

BASE = "http://127.0.0.1:8000"
UPLOAD = BASE + "/rag-coach/upload"
STATUS = BASE + "/rag-coach/status"
RESULT = BASE + "/rag-coach/processed-result"

# Create two small PDF-like byte contents (not a real PDF, but backend will still save and attempt to parse)
resume_bytes = b"%PDF-1.4\n%DummyResume\n1 0 obj<<>>endobj\n"
job_bytes = b"%PDF-1.4\n%DummyJobDesc\n1 0 obj<<>>endobj\n"

files = [
    ('files', ('resume.pdf', resume_bytes, 'application/pdf')),
    ('files', ('job_description.pdf', job_bytes, 'application/pdf')),
]

print('Uploading files and requesting processing...')
resp = requests.post(UPLOAD, files=files, data={'process_resume_job': 'true'}, timeout=120)
print('Upload status:', resp.status_code)
print(resp.text)

# Poll status until processing_ready or timeout
start = time.time()
while time.time() - start < 120:
    try:
        s = requests.get(STATUS, timeout=10).json()
        print('Status:', s)
        if s.get('processing_ready'):
            print('Processing ready, fetching result...')
            r = requests.get(RESULT, timeout=10)
            print('Result status:', r.status_code)
            try:
                print(json.dumps(r.json(), indent=2))
            except Exception as e:
                print('Result parse error:', e)
            break
        elif s.get('processing'):
            print('Still processing...')
        else:
            print('Not processing yet, waiting...')
    except Exception as e:
        print('Error polling status:', e)
    time.sleep(3)
else:
    print('Timeout waiting for processing')
