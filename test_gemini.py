import google.generativeai as genai
import os

# Paste your API key here
API_KEY = "AIzaSyBDlo4HrZmdHn6JvB56ohtp3-KcEKl_KJg"

# Configure the client
genai.configure(api_key=API_KEY)

# List available models
print("Available models:", [m.name for m in genai.list_models()])

# Create a model instance using an available model
model = genai.GenerativeModel('gemini-pro-latest')  # Using gemini-pro-latest which is available

# Generate a response
response = model.generate_content("Hello Gemini! Are you working?")

# Print the response
print("Gemini Response:", response.text)
