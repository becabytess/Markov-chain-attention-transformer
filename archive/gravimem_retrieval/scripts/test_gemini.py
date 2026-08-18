import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print("Testing API Key:", api_key[:10] + "...")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
res = requests.get(url)
print("ListModels Status:", res.status_code)
if res.status_code == 200:
    data = res.json()
    models = data.get("models", [])
    print(f"Total models available: {len(models)}")
    for m in models:
        name = m.get("name")
        methods = m.get("supportedGenerationMethods", [])
        if "embedContent" in methods or "embedding" in name:
            print(f" -> EMBEDDING: {name} (methods: {methods})")
        elif "flash" in name:
            print(f" -> LLM: {name}")
else:
    print("Error:", res.text)
