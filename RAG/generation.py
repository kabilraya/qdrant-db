from google import genai
import os
from dotenv import load_dotenv
from google.genai import types

load_dotenv()
def chat(prompt:str):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
    model = "gemini-2.5-flash-lite", contents=prompt,
    )
    print(response.text)