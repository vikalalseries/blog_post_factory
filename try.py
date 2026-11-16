from google import genai

from blog_factory import config

client = genai.Client(api_key=config.GEMINI_API_KEY)

response = client.models.generate_content(
    model=config.GEMINI_MODEL,
    contents="Write a one-sentence blog post intro about multi-agent AI systems.",
)

print(response.text)
