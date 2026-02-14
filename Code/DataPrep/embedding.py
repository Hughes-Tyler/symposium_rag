import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Loading environment for API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")
client = OpenAI(api_key=api_key)

# Loading articles dataset
ARTICLES_FILE = "Input/survey_monkey_help_articles_with_text.csv"
OUTPUT_FILE = "Results/article_embeddings.npy"

df = pd.read_csv(ARTICLES_FILE, encoding="latin1")
print(f"✅ Loaded {len(df)} articles from {ARTICLES_FILE}")

# Generating embeddings for each article’s text
embeddings = []
for text in tqdm(df["Text"].fillna(""), desc="Embedding articles"):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embeddings.append(response.data[0].embedding)

embeddings = np.array(embeddings)
np.save(OUTPUT_FILE, embeddings)

print(f"Total embeddings: {len(embeddings)} | Shape: {embeddings.shape}")
