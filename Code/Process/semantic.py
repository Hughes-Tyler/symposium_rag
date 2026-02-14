import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# Load environment and initialize
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")
client = OpenAI(api_key=api_key)

# File paths
ARTICLES_FILE = "Input/survey_monkey_help_articles_with_text.csv"
QUERIES_FILE = "Data/Survey Monkey Queries - Hughes.xlsx"
EMBEDDINGS_FILE = "Results/article_embeddings.npy"

# Load data
df = pd.read_csv(ARTICLES_FILE, encoding="latin1")
article_embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True)

encoding = tiktoken.encoding_for_model("text-embedding-3-small")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

# Semantic search function
def semantic_search(query: str, top_n: int = 10):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_emb = np.array(resp.data[0].embedding)
    norms_query = np.linalg.norm(query_emb)
    norms_articles = np.linalg.norm(article_embeddings, axis=1)
    similarities = (article_embeddings @ query_emb) / (norms_articles * norms_query)
    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = df.iloc[top_indices][['Title', 'URL']].copy()
    results['Score'] = similarities[top_indices]
    return results

# Precision and Recall at K functions
def precision_at_k(retrieved_urls, relevant_urls, k):
    """
    Precision@K: What proportion of the top K retrieved items are relevant?
    """
    top_k = retrieved_urls[:k]
    relevant_in_top_k = sum(1 for url in top_k if url in relevant_urls)
    return relevant_in_top_k / k if k > 0 else 0

def recall_at_k(retrieved_urls, relevant_urls, k):
    """
    Recall@K: What proportion of all relevant items appear in the top K?
    """
    top_k = retrieved_urls[:k]
    relevant_in_top_k = sum(1 for url in top_k if url in relevant_urls)
    total_relevant = len(relevant_urls)
    return relevant_in_top_k / total_relevant if total_relevant > 0 else 0

def mean_reciprocal_rank(retrieved_urls, relevant_urls):
    """
    MRR: 1/rank of the first relevant item (0 if none found)
    """
    for i, url in enumerate(retrieved_urls, 1):
        if url in relevant_urls:
            return 1.0 / i
    return 0.0

# Evaluate all queries from file
queries_df = pd.read_excel(QUERIES_FILE)
results = []
k_values = [1, 3, 5, 10]  # Different K values to evaluate

# Store metrics across all queries
all_metrics = {f"P@{k}": [] for k in k_values}
all_metrics.update({f"R@{k}": [] for k in k_values})
all_metrics["MRR"] = []

for _, row in queries_df.iterrows():
    query = row['Query']
    expected_url = str(row['Expected articles']).strip().lower()
    
    # Get top 10 articles
    top_articles = semantic_search(query, top_n=10)
    top_urls = [u.strip().lower() for u in top_articles['URL']]
    
    # For this evaluation we have 1 relevant article per query
    relevant_urls = {expected_url}
    
    match_found = expected_url in top_urls
    rank = top_urls.index(expected_url) + 1 if match_found else None
    
    # Calculate metrics for this query
    query_metrics = {}
    for k in k_values:
        p_at_k = precision_at_k(top_urls, relevant_urls, k)
        r_at_k = recall_at_k(top_urls, relevant_urls, k)
        query_metrics[f"P@{k}"] = p_at_k
        query_metrics[f"R@{k}"] = r_at_k
        all_metrics[f"P@{k}"].append(p_at_k)
        all_metrics[f"R@{k}"].append(r_at_k)
    
    mrr = mean_reciprocal_rank(top_urls, relevant_urls)
    query_metrics["MRR"] = mrr
    all_metrics["MRR"].append(mrr)
    
    result_row = {
        "Query": query,
        "Expected Article (URL)": row['Expected articles'],
        "Match Found": match_found,
        "Rank (if found)": rank,
        "Top Results (URLs)": "; ".join(top_articles['URL'].tolist()),
        "Top Result Titles": "; ".join(top_articles['Title'].tolist())
    }
    result_row.update(query_metrics)
    results.append(result_row)
    
    print(f"Processed query: {query} | Match found: {match_found} | Rank: {rank}")

# Save 
output_df = pd.DataFrame(results)
output_path = "Results/semantic_query_match_evaluation.csv"
output_df.to_csv(output_path, index=False)

# ---------------------------------------------------------------------
# Calculate and display aggregate metrics
# ---------------------------------------------------------------------
accuracy = output_df['Match Found'].mean() * 100

for k in k_values:
    avg_precision = np.mean(all_metrics[f"P@{k}"]) * 100

for k in k_values:
    avg_recall = np.mean(all_metrics[f"R@{k}"]) * 100

avg_mrr = np.mean(all_metrics["MRR"])

# Calculate average rank for found items
found_ranks = output_df[output_df['Match Found']]['Rank (if found)'].dropna()
if len(found_ranks) > 0:
    avg_rank = found_ranks.mean()

