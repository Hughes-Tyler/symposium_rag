import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re

# Load and initialize
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
df = pd.read_csv(ARTICLES_FILE)
article_embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True)
queries_df = pd.read_excel(QUERIES_FILE)

# Initialize TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df["Text"].fillna(""))

# Build BM25-style vectorizer
bm25_vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),
    max_features=10000,
    sublinear_tf=True,  # Use log scaling
    norm='l2'
)
bm25_matrix = bm25_vectorizer.fit_transform(df["Text"].fillna(""))

# Feature extraction for keyword matching
def extract_keywords(text):
    """Extract potential keywords from text."""
    if pd.isna(text):
        return set()
    # Convert to lowercase and extract words
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return set(words)

# Pre-compute keywords
df['keywords'] = df['Text'].apply(extract_keywords)
df['title_keywords'] = df['Title'].apply(extract_keywords)

# Individual search methods
def semantic_search(query: str, top_n: int = 20):
    """Semantic search using embeddings."""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_emb = np.array(resp.data[0].embedding)
    norms_query = np.linalg.norm(query_emb)
    norms_articles = np.linalg.norm(article_embeddings, axis=1)
    similarities = (article_embeddings @ query_emb) / (norms_articles * norms_query)
    
    # Return dict with index and score
    return {i: score for i, score in enumerate(similarities)}

def tfidf_search(query: str, top_n: int = 20):
    """TF-IDF based search."""
    query_vec = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return {i: score for i, score in enumerate(similarities)}

def bm25_search(query: str, top_n: int = 20):
    """BM25-style search (improved TF-IDF)."""
    query_vec = bm25_vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, bm25_matrix).flatten()
    return {i: score for i, score in enumerate(similarities)}

def keyword_boost_search(query: str):
    """Exact keyword matching with title boost."""
    query_keywords = extract_keywords(query)
    scores = {}
    
    for idx, row in df.iterrows():
        score = 0.0
        article_keywords = row['keywords']
        title_keywords = row['title_keywords']
        
        text_matches = len(query_keywords & article_keywords)
        score += text_matches * 0.5
        title_matches = len(query_keywords & title_keywords)
        score += title_matches * 2.0
        
        # Bonus for having all query keywords
        if query_keywords.issubset(article_keywords | title_keywords):
            score += 3.0
        
        scores[idx] = score
    
    # Normalize
    max_score = max(scores.values()) if scores else 1.0
    if max_score > 0:
        scores = {k: v / max_score for k, v in scores.items()}
    
    return scores

# Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(ranking_dicts, k=60):
    """
    Combine multiple rankings using Reciprocal Rank Fusion.
    
    Args:
        ranking_dicts: List of dicts with {index: score}
        k: RRF constant (typically 60)
    
    Returns:
        Dict with fused scores
    """
    fused_scores = defaultdict(float)
    
    for ranking in ranking_dicts:
        # Sort by score descending
        sorted_items = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
        
        # Add RRF score: 1 / (k + rank)
        for rank, (idx, score) in enumerate(sorted_items, 1):
            fused_scores[idx] += 1.0 / (k + rank)
    
    return dict(fused_scores)

# Weighted Ensemble
def weighted_ensemble(ranking_dicts, weights):
    """
    Combine rankings with custom weights.
    
    Args:
        ranking_dicts: List of dicts with {index: score}
        weights: List of weights (should sum to 1.0)
    
    Returns:
        Dict with weighted scores
    """
    ensemble_scores = defaultdict(float)
    
    for ranking, weight in zip(ranking_dicts, weights):
        # Normalize scores to 0-1 range
        scores = list(ranking.values())
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            for idx, score in ranking.items():
                normalized = (score - min_score) / score_range if score_range > 0 else 0
                ensemble_scores[idx] += normalized * weight
    
    return dict(ensemble_scores)

# Hybrid search with multiple strategies
def hybrid_search(query: str, top_n: int = 10, method='rrf'):
    """
    Perform hybrid search combining multiple methods.
    
    Args:
        query: Search query
        top_n: Number of results to return
        method: 'rrf' for Reciprocal Rank Fusion, 'weighted' for weighted ensemble
    
    Returns:
        DataFrame with top results
    """
    # Get rankings from all methods
    semantic_scores = semantic_search(query)
    tfidf_scores = tfidf_search(query)
    bm25_scores = bm25_search(query)
    keyword_scores = keyword_boost_search(query)
    
    if method == 'rrf':
        final_scores = reciprocal_rank_fusion([
            semantic_scores,
            tfidf_scores,
            bm25_scores,
            keyword_scores
        ], k=60)
    else:
        # Weights: [semantic, tfidf, bm25, keyword]
        weights = [0.40, 0.25, 0.25, 0.10]  # ADJUST based on performance - START AT 0.25 FOR ALL
        final_scores = weighted_ensemble([
            semantic_scores,
            tfidf_scores,
            bm25_scores,
            keyword_scores
        ], weights)
    
    top_indices = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:top_n]
    
    results = df.iloc[top_indices][['Title', 'URL']].copy()
    results['Score'] = [final_scores[i] for i in top_indices]
    
    return results

# Evaluation metrics
def precision_at_k(retrieved_urls, relevant_urls, k):
    top_k = retrieved_urls[:k]
    relevant_in_top_k = sum(1 for url in top_k if url in relevant_urls)
    return relevant_in_top_k / k if k > 0 else 0

def recall_at_k(retrieved_urls, relevant_urls, k):
    top_k = retrieved_urls[:k]
    relevant_in_top_k = sum(1 for url in top_k if url in relevant_urls)
    total_relevant = len(relevant_urls)
    return relevant_in_top_k / total_relevant if total_relevant > 0 else 0

def mean_reciprocal_rank(retrieved_urls, relevant_urls):
    for i, url in enumerate(retrieved_urls, 1):
        if url in relevant_urls:
            return 1.0 / i
    return 0.0

# Evaluate hybrid search
def evaluate_search(method='rrf'):
    """Evaluate the hybrid search method."""
    results = []
    k_values = [1, 3, 5, 10]
    
    all_metrics = {f"P@{k}": [] for k in k_values}
    all_metrics.update({f"R@{k}": [] for k in k_values})
    all_metrics["MRR"] = []
    
    for _, row in queries_df.iterrows():
        query = row['Query']
        expected_url = str(row['Expected articles']).strip().lower()
        
        # Perform hybrid search
        top_articles = hybrid_search(query, top_n=10, method=method)
        top_urls = [u.strip().lower() for u in top_articles['URL']]
        
        relevant_urls = {expected_url}
        match_found = expected_url in top_urls
        rank = top_urls.index(expected_url) + 1 if match_found else None
        
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
            
    # Save results
    output_df = pd.DataFrame(results)
    output_path = f"Results/hybrid_search_evaluation_{method}.csv"
    output_df.to_csv(output_path, index=False)
    
    accuracy = output_df['Match Found'].mean() * 100

    for k in k_values:
        avg_precision = np.mean(all_metrics[f"P@{k}"]) * 100
        print(f"  Precision@{k:2d}: {avg_precision:6.2f}%")
    
    print("RECALL @ K")
    for k in k_values:
        avg_recall = np.mean(all_metrics[f"R@{k}"]) * 100
        print(f"  Recall@{k:2d}:    {avg_recall:6.2f}%")
    
    print("OTHER METRICS")
    avg_mrr = np.mean(all_metrics["MRR"])
    print(f"  Mean Reciprocal Rank (MRR): {avg_mrr:.4f}")
    
    found_ranks = output_df[output_df['Match Found']]['Rank (if found)'].dropna()
    if len(found_ranks) > 0:
        avg_rank = found_ranks.mean()
        print(f"  Average Rank (when found):   {avg_rank:.2f}")
        
    return output_df, all_metrics

# ---------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Running Hybrid Search with Reciprocal Rank Fusion...")
    rrf_results, rrf_metrics = evaluate_search(method='rrf')
    
    
    print("🔍 Running Hybrid Search with Weighted Ensemble...")
    weighted_results, weighted_metrics = evaluate_search(method='weighted')
    
    # Compare methods
    print("📊 COMPARISON: RRF vs WEIGHTED")
    print(f"RRF Accuracy:      {rrf_results['Match Found'].mean() * 100:.2f}%")
    print(f"Weighted Accuracy: {weighted_results['Match Found'].mean() * 100:.2f}%")
    print(f"RRF MRR:           {np.mean(rrf_metrics['MRR']):.4f}")
    print(f"Weighted MRR:      {np.mean(weighted_metrics['MRR']):.4f}")
