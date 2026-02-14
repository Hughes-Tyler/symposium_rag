import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Articles with text and URLs
articles_df = pd.read_csv("Input/survey_monkey_help_articles_with_text.csv")

# Monkey queries with expected article URLs
queries_df = pd.read_excel("Data/Survey Monkey Queries - Hughes.xlsx")

# Prepare TF-IDF vectorizer on the article text
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(articles_df["Text"].fillna(""))

# Define search function
def search_articles(query, top_n=10):
    """Return top N most relevant articles (Title + URL) for a given query."""
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    return articles_df.iloc[top_indices][["Title", "URL"]]

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

# Loop through queries and evaluate accuracy by URL match
results = []
k_values = [1, 3, 5, 10]  # Different K values to evaluate

# Store metrics across all queries
all_metrics = {f"P@{k}": [] for k in k_values}
all_metrics.update({f"R@{k}": [] for k in k_values})
all_metrics["MRR"] = []

for _, row in queries_df.iterrows():
    query = row["Query"]
    expected_url = str(row["Expected articles"]).strip().lower()
    
    # Get top 10 articles
    top_articles = search_articles(query, top_n=10)
    
    # Normalize URLs for comparison
    top_urls = [str(u).strip().lower() for u in top_articles["URL"]]
    
    # 1 relevant article per query
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
        "Expected Article URL": row["Expected articles"],
        "Match Found": match_found,
        "Rank (if found)": rank,
        "Top URLs": "; ".join(top_articles["URL"].tolist())
    }
    result_row.update(query_metrics)
    results.append(result_row)
    
    print(f"Processed query: {query} | Match found: {match_found} | Rank: {rank}")

# Step 5: Export evaluation results
output_df = pd.DataFrame(results)
output_path = "Results/tfidf_url_match_evaluation.csv"
output_df.to_csv(output_path, index=False)

# ---------------------------------------------------------------------
# Calculate and display aggregate metrics
# ---------------------------------------------------------------------
accuracy = output_df['Match Found'].mean() * 100


print(f"\nOverall Accuracy: {accuracy:.2f}% ({output_df['Match Found'].sum()} of {len(output_df)})")

print("PRECISION @ K (Average across all queries)")
for k in k_values:
    avg_precision = np.mean(all_metrics[f"P@{k}"]) * 100
    print(f"  Precision@{k:2d}: {avg_precision:6.2f}%")

print("RECALL @ K (Average across all queries)")
for k in k_values:
    avg_recall = np.mean(all_metrics[f"R@{k}"]) * 100
    print(f"  Recall@{k:2d}:    {avg_recall:6.2f}%")

print("OTHER METRICS")
avg_mrr = np.mean(all_metrics["MRR"])
print(f"  Mean Reciprocal Rank (MRR): {avg_mrr:.4f}")

# Calculate average rank for found items
found_ranks = output_df[output_df['Match Found']]['Rank (if found)'].dropna()
if len(found_ranks) > 0:
    avg_rank = found_ranks.mean()
    print(f"  Average Rank (when found):   {avg_rank:.2f}")

print("\nSample results (first 10 queries):")
display_cols = ["Query", "Match Found", "Rank (if found)", "P@1", "P@3", "P@5", "MRR"]
print(output_df[display_cols].head(10).to_string(index=False))