import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from typing import List, Dict, Optional
from pathlib import Path

# ---------------------------------------------------------------------
# Load environment and initialize
# ---------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")
client = OpenAI(api_key=api_key)

# ---------------------------------------------------------------------
# File paths (robust: relative to the project root, not where you run from)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # rag_model.py is in Code/Process/

ARTICLES_FILE   = PROJECT_ROOT / "Input" / "survey_monkey_help_articles_with_text.csv"
EMBEDDINGS_FILE = PROJECT_ROOT / "Results" / "article_embeddings.npy"
QUERIES_FILE    = PROJECT_ROOT / "Data" / "Survey Monkey Queries - Hughes.xlsx"

# Optional debug prints
print("ARTICLES_FILE:", ARTICLES_FILE)
print("Exists:", ARTICLES_FILE.exists())

# ---------------------------------------------------------------------
# Load existing data
# ---------------------------------------------------------------------
df = pd.read_csv(ARTICLES_FILE)
article_embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True)

# Initialize tokenizer
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    return len(encoding.encode(text))

# ---------------------------------------------------------------------
# Semantic search function (same as before)
# ---------------------------------------------------------------------
def semantic_search(query: str, top_n: int = 5) -> pd.DataFrame:
    """
    Perform semantic search using OpenAI embeddings.
    
    Args:
        query: The search query
        top_n: Number of top results to return
        
    Returns:
        DataFrame with Title, URL, Text, and Score columns
    """
    # Get query embedding
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_emb = np.array(resp.data[0].embedding)
    
    # Calculate cosine similarities
    norms_query = np.linalg.norm(query_emb)
    norms_articles = np.linalg.norm(article_embeddings, axis=1)
    similarities = (article_embeddings @ query_emb) / (norms_articles * norms_query)
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = df.iloc[top_indices][['Title', 'URL', 'Text']].copy()
    results['Score'] = similarities[top_indices]
    
    return results

# ---------------------------------------------------------------------
# Context preparation for RAG
# ---------------------------------------------------------------------
def prepare_context(search_results: pd.DataFrame, max_tokens: int = 6000) -> tuple[str, List[Dict]]:
    """
    Prepare context from search results for the LLM.
    Truncates if necessary to stay within token limit.
    
    Args:
        search_results: DataFrame from semantic_search
        max_tokens: Maximum tokens for context (leaving room for query and response)
        
    Returns:
        Tuple of (formatted_context, sources_metadata)
    """
    context_parts = []
    sources = []
    current_tokens = 0
    
    for idx, row in search_results.iterrows():
        # Format each article section
        article_text = f"""
---
ARTICLE {len(context_parts) + 1}: {row['Title']}
URL: {row['URL']}
RELEVANCE SCORE: {row['Score']:.3f}

CONTENT:
{row['Text']}
---
"""
        
        article_tokens = count_tokens(article_text)
        
        # Check if adding this article would exceed token limit
        if current_tokens + article_tokens > max_tokens:
            # Try to add truncated version
            if current_tokens < max_tokens * 0.8:  # Only if we have room
                # Truncate the text to fit
                remaining_tokens = max_tokens - current_tokens - 200  # Buffer
                text_sample = row['Text'][:remaining_tokens * 4]  # Rough char estimate
                article_text = f"""
---
ARTICLE {len(context_parts) + 1}: {row['Title']}
URL: {row['URL']}
RELEVANCE SCORE: {row['Score']:.3f}

CONTENT (TRUNCATED):
{text_sample}...
---
"""
                context_parts.append(article_text)
                sources.append({
                    'number': len(sources) + 1,
                    'title': row['Title'],
                    'url': row['URL'],
                    'score': float(row['Score'])
                })
            break
        
        context_parts.append(article_text)
        sources.append({
            'number': len(sources) + 1,
            'title': row['Title'],
            'url': row['URL'],
            'score': float(row['Score'])
        })
        current_tokens += article_tokens
    
    context = "\n".join(context_parts)
    return context, sources

# ---------------------------------------------------------------------
# RAG Query Function
# ---------------------------------------------------------------------
def rag_query(
    query: str,
    top_k: int = 5,
    temperature: float = 0.3,
    max_response_tokens: int = 1000,
    model: str = "gpt-4o-mini"  # Cheapest good option
) -> Dict:
    """
    Perform RAG query: retrieve relevant documents and generate answer.
    
    Args:
        query: User's question
        top_k: Number of documents to retrieve
        temperature: LLM temperature (0-1, lower = more focused)
        max_response_tokens: Maximum tokens in response
        model: OpenAI model to use (gpt-4o-mini is cheapest)
        
    Returns:
        Dictionary containing:
            - query: Original query
            - answer: Generated answer
            - sources: List of source documents used
            - retrieved_docs: Full search results
            - tokens_used: Token usage breakdown
    """
    
    # Step 1: Retrieve relevant documents
    print(f"🔍 Searching for relevant articles...")
    search_results = semantic_search(query, top_n=top_k)
    
    # Step 2: Prepare context
    print(f"📚 Preparing context from {len(search_results)} articles...")
    context, sources = prepare_context(search_results)
    
    # Step 3: Build the prompt
    system_prompt = """You are a helpful SurveyMonkey support assistant. Your role is to answer user questions based ONLY on the provided knowledge base articles.

CRITICAL RULES:
1. Only use information from the provided articles to answer questions
2. If the answer is not in the provided articles, say "I don't have enough information in the knowledge base to answer that question."
3. Cite which article number(s) you're using (e.g., "According to Article 1...")
4. Be concise but complete - provide step-by-step instructions when relevant
5. If multiple articles have relevant info, synthesize them into a coherent answer
6. Always include relevant article URLs so users can read more

FORMAT YOUR RESPONSE:
- Start with a direct answer
- Include step-by-step instructions if applicable
- End with "📚 Source(s): [Article titles and URLs]"
"""

    user_prompt = f"""KNOWLEDGE BASE ARTICLES:
{context}

USER QUESTION:
{query}

Please provide a helpful answer based on the articles above."""
    
    # Step 4: Call the LLM
    print(f"🤖 Generating answer with {model}...")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_response_tokens
        )
        
        answer = response.choices[0].message.content
        
        # Token usage
        tokens_used = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }
        
        print(f"✅ Answer generated! Used {tokens_used['total_tokens']} tokens")
        
        return {
            'query': query,
            'answer': answer,
            'sources': sources,
            'retrieved_docs': search_results,
            'tokens_used': tokens_used,
            'model': model
        }
        
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return {
            'query': query,
            'answer': f"Error generating answer: {str(e)}",
            'sources': sources,
            'retrieved_docs': search_results,
            'tokens_used': None,
            'model': model
        }

# ---------------------------------------------------------------------
# Interactive RAG Session
# ---------------------------------------------------------------------
def interactive_rag_session():
    """
    Run an interactive RAG session where users can ask multiple questions.
    """
    print("=" * 70)
    print("🤖 SurveyMonkey Help RAG Assistant")
    print("=" * 70)
    print("Ask me anything about SurveyMonkey! Type 'quit' to exit.\n")
    
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not query:
            continue
        
        print()
        result = rag_query(query)
        
        print("\n" + "=" * 70)
        print("📝 ANSWER:")
        print("=" * 70)
        print(result['answer'])
        print("\n" + "=" * 70)
        print(f"💰 Cost: ~${calculate_cost(result['tokens_used']):.4f}")
        print("=" * 70 + "\n")

# ---------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------
def calculate_cost(tokens_used: Optional[Dict], model: str = "gpt-4o-mini") -> float:
    """
    Calculate the cost of a query based on token usage.
    Prices as of Feb 2025 for gpt-4o-mini.
    """
    if not tokens_used:
        return 0.0
    
    # Prices per 1M tokens
    pricing = {
        "gpt-4o-mini": {
            "input": 0.150,  # $0.150 per 1M input tokens
            "output": 0.600  # $0.600 per 1M output tokens
        },
        "gpt-3.5-turbo": {
            "input": 0.50,
            "output": 1.50
        }
    }
    
    model_pricing = pricing.get(model, pricing["gpt-4o-mini"])
    
    input_cost = (tokens_used['prompt_tokens'] / 1_000_000) * model_pricing['input']
    output_cost = (tokens_used['completion_tokens'] / 1_000_000) * model_pricing['output']
    
    return input_cost + output_cost

# ---------------------------------------------------------------------
# Batch evaluation on queries file
# ---------------------------------------------------------------------
def evaluate_rag_on_queries(queries_file: str = None):
    """
    Run RAG on all queries from the test file and save results.
    """
    if queries_file is None:
        queries_file = QUERIES_FILE
    
    print("📊 Running RAG evaluation on query file...")
    
    queries_df = pd.read_excel(queries_file)
    results = []
    total_cost = 0.0
    
    for idx, row in queries_df.iterrows():
        query = row['Query']
        expected_url = str(row['Expected articles']).strip().lower()
        
        print(f"\n[{idx + 1}/{len(queries_df)}] Processing: {query}")
        
        # Run RAG query
        rag_result = rag_query(query, top_k=5)
        
        # Check if expected URL is in retrieved docs
        retrieved_urls = [u.strip().lower() for u in rag_result['retrieved_docs']['URL']]
        match_found = expected_url in retrieved_urls
        rank = retrieved_urls.index(expected_url) + 1 if match_found else None
        
        # Calculate cost
        query_cost = calculate_cost(rag_result['tokens_used'])
        total_cost += query_cost
        
        results.append({
            "Query": query,
            "Expected Article (URL)": row['Expected articles'],
            "Match Found in Top 5": match_found,
            "Rank (if found)": rank,
            "Generated Answer": rag_result['answer'],
            "Sources Used": "; ".join([s['title'] for s in rag_result['sources']]),
            "Source URLs": "; ".join([s['url'] for s in rag_result['sources']]),
            "Tokens Used": rag_result['tokens_used']['total_tokens'] if rag_result['tokens_used'] else 0,
            "Cost ($)": query_cost
        })
        
        print(f"   ✓ Match found: {match_found} | Cost: ${query_cost:.4f}")
    
    # Save results
    output_df = pd.DataFrame(results)
    output_path = "Results/rag_query_evaluation.csv"
    output_df.to_csv(output_path, index=False)
    
    # Print summary
    accuracy = output_df['Match Found in Top 5'].mean() * 100
    avg_cost = output_df['Cost ($)'].mean()
    
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total queries processed: {len(output_df)}")
    print(f"Retrieval accuracy: {accuracy:.2f}%")
    print(f"Average cost per query: ${avg_cost:.4f}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Results saved to: {output_path}")
    print("=" * 70)
    
    return output_df

# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "interactive":
            # Run interactive session
            interactive_rag_session()
        
        elif mode == "evaluate":
            # Run batch evaluation
            evaluate_rag_on_queries()
        
        elif mode == "demo":
            # Run a single demo query
            demo_query = "How many contacts can I import to SurveyMonkey?"
            print(f"Demo query: {demo_query}\n")
            result = rag_query(demo_query)
            print("\n" + "=" * 70)
            print("ANSWER:")
            print("=" * 70)
            print(result['answer'])
            print("\n" + "=" * 70)
            print(f"Cost: ${calculate_cost(result['tokens_used']):.4f}")
            print("=" * 70)
        
        else:
            print("Unknown mode. Use: interactive, evaluate, or demo")
    
    else:
        # Default: show usage
        print("RAG Model - Usage:")
        print("  python rag_model.py interactive    # Interactive Q&A session")
        print("  python rag_model.py evaluate       # Batch evaluation on queries file")
        print("  python rag_model.py demo           # Run single demo query")