import numpy as np
from sklearn.decomposition import PCA  # For dimensionality reduction
from rank_bm25 import BM25Okapi
import numpy as np
import collections
import pandas as pd

def get_bm25_scores(t1, t2, columns, query, idf_dict, field="product", term_weights=None):
    
    df1 = prepare_df(t1, columns)
    df2 = prepare_df(t2, columns)
    
    product_corpus = [doc.split() for doc in df2["product"]]
    category_corpus = [doc.split() for doc in df2["category"]]
    product_detail_corpus = [doc.split() for doc in df2["product_detail"]]
    combined_corpus = product_corpus + category_corpus + product_detail_corpus

    bm25_model = BM25Okapi(combined_corpus, idf=idf_dict)

    tokenized_query = query.split()
    if field in ["product", "category", "product_detail"]:
        bm25_model.idf = locals()[field + "_idf"]  # Apply field-specific IDF
    if term_weights is not None:
        bm25_model.idf *= term_weights  # Apply term weights
    scores = bm25_model.get_scores(tokenized_query)

    return scores

def calculate_idf(corpus):
    term_doc_counts = collections.defaultdict(int)  # Count doc occurrences of each term
    doc_count = len(corpus)  # Total number of documents

    for doc in corpus:
        for term in set(doc):
            term_doc_counts[term] += 1

    idf_dict = {term: np.log((doc_count - df + 1) / df) for term, df in term_doc_counts.items()}
    return idf_dict

def prepare_df(t, columns):

    df = t[columns]
    df = pd.DataFrame(df)
    df.fillna("", inplace=True)
    # df['combined_text_bm25'] = df.apply(lambda row: format_rows(row, columns), axis=1)
    df['combined_text_bm25'] = df.apply(lambda row: custom_join(row), axis=1)

    df["index"] = range(len(df))
    df = df.set_index("index")

    return df

def score_bm25(t1, t2, columns):

    df1 = prepare_df(t1, columns)
    df2 = prepare_df(t2, columns)

    product_corpus = [doc.split() for doc in df2["product"]]
    category_corpus = [doc.split() for doc in df2["category"]]
    product_detail_corpus = [doc.split() for doc in df2["product_detail"]]


    product_name_idf = calculate_idf(product_name_corpus)
    description_idf = calculate_idf(description_corpus)
    ingredients_idf = calculate_idf(ingredients_corpus)
    np.log((total_documents - field_document_frequency + 0.5) / (field_document_frequency + 0.5))
    
    # ... existing code for bm25 model creation
    bm25_model = BM25Okapi(tokenized_corpus)

    bm25_scores_list = []
    for doc in df1["combined_text_bm25"]:
        # Access and apply relevant field-specific IDF based on the column
        field_idf = description_idf if column == "description" else product_name_idf # ... adapt for other fields
        bm25_model.idf = field_idf
        bm25_scores_list.append(bm25_model.get_scores(doc.split()))











# Sample product data
products = []

# Get embeddings for all products
product_embeddings = (products)

# Calculate unweighted cosine similarity
cosine_similarities = np.dot(product_embeddings, product_embeddings.T)

# Analyze dimension contributions using PCA (example)
pca = PCA(n_components=10)  # Reduce to 10 dimensions
reduced_similarities = pca.fit_transform(cosine_similarities)

# Identify important dimensions (example)
important_dimensions = np.argsort(np.abs(pca.components_[0]))[::-1][:3]  # Top 3 dimensions

# Assign weights (example)
weights = np.zeros_like(product_embeddings[0])
weights[important_dimensions] = 2  # Assign higher weights to important dimensions

# Calculate weighted cosine similarity
weighted_cosine_similarities = np.dot(product_embeddings * weights, product_embeddings.T * weights)

# Use the weighted_cosine_similarities for product matching and evaluation
