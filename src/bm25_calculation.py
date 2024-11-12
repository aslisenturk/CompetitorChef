import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np
import re


def format_rows(row, columns):
    translation_map = str.maketrans({'ë':'e', 'â':'a', 'à':'a', 'è':'e', 'é':'e', 'ê':'e', 'ô':'o'})
    row = row[columns].fillna("").apply(lambda x: x.lower().translate(translation_map).replace('-', ' ').replace(';', ' ').replace('\n', ' '))
    text = ' '.join(row)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_df(t, columns):

    df = t[columns]
    df = pd.DataFrame(df)
    df.fillna("", inplace=True)
    df['combined_text_bm25'] = df.apply(lambda row: format_rows(row, columns), axis=1)
    df["index"] = range(len(df))
    df = df.set_index("index")

    return df

def score_bm25(t1, t2, columns):
    df1 = prepare_df(t1, columns)
    df2 = prepare_df(t2, columns)

    tokenized_corpus = [doc.split() for doc in df2["combined_text_bm25"]]
    bm25_model = BM25Okapi(tokenized_corpus)

    bm25_scores_list = [bm25_model.get_scores(doc.split()) for doc in df1["combined_text_bm25"]]
    
    return np.array(bm25_scores_list).reshape(len(t1), -1)


def reciprocal_rank_fusion(result1: np.array, result2: np.array, threshold: float, k=60):

    """
    Apply Reciprocal Rank Fusion (RRF) to combine rankings from two search methods using numpy arrays.
    This function sorts the input scores before applying RRF.
 
    :param result1: Numpy array of scores from the first search method.
    :param result2: Numpy array of scores from the second search method.
    :param index: Pandas Index or array-like representing document IDs.
    :param k: Constant for RRF, typically set to 60.
    :return: A dataframe with the fused ranking.
    """
    # Convert numpy arrays and index to DataFrames
    df1 = pd.DataFrame({'score': result1}).sort_values(by='score', ascending=False)
    df2 = pd.DataFrame({'score': result2}).sort_values(by='score', ascending=False)
 
    # Assign RRF scores
    df1['rrf_score'] = 1 / (k + df1['score'].rank(method='first', ascending=False))
    df2['rrf_score'] = 1 / (k + df2['score'].rank(method='first', ascending=False))
    
    # Combine scores
    combined_scores = (df1['rrf_score'] + df2['rrf_score'])*10
    combined_ranking = pd.DataFrame({'rrf_score': combined_scores}).sort_values(by='rrf_score', ascending=False)
    # scores = combined_ranking["rrf_score"]
    # return combined_ranking.index, np.array(scores)
    filtered_ranking = combined_ranking[combined_ranking['rrf_score'] > threshold]

    return filtered_ranking.index, np.array(filtered_ranking["rrf_score"])
