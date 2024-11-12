import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
import re


def get_product_data(table: pd.DataFrame, row_index: int, j: int) -> dict:
    """
    Retrieves product data from a table and constructs a dictionary with tailored keys.
        Args:
            table: A pandas DataFrame containing product information.
            row_index: An integer or string representing the index of the row to retrieve data from. If " ", empty values are returned.
            j: An integer indicating the position of the product within a sequence (used for key formatting).
        Returns:
            A dictionary containing product information with keys formatted based on the value of j:
            - If j is 0, the keys are 'product_0', 'price_0', and 'product_detail_0'.
            - If j is not 0, the keys are 'brand_j', 'product_j', 'price_j', and 'product_detail_j'.
    """
    row_data = table.iloc[row_index]
    brand = row_data['brand']
    product = row_data['product']
    product_detail = row_data['product_detail']
    price = row_data['price']
    category = row_data['category']

    if j==0:
        data = {
            f'index_{j}': row_index,
            f'product_{j}': product,
            f'price_{j}': price,
            f'category_{j}': category,
            f'product_detail_{j}': product_detail,
        }
    else:
        data = {
            f'brand_{j}': brand,
            f'category_{j}': category,
            f'product_{j}': product,
            f'price_{j}': price,
            f'product_detail_{j}': product_detail,
        }
        
        if row_data["quantity_type"] is not None:
            quantity_type = row_data["quantity_type"]
            data[f'quantity_{j}'] = quantity_type
    
    return data


def match_data(config, t1: pd.DataFrame, t2: pd.DataFrame, matches: list) -> pd.DataFrame:
# def match_data(config, t1: pd.DataFrame, t2: pd.DataFrame, matches: list, category_map) -> pd.DataFrame:

    """
    Matches product data from two datasets based on similarity scores and filter by restaurant district.
    Args:
        t1 (pandas.DataFrame): The first dataset.
        t2 (pandas.DataFrame): The second dataset.
        similarity_score (numpy.ndarray): The similarity matrix.
        matches (list): A list of indices indicating matching pairs.
        restaurant (str): The name of the restaurant.
    Returns:
        pandas.DataFrame: A DataFrame containing the matched product data.
    """
    categories = {k: set(v) for k, v in config["category"]["categories"].items()}
    # categories = {k: set(v) for k, v in category_map.items()}

    category1_series = t1['category'].str.lower().str.strip()
    matched_data1 = [get_product_data(t1, i, 0) for i in range(len(matches))]

    matched_data2 = []
    for index, match in enumerate(matches):
        category1 = category1_series.iloc[index]
        data = []
        i = 0
        for m in match:
            data_two = get_product_data(t2, m[0], 1)
            c2 = str(data_two["category_1"]).lower().strip()
            if c2 in categories.get(category1, {}): # or category1 in ['yeni̇ veganlar']:
                i += 1
                data.append({"i": i, "index": index, "score": m[1], **data_two})
                if i >= 4:
                    break

        matched_data2.append(data)

    return matched_data1, matched_data2
    # return pd.DataFrame(matched_data1), pd.DataFrame(matched_data2)


def custom_join(row):
    """
    Joins column values with descriptive labels, applying formatting and cleaning.
    Args:
        row: A pandas Series containing the values to be joined.
    Returns:
        A string containing the joined text with descriptive labels and appropriate formatting.
    # """
    columns = ["product", "sector", "category", "product_detail", "quantity_type"]
    for col in columns:
        row[col] = row[col].lower().replace('.', ',').replace('\n', ' ').replace(r'\s+', ' ').replace('®', '')
    row["product"] = row["product"].replace(" (", ",").replace(" )", "").replace(" / ", "/")
    text = "sektör: {}, kategori: {}, ürün: {}".format(row['sector'], row['category'], row['product']).strip()

    if row["product_detail"] != " ":
        row["product_detail"] = row["product_detail"].replace(";", ",").replace("/", ",").replace(" &", ",")
        text += ", açıklama: {}".format(row["product_detail"]).strip()

    text += ", fiyat: {},".format(row["price"]).rstrip()

    if row["quantity"] != 0:
        text += " miktar: {} {}".format(row["quantity"], row["quantity_type"])
    
    text = re.sub(r"[,\s]+$", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.rstrip()


def merge_rows(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Combines rows from two DataFrames into a single DataFrame.
    Args:
        df1: A pandas DataFrame.
        df2: A pandas DataFrame.
    Returns:
        A pandas DataFrame containing the combined rows from df1 and df2.
    """
    rows = []
    for index, row in df1.iterrows():
        rows.append(row)
    for index, row in df2.iterrows():
        rows.append(row)

    return pd.DataFrame(rows)


def prepare_df(df):

    df = pd.DataFrame(df)
    df["quantity"].fillna(0, inplace=True)
    df["quantity"] = df["quantity"].astype(int)  
    df["price"] = df["price"].astype(int) 
    df.fillna(" ", inplace=True)
    df['combined_text'] = df.apply(custom_join, axis=1)
    df["index"] = range(len(df))
    df = df.set_index("index")

    return df

def score_bm25(t1, t2):
    df1 = prepare_df(t1)
    df2 = prepare_df(t2)

    tokenized_corpus = [doc.split() for doc in df2["combined_text"]]
    bm25_model = BM25Okapi(tokenized_corpus)
    bm25_scores_list = [bm25_model.get_scores(doc.split()) for doc in df1["combined_text"]]
    
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
    # filtered_ranking = combined_ranking[combined_ranking['rrf_score'] > threshold]
    filtered_ranking = combined_ranking[:30]


    return filtered_ranking.index, np.array(filtered_ranking["rrf_score"])


@st.cache_data
def load_data(restaurant_code: str):
    """
    Loads data and embeddings from paths got from restaurant name.
    Args:
        restaurant_code: Name of the restaurant to get the data of.
    Returns:
        tuple: A tuple containing the loaded tables and embeddings.
    """
    table_path1 = f"new_tables/{restaurant_code}.parquet"
    table_path2 = "new_tables/other_tables.parquet"

    t1 = pd.read_parquet(table_path1)
    t2 = pd.read_parquet(table_path2)

    embedding_path1 =f"new_tables/{restaurant_code}.npy"
    embedding_path2 = "new_tables/other_tables.npy"

    # Load the data and embeddings
    embedding1 = np.load(embedding_path1)
    embedding2 = np.load(embedding_path2)
    
    # Reshape embeddings to match data dimensions
    embedding1 = embedding1.reshape(len(t1), -1)
    embedding2 = embedding2.reshape(len(t2), -1)

    # Ensure embeddings match data dimensions
    assert len(embedding1) == len(t1), "Embedding1 and t1 sizes mismatch"
    assert len(embedding2) == len(t2), "Embedding2 and t2 sizes mismatch"
    
    return t1, t2, embedding1, embedding2
