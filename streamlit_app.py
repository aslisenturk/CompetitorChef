from similarity import load_data, match_data, score_bm25, reciprocal_rank_fusion
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as st
import yaml
import time
import numpy as np


with open('config.yaml', 'r', encoding="utf-8") as f:
    config = yaml.safe_load(f)


def get_scores(t1: pd.DataFrame, t2: pd.DataFrame, e1, e2, threshold):
    """
    Matches product data from two dataframes based on similarity scores.
    Args:
        t1: Dataframe containing first restaurant data.
        t2: Dataframe containing second restaurant data.
        e1: Embeddings of first restaurant data.
        e2: Embeddings of second restaurant data.
        filter (str): Parameter to filter matches.
    Returns:
        pandas.DataFrame: A DataFrame containing the matched product data.
    """
    cosine_score = cosine_similarity(e1, e2)
    bm25_score = score_bm25(t1, t2)

    match_list = []
    for i in range(len(t1)):
        r, s = reciprocal_rank_fusion(cosine_score[i], bm25_score[i], threshold)
        match_list.append([(r[j], s[j]) for j in range(len(r))])
        
    return match_list


def retrieve_results(list1, list2, category_filter):

    # Filter list1 based on category_filter if necessary
    list1 = [item for item in list1 if category_filter == "TÜMÜ" or item.get("category_0") == category_filter]

    # Define CSS style and column configuration outside the loop
    text_format = """<style>.text {border: 1.5px solid #0e78bd; border-radius: 5px; padding: 13px;}</style>"""
    column_config = {"i": " ", "score": " ", "brand_1": "Marka", "product_1": "Ürün", "price_1": "Fiyat", "category_1": "Kategori", "product_detail_1": "Açıklama"}
    column_order=["i", "score", "brand_1", "product_1", "price_1", "product_detail_1", "category_1"]
    
    for i in list1:
        # Retrieve related data from list2
        expander = st.expander(f"{i['product_0']}")
        formatted_text = f"<p class='text'> Fiyat: {i['price_0']} &emsp; Kategori: {i['category_0']} <br> Açıklama: {i['product_detail_0']}</p>"
        st.markdown(text_format, unsafe_allow_html=True)
        expander.markdown(formatted_text, unsafe_allow_html=True)
        expander.dataframe(list2[i.get("index_0")], use_container_width=True, column_config=column_config, column_order=column_order)


def main():

    st.set_page_config(page_title="Divan Menü", page_icon="🌀", layout="wide")

    title_alignment="""<style>
        .title {margin-top: 0; margin-bottom: 2px; padding-bottom: 5px; border-bottom: 1px solid white; font-weight: bold;}
        hr {margin-top: 1px; border-color: #0e78bd; border-bottom-width: 3px;}
        </style>"""

    with st.container():
        st.sidebar.markdown(title_alignment, unsafe_allow_html=True)
        st.sidebar.markdown('<h1 class="title">Divan Menü</h1>', unsafe_allow_html=True)
        st.sidebar.divider()

    # Get restaurant and category selection from sidebar
    restaurant = st.sidebar.selectbox("Restoran", config["table"]["divan_restaurants"], on_change=None)
    category = st.sidebar.selectbox("Kategori", config["table"][f"{config['table']['divan_dict'][restaurant]}_category"], on_change=None)
    threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=0.4, step=0.05, value=0.1)

    restaurant_code = config['table']['divan_dict'][restaurant]

    # if "list1" not in st.session_state:
    #     st.session_state["list1"] = list1

    # if "list2" not in st.session_state:
    #     st.session_state["list2"] = list2

    # Load data and match the products
    t1, t2, e1, e2 = load_data(restaurant_code)
    match_index_list = get_scores(t1, t2, e1, e2, threshold)
    list1, list2 = match_data(config, t1, t2, match_index_list)

    pd.DataFrame(list1).to_parquet("list1.parquet")
    pd.DataFrame(list2).to_parquet("list2.parquet")

    # Submit button for triggering data retrieval
    st.sidebar.button(label="Gönder", key="generate", type="primary", on_click=retrieve_results, 
                      args=(list1, list2, category))
    

if __name__ == "__main__":
    main()
    