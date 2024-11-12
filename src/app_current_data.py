from similarity import match_data, score_bm25, reciprocal_rank_fusion
from sklearn.metrics.pairwise import cosine_similarity
from database import fetch_other_tables, fetch_divan_tables
from gpt_model import get_embeddings
from completion import get_category_map
import pandas as pd
import streamlit as st
import yaml


with open('config.yaml', 'r', encoding="utf-8") as f:
    config = yaml.safe_load(f)


def get_data_embeddings(restaurant):
    restaurant_code = config['table']['divan_dict'][restaurant]

    t1 =fetch_divan_tables(restaurant_code ,columns)
    combined_text_list = t1['combined_text'].tolist()
    e1 = get_embeddings(config, combined_text_list)
    print("Embeddings created.")

    tables = config["table"]
    columns = config["table"]["columns"]
    t2 = fetch_other_tables(tables)
    combined_text_list = t2['combined_text'].tolist()
    e2 = get_embeddings(config, combined_text_list)
    print("Embeddings created.")

    return t1, t2, e1, e2


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
    bm25_score = score_bm25(t1, t2, columns=["sector", "category", "product", "product_detail"])

    matches = []
    match_list = []

    for i in range(len(t1)):
        r, s = reciprocal_rank_fusion(cosine_score[i], bm25_score[i], threshold)
        matches.append([r, s])

        m_list = [(r[j], s[j]) for j in range(len(r))]
        match_list.append(m_list)
        
    return match_list


def retrieve_results(list1, list2, category_filter):

    # Filter list1 based on category_filter if necessary
    list1 = [item for item in list1 if category_filter == "TÜMÜ" or item.get("category_0") == category_filter]

    # Pre-process list2 for efficient matching
    list2_dict = {}
    for item in list2:
        index = item.get("index")
        if index not in list2_dict:
            list2_dict[index] = []
        list2_dict[index].append(item)

    # Define CSS style and column configuration outside the loop
    text_format = """<style>.text {border: 2px solid #0e78bd; border-radius: 7px; background-color: #0e78bd; padding: 13px;}</style>"""
    column_config = {"i": " ", "score": " ", "brand_1": "Marka", "product_1": "Ürün", "price_1": "Fiyat", "category_1": "Kategori", "product_detail_1": "Açıklama"}
    
    for i in list1:
        # Retrieve related data from list2_dict
        filtered_data = list2_dict.get(i.get("index_0"), [])
        expander = st.expander(f"{i['product_0']}")
        formatted_text = f"<p class='text'>{i['product_0']} &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp Fiyat: {i['price_0']} &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp Kategori: {i['category_0']} <br> Açıklama: {i['product_detail_0']}</p>"
        st.markdown(text_format, unsafe_allow_html=True)
        expander.markdown(formatted_text, unsafe_allow_html=True)
        expander.dataframe(filtered_data, column_config=column_config)


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
    threshold = st.sidebar.slider("Threshold", min_value=0.1, max_value=0.4, step=0.05, value=0.2)

    t1, t2, e1, e2 = get_data_embeddings(restaurant)
    match_index_list = get_scores(t1, t2, e1, e2, threshold)
    category_map = get_category_map(t1, t2, chunk_size=10)
    list1, list2 = match_data(config, t1, t2, match_index_list, category_map)

    # Submit button for triggering data retrieval
    st.sidebar.button(label="Gönder", key="generate", type="primary", 
                      on_click=retrieve_results, args=(list1, list2, category, threshold))


if __name__ == "__main__":
    main()
    