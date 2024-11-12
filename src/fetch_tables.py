from database import Database
from similarity import custom_join, merge_rows
import pandas as pd
import yaml


with open('config.yaml', 'r', encoding="utf-8") as f:
    config = yaml.safe_load(f)


def fetch_divan_tables(tables, columns):
    """
    Fetches data from specified tables form the database and processes it, and saves as a Parquet file.
    Args:
        tables: A dictionary containing lists of table names, and a list of columns to fetch.
    Returns:
        A pandas DataFrame containing the processed data.
    """
    db = Database()
    data = pd.DataFrame()
    conn = db.connect_database(config)

    for table in tables:

        temp = db.fetch_data(conn, 
                         query=f"""select distinct * from daas.{table} where date=current_date""", 
                         columns=columns)
        print(f"Fetched data from {table}.")

        data = merge_rows(data, temp)
    
    db.close_database(conn)

    # Combine the relevant columns into a single text column and preprocess the text
    data[config["table"]["relevant_columns"]].fillna('')
    data['combined_text'] = data.apply(custom_join, axis=1)
    data["index"] = range(len(data))
    data = data.set_index("index")

    print("Columns merged.")

    data.to_parquet(f"new_tables/{table}.parquet", index=True)

    return data


def fetch_other_tables(tables):
    """
    Fetches data from specified tables form the database, merges and processes it, and saves as a Parquet file.
    Args:
        tables: A dictionary containing lists of table names, and a list of columns to fetch.
    Returns:
        A pandas DataFrame containing the processed data.
    """
    data = pd.DataFrame()
    db = Database()
    conn = db.connect_database(config)

    for table in tables["other_tables"]:

        temp = db.fetch_data(conn, 
                     query=f"""select distinct * from daas.{table} where date=current_date""", 
                     columns=tables["columns"])
        print(f"Fetched data from {table}.")

        data = merge_rows(data, temp)

    print("Merged rows.")

    for table in tables["more_tables"]:

        temp = db.fetch_data(conn,
                             query=f"""select distinct on (product) * from daas.{table} where date=current_date""",
                             columns=tables["columns"])
        print("Fetched more data.")

        data = merge_rows(data, temp)
        
    print("Merged more rows.")

    db.close_database(conn)

    # Combine the relevant columns into a single text column and preprocess the text
    data["quantity"] = pd.to_numeric(data["quantity"])
    data["extra_price"] = pd.to_numeric(data["extra_price"])
    data[config["table"]["relevant_columns"]].fillna('')

    data['combined_text'] = data.apply(custom_join, axis=1)
    data["index"] = range(len(data))
    data = data.set_index("index")

    print("Columns merged.")

    data.to_parquet("new_tables/other_tables.parquet", index=True)

    return data
