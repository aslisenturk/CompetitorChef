import pandas as pd
import numpy as np
import psycopg2 as pg
from similarity import custom_join, merge_rows
from typing import Dict
import yaml


with open('config.yaml', 'r', encoding="utf-8") as f:
    config = yaml.safe_load(f)


class Database():

    def __init__(self, ):
        pass


    def connect_database(self, config: Dict):
        """
        Creates a database connection using provided configuration.
        Returns:
        conn: An instance of the connection.
        """
        try:
            print("Connecting to database...")
            # Connect to postgres DB
            conn = pg.connect(
                host=config["database"]["DB_PATH"],
                database=config["database"]["DB_NAME"],
                port=config["database"]["DB_PORT"],
                user=config["database"]["DB_USER"],
                password=config["database"]["DB_PASSWORD"]
            )
            print("Connected.")

        except (Exception, pg.Error) as err:
            print(err)

        return conn


    def fetch_data(self, conn, query, columns):
        """
        Creates a database connection using provided config["database"]uration.
        Args:
        conn: An instance of the connection.
        query: The query to be executed.
        columns: Names of the columns for the dataframe.        
        Returns:
        df: The dataframe from query result.
        """
        # Open cursor to perform database operations
        cur = conn.cursor()
        try:
            # Execute query and retrieve query results
            cur.execute(query)
            tuples_list = cur.fetchall()

        except (Exception, pg.DatabaseError) as err:
            print(err)
        
        cur.close()
        # Transform the list into a pandas DataFrame
        df = pd.DataFrame(tuples_list, columns=columns)

        return df


    def close_database(self, conn): 
        """
        Closes the database connection.
        Args:
        conn: An instance of the connection. 
        """
        if conn is not None: 
            conn.close()
            print('Connection closed.')


def fetch_divan_tables(table, columns):
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

    # for table in tables:

    data = db.fetch_data(conn, 
                     query=f"""select distinct * from daas.{table} where date=current_date""", 
                     columns=columns)
    print(f"Fetched data from {table}.")

    # data = merge_rows(data, temp)
    
    db.close_database(conn)

    # Combine the relevant columns into a single text column and preprocess the text
    # data[config["table"]["relevant_columns"]].fillna('')
    # data['combined_text'] = data.apply(custom_join, axis=1)
    # data["index"] = range(len(data))
    # data = data.set_index("index")

    # print("Columns merged.")

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
    data['extra_price'] = data['extra_price'].astype(str)

    # data["extra_price"] = pd.to_numeric(data["extra_price"])
    # data[config["table"]["relevant_columns"]].fillna('')

    # data['combined_text'] = data.apply(custom_join, axis=1)
    # data["index"] = range(len(data))
    # data = data.set_index("index")

    print("Columns merged.")

    data.to_parquet("new_tables/other_tables.parquet", index=True)

    return data