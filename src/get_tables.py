from database import fetch_other_tables, fetch_divan_tables
from similarity import prepare_df
from gpt_model import get_embeddings
import numpy as np
import yaml


with open('config.yaml', 'r', encoding="utf-8") as f:
    config = yaml.safe_load(f)


tables = config["table"]
columns = config["table"]["columns"]

# Fetch data and create embeddings for all tables in other_tables merged
other_data = fetch_other_tables(tables)
other_data = prepare_df(other_data)
other_data.to_parquet("new_tables/other_combined.parquet")

combined_text_list = other_data['combined_text'].tolist()
other_embeddings = get_embeddings(config, "embedding", combined_text_list)
print("Embeddings created.")

# Save the numpy array to a .npy file
np.save("new_tables/other_tables.npy", other_embeddings)
print("Saved.")

# # Fetch data and create embeddings for a single table in divan_tables
# table = tables["divan_tables"][0]
# divan_data = fetch_divan_tables(table, columns)
# divan_data = prepare_df(divan_data)
# divan_data.to_parquet("new_tables/divan_combined.parquet")
# combined_text_list = divan_data['combined_text'].tolist()
# embeddings = get_embeddings(config, "embedding", combined_text_list)
# print("Embeddings created.")

# # Save the numpy array to a .npy file
# np.save(f"new_tables/{table}.npy", embeddings)
