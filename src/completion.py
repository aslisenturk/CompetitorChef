import pandas as pd
from gpt_model import get_completion
import yaml
import re

with open('config.yaml', 'r', encoding="utf-8") as f:
    config = yaml.safe_load(f)


def get_category_map(divan, other, chunk_size):
    divan_category = pd.DataFrame(divan["category"])
    divan_category = divan_category.drop_duplicates().reset_index(drop=True)
    divan_category["category"] = divan_category["category"].str.lower().str.strip()
    sublists = get_category_sublists(divan_category, chunk_size=chunk_size)

    other_category = pd.DataFrame(other["category"])
    other_category = other_category.drop_duplicates().reset_index(drop=True)
    other_category["category"] = other_category["category"].str.lower().str.strip()
    other_category_list = other_category["category"].tolist()
    other_category_text = "\n- ".join([""] + other_category_list)

    category_map = {}

    for s in sublists:
        prompt = f"Restoran 1: {s} \n Restoran 2: {other_category_text}"
        message = [{"role": "user", "content": prompt}]
        response = get_completion(config, model="completion", messages=message)
        category_map = {**category_map, **response_to_dict(response)}

    return category_map


def response_to_dict(response):
    response_text = str(response.content)
    lines = str.splitlines(response_text)
    lines = lines[1:]
    categories = {}
    for line in lines:
        line = line.replace("\n", " ")
        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"\(.*?\)", "", line)
        line = line.upper().lower()
        if line == "Restoran 2: ":
            break
        if line == " ":
            continue
        parts = line.split(' - ')
        category = parts[0].strip()
        category = category.replace("-", "").strip()
        
        if len(parts) == 2:
            items = parts[1].split(",")
            for i in range(len(items)):
                items[i] = items[i].strip()
            categories[category] = items

    return categories


def get_category_sublists(df, chunk_size):

    category = pd.DataFrame(df["category"])
    category = category.drop_duplicates().reset_index(drop=True)
    category["category"] = category["category"].str.lower().str.strip()

    sublists = []
    for i in range(0, len(category), chunk_size):
        sub_df = category["category"].iloc[i:i+chunk_size].tolist()
        sub_text = "\n- ".join([""] + sub_df)
        sublists.append(sub_text)

    return sublists

    # category_map = {}
    # for s in sublists:
    #     prompt = f"Restoran 1: {s} \n Restoran 2: {other_category_text}"
    #     message = [{"role": "user", "content": prompt}]
    #     response = get_completion(config, model="completion", messages=message)
    #     category = response_to_dict(response)
    #     category_map = {**category_map, **category}


def merge_dicts(dict1, dict2):
  """
  Merges two dictionaries with unique values and handles missing keys.
  Args:
    dict1: The first dictionary.
    dict2: The second dictionary.
  Returns:
    A new dictionary with merged values.
  """
  merged_dict = {}
  for key, values in dict1.items():
    if key in dict2:
      merged_dict[key] = list(set(values + dict2[key]))  # Merge unique values
    else:
      merged_dict[key] = values

  for key, values in dict2.items():
    if key not in dict1:
      merged_dict[key] = values  # Add keys from dict2 that are not in dict1

  return merged_dict
