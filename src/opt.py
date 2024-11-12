def match_data(config, t1: pd.DataFrame, t2: pd.DataFrame, matches: list) -> pd.DataFrame:
    # Preprocess categories for efficient lookup
    categories = {k: set(v) for k, v in config["category"]["categories"].items()}

    # Process t1 data outside of the loop
    category1_series = t1['category'].str.lower().str.strip()
    matched_data1 = [get_product_data(t1, i, 0) for i in range(len(matches))]

    # Process t2 data
    matched_data2 = []
    for index, match in enumerate(matches):
        category1 = category1_series.iloc[index]
        for m in match:
            data_two = get_product_data(t2, m[0], 1)
            c2 = data_two["category_1"].lower().strip()
            if c2 in categories.get(category1, {}):
                matched_data2.append({"index": index, "score": m[1], **data_two})

    return pd.DataFrame(matched_data1), pd.DataFrame(matched_data2)


#--------------
    matched_data1 = []
    matched_data2 = []
    # temp = {}

    categories = config["category"]["categories"]
    category1 = []
    index = -1

    for i in range(len(matches)):
        data_one = get_product_data(t1, i, 0)
        c1 = str(data_one["category_0"]).lower().strip()
        category1.append(c1)
        matched_data1.append({**data_one})

    for match in matches:
        index += 1
        for m in match:
            data_two =get_product_data(t2, m[0], 1)
            c2 = str(data_two["category_1"]).lower().strip()
            if c2 in categories[f"{category1[index]}"]:
                matched_data2.append({"index": index, "score":m[1], **data_two})
            # if c1 not in categories:
            #     top_5 = dict(islice(data_two.items(), 5))
            #     temp.update({"index": index, "score":m[1], **top_5})
            # print(temp)

            # matched_data2.append({**temp})

        
    return pd.DataFrame(matched_data1), pd.DataFrame(matched_data2)
