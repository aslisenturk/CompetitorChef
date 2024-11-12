import streamlit as st
import itertools
from collections import OrderedDict  # Use OrderedDict to preserve insertion order


# Create the first dictionary with 5 sentences
dict1 = {
    0: "This is the first sentence.",
    1: "This is the second sentence.",
    2: "This is the third sentence.",
    3: "This is the fourth sentence.",
    4: "This is the fifth sentence."
}

# Create the second dictionary with 15 sentences
dict2 = {
    0: "Another sentence here.",
    1: "More sentences...",
    2: "This is a sentence about nature.",
    3: "And another one about technology.",
    4: "But let's not forget about art and culture.",
    5: "Music fills the air with its vibrant melodies.",
    6: "The brushstrokes on a canvas sing a silent song.",
    7: "Sculptures capture the essence of movement and time.",
    8: "Poetry whispers secrets between the lines.",
    9: "Every story unfolds like a map of possibilities.",
    10: "And within each experience, wisdom grows.",
    11: "The laughter of children echoes through the playground.",
    12: "Curiosity guides our footsteps towards discovery.",
    13: "Challenges become stepping stones to greater heights.",
    14: "Dreams shimmer like distant stars, guiding us forward.",
    15: "With determination, we carve our own path in the world."
}

new_dict = {}

# Keep track of the current index in dict2
dict2_index = 0

# Iterate over values in dict1
for value1 in dict1.values():
    # Extract the next 3 values from dict2 based on the index
    values2 = list(itertools.islice(dict2.values(), dict2_index, dict2_index + 3))
    # Update the index for the next iteration
    dict2_index += 3
    # Add the key-value pair to the new dictionary
    new_dict[value1] = values2

# new_dict = OrderedDict()

# # Iterate over values in dict1
# for value1 in dict1.values():
#     # Create an empty list to store values from dict2
#     values2 = []
#     # Add the first 3 values from dict2 to the list
#     values2.extend(list(itertools.islice(dict2.values(), 3)))
#     # Add the key-value pair to the new dictionary
#     new_dict[value1] = values2

# # Create a new dictionary with 3 values from dict2 for each key from dict1, preserving order
# new_dict = {}
# for value1 in dict1.values():
#     values2 = list(itertools.islice(dict2.values(), 3))  # Get 3 values from dict2 in order
#     new_dict[value1] = values2




# new_dict = {value1: list(itertools.islice(dict2.values(), 3)) for value1 in dict1.values()}

# # Add quotation marks and indentation for readability (optional)
# for key, value in new_dict.items():
#     new_dict[key] = [f'"{v}"' for v in value]
#     new_dict[key] = "\n  ".join(new_dict[key])



st.write(new_dict)
