# Imports:
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import torch
from helper_functions import extract_high_scoring_features, compute_feature_similarities, compute_similarity, get_embedding, get_user_input
from tabulate import tabulate


# Optimizing Reloading with Pickle
# Embedding the text took around 7min. 
# Since this process might be time consuming I decied to 
# pickle the dataframe with the embedded columns using Python’s `pickle` module.  

# The full data exploration and embeding happens in the "vendor_qualification_System.ipynb" notebook 


# STEP 1: Load Data (Load data with embedded columns)
print("Loading Vendor Data Embeddings")
with open("data/vendors_data_with_embeddings.pkl", "rb") as f:
    embedded_vendors_data = pickle.load(f)
print("Vendor Data Embeddings Loaded Successfully!")

# STEP 2: Get user input (or test cases)
software_category, capabilities = get_user_input()
# # Print input values
print("\nSelected Input:")
print(f"Software Category: {software_category}")
print(f"Capabilities: {capabilities}")


# STEP 3: Generate input embeddings
software_category_embedding = get_embedding(software_category)  # Already a tensor
capability_embeddings = [get_embedding(feature) for feature in capabilities]  # List of tensors (list of all features/capabilites)


# STEP 4.1: Compute category similarity
embedded_vendors_data["category_similarity"] = embedded_vendors_data.apply(
    lambda row: max(compute_similarity(software_category_embedding, 
                                       torch.stack([row["main_category_embedding"], row["categories_text_embedding"]]))),
    axis=1
) 
# STEP 4.2: Compute feature similarity (list of scores for each vendor)
embedded_vendors_data["feature_similarities"] = embedded_vendors_data["feature_embeddings"].apply(
    lambda feature_emb: compute_feature_similarities(capability_embeddings, feature_emb)
)


# STEP 5: Filter any vender under the threshold of 0.6
filtered_vendors = embedded_vendors_data[
    embedded_vendors_data["feature_similarities"].apply(
        # Flatten nested lists, compare each feature/capability score with the feature the user wants
        lambda scores: any(score >= 0.6 for row in scores for score in row)  
    )
].copy()

ranked_vendors = filtered_vendors.copy()


# STEP 6.1: Weigthed feature similarity for final ranking
ranked_vendors.loc[:, "weighted_feature_similarity"] = ranked_vendors["feature_similarities"].apply(
    lambda scores: sum(score for row in scores for score in row) / sum(len(row) for row in scores) if scores else 0
)

# STEP 6.2: Normalize vendor ratings between 0 and 1
if not ranked_vendors.empty:
    min_rating = ranked_vendors["rating"].min()
    max_rating = ranked_vendors["rating"].max()
    if max_rating > min_rating:  # Avoid division by zero
        ranked_vendors.loc[:, "normalized_rating"] = ranked_vendors["rating"].apply(
            lambda r: (r - min_rating) / (max_rating - min_rating)
        )
    else:
        ranked_vendors.loc[:, "normalized_rating"] = 0
else:
    ranked_vendors["normalized_rating"] = []

# STEP 7: Final Ranking Score (70% feature similarity, 30% rating)
ranked_vendors.loc[:, "final_score"] = (
    0.7 * ranked_vendors["weighted_feature_similarity"] + 
    0.3 * ranked_vendors["normalized_rating"]
)

# Sort based on final score
ranked_vendors = ranked_vendors.sort_values(by="final_score", ascending=False)


# Store matched features
ranked_vendors.loc[:, "matched_features"] = ranked_vendors.apply(
    lambda row: extract_high_scoring_features(row["feature_names"], row["feature_similarities"], 0.6), axis=1
)

# relevant columns for output
top_vendors = ranked_vendors[["seller", "final_score", "weighted_feature_similarity", "category_similarity", "rating", "matched_features"]]

# Display top vendors in a well-structured table
if not top_vendors.empty:
    print("\n📌 **Top Software Vendors Ranked by Relevance**\n")
    print(tabulate(top_vendors.head(10), headers="keys", tablefmt="fancy_grid", showindex=False, floatfmt=".4f"))
else:
    print("\n⚠️ No vendors met the similarity threshold.")
