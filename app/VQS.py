# Imports:
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import pickle


# Load a pre-trained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text):
    """Generate embedding for a given text using the sentence transformer model."""
    return model.encode(text, convert_to_tensor=True)


# Optimizing Reloading with Pickle
# Embedding the text took around 7min. 
# Since this process might be time consuming I decied to 
# pickle the dataframe with the embedded columns using Python’s `pickle` module.  

# The full data exploration and embeding happens in the "vendor_qualification_System.ipynb" notebook 

# Load data with embedded columns
with open("data/vendors_data_with_embeddings.pkl", "rb") as f:
    embedded_vendors_data = pickle.load(f)



# Similarity computation of vectorized fetures and vectorized input
def compute_feature_similarities(capability_embeddings, vendor_feature_embeddings):
    """Compute pairwise similarity scores between user capabilities and vendor features."""
    if not vendor_feature_embeddings:
        return []  # Return an empty list if no features are available

    vendor_feature_embeddings = torch.stack(vendor_feature_embeddings)  # Convert list to tensor
    similarity_matrix = util.pytorch_cos_sim(torch.stack(capability_embeddings), vendor_feature_embeddings)

    return similarity_matrix.tolist()  # Keeping as list of lists for now


def compute_similarity(input_embedding, vendor_embeddings):
    """ Compute cosine similarity between software_category and (main_category + categories_text) """
    similarity_scores = util.pytorch_cos_sim(input_embedding, vendor_embeddings)
    return similarity_scores.squeeze().tolist()


# User Inputs
software_category = input("Enter the software category: ")
capabilities = input("Enter the capabilities (comma-separated): ").split(",")

# Generate embeddings input embeddings
software_category_embedding = get_embedding(software_category)  # Already a tensor
capability_embeddings = [get_embedding(feature) for feature in capabilities]  # List of tensors