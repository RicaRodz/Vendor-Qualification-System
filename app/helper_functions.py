import json
import torch
from sentence_transformers import SentenceTransformer, util


# Load a pre-trained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    """Generate embedding for a given text using the sentence transformer model."""
    return model.encode(text, convert_to_tensor=True)


# Get user input or use test case
def get_user_input():
    test_cases = load_test_cases()
    
    user_choice = input("Enter 1 to provide custom input or press Enter to use a predefined test case: ").strip()

    if user_choice == "1":
        software_category = input("Enter the software category: ").strip()
        capabilities = [cap.strip() for cap in input("Enter capabilities (comma-separated): ").split(",")]
    else:
        print("\nAvailable test cases:")
        for i, test in enumerate(test_cases, start=1):
            print(f"{i}. {test['software_category']} - {test['capabilities']}")

        test_choice = input("\nEnter test case number (or press Enter for the first one): ").strip()
        test_case = test_cases[int(test_choice) - 1] if test_choice.isdigit() and 1 <= int(test_choice) <= len(test_cases) else test_cases[0]

        software_category = test_case["software_category"]
        capabilities = test_case["capabilities"]

    return software_category, capabilities

# Load predefined test cases
def load_test_cases(filename="test/tests_input.json"):
    with open(filename, "r") as f:
        return json.load(f)


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



# store matched features with similarity scores
def extract_high_scoring_features(features, scores, threshold=0.6):
    """Extracts feature names where similarity score is above the threshold."""
    high_score_features = [
        f"{feature} ({score:.2f})"  # Format as "Feature (Score)"
        for feature, row in zip(features, scores)  # Match features with similarity scores
        for score in row if score >= threshold
    ]
    return ", ".join(high_score_features) if high_score_features else "No strong matches"
