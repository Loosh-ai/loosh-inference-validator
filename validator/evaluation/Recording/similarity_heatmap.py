import numpy as np
import matplotlib.pyplot as plt

#import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import os

# Default model for backward compatibility
_default_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
_model = None

def _get_model(model_name: str = None) -> SentenceTransformer:
    """Get or create sentence transformer model instance."""
    global _model
    model_to_use = model_name or _default_model_name
    
    # Only reuse cached model if it matches the requested model
    if _model is None or (hasattr(_model, 'model_name_or_path') and _model.model_name_or_path != model_to_use):
        _model = SentenceTransformer(model_to_use)
    
    return _model

def generate_similarity_heatmap(responses: list[str], save_path: str = "./similarity_heatmap.png") -> None:
    """
    Generates a cosine similarity heatmap from a list of text responses and saves it to the specified path.

    Args:
        responses (list[str]): A list of text responses.
        save_path (str): File path to save the heatmap image.
    """

    # Step 1: Convert text to TF-IDF embeddings
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(responses).toarray()

    # Step 2: Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Step 3: Plot heatmap (matplotlib)
    plot_similarity_imshow(similarity_matrix, save_path, title="Cosine Similarity Between Responses")

    # Step 4: Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Step 5: Save heatmap to file
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # Example responses
    example_responses = [
        "Yes, that’s correct.",
        "Absolutely, I agree.",
        "No, that’s not true.",
        "Yes.",
        "I don't think so."
    ]

    # Generate and save heatmap
    output_path = "./output/similarity_heatmap.png"
    generate_similarity_heatmap(example_responses, save_path=output_path)

    print(f"Similarity heatmap saved to: {output_path}")


def generate_semantic_similarity_heatmap(
    responses: list[str],
    save_path: str = "./semantic_similarity_heatmap.png",
    model_name: str = None
) -> None:
    """
    Generates a semantic similarity heatmap using sentence embeddings and saves it as an image.

    Args:
        responses (list[str]): List of text responses.
        save_path (str): Path to save the heatmap image.
        model_name (str): Name of the sentence-transformers model to use. 
                         Defaults to 'sentence-transformers/all-MiniLM-L6-v2' if not provided.
    """
    
    # Get model instance
    model = _get_model(model_name)

    # Compute sentence embeddings
    embeddings = model.encode(responses, convert_to_numpy=True)

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to file
    plt.savefig(save_path)
    plt.close()


def plot_similarity_imshow(similarity_matrix: np.ndarray, save_path: str, title: str = "Cosine Similarity Between Responses"):
    n = similarity_matrix.shape[0]
    labels = [f"R{i+1}" for i in range(n)]

    plt.figure(figsize=(10, 8))

    # Heatmap
    im = plt.imshow(similarity_matrix, aspect="equal", interpolation="nearest", cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # Axes labels/ticks
    plt.title(title)
    plt.xlabel("Responses")
    plt.ylabel("Responses")
    plt.xticks(range(n), labels, rotation=45, ha="right")
    plt.yticks(range(n), labels)

    # Annotate values (like sns.heatmap(annot=True))
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{similarity_matrix[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)  # bump DPI if you want crisp text
    plt.close()


if __name__ == "__main__":
    example_responses = [
        "Yes, that’s correct.",
        "Absolutely, I agree.",
        "No, that’s not true.",
        "Yes.",
        "I don't think so."
    ]

    output_path = "./output/semantic_similarity_heatmap.png"
    generate_semantic_similarity_heatmap(example_responses, save_path=output_path)

    print(f"Semantic similarity heatmap saved to: {output_path}")


