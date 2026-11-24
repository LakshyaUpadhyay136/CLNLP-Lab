
from pathlib import Path
from gensim.models import Word2Vec
import re


def load_corpus(file_path):
    """Read the corpus and tokenize each line into lowercase words."""
    sentences = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().lower()

            if line:
                tokens = re.findall(r"[a-z]+", line)

                if tokens:
                    sentences.append(tokens)

    return sentences


def train_word2vec_model(sentences):
    """Train a Word2Vec CBOW model."""
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=1,
        sg=0,
        seed=42,
        epochs=100
    )

    return model


def cosine_similarity_manual(vector_a, vector_b):
    """Calculate cosine similarity using the mathematical formula."""
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def main():
    # Locate corpus.txt in the same directory as this script.
    file_path = Path(__file__).with_name("corpus.txt")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path.name}. "
            "Place corpus.txt beside this Python script."
        )

    # Step 1: Load and preprocess the corpus.
    sentences = load_corpus(file_path)

    print("=" * 70)
    print("EXPERIMENT 5.7: WORD2VEC COSINE SIMILARITY")
    print("=" * 70)

    print(f"\nNumber of sentences: {len(sentences)}")

    # Step 2: Train the Word2Vec model.
    model = train_word2vec_model(sentences)

    print("Model training completed.")
    print(f"Vocabulary size: {len(model.wv)}")
    print(f"Vector dimensions: {model.wv.vector_size}")

    # Step 3: Define the word pairs specified in the lab.
    word_pairs = [
        ("machine", "learning"),
        ("natural", "language"),
        ("deep", "learning")
    ]

    # Step 4: Calculate and display cosine similarities.
    print("\nCosine similarity results:")
    print("-" * 70)
    print(f"{'Word 1':<15}{'Word 2':<15}{'Gensim Score':<18}{'Manual Score'}")
    print("-" * 70)

    for word_a, word_b in word_pairs:
        if word_a not in model.wv or word_b not in model.wv:
            print(f"Missing vocabulary word in pair: {word_a}, {word_b}")
            continue

        vector_a = model.wv[word_a]
        vector_b = model.wv[word_b]

        # Gensim's built-in cosine similarity.
        gensim_score = model.wv.similarity(word_a, word_b)

        # Independently calculate the same formula.
        manual_score = cosine_similarity_manual(vector_a, vector_b)

        print(
            f"{word_a:<15}{word_b:<15}"
            f"{gensim_score:<18.4f}{manual_score:.4f}"
        )

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
