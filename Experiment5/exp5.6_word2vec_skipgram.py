
from pathlib import Path
from gensim.models import Word2Vec
import re


def load_corpus(file_path):
    """Read the corpus and convert each line into lowercase tokens."""
    sentences = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().lower()

            if line:
                tokens = re.findall(r"[a-z]+", line)

                if tokens:
                    sentences.append(tokens)

    return sentences


def train_skipgram_model(sentences):
    """Train a Word2Vec model using the Skip-Gram architecture."""
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        seed=42,
        epochs=100
    )

    return model


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

    print("=" * 65)
    print("EXPERIMENT 5.6: WORD2VEC SKIP-GRAM MODEL")
    print("=" * 65)

    print("\nTokenized corpus:")
    for index, sentence in enumerate(sentences, start=1):
        print(f"{index}. {sentence}")

    # Step 2: Train the Skip-Gram model.
    model = train_skipgram_model(sentences)

    print("\nModel training completed.")
    print(f"Vocabulary size: {len(model.wv)}")
    print(f"Vector dimensions: {model.wv.vector_size}")

    # Step 3: Retrieve the vector for 'language'.
    selected_word = "language"

    if selected_word not in model.wv:
        raise ValueError(
            f"'{selected_word}' is not present in the vocabulary."
        )

    vector = model.wv[selected_word]

    print(f"\nVector representation of '{selected_word}':")
    print(vector)

    # Step 4: Find the five most similar words.
    print(f"\nFive most similar words to '{selected_word}':")

    similar_words = model.wv.most_similar(
        selected_word,
        topn=5
    )

    for word, similarity in similar_words:
        print(f"{word:<20} similarity = {similarity:.4f}")

    # Step 5: Save the trained model.
    model_path = file_path.with_name("word2vec_skipgram.model")
    model.save(str(model_path))

    print(f"\nModel saved to: {model_path.name}")


if __name__ == "__main__":
    main()
