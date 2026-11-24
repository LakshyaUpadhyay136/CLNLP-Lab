from pathlib import Path
from gensim.models import Word2Vec
import re


def load_corpus(file_path):
    """
    Read corpus.txt and convert each line into a list of
    lowercase word tokens.
    """
    sentences = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().lower()

            if line:
                # Keep alphabetic words and contractions-free tokens.
                tokens = re.findall(r"[a-z]+", line)

                if tokens:
                    sentences.append(tokens)

    return sentences


def train_cbow_model(sentences):
    """
    Train a Word2Vec CBOW model.
    sg=0 selects CBOW; sg=1 would select Skip-Gram.
    """
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


def main():
    # Find corpus.txt in the same directory as this script.
    file_path = Path(__file__).with_name("corpus.txt")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path.name}. "
            "Place corpus.txt beside this Python script."
        )

    # Step 1: Load and preprocess the corpus.
    sentences = load_corpus(file_path)

    print("=" * 65)
    print("EXPERIMENT 5.5: WORD2VEC CBOW MODEL")
    print("=" * 65)

    print("\nTokenized corpus:")
    for index, sentence in enumerate(sentences, start=1):
        print(f"{index}. {sentence}")

    # Step 2: Train the CBOW model.
    model = train_cbow_model(sentences)

    print("\nModel training completed.")
    print(f"Vocabulary size: {len(model.wv)}")
    print(f"Vector dimensions: {model.wv.vector_size}")

    # Step 3: Select a word and display its vector.
    selected_word = "machine"

    if selected_word not in model.wv:
        raise ValueError(
            f"'{selected_word}' is not present in the vocabulary."
        )

    vector = model.wv[selected_word]

    print(f"\nVector representation of '{selected_word}':")
    print(vector)

    # Step 4: Find the three most similar words.
    print(f"\nThree most similar words to '{selected_word}':")

    similar_words = model.wv.most_similar(
        selected_word,
        topn=3
    )

    for word, similarity in similar_words:
        print(f"{word:<20} similarity = {similarity:.4f}")

    # Step 5: Save the trained model for reuse.
    model_path = file_path.with_name("word2vec_cbow.model")
    model.save(str(model_path))

    print(f"\nModel saved to: {model_path.name}")


if __name__ == "__main__":
    main()
