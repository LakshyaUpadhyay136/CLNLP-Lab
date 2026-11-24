import spacy

# Load the trained English language pipeline.
nlp = spacy.load("en_core_web_sm")

sentence = (
    "The young student is reading an interesting book "
    "in the library."
)


def run_pos_tagging():
    doc = nlp(sentence)

    # Keep each token only once, preserving its original order.
    unique_tokens = []
    seen = set()

    for token in doc:
        if token.text not in seen:
            unique_tokens.append(token)
            seen.add(token.text)

    print("=" * 75)
    print("EXPERIMENT 5.3(a): POS TAGGING USING SPACY")
    print("=" * 75)
    print(f"Input sentence: {sentence}\n")

    print(f"{'Token':<18}{'POS Tag':<15}{'Description'}")
    print("-" * 65)

    for token in unique_tokens:
        pos_tag = token.pos_
        description = spacy.explain(pos_tag) or "No description available"

        print(f"{token.text:<18}{pos_tag:<15}{description}")

    print(f"\nTotal unique tokens: {len(unique_tokens)}")


if __name__ == "__main__":
    run_pos_tagging()
