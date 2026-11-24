import nltk
from nltk.tokenize import TreebankWordTokenizer

sentence = (
    "The young student is reading an interesting book "
    "in the library."
)

# Download the pretrained English POS tagger if required.
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# Tokenize without needing Punkt sentence-tokenization data.
tokenizer = TreebankWordTokenizer()


def describe_tag(tag):
    """Return a readable description of common Penn Treebank tags."""
    descriptions = {
        "DT": "Determiner",
        "JJ": "Adjective",
        "JJR": "Comparative adjective",
        "JJS": "Superlative adjective",
        "NN": "Singular or mass noun",
        "NNS": "Plural noun",
        "NNP": "Proper noun, singular",
        "NNPS": "Proper noun, plural",
        "VB": "Base-form verb",
        "VBD": "Past-tense verb",
        "VBG": "Verb, gerund or present participle",
        "VBN": "Verb, past participle",
        "VBP": "Verb, non-3rd-person singular present",
        "VBZ": "Verb, 3rd-person singular present",
        "AUX": "Auxiliary verb",
        "IN": "Preposition or subordinating conjunction",
        "TO": "The word 'to'",
        "RB": "Adverb",
        "RBR": "Comparative adverb",
        "RBS": "Superlative adverb",
        "PRP": "Personal pronoun",
        "PRP$": "Possessive pronoun",
        "CC": "Coordinating conjunction",
        "CD": "Cardinal number",
        "MD": "Modal auxiliary",
        "RP": "Particle",
        "WDT": "Wh-determiner",
        "WP": "Wh-pronoun",
        "WRB": "Wh-adverb",
        ".": "Sentence-final punctuation",
        ",": "Comma",
        ":": "Colon or semicolon",
        "(": "Opening bracket",
        ")": "Closing bracket",
        "''": "Closing quotation mark",
        "``": "Opening quotation mark",
    }

    return descriptions.get(tag, "Other or punctuation tag")


def run_pos_tagging():
    tokens = tokenizer.tokenize(sentence)

    # Apply NLTK's pretrained POS tagger.
    tagged_tokens = nltk.pos_tag(tokens)

    # Keep each token only once, preserving order.
    unique_tagged_tokens = []
    seen = set()

    for token, tag in tagged_tokens:
        if token not in seen:
            unique_tagged_tokens.append((token, tag))
            seen.add(token)

    print("=" * 75)
    print("EXPERIMENT 5.3(b): POS TAGGING USING NLTK")
    print("=" * 75)
    print(f"Input sentence: {sentence}\n")

    print(f"{'Token':<18}{'POS Tag':<15}{'Description'}")
    print("-" * 65)

    for token, tag in unique_tagged_tokens:
        description = describe_tag(tag)
        print(f"{token:<18}{tag:<15}{description}")

    print(f"\nTotal unique tokens: {len(unique_tagged_tokens)}")


if __name__ == "__main__":
    run_pos_tagging()
