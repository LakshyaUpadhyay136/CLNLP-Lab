from pathlib import Path
from collections import Counter

# Locate the input file in the same folder as this script
file_path = Path(__file__).with_name("input_sub_word_data.txt")

# Read the corpus
text = file_path.read_text(encoding="utf-8")

# Convert text to lowercase and split into words
words = text.lower().split()
# print(words)

# Represent each word as a list of characters
corpus = [list(word) for word in words]

print("First 5 words represented as characters:")
for word in corpus[:5]:
    print(word)

# GETTING STATS FOR THE CORPUS
def get_stats(corpus):
    # """Return a Counter of all symbol pairs in the corpus."""
    # Count all adjacent character pairs
    pairs = Counter()
    for word in corpus:
        for i in range(len(word) - 1):
            pairs[word[i], word[i+1]] += 1
    return pairs

pair_counts = get_stats(corpus)

print("\nMost frequent adjacent pairs:")
for pair, frequency in pair_counts.most_common(10):
    print(pair, ":", frequency)

# MERGING PAIRS THAT ARE MOST FREQUENT
def merge_pair(corpus, pair):
    merged_corpus = []

    for word in corpus:
        new_word = []
        i = 0

        while i < len(word):
            # If the current and next symbols match the pair
            if (
                i < len(word) - 1
                and word[i] == pair[0]
                and word[i + 1] == pair[1]
            ):
                # Merge both symbols into one
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        merged_corpus.append(new_word)

    return merged_corpus

def encode_word(word, merge_rules):
    symbols = list(word)

    for pair in merge_rules:
        symbols = merge_pair([symbols], pair)[0]

    return symbols

# TRAINING BPE
def train_bpe(corpus, num_merges):
    merge_rules = []

    for step in range(num_merges):
        # Count adjacent pairs in the current corpus
        pair_counts = Counter()

        for word in corpus:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += 1

        # Stop if no pairs remain
        if not pair_counts:
            break

        # Select the most frequent pair
        best_pair = pair_counts.most_common(1)[0][0]

        # Merge that pair throughout the corpus
        corpus = merge_pair(corpus, best_pair)

        # Save the learned merge rule
        merge_rules.append(best_pair)

        print(f"Merge {step + 1}: {best_pair}")

    return corpus, merge_rules

# example_corpus = [
#     list("tree"),
#     list("trees"),
#     list("tree"),
#     list("train")
# ]

# trained_corpus, merge_rules = train_bpe(
#     example_corpus,
#     num_merges=5
# )

# print("\nFinal corpus:")
# print(trained_corpus)

# print("\nLearned merge rules:")
# print(merge_rules)


# Load and prepare the actual corpus
file_path = Path(__file__).with_name("input_sub_word_data.txt")
text = file_path.read_text(encoding="utf-8").lower()

words = text.split()
corpus = [list(word) for word in words]

# Train BPE by learning 100 merge rules
trained_corpus, merge_rules = train_bpe(corpus, num_merges=150)

print("\nTotal merge rules learned:", len(merge_rules))

print("\nFirst 20 learned merge rules:")
for rule in merge_rules[:20]: 
    print(rule)


# Build a vocabulary from the learned subword units
vocab = {}

for word in trained_corpus:
    for token in word:
        if token not in vocab:
            vocab[token] = len(vocab)

print("\nVocabulary size:", len(vocab))

print("\nFirst 20 vocabulary entries:")
for token, token_id in list(vocab.items())[:20]:
    print(f"Token: {token!r}, ID: {token_id}")

for word in words[:30]:
    tokens = encode_word(word, merge_rules)
    token_ids = [vocab[token] for token in tokens]

    print(f"Word: {word}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")