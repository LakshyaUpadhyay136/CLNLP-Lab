
import sys
from transformers import AutoTokenizer

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_pretrained_bpe():
    # 1. Load the pretrained GPT-2 BPE tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Sample sentence from input_sub_word_data.txt
    sentence = (
        "Natural language processing combines linguistics, computer science, "
        "and machine learning to process large amounts of natural language data."
    )

    print("=" * 60)
    print("EXPERIMENT 5.1 (a): PRETRAINED BPE TOKENIZER (GPT-2)")
    print("=" * 60)
    print(f"Input Sentence:\n{sentence}\n")

    # 3. Tokenize into subword units
    tokens = tokenizer.tokenize(sentence)

    # 4. Get corresponding token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 5. Display tokens and token IDs
    print(f"{'Index':<6} {'Subword Token':<25} {'Token ID':<10}")
    print("-" * 45)

    for idx, (token, token_id) in enumerate(zip(tokens, token_ids)):
        print(f"{idx + 1:<6} {token:<25} {token_id:<10}")


if __name__ == "__main__":
    run_pretrained_bpe()
