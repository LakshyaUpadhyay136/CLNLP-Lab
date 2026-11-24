from pathlib import Path
from transformers import AutoTokenizer


def run_pretrained_sentencepiece():
    # Locate the input file in the same directory as this script.
    file_path = Path(__file__).with_name("input_sub_word_data.txt")

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # Read the complete input file.
    text = file_path.read_text(encoding="utf-8").strip()

    # Load a pretrained tokenizer that uses SentencePiece.
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False
    )

    # Generate subword tokens and corresponding IDs.
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print("=" * 75)
    print("EXPERIMENT 5.2(a): PRETRAINED SENTENCEPIECE TOKENIZER")
    print("=" * 75)
    print(f"Pretrained tokenizer: {model_name}")
    print(f"Input file: {file_path.name}")
    print(f"Total tokens: {len(tokens)}\n")

    print(f"{'Index':<8}{'Subword Token':<35}{'Token ID':<10}")
    print("-" * 53)

    for index, (token, token_id) in enumerate(
        zip(tokens, token_ids), start=1
    ):
        print(f"{index:<8}{token:<35}{token_id:<10}")

    print("\nTokenization completed successfully.")


if __name__ == "__main__":
    run_pretrained_sentencepiece()
