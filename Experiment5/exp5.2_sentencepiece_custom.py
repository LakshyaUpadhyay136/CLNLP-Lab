
from pathlib import Path
import sentencepiece as spm


def run_custom_sentencepiece():
    # Locate the input file in the same directory as this script.
    file_path = Path(__file__).with_name("input_sub_word_data.txt")

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8").strip()

    # Train a new SentencePiece BPE tokenizer on the supplied corpus.
    model_prefix = file_path.parent / "sentencepiece_custom"

    spm.SentencePieceTrainer.train(
        input=str(file_path),
        model_prefix=str(model_prefix),
        model_type="bpe",
        vocab_size=300,
        character_coverage=1.0,
        hard_vocab_limit=False
    )

    # Load the tokenizer that was just trained.
    model_path = model_prefix.with_suffix(".model")
    processor = spm.SentencePieceProcessor(
        model_file=str(model_path)
    )

    # Generate subword tokens and their IDs.
    tokens = processor.encode(text, out_type=str)
    token_ids = processor.encode(text, out_type=int)

    print("=" * 75)
    print("EXPERIMENT 5.2(b): SENTENCEPIECE TRAINED FROM SCRATCH")
    print("=" * 75)
    print(f"Input file: {file_path.name}")
    print("Tokenizer algorithm: SentencePiece BPE")
    print(f"Requested vocabulary size: 300")
    print(f"Actual vocabulary size: {processor.get_piece_size()}")
    print(f"Total tokens: {len(tokens)}\n")

    print(f"{'Index':<8}{'Subword Token':<35}{'Token ID':<10}")
    print("-" * 53)

    for index, (token, token_id) in enumerate(
        zip(tokens, token_ids), start=1
    ):
        print(f"{index:<8}{token:<35}{token_id:<10}")

    print("\nTokenization completed successfully.")
    print(f"Saved tokenizer model: {model_path.name}")
    print(f"Saved vocabulary file: {model_prefix.name}.vocab")

pip install spacy nltk
python -m spacy download en_core_web_sm
if __name__ == "__main__":
    run_custom_sentencepiece()
