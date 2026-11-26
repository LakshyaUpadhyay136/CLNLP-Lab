# Experiment 2: Basic Text Preprocessing

**Name:** Lakshya Upadhyay  
**SAP ID:** 500119462  
**Specialisation Batch:** AI/ML B1  
**Subject:** Computational Linguistics & Natural Language Processing  

## Objective
To implement foundational text preprocessing pipelines essential for Natural Language Processing by transforming raw, unstructured text files into clean, normalized tokens while eliminating textual noise, punctuation, digits, and stop words.

## Procedure & Implementation
The experiment operates entirely on local input text files (`2.1_text_data.txt`, `2.2_tokenization_data.txt`, `2.3_clean_data.txt`) and is organized into three distinct sub-experiments matching `Experiment2.ipynb`:

### 1. Basic Text Cleaning (`2.1_text_data.txt`)
* **Uppercase Character Frequency:** Count total uppercase letters (`sum(1 for c in text if c.isupper())`).
* **Case Normalization:** Convert the complete text corpus to lowercase (`text.lower()`).
* **Punctuation Stripping:** Identify all punctuation marks and types using Python's `string.punctuation` module, then eliminate them via string comprehension.
* **Digit Removal:** Detect numeric digits using `str.isdigit()` and filter them out completely.
* **Whitespace Normalization:** Compress irregular and excess whitespace using `" ".join(text.split())` and calculate the exact count of removed whitespace characters.

### 2. Tokenization Across Three Frameworks (`2.2_tokenization_data.txt`)
Comparative implementation of word-level and sentence-level tokenization across three paradigms:
* **NLTK:**
  * Word tokenization via `nltk.word_tokenize(raw_text)` leveraging the `punkt_tab` pre-trained model.
  * Sentence tokenization via `nltk.sent_tokenize(raw_text)` to split on sentence boundaries.
* **spaCy:**
  * Load and process text using the `en_core_web_sm` English pipeline model (`nlp = spacy.load('en_core_web_sm')`).
  * Extract discrete word tokens using list comprehension over document token objects (`[t.text for t in doc]`).
  * Extract sentence boundaries using document sentence spans (`[s.text for s in doc.sents]`).
* **Native Python & Regular Expressions:**
  * Basic whitespace-based word splitting via `str.split()`.
  * Sentence boundary segmentation using regex positive lookbehind: `re.split(r'(?<=[.!?]) +', raw_text)`, ensuring sentence-ending punctuation remains attached to the preceding sentence.

### 3. Stop Words Removal (`2.3_clean_data.txt`)
* **Stop Word Corpus:** Download and load standard English stop words from `nltk.corpus.stopwords` (`set(stopwords.words('english'))`).
* **Tokenization & Case Uniformity:** Lowercase raw input text and generate tokens via `nltk.word_tokenize()`.
* **Stop Word Isolation:** Identify and extract the specific unique stop words present in the corpus (`[t for t in tokens if t in stop_words]`).
* **Cleaned Semantic Tokens:** Filter tokens to exclude all stop words while retaining strictly alphanumeric tokens (`[t for t in tokens if t not in stop_words and t.isalnum()]`).

## Observations
| Preprocessing Stage | Method / Framework | Action Performed | Result Output |
| :--- | :--- | :--- | :--- |
| **Text Cleaning** | Built-in Python (`string`, `str`) | Uppercase count, lowercasing, punctuation stripping, digit removal | Normalized lowercased string without digits or punctuation |
| **Whitespace Normalization** | Built-in Python (`split`, `join`) | Multi-space collapse, stripped redundant spacing | Compressed clean single-spaced text |
| **Tokenization (NLTK)** | `nltk.word_tokenize`, `sent_tokenize` | Rule-based statistical boundary parsing (`punkt_tab`) | Token list separating words and punctuation symbols |
| **Tokenization (spaCy)** | `en_core_web_sm` pipeline | Model-based document parsing via `.text` and `.sents` | Context-aware token and sentence span objects |
| **Tokenization (Native)** | `str.split()`, `re.split(r'(?<=[.!?]) +')` | Whitespace splitting & regex lookbehind | Native lists of words and sentences preserving punctuation |
| **Stop Word Isolation** | `nltk.corpus.stopwords` | Set-membership check against English stop words | Set of removed grammatical helper words (`the`, `is`, `in`, etc.) |
| **Stop Word Filtering** | List comprehension & `str.isalnum()` | Removal of stop words and remaining punctuation | Filtered list containing only core semantic content words |

## Result Interpretation & Learnings
* **Text Normalization Foundation:** Cleaning operations systematically convert noisy unstructured text into a canonical format, removing non-informative symbols while quantifying noise removal.
* **Tokenization Paradigms Compared:** 
  * NLTK isolates punctuation into individual tokens effectively using pre-trained boundary rules.
  * spaCy provides a unified, object-oriented pipeline yielding rich linguistic token attributes.
  * Regex with lookbehind (`(?<=[.!?]) +`) enables precise custom boundary splits without consuming delimiter tokens.
* **Vocabulary Reduction:** Removing high-frequency stop words significantly reduces corpus dimensionality while preserving words with core semantic meaning for downstream NLP tasks.
