# Experiment 4: Term Frequency and Named Entity Recognition

**Name:** Lakshya Upadhyay  
**SAP ID:** 500119462  
**Specialisation Batch:** AI/ML B1  
**Subject:** Computational Linguistics & Natural Language Processing  

## Objective
To implement, compare, and analyze statistical word importance metrics—including Term Frequency (TF) and Term Frequency-Inverse Document Frequency (TF-IDF)—utilizing both production NLP toolkits (`spaCy`) and native Python implementations from scratch, followed by automated entity extraction using Named Entity Recognition (NER).

## Procedure & Implementation
The experiment operates on raw input text (`4.1_4.2_input.txt`) and multi-document corpora, structured into three primary sub-experiments matching `Experiment4.ipynb`:

### 1. Toolkit-based Term Frequency & NER (`Experiment 4.1`)
* **Pipeline Setup:** Load the English pipeline model `en_core_web_sm` via `spaCy` (with auto-download fallback).
* **Text Processing & Filtering:** Parse `4.1_4.2_input.txt`, filtering out punctuation (`token.is_punct`) and whitespace tokens (`token.is_space`), lowercasing all tokens.
* **Frequency Counting:** Compute term frequencies using Python's `collections.Counter`, identifying and displaying the top 10 most frequent terms.
* **CSV Export:** Structure frequencies into a Pandas DataFrame and persist to [`4.1_TF_Toolkit.csv`](4.1_TF_Toolkit.csv).
* **Named Entity Recognition (NER):** Extract entity mentions along with their categorical entity labels (`doc.ents`, `ent.text`, `ent.label_`), rendering the top 10 detected entities in tabular format.

### 2. Native Term Frequency Analysis (`Experiment 4.2`)
* **Pure Python Pipeline:** Read `4.1_4.2_input.txt` without external NLP libraries.
* **Text Preprocessing:** Convert input text to lowercase and remove all punctuation characters using `str.maketrans('', '', string.punctuation)` and `translate()`.
* **Tokenization:** Segment the cleaned string using whitespace splitting (`str.split()`).
* **Frequency Computation:** Calculate frequency counts using a native Python dictionary (`dict.get(token, 0) + 1`), sorting terms descending by count to retrieve the top 10 most frequent words.
* **CSV Export:** Output the full vocabulary frequency distribution to [`4.2_TF_Native.csv`](4.2_TF_Native.csv) using Python's built-in `csv.writer`.

### 3. Native TF-IDF Computation from Scratch (`Experiment 4.3`)
Implement the end-to-end mathematical formulation of TF-IDF across a three-document NLP corpus:
1. **Document Preprocessing:** Clean, lowercase, strip punctuation, and tokenize each document.
2. **Term Frequency (TF):** Normalized term occurrences relative to document length:
   $$\text{TF}(t, d) = \frac{\text{Count of } t \text{ in } d}{\text{Total words in } d}$$
3. **Document Frequency (DF):** Count the number of distinct documents containing term $t$:
   $$\text{DF}(t) = |\{d \in D : t \in d\}|$$
4. **Inverse Document Frequency (IDF):** Measure term rarity across the total document collection $N$:
   $$\text{IDF}(t) = \ln\left(\frac{N}{\text{DF}(t)}\right)$$
5. **TF-IDF Calculation:** Composite weighting reflecting word specificity:
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
6. **Ranking:** Tabulate term-level metrics ($\text{TF}$, $\text{DF}$, $\text{IDF}$, $\text{TF-IDF}$) and identify top-ranking terms per document.

## Observations
| Sub-Experiment | Framework / Method | Input Dataset | Output Metric / Artifact | Key Findings |
| :--- | :--- | :--- | :--- | :--- |
| **4.1 Toolkit TF** | spaCy (`en_core_web_sm`) + `Counter` | `4.1_4.2_input.txt` | Top 10 terms, [`4.1_TF_Toolkit.csv`](4.1_TF_Toolkit.csv) | High-precision tokenization with automated punctuation/whitespace handling |
| **4.1 Toolkit NER** | spaCy statistical NER pipeline | `4.1_4.2_input.txt` | Tabular entities (`Entity`, `Label`) | Automatically tagged categories (e.g., `PERSON`, `ORG`, `GPE`, `DATE`) |
| **4.2 Native TF** | Built-in Python (`dict`, `str.translate`) | `4.1_4.2_input.txt` | Top 10 terms, [`4.2_TF_Native.csv`](4.2_TF_Native.csv) | Direct string translation matches toolkit counts closely with zero library overhead |
| **4.3 Native TF-IDF**| Pure Python math (`math.log`) | 3 NLP Documents | Detailed breakdown ($\text{TF}$, $\text{DF}$, $\text{IDF}$, $\text{TF-IDF}$) | Terms common across all documents receive $\text{IDF} = 0$, elevating domain-distinctive keywords |

## Result Interpretation & Learnings
* **Toolkit vs. Native Analysis:** 
  * spaCy's linguistic models excel at context-aware token isolation and provide built-in NER capabilities out-of-the-box.
  * The native Python implementation reproduces fundamental frequency counting accurately, demonstrating how statistical NLP functions at the algorithmic level.
* **Power of TF-IDF Weighting:**
  * Ubiquitous terms appearing in all documents (such as `is`, `a`, `of`) receive an $\text{IDF} = \ln(3/3) = 0$, completely neutralizing their weighting.
  * Rare or document-specific keywords (e.g., `helps`, `computers`, `machine`, `learning`) receive high composite scores, effectively summarizing document uniqueness.
* **Information Extraction via NER:** Demonstrates how pre-trained sequence taggers classify semantic entities into predefined categories without requiring hand-crafted rules.
