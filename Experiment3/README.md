# Experiment 3: Stemming, Lemmatization, and Regular Expressions

**Name:** Lakshya Upadhyay  
**SAP ID:** 500119462  
**Specialisation Batch:** AI/ML B1  
**Subject:** Computational Linguistics & Natural Language Processing  

## Objective
To implement and compare rule-based affix stripping (stemming) against vocabulary-driven morphological reduction (lemmatization), and to utilize Regular Expressions (Regex) for structured information extraction from raw unstructured text.

## Procedure & Implementation
The experiment is organized into three distinct sub-experiments matching `Experiment3.ipynb`:

### 1. Stemming with Porter Stemmer (`Experiment 3.1`)
* **Algorithm:** Utilize NLTK's `PorterStemmer` algorithm, which applies heuristic suffix-stripping rules across progressive reduction phases.
* **Input Dataset:** `playing`, `played`, `plays`, `studies`, `studying`, `connected`, `connection`, `computers`.
* **Execution:** Iteratively apply `stemmer.stem(word)` to reduce words to their base root/stem forms.
* **Observation Focus:** Evaluate inflection removal and identify instances where the resulting stem is a non-dictionary form (e.g., `studi`, `comput`).

### 2. Lemmatization with WordNet Lemmatizer (`Experiment 3.2`)
* **Algorithm:** Employ NLTK's `WordNetLemmatizer` leveraging lexical relations from the WordNet database (`wordnet`).
* **Input Dataset:** `cats`, `dogs`, `running`, `runs`, `ran`, `studies`, `studying`, `better`, `children`, `mice`, `went`, `ate`, `leaves`, `caring`.
* **Part-of-Speech (POS) Mapping:** Explicitly supply grammatical category tags (`pos=wordnet.NOUN`, `VERB`, `ADJ`) to handle irregular conjugations, plurals, and comparative forms:
  * Irregular past verbs: `ran` $\rightarrow$ `run`, `went` $\rightarrow$ `go`, `ate` $\rightarrow$ `eat`.
  * Irregular plurals: `children` $\rightarrow$ `child`, `mice` $\rightarrow$ `mouse`, `leaves` $\rightarrow$ `leaf`.
  * Comparative adjectives: `better` $\rightarrow$ `good`.

### 3. Structured Information Extraction using Regex (`Experiment 3.3`)
Implement robust regular expression patterns to extract structured entities from a multi-line workshop announcement text:
* **Email Addresses:** Pattern `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b` matching standard RFC email formats.
* **URLs:** Pattern `https?://[^\s]+|\bwww\.[^\s]+\b` with trailing punctuation sanitization (`u.rstrip('.,')`).
* **Mobile Numbers:** Pattern `(?:\+91[-\s]?)?[6-9]\d{9}` extracting 10-digit phone numbers with optional `+91` country code.
* **Hashtags:** Pattern `#[A-Za-z0-9_]+` isolating hashtag topic tags.
* **Social Mentions:** Pattern `(?<!\S)@[A-Za-z0-9_]+` matching user handle tags preceded by whitespace/string start.

## Observations
| Sub-Experiment | Framework / Module | Input Sample | Extracted / Transformed Output | Key Characteristic |
| :--- | :--- | :--- | :--- | :--- |
| **3.1 Stemming** | `nltk.stem.PorterStemmer` | `studies`, `computers`, `connection` | `studi`, `comput`, `connect` | Heuristic suffix chop; leaves non-dictionary stems |
| **3.2 Lemmatization** | `WordNetLemmatizer` (POS-aware) | `mice`, `went`, `better`, `caring` | `mouse`, `go`, `good`, `care` | Morphological root lookup; always yields valid words |
| **3.3 Regex (Emails)** | `re.findall()` | Text corpus | `nlpworkshop@gmail.com`, `support@python.org` | Boundary-delimited email pattern |
| **3.3 Regex (URLs)** | `re.findall()` | Text corpus | `https://www.nlpworkshop.com`, `www.python.org`, `https://github.com/NLPWorkshop` | Web protocol and domain matching |
| **3.3 Regex (Phones)** | `re.findall()` | Text corpus | `+91-9876543210`, `9123456789` | Standard Indian 10-digit mobile formats |
| **3.3 Regex (Tags/Handles)**| `re.findall()` | Text corpus | `#NLP`, `#Python`, `#MachineLearning` / `@NLPWorkshop`, `@PythonLearner` | Social media metadata identifiers |

## Result Interpretation & Learnings
* **Stemming vs. Lemmatization:**
  * **Porter Stemmer** is computationally faster and rule-based, but cuts off affixes indiscriminately, often resulting in non-words (`studi`).
  * **WordNet Lemmatizer** guarantees legitimate dictionary lemmas and accurately handles irregular words (`went` $\rightarrow$ `go`, `children` $\rightarrow$ `child`), but requires accurate POS context tags.
* **Regex Pattern Precision:** Demonstrated how character classes, quantifiers (`+`, `\d{9}`), non-capturing groups (`(?:...)`), word boundaries (`\b`), and negative lookbehinds (`(?<!\S)`) enable precise extraction of target entities without false positive capture.
