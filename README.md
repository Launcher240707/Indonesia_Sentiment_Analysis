# Indonesian Sentiment Analysis

An Indonesian text classification project that uses machine learning to classify text into three sentiment categories: **positive, neutral, and negative**.

The project explores TF-IDF feature extraction, Logistic Regression, Linear Support Vector Classification (Linear SVM), class weighting, and n-gram configurations. It also includes test-set evaluation and error analysis.

## Project Overview

This project aims to build a baseline sentiment analysis model for Indonesian-language text using the SmSA dataset from IndoNLU.

The main workflow includes:

- Dataset exploration and quality checks
- Text preprocessing
- TF-IDF feature extraction
- Model training and validation
- Class-weighting and n-gram experiments
- Test-set evaluation
- Error analysis and prediction

## Dataset

The project uses the **SmSA (Sentiment Analysis)** dataset from IndoNLU.

The dataset contains three sentiment labels:

- `positive`
- `neutral`
- `negative`

| Dataset split | Number of samples |
| ------------- | ----------------: |
| Training      |            11,000 |
| Validation    |             1,260 |
| Test          |               500 |
| **Total**     |        **12,760** |

Dataset source: [IndoNLU — SmSA](https://github.com/IndoNLP/indonlu/tree/master/dataset/smsa_doc-sentiment-prosa)

The dataset files are not included in this repository. Place the corresponding TSV files in `data/raw/` before running the scripts.

## Methodology

### 1. Text Preprocessing

The preprocessing pipeline performs:

- Unicode normalization
- Lowercasing
- URL and email replacement
- Punctuation removal
- Whitespace normalization

The preprocessing is intentionally minimal to preserve potentially useful sentiment information, including negation words.

### 2. Feature Extraction

`TfidfVectorizer` transforms text into numerical features.

The main configuration uses:

- Unigrams and bigrams: `ngram_range=(1, 2)`
- Minimum document frequency: `min_df=2`
- Maximum features: `max_features=100000`
- Sublinear term frequency: `sublinear_tf=True`

### 3. Model Experiments

The project explores:

- Logistic Regression
- Linear SVM
- Balanced Linear SVM using `class_weight="balanced"`
- Unigram-only versus unigram-and-bigram features

Model selection is based on validation performance, with Macro-F1 used to assess performance across all three classes.

## Results

### Validation Results

| Model / configuration                 | Accuracy | Macro-F1 |
| ------------------------------------- | -------: | -------: |
| Logistic Regression                   |   88.02% |   83.23% |
| Linear SVM                            |   89.21% |   85.23% |
| Balanced Linear SVM                   |   89.60% |   86.33% |
| Balanced Linear SVM, unigram only     |   86.27% |   82.99% |
| Balanced Linear SVM, unigram + bigram |   89.60% |   86.33% |

The balanced Linear SVM with unigram-and-bigram features achieved the highest validation Macro-F1 among the evaluated configurations.

### Final Test Results

The selected configuration was evaluated on the labeled test set of 500 samples.

| Metric   |  Score |
| -------- | -----: |
| Accuracy | 82.00% |
| Macro-F1 | 78.03% |

#### Per-class performance

| Sentiment | Precision | Recall | F1-score | Support |
| --------- | --------: | -----: | -------: | ------: |
| Negative  |      0.78 |   0.95 |     0.85 |     204 |
| Neutral   |      0.76 |   0.55 |     0.64 |      88 |
| Positive  |      0.90 |   0.81 |     0.85 |     208 |

The model correctly classified 410 out of 500 test samples.

**Key observation:** Neutral sentiment remains the most challenging class. Its recall of 55% indicates that the model misses a substantial proportion of genuinely neutral examples.

## Error Analysis

The project includes scripts to examine misclassified samples, compare model predictions, and investigate neutral-sentiment errors.

The test results reveal that:

- Neutral samples are frequently classified as negative.
- Some neutral samples are classified as positive.
- Descriptive or factual text can be confused with sentiment-bearing text.
- Class weighting improves neutral-class recall compared with the initial baseline.

These findings suggest that additional work on neutral-sentiment representation and contextual features may be beneficial.

## Project Structure

```text
indonesian-sentiment-analysis/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── exploration.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── experiment_balanced.py
│   ├── evaluate_balanced.py
│   ├── compare_error_analysis.py
│   ├── analyze_neutral_errors.py
│   ├── explain_neutral_errors.py
│   ├── experiment_ngrams.py
│   ├── evaluate_ngrams.py
│   └── compare_ngram_predictions.py
├── models/
├── results/
│   ├── figures/
│   └── reports/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and navigate to the project directory.

```bash
git clone <YOUR_REPOSITORY_URL>
cd indonesian-sentiment-analysis
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies.

```bash
python -m pip install -r requirements.txt
```

## Dataset Setup

Place these files in `data/raw/`:

```text
train_preprocess.tsv
valid_preprocess.tsv
test_preprocess.tsv
```

Use the labeled test file for final evaluation. Keep the test set separate from model training and model selection.

## Running the Project

Run the scripts from the project root.

Explore the dataset:

```bash
python src/exploration.py
```

Train and compare the baseline models:

```bash
python src/train.py
```

Evaluate the baseline model:

```bash
python src/evaluate.py
```

Run the class-weighting experiment:

```bash
python src/experiment_balanced.py
```

Evaluate the balanced model:

```bash
python src/evaluate_balanced.py
```

Run the n-gram experiment:

```bash
python src/experiment_ngrams.py
```

Evaluate the selected n-gram configuration:

```bash
python src/evaluate_ngrams.py
```

Run interactive sentiment prediction:

```bash
python src/predict.py
```

**Note:** Run the relevant training or evaluation script first to generate the model file required by the prediction script. Generated models and result files may be excluded from version control by `.gitignore`.

## Limitations

- The test set contains 500 samples, so the reported metrics should be interpreted in the context of this dataset.
- Neutral sentiment has lower recall than the other classes.
- TF-IDF and linear models may not capture complex contextual meaning, sarcasm, or subtle expressions.
- Results on this dataset do not guarantee equivalent performance on other Indonesian text sources.

## Future Improvements

Potential next steps include:

- Improving neutral-sentiment classification
- Evaluating alternative preprocessing strategies
- Testing additional feature representations
- Comparing TF-IDF models with Indonesian pretrained language models
- Adding automated tests and a more convenient prediction interface

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Joblib

## License

This project is distributed under the license specified in the repository's `LICENSE` file.
