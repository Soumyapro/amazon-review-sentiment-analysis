# Amazon Review Sentiment Analysis

A machine learning project that predicts the sentiment of Amazon product reviews from their text. The project starts with 5-star rating prediction, analyzes why that framing is hard, and then reframes the problem as binary classification (positive vs negative), reaching 88% accuracy with a fully balanced performance across both classes.

## Dataset

25,000 Amazon product reviews with two columns:

| Column | Description |
|--------|-------------|
| Review | Raw review text |
| Sentiment | Star rating from 1 to 5 |

Null values and duplicates are removed before modeling.

## Approach

**1. Text cleaning**
- Strip HTML tags and HTML entities
- Lowercase and keep only letters and apostrophes
- Tokenize with NLTK
- Remove stopwords while keeping negation words (not, don't, wasn't, etc.), since negations carry sentiment
- POS tag each token and lemmatize using the correct part of speech

**2. Feature extraction**
- TF-IDF vectorization with unigrams and bigrams, 20,000 max features, min_df of 2
- The vectorizer is fit on training data only to prevent data leakage

**3. Modeling**

*Stage 1: 5-class rating prediction*
- Logistic Regression and Multinomial Naive Bayes
- Both reach about 48% accuracy against a 20% random baseline
- The confusion matrix shows errors are almost always off by one star (predicting 4 when the answer is 5), which mirrors how humans also struggle to distinguish adjacent ratings

*Stage 2: Binary classification*
- 3-star reviews are dropped since they mix positive and negative language
- Ratings 1 and 2 become negative, ratings 4 and 5 become positive
- Logistic Regression reaches about 88% accuracy with balanced precision and recall on both classes

**4. Interpretation and inference**
- The most influential positive and negative words are extracted from the model coefficients
- A `predict_sentiment` function applies the same cleaning and vectorizer to new text and returns a label with a confidence score

## Results

| Task | Model | Accuracy |
|------|-------|----------|
| 5-class rating | Logistic Regression | ~48% |
| 5-class rating | Multinomial Naive Bayes | ~48% |
| Binary sentiment | Logistic Regression | ~88% |

## Requirements

```
numpy
pandas
matplotlib
seaborn
nltk
scikit-learn
```

NLTK resources (punkt, stopwords, wordnet, averaged_perceptron_tagger) are downloaded automatically in the first cell.

## Usage

1. Place `amazon_review.csv` in the project directory
2. Open and run `amazon_review_sentiment.ipynb` from top to bottom
3. Use the `predict_sentiment` function at the end to classify your own review text

```python
predict_sentiment([
    "This product broke after two days, complete waste of money.",
    "Absolutely love it, works perfectly and great quality!",
])
```

## Key Takeaways

- Fine-grained star prediction is inherently noisy because adjacent ratings overlap heavily in language
- Reframing to binary sentiment gives a far more reliable and useful model
- Preserving negation words during stopword removal is essential for sentiment tasks
- Fitting the vectorizer on training data only keeps evaluation honest

## Future Work

- Hyperparameter tuning with cross validation
- Report macro F1 and ordinal error metrics for the 5-class task
- Try word embeddings or transformer models for comparison
- Handle genuinely mixed reviews instead of dropping neutral ratings
