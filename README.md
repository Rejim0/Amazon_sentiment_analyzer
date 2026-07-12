# Amazon Review Sentiment Analyzer

A machine learning web app that analyzes the sentiment of Amazon product reviews.

🔗 **Live Demo**: [amazonsentimentanalyzer-production.up.railway.app](https://amazonsentimentanalyzer-production.up.railway.app)

---

## Overview

This project classifies Amazon reviews as **Positive** or **Negative** using NLP techniques and machine learning. It also shows the **polarity** and **subjectivity** score of the review using TextBlob.

---

## Dataset

Amazon Fine Food Reviews — 568,454 reviews from Kaggle.

Download: [Kaggle Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

After balancing:
- 40,000 Positive reviews (4-5 stars)
- 40,000 Negative reviews (1-2 stars)

---

## Tech Stack

- **Python** — core language
- **Flask** — web framework
- **scikit-learn** — ML models
- **NLTK** — text preprocessing
- **TextBlob** — polarity and subjectivity
- **Gunicorn** — production server
- **Railway** — cloud deployment
- **GitHub** — version control

---

## ML Pipeline

### Text Preprocessing
- Lowercasing
- Tokenization
- Removing stopwords (with negation handling)
- Lemmatization
- HTML artifact removal

### Feature Extraction
- TF-IDF with bigrams (ngram_range=(1,2))
- Vocabulary size: ~50,000 features

### Models Compared

| Model | Accuracy | Precision |
|---|---|---|
| Multinomial NB | 86.5% | 85.6% |
| Bernoulli NB | 85.3% | 82.7% |
| Logistic Regression | 91.1% | 91.3% |
| Random Forest | 89.9% | 90.2% |
| XGBoost | 85.7% | 86.5% |
| **SVM (Linear)** | **92.1%** | **92.2%** |

### Best Model: SVM + TF-IDF Bigrams
- Accuracy: 92.1%
- Precision: 92.2%

---

## Key Findings

- Negative reviews are slightly longer than positive (509 vs 472 chars)
- Both classes overlap heavily in length — sentiment lives in the words
- Bigrams significantly improved performance (+2.3% accuracy)
- Negation handling ("not good", "n't like") preserved for better accuracy
- TextBlob polarity added no improvement — TF-IDF already captures sentiment

---

## Project Structure
├── app.py                  # Flask application
├── utils.py                # Text preprocessing function
├── model.pkl               # Trained SVM model
├── vectorizer.pkl          # TF-IDF vectorizer
├── requirements.txt        # Dependencies
├── Procfile                # Railway deployment config
├── templates/
│   └── index.html          # Frontend
└── static/
└── style.css           # Styling

---

## How to Run Locally

```bash
# clone the repo
git clone https://github.com/Rejim0/Amazon_sentiment_analyzer.git

# install dependencies
pip install -r requirements.txt

# run the app
python app.py
```

Visit `http://127.0.0.1:5000`

---

## Limitations

- Trained on Amazon food reviews only
- Struggles with fashion/electronics domain reviews
- Modern phishing-style language not well detected
- A transformer model like BERT would handle context better