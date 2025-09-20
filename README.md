# ChatGPT Reviews Analysis (Python)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_GITHUB_USERNAME>/<YOUR_REPO>/blob/main/ChatGPT_Reviews_Analysis_with_Python.ipynb)
> After pushing to GitHub, replace `<YOUR_GITHUB_USERNAME>/<YOUR_REPO>` in the link above.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
![pandas](https://img.shields.io/badge/pandas-data--wrangling-150458)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Plotly](https://img.shields.io/badge/Plotly-interactive%20charts-lightgrey)
![TextBlob](https://img.shields.io/badge/TextBlob-sentiment-green)

A compact, portfolio‑friendly notebook that explores **user reviews about ChatGPT**.  
It walks through data loading, quick EDA, baseline **sentiment scoring with TextBlob**, and interactive visualizations with **Plotly**. You can extend it with scikit‑learn models for classification.

---

## ✨ Highlights
- **EDA**: counts, missing values, text length distributions, top tokens/bigrams.
- **Sentiment**: baseline polarity/subjectivity from **TextBlob**.
- **Interactive visuals**: histograms and bar charts built with **Plotly**.
- **Ready for ML**: cells prepared to plug in `scikit‑learn` vectorizers & models.

> The notebook file is: **`ChatGPT_Reviews_Analysis_with_Python.ipynb`**

---


## 📦 Data
- Expected CSV: **`chatgpt_reviews.csv`** (adjust the path in the notebook if needed).
- At minimum, provide a column with review text (e.g., `text`). Optional columns like rating, date, or source can be used for richer analysis.

If your dataset is private, place a small **sample** (10–50 rows) in `data/` and update the path in the notebook.

---

## 🧠 Method in Brief
1. **Load & inspect** the dataset; basic cleaning.
2. **Text preprocessing** (lowercasing, punctuation removal, stopwords where needed).
3. **Sentiment** via **TextBlob**: polarity in `[-1, 1]`, subjectivity in `[0, 1]`.
4. **Visualization** with Plotly (distributions, frequent terms).
5. *(Optional)* **Vectorize** (TF‑IDF) and train a simple classifier with scikit‑learn.

### Minimal sentiment example
```python
from textblob import TextBlob
import pandas as pd

df = pd.read_csv("chatgpt_reviews.csv")
df["polarity"] = df["text"].astype(str).apply(lambda t: TextBlob(t).sentiment.polarity)
df["subjectivity"] = df["text"].astype(str).apply(lambda t: TextBlob(t).sentiment.subjectivity)
df.head()
```

---

