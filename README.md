# ChatGPT Reviews Analysis (Python)


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


## 🚀 Quickstart
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook ChatGPT_Reviews_Analysis_with_Python.ipynb
```

Run all cells top‑to‑bottom. In **Google Colab**, just open the notebook via the badge above and upload/mount your data.

---

## 🔁 Reproducing Results
1. Install dependencies (`requirements.txt`).  
2. Put your CSV next to the notebook (or in `data/`) and update the file path.  
3. Execute the notebook.  
4. Export a few figures into `assets/` and embed them in this README (optional).

---

## 📁 Suggested Repository Layout
```
.
├── ChatGPT_Reviews_Analysis_with_Python.ipynb
├── requirements.txt
├── data/                     # raw/sample datasets (gitignored)
├── assets/                   # screenshots/figures for README
└── README.md
```

---

## 🧰 Tech Stack
- **Python** (3.10+) • **pandas** • **TextBlob** • **Plotly** • **scikit‑learn** • 

> Detected in the notebook: `pandas`, `plotly`, `sklearn`, `textblob`.

---

## 🛣️ Roadmap
- Add TF‑IDF + a baseline classifier (e.g., Logistic Regression or Naive Bayes).  
- Add evaluation: accuracy, F1, precision/recall, ROC‑AUC, confusion matrix.  
- Interpretability: word importance, error analysis, SHAP/LIME (optional).  
- Export a trained model (`joblib`) and serve it with a small FastAPI endpoint.

---

---

## 🙌 Author
**Nikita Marshchonok**  
GitHub: https://github.com/NikitaMarshchonok  
LinkedIn: http://www.linkedin.com/in/nikita-marshchonok  
Email: n.marshchonok@gmail.com
telegram: @nikitamarshchonok




