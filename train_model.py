import os
import re
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
STOPWORDS = set(stopwords.words('english'))
LEMMA = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^A-Za-z]', ' ', text)
    text = text.lower().split()
    text = [LEMMA.lemmatize(tok) for tok in text if tok not in STOPWORDS and len(tok) > 2]
    return ' '.join(text)

def normalize_labels(series):
    mapping = {
        "fake": 0, "false": 0, "0": 0, "no": 0,
        "true": 1, "real": 1, "1": 1, "yes": 1
    }
    series = series.astype(str).str.strip().str.lower().replace(mapping)
    return pd.to_numeric(series, errors="coerce")

def load_all_csvs(input_paths):
    all_dfs = []
    for path in input_paths:
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
        else:
            files = [path]

        for file in files:
            try:
                df = pd.read_csv(file)
                text_col = None
                label_col = None

                # 1. Detect text column
                for col in df.columns:
                    if col.lower() in ["text", "body", "article", "content", "news", "headline"]:
                        text_col = col
                        break
                if not text_col:
                    text_col = df.select_dtypes(include="object").apply(lambda x: x.str.len().mean()).idxmax()

                # 2. Detect label column
                for col in df.columns:
                    if col.lower() in ["label", "category", "target", "class", "truth"]:
                        label_col = col
                        break

                # 3. If no label col, guess from filename
                if not label_col:
                    if "fake" in file.lower():
                        df["label"] = 0
                        label_col = "label"
                    elif "true" in file.lower() or "real" in file.lower():
                        df["label"] = 1
                        label_col = "label"
                    else:
                        print(f"⚠️ No label column found in {file} and filename gives no hint — skipping")
                        continue

                # 4. Keep only text + label
                df = df[[text_col, label_col]].dropna()
                df[label_col] = normalize_labels(df[label_col])

                # If still NaN labels, drop them
                df = df.dropna(subset=[label_col])
                df["text"] = df[text_col].apply(clean_text)
                df.rename(columns={label_col: "label"}, inplace=True)

                all_dfs.append(df[["text", "label"]])
                print(f"✅ Loaded {len(df)} rows from {file} (text='{text_col}', label='{label_col}')")

            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Train a fake news classifier from multiple CSVs")
    parser.add_argument("--input", nargs="+", default=["datasets"], help="Paths to CSV files or a folder")
    parser.add_argument("--model_out", default="model.joblib")
    parser.add_argument("--vectorizer_out", default="vectorizer.joblib")
    parser.add_argument("--metrics_out", default="metrics.txt")
    args = parser.parse_args()

    df = load_all_csvs(args.input)
    if df.empty:
        raise ValueError("No valid CSVs found for training.")

    print(f"\n📊 Total merged samples: {len(df)}")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, strip_accents='unicode')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced')
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 (weighted): {f1:.4f}")
    print("\nClassification report:\n", report)
    print("\nConfusion matrix:\n", cm)

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (weighted): {f1:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report + "\n\n")
        f.write("Confusion matrix:\n")
        f.write(np.array2string(cm))

    joblib.dump(model, args.model_out)
    joblib.dump(vectorizer, args.vectorizer_out)
    print(f"\n💾 Model saved to {args.model_out}")
    print(f"💾 Vectorizer saved to {args.vectorizer_out}")

if __name__ == "__main__":
    main()
