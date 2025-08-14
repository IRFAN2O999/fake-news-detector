import streamlit as st
import pandas as pd
import joblib
import re
import os

# Load model & vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# ------------------------
# Helper functions
# ------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^A-Za-z]', ' ', text)
    return ' '.join([w.lower() for w in text.split() if len(w) > 2])

def detect_text_column(df: pd.DataFrame):
    for col in df.columns:
        if col.lower() in ["text", "body", "article", "content", "news", "headline"]:
            return col
    return df.select_dtypes(include="object").apply(lambda x: x.str.len().mean()).idxmax()

def detect_label_column(df: pd.DataFrame, file_name: str):
    for col in df.columns:
        if col.lower() in ["label", "category", "target", "class", "truth"]:
            return col
    # No label col → infer from filename
    if "fake" in file_name.lower():
        df["label"] = 0
        return "label"
    elif "true" in file_name.lower() or "real" in file_name.lower():
        df["label"] = 1
        return "label"
    return None

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 Fake News Detector")
st.write("Detect whether news articles are **Real** or **Fake**. You can paste text or upload CSV files.")

mode = st.radio("Choose Mode:", ["Single Article", "Upload CSV Files"])

# ------------------------
# Single Article Mode
# ------------------------
if mode == "Single Article":
    user_input = st.text_area("Paste your news text here:", height=200)
    if st.button("Check"):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0][pred]
            if pred == 1:
                st.success(f"✅ Real News ({prob*100:.2f}% confidence)")
            else:
                st.error(f"🚨 Fake News ({prob*100:.2f}% confidence)")
        else:
            st.warning("Please enter some text.")

# ------------------------
# CSV Upload Mode
# ------------------------
else:
    uploaded_files = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        all_results = []
        for file in uploaded_files:
            try:
                df = pd.read_csv(file)
                text_col = detect_text_column(df)
                label_col = detect_label_column(df, file.name)

                # Clean & prepare text
                df["cleaned_text"] = df[text_col].apply(clean_text)
                X_vec = vectorizer.transform(df["cleaned_text"])
                preds = model.predict(X_vec)
                probs = model.predict_proba(X_vec).max(axis=1)

                # Store results
                df["Prediction"] = ["Real" if p == 1 else "Fake" for p in preds]
                df["Confidence"] = (probs * 100).round(2)

                # If there is a body/content column, call it "Explanation"
                extra_cols = [c for c in df.columns if c.lower() in ["body", "content", "article"]]
                if extra_cols:
                    df["Explanation"] = df[extra_cols[0]]

                all_results.append(df)

                st.success(f"✅ Processed {file.name} ({len(df)} rows)")
                st.dataframe(df[["Prediction", "Confidence"] + ([ "Explanation" ] if "Explanation" in df else [])].head(10))

            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")

        # Merge all CSV results into one download
        merged = pd.concat(all_results, ignore_index=True)
        csv_download = merged.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download All Predictions", csv_download, "predictions.csv", "text/csv")
