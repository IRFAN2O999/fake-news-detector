# 📰 Fake News Detector

A machine learning web application built with **Streamlit** that detects whether a given news article is **Real** or **Fake**.  
It can:
- Predict single articles typed by the user
- Upload multiple CSV files (datasets) with **different column formats**
- Automatically merge and train on multiple datasets (news, tweets, Reddit posts, etc.)
- Provide prediction explanations (shows article body if available)

---

## 📂 Project Structure


fake-news-detector/
│
├── app.py # Streamlit web app
├── train_model.py # Training script (handles multiple CSV formats)
├── requirements.txt # Required Python packages
├── README.md # Project documentation
├── .gitignore # Files to ignore in GitHub
├── datasets/ # Place your CSV datasets here
│ ├── data.csv
│ ├── Fake.csv
│ ├── True.csv
│ └── ...other datasets...
├── model.pkl # Trained ML model
├── vectorizer.pkl # Text vectorizer
└── datasets/README.txt # Dataset info or download instructions



---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/IRFAN2O999/fake-news-detector.git
cd fake-news-detector



Create a virtual environment

python -m venv .venv


Activate the environment

Windows (PowerShell)

.\.venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


📊 Dataset

You can use your own dataset or public ones like:

Kaggle: Fake and Real News Dataset

LIAR Dataset

Format Flexibility:

The training script supports datasets with different column names.

It will auto-detect text and label columns (fake=0, true=1).

Place CSV files inside the datasets/ folder before training.


1️⃣ Train the Model
python train_model.py


Loads all CSVs from datasets/

Detects text & label columns automatically

Merges multiple datasets

Saves model.pkl and vectorizer.pkl

Run the Web App
streamlit run app.py

3️⃣ Features in Web App

Manual Input: Type or paste an article to check

CSV Upload: Upload multiple CSVs with different column formats

Predictions Table: Shows Article, Predicted Label, and Explanation/Body

Supports large datasets

🛠 Requirements
streamlit
scikit-learn
pandas
numpy
joblib

📷 Example

Web App Preview:


👨‍💻 Author

IRFAN2099
GitHub: https://github.com/IRFAN2O999

