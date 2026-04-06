
import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

df = pd.read_csv("train.csv")
print(df.head())
df['unsafe'] = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].max(axis=1)
safe_rows = df[df['unsafe'] == 0]
unsafe_rows = df[df['unsafe'] == 1]

sample_size = min(10000, len(safe_rows), len(unsafe_rows))
if sample_size == 0:
    raise ValueError("Unable to train: one of the classes has zero rows in train.csv")

safe_df = safe_rows.sample(sample_size, random_state=42)
unsafe_df = unsafe_rows.sample(sample_size, random_state=42)

df = pd.concat([safe_df, unsafe_df])
df['comment_text'] = df['comment_text'].str.lower()

# Load dataset
data = df.copy()

# Text cleaning function
def clean_text(text):
    text = str(text) if pd.notna(text) else ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)

data['comment_text'] = data['comment_text'].apply(clean_text)

# Features & labels (as requested)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

X = vectorizer.fit_transform(data['comment_text'])
y = data['unsafe']

model = LogisticRegression(class_weight='balanced')
model.fit(X, y)

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")
