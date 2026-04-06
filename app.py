from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# 🔹 Home Route
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    confidence = 0.0
    if request.method == "POST":
        text = request.form["text"]
        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]
        conf = max(probability) * 100

        if prediction == 1:
            result = "Unsafe"
        else:
            result = "Safe"

        confidence = round(conf, 1)

    return render_template("index.html", result=result, confidence=confidence)

# 🔹 Test Route (Add this here)
@app.route("/test")
def test():
    return "Flask is working"

# 🔹 Run App
if __name__ == "__main__":
    app.run(debug=True)