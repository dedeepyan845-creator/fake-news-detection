from flask import Flask, request, render_template_string
import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
df = pd.read_csv("news.csv")

# Features
X = df["text"]
y = df["label"]

# Train model if not exists
if not os.path.exists("model.pkl"):

    vectorizer = TfidfVectorizer(stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# HTML
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Fake News Detector</title>
<style>
body{font-family:Arial;text-align:center;background:#f2f2f2;padding:40px;}
textarea{width:80%;height:200px;padding:10px;}
button{padding:10px 20px;margin-top:10px;}
.result{font-size:28px;margin-top:20px;font-weight:bold;}
</style>
</head>
<body>

<h1>Fake News Detection</h1>

<form method="POST">
<textarea name="news" placeholder="Paste news here"></textarea><br>
<button type="submit">Check</button>
</form>

<div class="result">{{ result }}</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        news_list = request.form["news"].strip().split("\n")

        output = []

        for news in news_list:
            if news.strip():

                data = vectorizer.transform([news])
                prediction = model.predict(data)[0]

                if prediction == "FAKE":
                    output.append(f"❌ FAKE: {news}")
                else:
                    output.append(f"✅ REAL: {news}")

        result = "<br>".join(output)

    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(debug=True)