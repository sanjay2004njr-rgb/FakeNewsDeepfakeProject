import joblib
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load("fakenews/fake_news_model.pkl")
vectorizer = joblib.load("fakenews/vectorizer.pkl")

API_KEY = "YOUR_NEWSAPI_KEY"


def predict_news(news):
    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)
    return prediction[0]


def verify_news(news_text):
    url = f"https://newsapi.org/v2/everything?q={news_text[:50]}&apiKey={API_KEY}"

    try:
        response = requests.get(url)
        data = response.json()

        if data.get("totalResults", 0) > 0:
            return "LIKELY REAL"
        else:
            return "SUSPICIOUS"

    except Exception as e:
        return f"ERROR: {e}"


@app.route("/", methods=["GET", "POST"])
def home():

    ml_result = ""
    online_result = ""
    final_result = ""

    if request.method == "POST":

        news = request.form["news"]

        ml_result = predict_news(news)
        online_result = verify_news(news)

        if ml_result == "FAKE" and online_result == "LIKELY REAL":
            final_result = "POSSIBLY REAL (Verified by online news sources)"

        elif ml_result == "REAL" and online_result == "LIKELY REAL":
            final_result = "REAL"

        elif ml_result == "FAKE" and online_result == "SUSPICIOUS":
            final_result = "FAKE"

        else:
            final_result = "NEEDS FURTHER VERIFICATION"

    return render_template(
        "index.html",
        ml_result=ml_result,
        online_result=online_result,
        final_result=final_result
    )


if __name__ == "__main__":
    app.run(debug=True)