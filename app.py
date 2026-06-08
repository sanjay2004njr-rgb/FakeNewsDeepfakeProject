import joblib
import requests
import tensorflow as tf
import numpy as np

from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load Fake News Model
news_model = joblib.load("fakenews/fake_news_model.pkl")
vectorizer = joblib.load("fakenews/vectorizer.pkl")

# Load Deepfake Model
deepfake_model = tf.keras.models.load_model(
    "deepfake/deepfake_model.h5"
)

# NewsAPI Key
API_KEY = "e9d0cb992eae4be78cce397d41864f21"


@app.route("/")
def home():
    return render_template("index.html")


def verify_news(news_text):

    url = f"https://newsapi.org/v2/everything?q={news_text[:50]}&apiKey={API_KEY}"

    try:
        response = requests.get(url)
        data = response.json()

        total_results = data.get("totalResults", 0)

        print("NewsAPI Response:")
        print(data)

        if total_results > 0:
            return "LIKELY REAL", total_results
        else:
            return "SUSPICIOUS", total_results

    except Exception as e:
        print("Error:", e)
        return "ERROR", 0


@app.route("/predict_news", methods=["POST"])
def predict_news():

    news = request.form["news"]

    news_vector = vectorizer.transform([news])
    prediction = news_model.predict(news_vector)

    if prediction[0] == 1:
        ml_result = "REAL NEWS"
    else:
        ml_result = "FAKE NEWS"

    online_result, total_results = verify_news(news)

    if online_result == "LIKELY REAL" and total_results >= 5:
        final_result = "VERIFIED REAL NEWS"

    elif ml_result == "REAL NEWS" and online_result == "LIKELY REAL":
        final_result = "REAL NEWS"

    elif ml_result == "FAKE NEWS" and online_result == "LIKELY REAL":
        final_result = "POSSIBLY REAL (Verified by online sources)"

    elif ml_result == "FAKE NEWS" and online_result == "SUSPICIOUS":
        final_result = "FAKE NEWS"

    else:
        final_result = "NEEDS FURTHER VERIFICATION"

    return render_template(
        "index.html",
        ml_result=ml_result,
        online_result=online_result,
        final_result=final_result
    )


@app.route("/predict_image", methods=["POST"])
def predict_image():

    file = request.files["image"]

    filepath = "temp.jpg"
    file.save(filepath)

    img = image.load_img(
        filepath,
        target_size=(128, 128)
    )

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = deepfake_model.predict(img_array)

    if prediction[0][0] > 0.5:
        image_result = "REAL IMAGE"
    else:
        image_result = "FAKE IMAGE"

    return render_template(
        "index.html",
        image_result=image_result
    )


if __name__ == "__main__":
    app.run(debug=True)