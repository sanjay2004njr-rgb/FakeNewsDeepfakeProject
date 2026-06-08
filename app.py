from flask import Flask, render_template, request

import joblib
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load Fake News model
news_model = joblib.load(
    "fakenews/fake_news_model.pkl"
)

vectorizer = joblib.load(
    "fakenews/vectorizer.pkl"
)

# Load Deepfake model
deepfake_model = tf.keras.models.load_model(
    "deepfake/deepfake_model.h5"
)


# HOME PAGE
@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# FAKE NEWS PREDICTION
@app.route(
    "/predict_news",
    methods=["POST"]
)
def predict_news():

    news = request.form["news"]

    news_vector = vectorizer.transform([news])

    prediction = news_model.predict(news_vector)

    if prediction[0] == 1:
        result = "REAL NEWS"
    else:
        result = "FAKE NEWS"

    return render_template(
        "index.html",
        news_result=result
    )


# DEEPFAKE IMAGE PREDICTION
@app.route(
    "/predict_image",
    methods=["POST"]
)
def predict_image():

    file = request.files["image"]

    filepath = "temp.jpg"

    file.save(filepath)

    img = image.load_img(
        filepath,
        target_size=(128,128)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = deepfake_model.predict(
        img_array
    )

    if prediction[0][0] > 0.5:
        result = "REAL IMAGE"
    else:
        result = "FAKE IMAGE"

    return render_template(
        "index.html",
        image_result=result
    )


if __name__ == "__main__":

    app.run(debug=True)