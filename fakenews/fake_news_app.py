import joblib

model = joblib.load("fakenews/fake_news_model.pkl")

vectorizer = joblib.load("fakenews/vectorizer.pkl")


def predict_news(news):

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)

    return prediction[0]


if __name__ == "__main__":

    news = input("Enter News: ")

    result = predict_news(news)

    print("Prediction:", result)
    