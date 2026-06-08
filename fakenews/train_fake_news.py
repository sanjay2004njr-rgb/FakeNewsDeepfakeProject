import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading datasets...")

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print("Datasets loaded")

# Labels
fake["label"] = "FAKE"
true["label"] = "REAL"

# Combine datasets
data = pd.concat([fake, true])

# Shuffle dataset
data = data.sample(frac=1, random_state=42)

# Input and output
x = data["text"]
y = data["label"]

print("Vectorizing text...")

# Convert text into vectors
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7
)

x = vectorizer.fit_transform(x)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

# Train model
model = LogisticRegression(
    max_iter=10000,
    C=10
)

model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully")