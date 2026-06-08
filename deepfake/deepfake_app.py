import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("deepfake_model.h5")


def predict_image(img_path):

    # Load image
    img = image.load_img(
        img_path,
        target_size=(128,128)
    )

    # Convert image to array
    img_array = image.img_to_array(img)

    # Normalize image
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # Output
    if prediction[0][0] > 0.5:
        return "REAL IMAGE"
    else:
        return "FAKE IMAGE"


# Main program
if __name__ == "__main__":

    img_path = input("Enter Image Path: ")

    result = predict_image(img_path)

    print("Prediction:", result)