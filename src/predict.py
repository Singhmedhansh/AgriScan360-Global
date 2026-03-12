# src/predict.py
import argparse
import json
import tensorflow as tf
from src.dataset import preprocess_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    with open(args.classes, "r") as f:
        class_names = json.load(f)

    image = preprocess_image(args.image)
    preds = model.predict(image)[0]

    index = preds.argmax()
    confidence = float(preds[index])

    print("Prediction:", class_names[index])
    print("Confidence:", round(confidence, 4))


if __name__ == "__main__":
    main()
