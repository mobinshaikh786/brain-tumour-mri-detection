"""Standalone prediction script — usage: python predict.py path/to/image.jpg"""
import sys, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 150

def predict(img_path, model_path="model/brain_tumor_model.h5"):
    model = load_model(model_path)
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)
    return CLASSES[idx], float(preds[idx]), {c: float(p) for c, p in zip(CLASSES, preds)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    label, conf, all_p = predict(sys.argv[1])
    print(f"\n🧠 Prediction: {label.upper()} ({conf*100:.2f}%)")
    print("All probabilities:")
    for c, p in all_p.items():
        print(f"  {c:12s} : {p*100:5.2f}%")
