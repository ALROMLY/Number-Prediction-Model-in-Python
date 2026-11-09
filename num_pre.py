"""Handwritten Digit Recognition (MNIST CNN + Tkinter GUI)."""

import os
import threading

import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps, ImageTk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MNIST_PATH = os.path.join(BASE_DIR, "mnist.npz")        # optional local copy
MODEL_PATH = os.path.join(BASE_DIR, "digit_model.keras")  # cached trained model

# Region of the photo that contains the writing, as fractions:
# (top, bottom, left, right).  Set to None to use the whole image.
ROI = (0.18, 0.62, 0.20, 0.80)


# ---------------------------------------------------------------
# IMAGE PREPROCESSING  (kept outside the GUI so it can be tested)
# ---------------------------------------------------------------

def preprocess_digit(img, roi=ROI):
    """Turn a grayscale photo into a 28x28 MNIST-style digit.

    Returns (canvas_uint8_28x28, None) on success or (None, message).
    """
    if roi is not None:
        h, w = img.shape
        top, bottom, left, right = roi
        img = img[int(h * top):int(h * bottom), int(w * left):int(w * right)]

    if img.size == 0:
        return None, "Image area is empty."

    # Shrink very large photos so the blur / thresholds behave consistently
    longest = max(img.shape)
    if longest > 800:
        s = 800.0 / longest
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    # Light ink on a dark page -> flip to dark ink on light page
    if np.median(img) < 90:
        img = 255 - img

    # Correct uneven lighting (blur size scales with image size)
    sigma = max(10, max(img.shape) / 30)
    background = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    corrected = cv2.divide(img, background, scale=255)

    # Black & white (ink = white)
    _, thresh = cv2.threshold(
        corrected, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Join small gaps in pen strokes
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    # Find the digit's connected pieces
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        thresh, connectivity=8
    )
    if num_labels <= 1:
        return None, "No digit detected."

    H, W = thresh.shape
    min_area = max(30, 0.0003 * H * W)

    def touches_border(i):
        x, y, bw, bh = stats[i, :4]
        return x == 0 or y == 0 or x + bw >= W or y + bh >= H

    big = [i for i in range(1, num_labels)
           if stats[i, cv2.CC_STAT_AREA] >= min_area]
    # Ignore shadows / paper edges that touch the border, if possible
    candidates = [i for i in big if not touches_border(i)] or big
    if not candidates:
        return None, "No digit detected."

    # Keep the main piece plus other substantial pieces (multi-stroke digits)
    largest = max(stats[i, cv2.CC_STAT_AREA] for i in candidates)
    keep = [i for i in candidates if stats[i, cv2.CC_STAT_AREA] >= 0.1 * largest]
    mask = (np.isin(labels, keep) * 255).astype(np.uint8)

    # Thicken thin pen strokes to MNIST-like thickness (~12% of digit height)
    x, y, bw, bh = cv2.boundingRect(mask)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    stroke = 4.0 * dist[mask > 0].mean()
    target = 0.12 * max(bw, bh)
    r = int(round((target - stroke) / 2))
    if r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mask = cv2.dilate(mask, k)

    x, y, bw, bh = cv2.boundingRect(mask)
    digit = mask[y:y + bh, x:x + bw]

    # Resize to fit 20x20, keeping proportions
    scale = 20.0 / max(bw, bh)
    new_w = max(1, int(round(bw * scale)))
    new_h = max(1, int(round(bh * scale)))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Paste in the middle of a 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    xo, yo = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[yo:yo + new_h, xo:xo + new_w] = digit

    # Centre by centre of mass (like MNIST)
    m = cv2.moments(canvas)
    if m["m00"] != 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        shift = np.float32([[1, 0, 13.5 - cx], [0, 1, 13.5 - cy]])
        canvas = cv2.warpAffine(
            canvas, shift, (28, 28), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

    # Slight softening to mimic MNIST's anti-aliased strokes
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return canvas, None


# ---------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------

def load_mnist():
    if os.path.exists(MNIST_PATH):
        with np.load(MNIST_PATH) as d:
            return (d["x_train"], d["y_train"]), (d["x_test"], d["y_test"])
    return tf.keras.datasets.mnist.load_data()  # downloads if no local file


def build_and_train_model():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    L = tf.keras.layers
    model = tf.keras.Sequential([
        L.Input(shape=(28, 28, 1)),
        # Augmentation (only active during training) -> robust to real photos
        L.RandomRotation(0.08, fill_mode="constant", fill_value=0.0),
        L.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=0.0),
        L.RandomZoom(0.1, fill_mode="constant", fill_value=0.0),
        L.Conv2D(32, 3, activation="relu"),
        L.MaxPooling2D(),
        L.Conv2D(64, 3, activation="relu"),
        L.MaxPooling2D(),
        L.Flatten(),
        L.Dropout(0.3),
        L.Dense(128, activation="relu"),
        L.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    print("Training model...")
    model.fit(x_train, y_train, epochs=6, batch_size=128,
              validation_data=(x_test, y_test))
    print("Model training completed!")
    return model


# ---------------------------------------------------------------
# GUI
# ---------------------------------------------------------------

class DeepImages:

    def __init__(self):
        self.model = None
        self.model_error = None
        self.model_ready = threading.Event()
        self.image_path = None
        self.photo = None
        self.preview = None

        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("500x700")

        tk.Label(self.root, text="Handwritten Digit Recognition",
                 font=("Arial", 18, "bold")).pack(pady=20)

        tk.Button(self.root, text="Browse Image", font=("Arial", 12),
                  command=self.upload_image).pack(pady=10)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        self.predict_button = tk.Button(
            self.root, text="Predict Digit", font=("Arial", 12, "bold"),
            command=self.predict_digit, state="disabled")
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(
            self.root, text="Preparing model, please wait...",
            font=("Arial", 16, "bold"))
        self.result_label.pack(pady=10)

        tk.Label(self.root, text="What the model sees (28x28):",
                 font=("Arial", 10)).pack()
        self.preview_label = tk.Label(self.root)
        self.preview_label.pack(pady=5)

        # Train / load in the background so the window opens immediately
        threading.Thread(target=self._prepare_model, daemon=True).start()
        self.root.after(300, self._check_model_ready)

    def run(self):
        self.root.mainloop()

    # ---------------- model loading ----------------

    def _prepare_model(self):
        try:
            model = None
            if os.path.exists(MODEL_PATH):
                try:
                    model = tf.keras.models.load_model(MODEL_PATH)
                except Exception:
                    model = None  # corrupt / incompatible -> retrain
            if model is None:
                model = build_and_train_model()
                try:
                    model.save(MODEL_PATH)
                except Exception:
                    pass
            self.model = model
        except Exception as e:
            self.model_error = e
        finally:
            self.model_ready.set()

    def _check_model_ready(self):
        if not self.model_ready.is_set():
            self.root.after(300, self._check_model_ready)
            return
        if self.model_error is not None:
            self.result_label.config(text=f"Model error: {self.model_error}")
        else:
            self.predict_button.config(state="normal")
            self.result_label.config(text="Upload an image to begin.")

    # ---------------- upload ----------------

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            img = ImageOps.exif_transpose(Image.open(path))  # match OpenCV
            img.thumbnail((250, 250))
            self.photo = ImageTk.PhotoImage(img)
        except Exception:
            self.result_label.config(text="Unable to open image.")
            return

        self.image_path = path
        self.image_label.config(image=self.photo)
        self.preview_label.config(image="")
        self.result_label.config(text="Image uploaded. Click Predict Digit.")

    # ---------------- predict ----------------

    def predict_digit(self):
        if self.image_path is None:
            self.result_label.config(text="Please upload an image first.")
            return

        # np.fromfile + imdecode also works with non-ASCII paths on Windows
        try:
            data = np.fromfile(self.image_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        except Exception:
            img = None
        if img is None:
            self.result_label.config(text="Unable to read image.")
            return

        canvas, error = preprocess_digit(img)
        if error:
            self.result_label.config(text=error)
            return

        # Show what the model actually receives
        big = Image.fromarray(canvas).resize((112, 112), Image.NEAREST)
        self.preview = ImageTk.PhotoImage(big)
        self.preview_label.config(image=self.preview)

        x = (canvas.astype("float32") / 255.0).reshape(1, 28, 28, 1)
        probs = self.model(x, training=False).numpy()[0]
        digit = int(np.argmax(probs))
        confidence = float(probs[digit]) * 100

        text = f"Predicted Digit: {digit}\nConfidence: {confidence:.2f}%"
        if confidence < 60:
            text += "\n(low confidence - try a clearer photo)"
        self.result_label.config(text=text)


if __name__ == "__main__":
    DeepImages().run()