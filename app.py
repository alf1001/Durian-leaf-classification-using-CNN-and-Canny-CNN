import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import zipfile
import tempfile
import os
import pandas as pd

IMG_SIZE = (224, 224)

# =========================
# PILIH MODE
# =========================
mode = st.radio(
    "Pilih Mode Model:",
    ["Canny + CNN (4 Channel)", "CNN saja (RGB)"]
)

# =========================
# LOAD MODEL RGB
# =========================
def build_model(filters, dropout, learning_rate, num_classes):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224,224,3)),

        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

@st.cache_resource
def load_model_rgb():
    import zipfile, tempfile, os

    model = build_model(
        filters=None,
        dropout=0.5,
        learning_rate=0.001,
        num_classes=6
    )

    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile("best_model_fold_4(1).keras", 'r') as z:
        z.extractall(temp_dir)

    weight_path = os.path.join(temp_dir, "model.weights.h5")

    model.load_weights(weight_path)

    return model

    # =========================
    # LOAD WEIGHTS
    # =========================
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile("best_model_fold_4 (1).keras", 'r') as z:
        z.extractall(temp_dir)

    weight_path = os.path.join(temp_dir, "model.weights.h5")

    model.load_weights(weight_path)

    return model

# =========================
# LOAD MODEL CANNY-CNN
# =========================
@st.cache_resource
def load_model_canny():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224,224,4)),

        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile("best_model(1).keras", 'r') as z:
        z.extractall(temp_dir)

    weight_path = os.path.join(temp_dir, "model.weights.h5")
    model.load_weights(weight_path)

    return model


# =========================
# INIT MODEL
# =========================
model_rgb = None
model_canny = None

try:
    model_rgb = load_model_rgb()
    model_canny = load_model_canny()
    st.success("Semua model berhasil dimuat ✅")
except Exception as e:
    st.error("Model gagal dimuat ❌")
    st.text(str(e))


# =========================
# CLASS NAMES
# =========================
class_names = [
    "Bawor",
    "DuriHitam",
    "Matahari",
    "Montong",
    "Musangking",
    "SuperTembaga"
]


# =========================
# AUTO CROP
# =========================
def canny_auto_crop_leaf(img):

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    return img[y:y+h, x:x+w]


# =========================
# PREPROCESS CANNY
# =========================
def preprocess_canny(img):

    cropped = canny_auto_crop_leaf(img)

    rgb = cv2.resize(cropped, IMG_SIZE)
    rgb = rgb.astype(np.float32) / 255.0

    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 30, 100)

    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.medianBlur(edges, 3)

    edge = cv2.resize(edges, IMG_SIZE)
    edge = edge.astype(np.float32) / 255.0
    edge = np.expand_dims(edge, axis=-1)

    fused = np.concatenate([rgb, edge], axis=-1)

    return cropped, edges, fused


# =========================
# PREPROCESS RGB
# =========================
def preprocess_rgb(img):
    cropped = canny_auto_crop_leaf(img)
    rgb = cv2.resize(cropped, IMG_SIZE)
    rgb = rgb.astype(np.float32) / 255.0
    return cropped, rgb


# =========================
# OVERLAY
# =========================
def overlay_edges(rgb, edges):
    overlay = rgb.copy()
    overlay[edges > 0] = [255, 0, 0]
    return overlay


# =========================
# UI
# =========================
st.title("🌿 Klasifikasi Daun Durian")

uploaded_file = st.file_uploader("Upload gambar daun", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(image, caption="Gambar Asli", width="stretch")

    st.subheader("Tahapan Preprocessing")

    # =========================
    # MODE CANNY + CNN
    # =========================
    if mode == "Canny + CNN (4 Channel)":

        cropped, edges, fused = preprocess_canny(img)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(cropped, caption="Crop Daun", width="stretch")

        with col2:
            st.image(edges, caption="Canny Edge", width="stretch")

        with col3:
            st.image(fused[:,:,:3], caption="Input CNN (RGB)", width="stretch")

        overlay = overlay_edges(cropped, edges)
        st.image(overlay, caption="RGB + Canny Overlay", width="stretch")

        input_data = np.expand_dims(fused, axis=0)
        pred = model_canny.predict(input_data)

    # =========================
    # MODE CNN SAJA
    # =========================
    else:

        cropped, rgb = preprocess_rgb(img)

        st.image(cropped, caption="Input CNN (RGB)", width="stretch")

        input_data = np.expand_dims(rgb, axis=0)
        pred = model_rgb.predict(input_data)


    # =========================
    # HASIL
    # =========================
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    st.subheader("Hasil Prediksi")
    st.write(f"Jenis: **{class_names[class_idx]}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")

    # Top 3
    top_indices = np.argsort(pred[0])[::-1][:3]

    st.subheader("Top 3 Prediksi:")
    for i in top_indices:
        st.write(f"{class_names[i]}: {pred[0][i]*100:.2f}%")

    # Grafik
    prob_df = pd.DataFrame({
        "Kelas": class_names,
        "Probabilitas": pred[0]
    })

    st.bar_chart(prob_df.set_index("Kelas"))

    # Status
    if confidence < 0.6:
        st.warning("Model kurang yakin ⚠️")
    else:
        st.success("Model yakin terhadap prediksi ✅")