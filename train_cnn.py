#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Proje: CNN Tabanlı Görüntü Sınıflandırma Modeli (Derin Öğrenme)
Geliştirici: Berkay Bilgin (https://github.com/BERKAY081014)
Kütüphaneler: TensorFlow 2.x, Keras, NumPy, Matplotlib, Scikit-learn
Model: Custom Deep Convolutional Neural Network with Dropout & Batch Normalization
===============================================================================
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# Donanım Hızlandırma Kontrolü (GPU / CPU)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"[BİLGİ] GPU Etkinleştirildi: {physical_devices[0]}")
else:
    print("[BİLGİ] CPU modunda çalışılıyor.")

# ---------- Hiperparametreler ----------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 2
LEARNING_RATE = 1e-4

def build_cnn_model(input_shape=(128, 128, 3), num_classes=2):
    """
    Özelleştirilmiş 4 bloklu CNN Mimarisi
    Conv2D -> BatchNorm -> ReLU -> MaxPooling -> Dropout
    """
    model = models.Sequential([
        # 1. Giriş ve Veri Artırma Katmanları (Data Augmentation)
        layers.Input(shape=input_shape),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.1),

        # 1. Evrişim Bloğu (32 Filtre)
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.15),

        # 2. Evrişim Bloğu (64 Filtre)
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        # 3. Evrişim Bloğu (128 Filtre)
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),

        # 4. Evrişim Bloğu (256 Filtre)
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Küresel Ortalama Havuzlama ve Tam Bağlantılı (Dense) Katman
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # Çıkış Katmanı (İkili Sınıflandırma için Sigmoid / Çoklu için Softmax)
        layers.Dense(1 if num_classes == 2 else num_classes, 
                     activation="sigmoid" if num_classes == 2 else "softmax")
    ])

    return model

def create_synthetic_dataset():
    """
    Bağımsız çalıştırılabilir test için sentetik veri üreteci
    """
    print("[VERİ] 800 örnekli eğitim veri seti simüle ediliyor...")
    x = np.random.uniform(0.0, 1.0, size=(800, 128, 128, 3)).astype(np.float32)
    y = np.random.randint(0, 2, size=(800, 1)).astype(np.float32)
    return x, y

def main():
    print("=" * 60)
    print("Evrişimli Sinir Ağı (CNN) Model Eğitimi - Berkay Bilgin")
    print("=" * 60)

    model = build_cnn_model(input_shape=(128, 128, 3), num_classes=NUM_CLASSES)
    model.summary()

    # Optimizasyon ve Kayıp Fonksiyonu
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy" if NUM_CLASSES == 2 else "categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )

    # Geri Çağırma (Callbacks) Mekanizmaları
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    # Veri hazırlığı
    x_train, y_train = create_synthetic_dataset()
    x_val, y_val = x_train[:150], y_train[:150]
    x_train, y_train = x_train[150:], y_train[150:]

    print(f"\n[EĞİTİM] {EPOCHS} epoch boyunca eğitim başlatılıyor...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=cb_list,
        verbose=1
    )

    # Modeli Kaydet
    output_model_path = "berkay_cnn_model.keras"
    model.save(output_model_path)
    print(f"\n[BAŞARILI] Eğitilen model '{output_model_path}' olarak kaydedildi.")

if __name__ == "__main__":
    main()\n