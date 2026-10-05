# CNN Tabanlı Görüntü Sınıflandırma Modeli

TensorFlow 2.x ve Keras kullanılarak geliştirilen çok katmanlı Evrişimli Sinir Ağları (Deep Convolutional Neural Networks) görüntü sınıflandırma mimarisi.

## 🧠 Model Mimarisi
- **Data Augmentation:** Rastgele döndürme, yatay çevirme ve yakınlaştırma ile aşırı öğrenme (overfitting) engellendi.
- **Evrişim Katmanları:** 32, 64, 128 ve 256 filtreli 4 aşamalı konvolüsyon hiyerarşisi.
- **Normalizasyon:** Her blokta Batch Normalization ve %15-%40 kademeli Dropout.
- **Global Average Pooling:** Düzleştirme (Flatten) yerine parametre sayısını azaltıp genelleme kabiliyetini artıran küresel havuzlama.

## 🚀 Çalıştırma
```bash
pip install tensorflow matplotlib numpy scikit-learn
python train_cnn.py
```

**Geliştirici:** Berkay Bilgin ([@BERKAY081014](https://github.com/BERKAY081014))\n