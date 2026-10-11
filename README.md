# MNIST El Yazısı Rakam Tanıma Projesi (PyTorch MLP)

Bu proje, MNIST veri setini kullanarak el yazısı rakamları sınıflandırmak amacıyla tasarlanmış, tamamen Multi-Layer Perceptron (MLP / Fully Connected) mimarisine dayalı bir derin öğrenme çözümüdür. Kural gereği Evrişimli Sinir Ağları (CNN) veya önceden eğitilmiş modeller kullanılmamıştır; model sıfırdan eğitilmektedir.

---

## 1. Mimari ve Hiperparametreler

- **Mimari:**
  - **Girdi Katmanı:** 784 düğüm ($28 \times 28$ piksellik düzleştirilmiş vektör)
  - **Gizli Katman 1:** 256 nöron + ReLU aktivasyon fonksiyonu
  - **Gizli Katman 2:** 128 nöron + ReLU aktivasyon fonksiyonu
  - **Çıktı Katmanı:** 10 sınıf nöronu (0–9 arası rakamlar için raw logits)
- **Kayıp Fonksiyonu:** `CrossEntropyLoss` (Raw logit çıktıları ile çalışır)
- **Optimizasyon:** `Adam` Optimizer (Öğrenme oranı / Learning Rate: 0.001)
- **Batch Size:** 64
- **Epoch Sayısı:** 15
- **Veri Ayrımı:**
  - Orijinal Train Kümesi (60.000 örnek): %85 Eğitim (51.000), %15 Doğrulama (9.000)
  - Orijinal Test Kümesi: 10.000 örnek (Eğitim bittikten sonra bağımsız test için kullanılır)

---

## 2. Proje Yapısı

