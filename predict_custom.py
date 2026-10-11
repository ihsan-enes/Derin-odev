import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.model import MNIST_MLP

def preprocess_image(image_path: str):
    # 1. Gri tona çevir ve kontrastı uç noktalara çek
    original_img = Image.open(image_path).convert('L')
    img_contrast = ImageOps.autocontrast(original_img)
    
    # 2. MNIST formatı: Siyah zemin, beyaz yazı
    inverted_img = ImageOps.invert(img_contrast)
    
    # 3. Adaptif Eşikleme (Binarize) - Fotoğraftaki kağıt gölgelerini tamamen yok eder
    np_img = np.array(inverted_img)
    thresh = np.mean(np_img) + 0.4 * np.std(np_img)
    binary = np.where(np_img > thresh, 255, 0).astype(np.uint8)
    bin_img = Image.fromarray(binary)
    
    # 4. Çizgileri Kalınlaştırma (Kurşun kalem inceliğini MNIST dolgunluğuna getirir)
    bin_img = bin_img.filter(ImageFilter.MaxFilter(3))
    
    # 5. Rakamı sınırlayıcı kutusuna (bounding box) göre kesip kare tuvalde ortala
    bbox = bin_img.getbbox()
    if bbox:
        cropped = bin_img.crop(bbox)
    else:
        cropped = bin_img

    w, h = cropped.size
    max_dim = max(w, h)
    
    # MNIST standartlarına uygun kenar payı (%25 siyah boşluk)
    pad_size = int(max_dim * 1.5)
    canvas = Image.new('L', (pad_size, pad_size), 0)
    canvas.paste(cropped, ((pad_size - w) // 2, (pad_size - h) // 2))
    
    # 6. 28x28 boyutuna indirgeme
    resized_img = canvas.resize((28, 28), Image.Resampling.BILINEAR)

    # 7. MNIST Normalizasyonu ile Tensöre dönüştür
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    tensor_img = transform(resized_img)
    flattened_tensor = tensor_img.view(1, 784)
    
    return original_img, resized_img, flattened_tensor

def main():
    custom_dir = 'custom_digits'
    model_path = 'models/best_mlp_model.pth'
    report_output_path = 'reports/custom_predictions.png'
    
    os.makedirs('reports', exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Hata: Model ağırlıkları bulunamadı: {model_path}. Lütfen önce train.py dosyasını çalıştırın.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNIST_MLP(input_size=784, hidden1_size=256, hidden2_size=128, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    supported_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    image_paths = []
    for ext in supported_extensions:
        image_paths.extend(glob.glob(os.path.join(custom_dir, ext)))

    if not image_paths:
        print(f"'{custom_dir}' klasöründe geçerli resim dosyası bulunamadı. Lütfen test etmek istediğiniz görüntüleri bu klasöre ekleyin.")
        return

    image_paths.sort()
    num_images = len(image_paths)

    fig, axes = plt.subplots(num_images, 2, figsize=(6, 3 * num_images))
    if num_images == 1:
        axes = [axes]

    print("\n--- Kendi Rakam Görselleri İçin Tahmin Sonuçları ---")
    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            original_img, processed_img, input_tensor = preprocess_image(img_path)
            input_tensor = input_tensor.to(device)

            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

            pred_digit = predicted_class.item()
            conf_percent = confidence.item() * 100.0

            file_name = os.path.basename(img_path)
            print(f"Dosya: {file_name:<20} | Tahmin Edilen: {pred_digit} | Güven Skoru: %{conf_percent:.2f}")

            # Orijinal Görseli Göster
            axes[idx][0].imshow(original_img, cmap='gray')
            axes[idx][0].set_title(f"Orijinal: {file_name}")
            axes[idx][0].axis('off')

            # 28x28 Model Girdisini Göster
            axes[idx][1].imshow(processed_img, cmap='gray')
            axes[idx][1].set_title(f"Model Girdisi (28x28)\nTahmin: {pred_digit} (%{conf_percent:.1f})")
            axes[idx][1].axis('off')

    plt.tight_layout()
    plt.savefig(report_output_path)
    plt.close()
    print(f"\nKarşılaştırma görseli kaydedildi: {report_output_path}")

if __name__ == '__main__':
    main()