# PROGIDY_WD_03
Implemented a Support Vector Machine (SVM) to classify Cats vs Dogs using the Kaggle PetImages dataset. Preprocessed images with resizing, grayscale conversion, and normalization. Trained an SVM classifier achieving baseline accuracy, with scope for improvement using feature extraction.
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ✅ Dataset path (make sure it points to PetImages folder)
data_dir = r"C:\Users\ankia\OneDrive\Desktop\internship pro\archive\PetImages"

categories = ["Cat", "Dog"]
img_size = 64                  # size for training (small & fast)
preview_size = 128             # size for clear preview
max_images_per_class = 1000    # ✅ limit to 1000 images per class

data = []
labels = []
preview_images = []   # store clear images for display
preview_labels = []

print("Loading images...")

for category in categories:
    folder = os.path.join(data_dir, category)
    label = categories.index(category)

    count = 0
    for img_name in os.listdir(folder):
        if count >= max_images_per_class:
            break

        img_path = os.path.join(folder, img_name)
        try:
            # ---------- Training image (grayscale, small size) ----------
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue
            img_gray = cv2.resize(img_gray, (img_size, img_size))
            img_gray = img_gray.flatten() / 255.0   # normalize
            data.append(img_gray)
            labels.append(label)

            # ---------- Preview image (color, bigger size) ----------
            if count < 5:   # only store first 5 per class for preview
                img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
                img_color = cv2.resize(img_color, (preview_size, preview_size))
                preview_images.append(img_color)
                preview_labels.append(label)

            count += 1
        except Exception as e:
            continue

data = np.array(data)
labels = np.array(labels)

print(f"Dataset loaded: {len(data)} samples")

# ✅ Show preview images (clear color, 128x128)
for i in range(len(preview_images)):
    plt.imshow(preview_images[i])
    plt.title("Cat" if preview_labels[i] == 0 else "Dog")
    plt.axis("off")
    plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train SVM
print("Training SVM...")
svm = SVC(kernel='linear', verbose=True)
svm.fit(X_train, y_train)

# Prediction
y_pred = svm.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")
