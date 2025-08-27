from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- 1. Veri Setini Yükleme ---
oli = fetch_olivetti_faces()
X = oli.data
y = oli.target

# --- 2. Eğitim ve Test Verisine Ayırma ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

# --- 3. Örnek Yüzleri Görselleştirme ---
plt.figure(figsize=(6, 3))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(oli.images[i+40], cmap="gray")
    plt.axis("off")
plt.suptitle("Olivetti Faces Sample")
plt.show()

# --- 4. Random Forest ile n_estimators Denemeleri ---
accuracy_list = []
estimators = list(range(100, 501, 100))  # 100, 200, 300, 400, 500

for n in estimators:
    rf_clf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_list.append(acc)

# --- 5. En Yüksek Doğruluk ve Kaç Ağaçla Geldiğini Yazdır ---
best_index = accuracy_list.index(max(accuracy_list))
print(f"Max Accuracy: {accuracy_list[best_index]:.4f} with n_estimators = {estimators[best_index]}")

# --- 6. Doğrulukları Grafikle Göster ---
plt.plot(estimators, accuracy_list, marker="o", linestyle="-")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.title("Random Forest Performance on Olivetti Faces")
plt.grid(True)
plt.show()

