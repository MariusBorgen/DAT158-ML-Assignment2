
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt

# Last ned ansiktsdatasettet
data = fetch_olivetti_faces()

X = data.images
y = data.target

print("Antall bilder:", len(X))
print("Bildestørrelse:", X[0].shape)
print("Antall personer:", len(set(y)))

# Vis de første 10 ansiktene
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(X[i], cmap="gray")
    ax.set_title(f"Person {y[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()
