import numpy as np 


import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

# ماتریس نمونه نامنفی
V = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

print("ماتریس اصلی V:")
print(V)

# تعداد ویژگی‌های مخفی (hidden features)
n_components = 2

# ایجاد مدل NMF
model = NMF(n_components=n_components, init='random', random_state=42)

# فیت کردن مدل روی داده‌ها
W = model.fit_transform(V)
H = model.components_

print("\nماتریس W (ضرایب پایه):")
print(np.round(W, 2))

print("\nماتریس H (پایه‌های مخفی):")
print(np.round(H, 2))

# بازسازی ماتریس V از W و H
V_reconstructed = np.dot(W, H)
print("\nماتریس بازسازی شده V ≈ W × H:")
print(np.round(V_reconstructed, 2))

# نمایش گرافیکی مقایسه
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(V, cmap='viridis')
axes[0].set_title("Original V")
axes[1].imshow(V_reconstructed, cmap='viridis')
axes[1].set_title("Reconstructed V")
plt.tight_layout()
plt.show()