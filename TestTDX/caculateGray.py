import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import csv

# === 设置你的 PNG 文件夹路径 ===
folder = r"D:\Codes\PI Inkjet\0_250731_Corrected_Connected"

# 找到所有 PNG 文件
png_files = sorted(glob.glob(os.path.join(folder, "*.png")))

if not png_files:
    print("没有找到 PNG 文件，请检查路径")
    exit()

# 读取第一个 PNG 文件
file = png_files[0]
print(f"读取文件：{file}")

# 读取图像（灰度）
img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
h, w = img.shape

# === 截取下方 5% ===
crop_h = int(h * 0.05)
cropped = img[h - crop_h : h, :]

# === 计算按列的灰度累积值 ===
col_sum = np.sum(cropped, axis=0)

# === 找峰值 ===
peak_x = np.argmax(col_sum)
peak_value = col_sum[peak_x]
print(f"峰值位置（X）: {peak_x}, 峰值灰度: {peak_value}")

# === 将结果保存为 CSV ===
csv_path = os.path.join(folder, "gray_projection.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["X", "GraySum"])
    for x, val in enumerate(col_sum):
        writer.writerow([x, val])

print(f"CSV 已保存到: {csv_path}")

# === 作图 ===
plt.figure(figsize=(10, 4))
plt.plot(col_sum, label="Gray Sum")

# 标注峰值
plt.scatter(peak_x, peak_value, color="red")
plt.text(peak_x, peak_value, f"Peak: {peak_x}", color="red", fontsize=10, ha='left')

plt.title("Gray Projection (Bottom 5%)")
plt.xlabel("X pixel position")
plt.ylabel("Gray sum over bottom 5%")
plt.grid(True)
plt.tight_layout()
plt.show()
