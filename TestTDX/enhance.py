import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ===============================
# 参数设置
# ===============================
folder_path = r"D:\Codes\PI Inkjet\connecttest"  # 你的 BMP 文件夹路径
scale_factor = 8              # 放大倍数（例如 2 倍、4 倍）

# ===============================
# Step 1：读取文件夹第一张 BMP
# ===============================
bmp_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".bmp")]
bmp_files.sort()

if not bmp_files:
    raise ValueError("文件夹内没有 BMP 文件")

input_path = os.path.join(folder_path, bmp_files[0])
img = cv2.imread(input_path, 0)   # 灰度读取

# ===============================
# Step 2：双三次插值放大（最清晰）
# ===============================
h, w = img.shape
new_w = int(w * scale_factor)
new_h = int(h * scale_factor)

up = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

# ===============================
# Step 3：多步骤背景去噪
# ===============================

# 3.1 中值滤波（去散点）
med = cv2.medianBlur(up, 3)

# 3.2 双边滤波（保边去噪）
bil = cv2.bilateralFilter(med, d=9, sigmaColor=75, sigmaSpace=75)

# 3.3 形态学开运算（去孤立点）
kernel = np.ones((3, 3), np.uint8)
denoised = cv2.morphologyEx(bil, cv2.MORPH_OPEN, kernel)

# ===============================
# Step 4：增强圆环边界强度
# ===============================

# 4.1 CLAHE 对比度增强
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
contrast = clahe.apply(denoised)

# 4.2 工业级 Unsharp Mask 锐化
blur = cv2.GaussianBlur(contrast, (0,0), sigmaX=1.2)
sharp = cv2.addWeighted(contrast, 1.6, blur, -0.6, 0)

# 4.3 DoG（Difference of Gaussian）强化圆环结构
g1 = cv2.GaussianBlur(sharp, (0,0), sigmaX=1.0)
g2 = cv2.GaussianBlur(sharp, (0,0), sigmaX=2.0)
dog = cv2.normalize(cv2.subtract(g1, g2), None, 0, 255, cv2.NORM_MINMAX)
ring_enhanced = cv2.addWeighted(sharp, 1.0, dog, 0.6, 0)

# 输入图像（ring_enhanced）
gray = ring_enhanced.copy()

# 1. 二值化（找到亮环区域）
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 2. 找轮廓（外圈即可）
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 3. 创建黑色背景
filled_circles = np.zeros_like(gray)

# 4. 直接把每个外轮廓填满（实心白色圆）
cv2.drawContours(filled_circles, contours, -1, 255, thickness=cv2.FILLED)


img_gray = ring_enhanced.copy()

# -------- 1. 边缘检测 --------
blur = cv2.GaussianBlur(img_gray, (5,5), 0)
edges = cv2.Canny(blur, 50, 150)

# -------- 2. 查找轮廓（外圈）--------
contours, _ = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# -------- 3. 转为伪彩色图像，用绿色绘制更明显 --------
img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# -------- 4. 用绿色细线绘制轮廓 --------
cv2.drawContours(img_color, contours, -1, (0,255,0), 1)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(gray, cmap='gray')
plt.title("Input: ring_enhanced")
plt.axis("off")

plt.figure(figsize=(10,8))
plt.imshow(img_color[:,:,::-1])
plt.title("Ring Boundaries (Green Contours)")
plt.axis("off")
plt.show()

plt.subplot(1,2,2)
plt.imshow(filled_circles, cmap='gray')
plt.title("Output: Solid White Circles")
plt.axis("off")

plt.show()
