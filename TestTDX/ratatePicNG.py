import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# 1. 读取原始图像
# =========================================
path = "D:\\Codes\\PI Inkjet\\connecttest\\250731_134232_0000000944_0_&Cam1Img.bmp"
img = cv2.imread(path, 0)
H, W = img.shape

# =========================================
# 2. 图像增强（去噪 + 放大 + 锐化）
# =========================================
med = cv2.medianBlur(img, 3)
bil = cv2.bilateralFilter(med, 9, 75, 75)
kernel = np.ones((3,3), np.uint8)
morph = cv2.morphologyEx(bil, cv2.MORPH_OPEN, kernel)
up = cv2.resize(morph, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
contrast = clahe.apply(up)

blur = cv2.GaussianBlur(contrast, (0,0), 1.2)
sharp = cv2.addWeighted(contrast, 1.6, blur, -0.6, 0)

g1 = cv2.GaussianBlur(sharp, (0,0), 1.0)
g2 = cv2.GaussianBlur(sharp, (0,0), 2.0)
dog = cv2.subtract(g1, g2)
enhanced = cv2.addWeighted(sharp, 1.0, dog, 0.6, 0)

# =========================================
# 3. 提取圆心
# =========================================
blur2 = cv2.GaussianBlur(enhanced, (5,5), 0)
edges = cv2.Canny(blur2, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

centers = []
for c in contours:
    M = cv2.moments(c)
    if M["m00"] != 0:
        centers.append([M["m10"]/M["m00"], M["m01"]/M["m00"]])

centers = np.array(centers)
mean_center = centers.mean(axis=0)

# =========================================
# 4. 计算旋转角度（PCA 主方向）
# =========================================
X = centers - mean_center
cov = np.cov(X.T)
eigvals, eigvecs = np.linalg.eig(cov)
v = eigvecs[:, np.argmax(eigvals)]

theta = np.arctan2(v[1], v[0]) * 180 / np.pi
if theta > 45: theta -= 90
if theta < -45: theta += 90

# =========================================
# 5. 旋正（去倾斜）
# =========================================
(h2, w2) = enhanced.shape
rot_matrix = cv2.getRotationMatrix2D((w2/2, h2/2), theta, 1.0)
rotated = cv2.warpAffine(enhanced, rot_matrix, (w2, h2), flags=cv2.INTER_CUBIC)

# =========================================
# 6. 再次计算圆心用于居中
# =========================================
blur3 = cv2.GaussianBlur(rotated, (5,5), 0)
edges_r = cv2.Canny(blur3, 50, 150)
contours_r, _ = cv2.findContours(edges_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

centers_r = []
for c in contours_r:
    M = cv2.moments(c)
    if M["m00"] != 0:
        centers_r.append([M["m10"]/M["m00"], M["m01"]/M["m00"]])

centers_r = np.array(centers_r)
mean_center_r = centers_r.mean(axis=0)

# =========================================
# 7. 平移，使液晶阵列居中
# =========================================
# 图像中心
img_center = np.array([w2/2, h2/2])

# 阵列中心相对图像中心的位移（阵列中心 - 图像中心）
dx = mean_center_r[0] - img_center[0]
dy = mean_center_r[1] - img_center[1]

# warpAffine 需要反向移动图像，所以取负号
trans = np.float32([
    [1, 0, -dx],
    [0, 1, -dy]
])

aligned = cv2.warpAffine(rotated, trans, (w2, h2), flags=cv2.INTER_CUBIC)


# =========================================
# 8. 显示最终结果
# =========================================
plt.figure(figsize=(10,10))
plt.imshow(aligned, cmap='gray')
plt.title("Final Aligned Image (Rotation + Centering)")
plt.axis("off")
plt.show()

cv2.imwrite("/mnt/data/final_aligned.bmp", aligned)

print("图像已成功旋正并居中 → /mnt/data/final_aligned.bmp")
