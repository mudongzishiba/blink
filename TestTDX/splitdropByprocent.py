import cv2
import numpy as np
import os
import random

# 用于计算液滴的占用面积
# 1. 修改这里：读取您的 BMP 图片

def get_random_bmp_file(directory):
    bmp_files = [f for f in os.listdir(directory) if f.lower().endswith('.bmp')]
    if not bmp_files:
        raise ValueError("指定目录中没有 BMP 文件")
    return os.path.join(directory, random.choice(bmp_files))

def cv_imread(file_path):
    # 使用 numpy 读取文件流，然后用 cv2 解码
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img

# --- 2. 图像增强算法：Gamma校正 ---
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# directory = r"D:\B9实验室项目\Inkjet智能膜面调节\0_250731"
directory = r"D:\B9实验室项目\Inkjet智能膜面调节\PI Inkjet\分类\正常吐出"


try:
    filepath = get_random_bmp_file(directory)
    print(f"随机选择的 BMP 文件路径: {filepath}")
    img = cv_imread(filepath)
    if img is None:
        print("错误：未找到图片或格式不支持")
        exit()
except Exception as e:
    print(f"发生错误: {e}")
    exit()

# ==========================================
#              核心修改区域开始
# ==========================================

# 1. 获取原图尺寸
h, w = img.shape[:2]

# 2. 定义 ROI (Region of Interest) 截取高度，即上方 80%
roi_h = int(h * 0.55)

# 3. 截取图片上方 60% 的区域进行处理
# img[y:y+h, x:x+w] -> 这里是 [0到60%高度, 所有宽度]
roi_img = img[0:roi_h, 0:w]

# --- 中间处理逻辑 (针对 ROI 区域进行) ---
gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

# gray_roi = adjust_gamma(gray_roi, gamma=0.5)  # 如果需要可以开启
# 对 ROI 图像进行高斯模糊处理
gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

# 阈值分割
_, binary_roi = cv2.threshold(gray_roi, 45, 255, cv2.THRESH_BINARY)

# 查找轮廓 (在 ROI 区域内查找)
contours, hierarchy = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- 结果可视化与合并 ---

# 创建全尺寸的 Mask (为了最后能和原图合并显示)
# 初始化全黑
mask_full = np.zeros((h, w), dtype=np.uint8)

# 将检测到的轮廓画在全尺寸 mask 上
# 注意：因为是从 (0,0) 开始截取的，ROI 中的坐标直接对应原图坐标，不需要偏移
cv2.drawContours(mask_full, contours, -1, 255, thickness=cv2.FILLED)

# 合并通道 (使用原图 img 和 全尺寸 mask)
b, g, r = cv2.split(img)
dst = cv2.merge([b, g, r, mask_full])

# 在结果图上画出轮廓 (绿色)
cv2.drawContours(image=dst, contours=contours, contourIdx=-1, color=(0, 255, 0, 255), thickness=1)

# 为了视觉确认，在结果图上画一条红线，显示 55% 的分界线
cv2.line(dst, (0, roi_h), (w, roi_h), (0, 0, 255, 255), 2)
cv2.putText(dst, "55% Cutoff", (10, roi_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255, 255), 2)

# --- 统计计算 ---
# 统计绿色区域面积 (液滴面积)
green_area = cv2.countNonZero(mask_full)

# 统计分析区域的总面积 (这里只计算 ROI 的面积，即上方 50% 的面积)
roi_total_area = roi_h * w

# 计算比例
green_ratio = green_area / roi_total_area

# ==========================================
#              核心修改区域结束
# ==========================================

# 展示出 dst，在图片上写出 green_ratio 的值
print(f"分析区域(Top 55%)总像素: {roi_total_area}")
print(f"液滴像素: {green_area}")
print(f"占比: {green_ratio:.2%}")

cv2.putText(dst, f"ROI Green Ratio: {green_ratio:.2%}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0, 255), 2)

# 显示窗口可调整大小
cv2.namedWindow('Result', cv2.WINDOW_NORMAL) 
cv2.imshow('Result', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
# cv2.imwrite('droplet_roi_result.png', dst)