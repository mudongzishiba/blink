import cv2
import numpy as np
import os
import random

from torch import ge

# 用于计算液滴的占用面积
# 1. 修改这里：读取您的 BMP 图片
# 注意：OpenCV 读取 BMP 不需要额外参数，它会自动处理

# 改为从指定路径里随机挑选一张bmp图片

def get_random_bmp_file(directory):
    bmp_files = [f for f in os.listdir(directory) if f.lower().endswith('.bmp')]
    if not bmp_files:
        raise ValueError("指定目录中没有 BMP 文件")
    return os.path.join(directory, random.choice(bmp_files))

def cv_imread(file_path):
    # 使用 numpy 读取文件流，然后用 cv2 解码
    # cv2.IMREAD_COLOR: 读取彩色图
    # cv2.IMREAD_UNCHANGED: 如果图片本来就有透明通道或其他格式，用这个
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img

# --- 2. 图像增强算法：Gamma校正 ---
def adjust_gamma(image, gamma=1.0):
    # 建立查找表，加快计算速度
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

directory = r"D:\B9实验室项目\Inkjet智能膜面调节\0_250731"
# file = r"250731_134211_0000000361_0_&Cam1Img.bmp"
filepath = get_random_bmp_file(directory)
# 打印filepath
# filepath = os.path.join(directory, file)
print(f"随机选择的 BMP 文件路径: {filepath}")
img = cv_imread(filepath)
if img is None:
    print("错误：未找到图片或格式不支持")
    exit()


# --- 中间处理逻辑完全一样 ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gray = adjust_gamma(gray, gamma=0.5)  # 调整gamma值，增强对比度
# 对 gray图像进行高斯模糊处理
gray = cv2.GaussianBlur(gray, (5, 5), 0)


# 阈值分割 (如果bmp图片质量很高，背景纯黑，这个阈值25通常效果很好)
_, binary = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建Mask
mask = np.zeros_like(gray)
cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# 合并通道
b, g, r = cv2.split(img)
dst = cv2.merge([b, g, r, mask])
cv2.drawContours(image=dst, contours=contours, contourIdx=-1, color=(0, 255, 0, 255), thickness=1)

#分别统计绿色围绕区域面积和总面积，计算绿色所占比例
total_area = mask.size
green_area = cv2.countNonZero(mask)
green_ratio = green_area / total_area

# 展示出dst，在图片上写出green_ratio的值
cv2.putText(dst, f"Green Ratio: {green_ratio:.2%}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0, 255), 2)
cv2.imshow('Result', dst)
cv2.waitKey(0)
cv2.destroyAllWindows() 
                
# -------------------------

# 3. 修改这里：保存结果
# 强烈建议保存为 .png 以保留透明通道
# cv2.imwrite('droplet_result.png', dst)

# 如果您必须保存为 bmp (注意：很多软件可能无法正确显示透明bmp)
# cv2.imwrite('droplet_result.bmp', dst)
# img = cv2.imread("D:\\B9实验室项目\\Inkjet智能膜面调节\\0_250731\\250731_134217_0000000529_0_&Cam1Img.bmp", cv2.IMREAD_GRAYSCALE)

# if __name__ == '__main__':
#     main()
