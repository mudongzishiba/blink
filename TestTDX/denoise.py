import cv2
import numpy as np
import os
import glob

def process_industrial_image(folder_path):
    # --- 1. 获取第一张BMP ---
    search_pattern = os.path.join(folder_path, "*.bmp")
    bmp_files = glob.glob(search_pattern)
    if not bmp_files:
        print(f"错误：在 '{folder_path}' 未找到BMP文件")
        return
    first_image_path = bmp_files[0]
    print(f"正在处理: {first_image_path}")

    img = cv2.imread(first_image_path, 0) # 读取灰度
    if img is None: return

    # ========================================================
    # --- 改进点：使用双边滤波替代高斯滤波 ---
    # ========================================================
    # 参数说明：
    # d (9): 过滤时周围像素的直径。通常 5-9 之间。
    # sigmaColor (75): 颜色/灰度空间的标准差。
    #      值越大 -> 允许跨越的灰度差异越大（越接近高斯模糊）。
    #      值越小 -> 只有灰度非常接近的像素才会被混合（保留更多细节）。
    #      建议尝试：50 ~ 100 之间。
    # sigmaSpace (75): 坐标空间的标准差。控制空间范围。
    
    denoised = cv2.bilateralFilter(img, d=5, sigmaColor=45, sigmaSpace=75)

    # 备选方案：如果双边滤波还是觉得不够干净，可以尝试“中值滤波”
    # 中值滤波对“颗粒感”很强的椒盐噪声效果最好，且不模糊边缘
    # denoised = cv2.medianBlur(img, 3) 

    # ========================================================
    # --- 后续：背景分离 (形态学) ---
    # ========================================================
    
    # 核心：根据你的物体大小调整 kernel_size
    # 如果物体很细小，必须调小这个值 (比如 (15,15))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 假设：背景是灰的，物体/瑕疵比背景【亮】 -> 使用 TopHat
    # 如果物体比背景【暗】，请改用 cv2.MORPH_BLACKHAT
    background_removed = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)

    # 增强：为了让分割更清晰，可以稍微拉伸一下对比度
    # 将灰度范围拉伸到 0-255
    normalized = cv2.normalize(background_removed, None, 0, 255, cv2.NORM_MINMAX)

    # 二值化
    ret, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- 显示结果 ---
    def show(name, img, scale=4):
        h, w = img.shape[:2]
        cv2.imshow(name, cv2.resize(img, (int(w*scale), int(h*scale))))

    show("Original", img)
    show("Bilateral Denoised", denoised) # 观察这张图，确认噪点没了但边缘还在
    show("Result", binary)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

target_folder = r"D:\Codes\PI Inkjet\connecttest" 
if __name__ == '__main__':
    process_industrial_image(target_folder)