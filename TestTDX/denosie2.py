import cv2
import numpy as np
import os
import glob

def process_and_fill_circles(folder_path):
    # --- 1. 读取图片 ---
    bmp_files = glob.glob(os.path.join(folder_path, "*.bmp"))
    if not bmp_files:
        print("未找到图片")
        return
    
    img = cv2.imread(bmp_files[0], 0) # 读取灰度图
    if img is None: return

    # --- 2. 降噪 (双边滤波) ---
    # 去除噪点，同时保留边缘
    denoised = cv2.bilateralFilter(img, d=5, sigmaColor=45, sigmaSpace=75)

    # --- 3. 背景去除 (顶帽变换) ---
    # 原理：原图 - 开运算图。
    # 效果：大面积的灰色背景会变成黑色(0)，亮色物体保留。
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel_bg)

    # --- 4. 二值化 ---
    # 简单的固定阈值或OTSU，将残留的微弱背景彻底变黑
    # 这里的阈值 20 可以根据实际情况微调
    ret, binary = cv2.threshold(tophat, 25, 255, cv2.THRESH_BINARY)

    # --- 5. 核心步骤：查找轮廓并使用凸包填充 ---
    # 这一步能同时解决两个问题：
    # A. 内部是灰色的空心圆 -> 变成实心
    # B. 边缘断裂不完整的圆 -> 强制补全
    
    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建一张纯黑画布用于绘制最终结果
    final_result = np.zeros_like(img)

    for cnt in contours:
        # 面积过滤：忽略掉太小的噪点 (例如小于 50 像素)
        if cv2.contourArea(cnt) > 9:
            # 获取凸包 (Convex Hull)
            # 凸包会把断裂的“C”形或空心的“O”形，全部包成一个实心区域
            hull = cv2.convexHull(cnt)
            
            # 在画布上用白色(255)填充凸包区域
            cv2.drawContours(final_result, [hull], -1, 255, thickness=cv2.FILLED)

    # --- 6. 显示结果 ---
    def show(name, img, scale=2):
        h, w = img.shape[:2]
        cv2.imshow(name, cv2.resize(img, (int(w*scale), int(h*scale))))

    show("Original", img)
    show("Processed (Binary)", binary)      # 此时可能还是断裂/空心的
    show("Final Result (Filled)", final_result) # 最终补全后的实心圆

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 你的文件夹路径
target_folder = r"D:\Codes\PI Inkjet\connecttest" 

if __name__ == '__main__':
    process_and_fill_circles(target_folder)