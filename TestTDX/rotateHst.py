import cv2
import numpy as np
import os
import glob
import csv

def rotate_image(image, angle):
    """
    辅助函数：保持中心旋转图像
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 这里的 borderValue=0 (黑色填充) 对二值图计算方差至关重要
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return rotated

def calculate_best_angle(img_binary, step=0.5, range_limit=2):
    """
    输入二值化图像，计算投影方差最大的角度
    """
    best_angle = 0
    max_variance = -1
    # step = 0.01
    # 生成角度序列 (-10 到 10)
    angles = np.arange(-range_limit, range_limit + step, step)

    for angle in angles:
        # 1. 旋转
        rotated = rotate_image(img_binary, angle)
        
        # 2. 统计行平均值 (原X轴方向的灰度平均值)
        # axis=1: 对每一行计算均值 -> 得到一个垂直方向的分布数组
        row_means = np.mean(rotated, axis=1)
        
        # 3. 计算方差
        variance = np.var(row_means)
        
        # 4. 比较最大值
        if variance > max_variance:
            max_variance = variance
            best_angle = angle
            
    return best_angle, max_variance

def batch_process_and_save_csv(folder_path):
    # --- 1. 准备文件列表 ---
    search_pattern = os.path.join(folder_path, "*.bmp")
    bmp_files = glob.glob(search_pattern)
    
    if not bmp_files:
        print(f"错误：在 '{folder_path}' 未找到BMP文件")
        return

    # 定义输出 CSV 文件的路径 (保存在图片同级目录下)
    csv_path = os.path.join(folder_path, "rotation_results.csv")
    
    print(f"找到 {len(bmp_files)} 张图片，开始处理...")
    print(f"结果将保存至: {csv_path}\n")

    # --- 2. 打开 CSV 文件准备写入 ---
    # newline='' 是为了防止在Windows下出现空行
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # 写入表头
        writer.writerow(["Filename", "Best_Angle", "Max_Variance"])

        # --- 3. 循环处理每一张图片 ---
        for index, file_path in enumerate(bmp_files):
            # 获取单纯的文件名 (例如 "image_01.bmp")
            file_name = os.path.basename(file_path)
            
            try:
                # 读取灰度图
                img = cv2.imread(file_path, 0)
                if img is None:
                    print(f"[{index+1}/{len(bmp_files)}] 跳过: 无法读取 {file_name}")
                    continue
                
                h, w = img.shape
                cut_h = int(h * 0.75) # 保留顶部 75% 的高度
                
                # numpy切片语法: [y_start : y_end, x_start : x_end]
                # 这里取 0 到 cut_h 行，所有列
                crop_img = img[0:cut_h, :] 
                # --- 预处理 (为了提高计算精度，必须做二值化) ---
                # A. 双边滤波去噪
                denoised = cv2.bilateralFilter(crop_img, 5, 30, 75)
                # B. Otsu 二值化 (背景黑，物体白)
                _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # --- 核心计算 ---
                # step=0.5 表示精度为 0.5度。如果需要更准，可以设为 0.1，但速度会慢5倍
                best_angle, max_var = calculate_best_angle(binary, step=0.01, range_limit=2)

                # --- 写入 CSV ---
                # 格式化角度保留2位小数
                writer.writerow([file_name, f"{best_angle:.2f}", f"{max_var:.2f}"])
                
                print(f"[{index+1}/{len(bmp_files)}] 处理完成: {file_name} -> 角度: {best_angle:.2f}°")

            except Exception as e:
                print(f"[{index+1}/{len(bmp_files)}] 出错: {file_name}, 错误信息: {e}")

    print("\n------------------------------------------------")
    print("全部处理完毕！")
    print(f"CSV文件已生成: {csv_path}")

# ==========================================
# 配置区域
# ==========================================
target_folder = r"D:\Codes\PI Inkjet\0_250731" 

if __name__ == '__main__':
    batch_process_and_save_csv(target_folder)