import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def rotate_image(image, angle):
    """
    辅助函数：保持中心旋转图像
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1)
    
    # 执行旋转，背景填充黑色
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return rotated

def find_best_rotation_angle(img, step=0.5, range_limit=10):
    """
    在 ±range_limit 范围内，以 step 为步长旋转。
    计算每一行的平均灰度，求其方差。
    返回方差最大的角度和矫正后的图像。
    """
    
    best_angle = 0
    max_variance = -1
    best_image = None
    
    # 记录数据用于画图分析 (可选)
    angles_list = []
    variances_list = []

    # 生成角度序列: -10, -9.5, ... 0 ... 9.5, 10
    # np.arange 不包含终点，所以加一点余量
    angles = np.arange(-range_limit, range_limit + step, step)

    print(f"开始计算旋转方差 (范围: ±{range_limit}°, 步长: {step}°)...")

    # 全局灰度平均值 (其实在方差计算公式中，对比的是当前旋转状态下的均值，
    # 但为了严格符合你的描述，我们可以计算 row_means 的方差)
    
    for angle in angles:
        # 1. 旋转
        rotated = rotate_image(img, angle)
        
        # 2. 统计原X轴方向的灰度平均值 
        # axis=1 表示沿着水平方向(X轴)计算平均值，结果是每一行(Y)有一个值
        row_means = np.mean(rotated, axis=1)
        
        # 3. 计算方差
        # np.var 计算的是这些行平均值相对于它们整体平均值的离散程度
        variance = np.var(row_means)
        
        # 记录
        angles_list.append(angle)
        variances_list.append(variance)
        
        # 4. 更新最大值
        if variance > max_variance:
            max_variance = variance
            best_angle = angle
            best_image = rotated

    print(f"计算完成。最大方差: {max_variance:.2f}，最佳角度: {best_angle}°")
    
    return best_angle, best_image, angles_list, variances_list

def main(folder_path):
    # --- 1. 读取图片 ---
    bmp_files = glob.glob(os.path.join(folder_path, "*.bmp"))
    if not bmp_files:
        print("未找到图片")
        return
    
    # 读取原始灰度图
    original = cv2.imread(bmp_files[0], 0)
    
    # --- 2. 预处理 (建议) ---
    # 为了让方差计算更准确，建议先二值化，把背景变纯黑(0)，物体变纯白(255)
    # 这样对比度最强，方差峰值最明显
    denoised = cv2.bilateralFilter(original, 5, 30, 75)
    # 简单的OTSU二值化
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # --- 3. 执行核心算法 ---
    # 输入二值化图进行计算（精度更高），但最后你可以应用这个角度到原图上
    step_angle = 0.01 # 步进角度，越小越慢但越准 (例如 0.1)
    best_angle, corrected_binary, angles, variances = find_best_rotation_angle(binary, step=step_angle, range_limit=10)

    # 将计算出的最佳角度应用到原图(灰度图)上
    final_result = rotate_image(original, best_angle)

    # --- 4. 绘制方差变化曲线 (直观展示) ---
    plt.figure(figsize=(10, 5))
    plt.plot(angles, variances, label='Variance Profile')
    plt.axvline(x=best_angle, color='r', linestyle='--', label=f'Best Angle: {best_angle}°')
    plt.title("Variance vs Rotation Angle")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Variance of Row Means")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 5. 显示图像结果 ---
    def show(name, img, scale=3):
        h, w = img.shape[:2]
        cv2.imshow(name, cv2.resize(img, (int(w*scale), int(h*scale))))

    show("Original", original)
    show(f"Corrected ({best_angle} deg)", final_result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

target_folder = r"D:\Codes\PI Inkjet\connecttest" 
if __name__ == '__main__':
    main(target_folder)