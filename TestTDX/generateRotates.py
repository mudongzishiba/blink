import cv2
import numpy as np
import os
import glob
import csv

# 用于批量将文件夹里面的图片进行旋转，好用

def rotate_image(image, angle):
    """
    绕中心旋转图像，保持原图尺寸，填充黑色背景
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 关键点：
    # getRotationMatrix2D 的第二个参数是旋转角度
    # 正数表示逆时针旋转，负数表示顺时针旋转
    # 如果之前的算法算出的是“物体倾斜角”，要把它扶正通常需要取反（-angle）
    # 这里默认按照传入的 angle 旋转，如果方向反了，请修改为 -angle
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # borderValue=(0,0,0) 表示旋转后的空余部分填充黑色
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return rotated

def batch_rotate_and_save(folder_path, use_csv=True, fixed_angle=0):
    # --- 1. 确定新文件夹路径 (同一级) ---
    # 获取父目录
    parent_dir = os.path.dirname(folder_path.rstrip(os.sep))
    # 获取当前文件夹名
    current_folder_name = os.path.basename(folder_path.rstrip(os.sep))
    # 新文件夹名字
    new_folder_name = current_folder_name + "_Corrected"
    new_folder_path = os.path.join(parent_dir, new_folder_name)

    # 创建新文件夹
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"新建文件夹: {new_folder_path}")
    else:
        print(f"输出文件夹已存在: {new_folder_path}")

    # --- 2. 读取 CSV 中的角度信息 (如果需要) ---
    angle_map = {} # 字典: {'文件名': 角度}
    csv_path = os.path.join(folder_path, "rotation_results.csv")
    
    if use_csv:
        if os.path.exists(csv_path):
            print(f"正在读取角度文件: {csv_path}")
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None) # 跳过表头
                for row in reader:
                    if len(row) >= 2:
                        # row[0]是文件名, row[1]是最佳角度
                        filename = row[0]
                        angle = float(row[1])
                        angle_map[filename] = angle
        else:
            print("警告：未找到 CSV 文件，将使用默认固定角度 0度 (或请手动设置 fixed_angle)")
            use_csv = False # 降级为普通模式

    # --- 3. 遍历图片并处理 ---
    bmp_files = glob.glob(os.path.join(folder_path, "*.bmp"))
    print(f"找到 {len(bmp_files)} 张图片，开始旋转处理...")

    for i, file_path in enumerate(bmp_files):
        file_name = os.path.basename(file_path)
        
        # 确定旋转角度
        current_angle = 0
        if use_csv:
            # 从字典里查表，查不到就默认为0
            # 注意：之前的算法算出的是物体的倾斜角。
            # 如果物体向左歪了5度，我们需要向右转5度来矫正。
            # 通常矫正角度 = - (检测角度)
            # 请根据实际结果观察：如果是越转越歪，请去掉下面的负号
            detected_angle = angle_map.get(file_name, 0.0)
            current_angle = -detected_angle 
        else:
            current_angle = fixed_angle

        # 读取图片
        img = cv2.imread(file_path,0) # 这里读取彩色图或灰度图均可
        if img is None: continue

        # 执行旋转
        result_img = rotate_image(img, current_angle)

        # 保存图片到新文件夹
        save_path = os.path.join(new_folder_path, file_name)
        cv2.imwrite(save_path, result_img)

        # 打印进度 (每10张打印一次)
        if (i + 1) % 10 == 0:
            print(f"进度 [{i+1}/{len(bmp_files)}] Saved: {file_name} (Rotated {current_angle:.2f}°)")

    print("\n处理完,所有图片已保存至新文件夹。")

# ==========================================
# 配置区域
# ==========================================
target_folder = r"D:\Codes\PI Inkjet\connecttest" 

if __name__ == '__main__':
    # 模式 A: 使用 CSV 文件里的角度自动矫正 (推荐)
    # batch_rotate_and_save(target_folder, use_csv=True)
    
    # 模式 B: 所有图片都旋转固定的角度 (例如全部旋转 90度)
    batch_rotate_and_save(target_folder, use_csv=False, fixed_angle=-0.98)