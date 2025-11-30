import cv2
import numpy as np
import os
import glob

def stitch_images_transparent_grid(folder_path, overlap_x, step_y, direction_y='up', alpha=0.5):
    """
    alpha: 透明度 (0.0 - 1.0)。
           0.3 比较淡，能看清下面。
           0.8 比较明显。
    """
    
    # --- 1. 获取图片 ---
    bmp_files = glob.glob(os.path.join(folder_path, "*.bmp"))
    bmp_files.sort()
    
    if not bmp_files:
        print("错误：未找到图片")
        return

    print(f"找到 {len(bmp_files)} 张图片，开始计算坐标...")

    # --- 2. 预读取获取尺寸 ---
    first_img = cv2.imread(bmp_files[0], cv2.IMREAD_UNCHANGED)
    if first_img is None: return
    h_img, w_img = first_img.shape[:2]

    # --- 3. 位移量计算 ---
    move_x = w_img - overlap_x
    move_y = step_y

    # --- 4. 计算相对坐标 ---
    img_positions = []
    curr_x, curr_y = 0, 0
    
    for _ in range(len(bmp_files)):
        img_positions.append((curr_x, curr_y))
        curr_x += move_x
        if direction_y == 'down':
            curr_y += move_y
        else: # 'up'
            curr_y -= move_y

    # --- 5. 坐标修正 ---
    all_x = [pos[0] for pos in img_positions]
    all_y = [pos[1] for pos in img_positions]
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    total_w = (max_x - min_x) + w_img
    total_h = (max_y - min_y) + h_img
    offset_x = -min_x
    offset_y = -min_y

    print(f"画布尺寸: {total_w}x{total_h}")

    # --- 6. 创建画布 (3通道 BGR) ---
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    # 定义绿色 (B, G, R) 和 透明度参数
    color_green = np.array([0, 255, 0], dtype=np.float32) # 绿色
    alpha_line = alpha            # 线条的不透明度
    beta_orig = 1.0 - alpha_line  # 原图的不透明度

    # --- 7. 贴图并混合线条 ---
    for i, file_path in enumerate(bmp_files):
        if i == 0:
            img = first_img
        else:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None: continue

        # 转彩色
        if len(img.shape) == 2:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img

        # 坐标计算
        rel_x, rel_y = img_positions[i]
        abs_x = rel_x + offset_x
        abs_y = rel_y + offset_y
        
        h_curr, w_curr = img_color.shape[:2]
        
        # 边界 (不越界)
        end_y = min(abs_y + h_curr, total_h)
        end_x = min(abs_x + w_curr, total_w)
        
        # A. 粘贴图片
        canvas[abs_y:end_y, abs_x:end_x] = img_color[:end_y-abs_y, :end_x-abs_x]

        # ==========================================================
        # --- B. 绘制透明绿色边框 (NumPy 加权混合) ---
        # ==========================================================
        
        # 我们只修改边缘的像素。
        # 坐标范围：行 y=[abs_y, end_y-1], 列 x=[abs_x, end_x-1]
        
        # 1. 上边缘 (Top Edge)
        # if abs_y < total_h:
        #     # 取出当前行的像素 (float便于计算)
        #     roi = canvas[abs_y, abs_x:end_x].astype(np.float32)
        #     # 混合公式: 原图*beta + 绿色*alpha
        #     blended = roi * beta_orig + color_green * alpha_line
        #     # 赋值回去 (转回 uint8)
        #     canvas[abs_y, abs_x:end_x] = blended.astype(np.uint8)

        # # 2. 下边缘 (Bottom Edge)
        # # 注意 end_y 是开区间，所以最后一行是 end_y - 1
        # if end_y - 1 >= 0:
        #     roi = canvas[end_y - 1, abs_x:end_x].astype(np.float32)
        #     blended = roi * beta_orig + color_green * alpha_line
        #     canvas[end_y - 1, abs_x:end_x] = blended.astype(np.uint8)

        # # 3. 左边缘 (Left Edge)
        # if abs_x < total_w:
        #     roi = canvas[abs_y:end_y, abs_x].astype(np.float32)
        #     blended = roi * beta_orig + color_green * alpha_line
        #     canvas[abs_y:end_y, abs_x] = blended.astype(np.uint8)

        # # 4. 右边缘 (Right Edge)
        # if end_x - 1 >= 0:
        #     roi = canvas[abs_y:end_y, end_x - 1].astype(np.float32)
        #     blended = roi * beta_orig + color_green * alpha_line
        #     canvas[abs_y:end_y, end_x - 1] = blended.astype(np.uint8)

        # if (i + 1) % 10 == 0:
        #     print(f"进度: [{i+1}/{len(bmp_files)}]")

    # --- 8. 保存 ---
    parent_dir = os.path.dirname(folder_path.rstrip(os.sep))
    folder_name = os.path.basename(folder_path.rstrip(os.sep))
    output_folder = os.path.join(parent_dir, folder_name + "_Connected")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    save_path = os.path.join(output_folder, f"Result_AlphaGrid_{direction_y}.png")
    print(f"正在保存... {save_path}")
    cv2.imwrite(save_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print("完成。")

# ==========================================
# 配置区域
# ==========================================
target_folder = r"D:\Codes\PI Inkjet\0_250731_Corrected" 

overlap_x_val = 30 
step_y_val = 0 
stitch_direction = 'up' 

# 透明度设置 (0.0 ~ 1.0)
# 0.5 = 半透明 (推荐)
# 0.2 = 很淡
# 0.8 = 很亮
line_opacity = 0.8

if __name__ == '__main__':
    stitch_images_transparent_grid(target_folder, overlap_x_val, step_y_val, direction_y=stitch_direction, alpha=line_opacity)