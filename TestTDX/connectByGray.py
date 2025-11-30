import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def stitch_force_overlap(img1_path, img2_path, expected_overlap=30, tolerance=10):
    """
    expected_overlap: 你期望的重叠宽度 (例如 30)
    tolerance: 允许的误差范围 (例如 10，代表在 20~40 之间搜索)
    """
    print(f"--- 强制搜索模式 ---\nP1: {os.path.basename(img1_path)}\nP2: {os.path.basename(img2_path)}")
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None: return

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape

    # 1. 设定模板：取 img1 最右侧
    # 模板宽度设为 15，保证小于最小重叠 (30-10=20)
    tmpl_w = 15  
    
    # 智能寻找纹理最丰富的区域 (同上一版逻辑)
    best_template = None
    best_h_start = 0
    max_std_dev = -1
    tmpl_h_block = 80 # 增大取样高度以增加稳定性
    
    for y in range(0, h1 - tmpl_h_block, 20):
        roi = gray1[y : y + tmpl_h_block, w1 - tmpl_w : w1]
        score = np.std(roi)
        if score > max_std_dev:
            max_std_dev = score
            best_template = roi
            best_h_start = y
            
    print(f"特征提取: 高度={best_h_start}, 纹理强度={max_std_dev:.2f}")

    # ================= 核心修改：根据期望重叠计算 Img2 的搜索范围 =================
    # 逻辑推导：
    # Overlap = (Template_Width) + (Match_X_in_Img2)
    # 所以: Match_X_in_Img2 = Overlap - Template_Width
    
    # 也就是我们期望在 Img2 的 x = 15 (30-15) 附近找到匹配
    target_x_center = expected_overlap - tmpl_w
    
    # 确定搜索的 X 范围 (ROI)
    search_x_start = max(0, target_x_center - tolerance)
    search_x_end   = min(w2, target_x_center + tolerance + tmpl_w) #稍微多给一点余量
    
    print(f"约束搜索: 只在 Img2 的 x=[{search_x_start}, {search_x_end}] 范围内寻找")
    
    # 确定搜索的 Y 范围 (限制竖直位移在 ±40 像素内)
    v_search_range = 10
    y_search_start = max(0, best_h_start - v_search_range)
    y_search_end = min(h2, best_h_start + best_template.shape[0] + v_search_range)
    
    # 截取搜索区域
    search_roi = gray2[y_search_start:y_search_end, search_x_start:search_x_end]

    # ================= 匹配 =================
    res = cv2.matchTemplate(search_roi, best_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    match_x_local, match_y_local = max_loc
    
    # 还原全局坐标
    # X 坐标要加上 search_x_start
    match_x_global = match_x_local + search_x_start
    # Y 坐标要加上 y_search_start
    match_y_global = match_y_local + y_search_start
    
    print(f"匹配得分: {max_val:.5f}")
    
    # ================= 计算结果 =================
    # 1. 竖直位移
    dy = match_y_global - best_h_start
    print(f"★ 竖直位移 (dy): {dy} px")

    # 2. 重叠宽度
    # 此时: Match_X_Global 就是模板在 Img2 中的起始位置
    # Overlap = Template_Width + Match_X_Global (因为模板取自 Img1 边缘)
    # 更严谨的拼接点算法：
    stitch_point_x = (w1 - tmpl_w) - match_x_global
    overlap_width = w1 - stitch_point_x
    print(f"★ 重叠宽度: {overlap_width} px")

    # ================= 拼接与展示 =================
    final_w = w1 + w2 - overlap_width
    y_min = min(0, dy)
    y_max = max(h1, dy + h2)
    final_h = y_max - y_min
    
    canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)
    
    offset_y1 = -y_min 
    canvas[offset_y1:offset_y1+h1, 0:w1] = img1
    
    offset_y2 = dy - y_min
    if stitch_point_x < final_w:
        # 注意：这里需要处理边界，防止数组越界
        paste_w = min(w2, final_w - stitch_point_x)
        canvas[offset_y2:offset_y2+h2, stitch_point_x:stitch_point_x+paste_w] = img2[:, 0:paste_w]

    # 保存结果
    save_path = "stitched_forced.bmp"
    cv2.imwrite(save_path, canvas)
    print(f"结果已保存: {save_path}")

    # 绘图确认
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    # 画出匹配框
    img2_disp = img2.copy()
    cv2.rectangle(img2_disp, (match_x_global, match_y_global), 
                  (match_x_global+tmpl_w, match_y_global+best_template.shape[0]), (0, 0, 255), 2)
    plt.imshow(cv2.cvtColor(img2_disp, cv2.COLOR_BGR2RGB))
    plt.title(f"Match Location\nX={match_x_global}, Y={match_y_global}")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title(f"Stitched: dy={dy}, ov={overlap_width}")
    plt.show()

# ================= 主程序入口 =================
if __name__ == "__main__":
    target_dir = r"D:\Codes\PI Inkjet\connecttest"
    
    if os.path.exists(target_dir):
        files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith('.bmp')])
        if len(files) >= 2:
            p1 = os.path.join(target_dir, files[0])
            p2 = os.path.join(target_dir, files[1])
            
            # ★★★ 重点：在这里修改你的期望值 ★★★
            # 告诉算法：重叠大概是 30，允许误差 ±10 (即 20~40)
            stitch_force_overlap(p1, p2, expected_overlap=30, tolerance=10)