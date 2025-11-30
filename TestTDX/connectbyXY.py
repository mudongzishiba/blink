import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def stitch_force_negative_with_vis(img1_path, img2_path, target_score=0.90, expected_overlap=30, dy_limits=(-25, 2)):
    print(f"--- 正在处理: {os.path.basename(img1_path)} & {os.path.basename(img2_path)} ---")
    
    # 1. 读取与预处理
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None: return

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape

    # 2. 参数设置
    tmpl_w = 15      
    tmpl_h = 60      
    step = 10        
    x_tolerance = 10
    limit_min_dy, limit_max_dy = dy_limits

    best_record = {"score": -1, "dy": 0, "overlap": 0}
    found_match = False

    # 3. 遍历搜索 (逻辑同前)
    search_start = int(h1 * 0.1)
    search_end = int(h1 * 0.9) - tmpl_h

    for y_start in range(search_start, search_end, step):
        template = gray1[y_start : y_start + tmpl_h, w1 - tmpl_w : w1]
        if np.std(template) < 10: continue

        y_roi_min = max(0, y_start + limit_min_dy)
        y_roi_max = min(h2, y_start + limit_max_dy + tmpl_h)
        
        if y_roi_max - y_roi_min < tmpl_h: continue

        target_x = expected_overlap - tmpl_w
        x_roi_min = max(0, target_x - x_tolerance)
        x_roi_max = min(w2, target_x + x_tolerance + tmpl_w)
        
        search_roi = gray2[y_roi_min:y_roi_max, x_roi_min:x_roi_max]
        if search_roi.shape[0] < tmpl_h or search_roi.shape[1] < tmpl_w: continue

        res = cv2.matchTemplate(search_roi, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        match_x_local, match_y_local = max_loc
        match_x_global = match_x_local + x_roi_min
        match_y_global = match_y_local + y_roi_min
        
        current_dy = match_y_global - y_start
        
        if max_val > best_record["score"]:
            stitch_point_x = (w1 - tmpl_w) - match_x_global
            best_record = {
                "score": max_val,
                "dy": current_dy,
                "overlap": w1 - stitch_point_x
            }

        if max_val >= target_score:
            found_match = True
            break

    # 4. 获取最佳结果
    res = best_record
    dy = res['dy']
    overlap = res['overlap']
    score = res['score']
    print(f"计算结果 -> 竖直位移: {dy} px | 重叠宽度: {overlap} px | 匹配得分: {score:.4f}")

    # 5. 生成拼接图 (Canvas)
    stitch_point_x = w1 - overlap
    final_w = w1 + w2 - overlap
    y_min = min(0, dy)
    y_max = max(h1, dy + h2)
    final_h = y_max - y_min
    
    canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)
    
    # 贴图 P1
    offset_y1 = -y_min 
    canvas[offset_y1:offset_y1+h1, 0:w1] = img1
    
    # 贴图 P2
    offset_y2 = dy - y_min
    if stitch_point_x < final_w:
        paste_w = min(w2, final_w - stitch_point_x)
        if paste_w > 0:
            canvas[offset_y2:offset_y2+h2, stitch_point_x:stitch_point_x+paste_w] = img2[:, 0:paste_w]

    # 保存
    cv2.imwrite("result_stitched.bmp", canvas)

    # ==========================================
    # ★★★ 可视化展示部分 (Visualization) ★★★
    # ==========================================
    fig = plt.figure(figsize=(14, 8))
    
    # --- 子图1: 完整拼接结果 ---
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=4) # 占据第一行
    ax1.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Stitched Result (dy={dy}, overlap={overlap})")
    
    # 在图上画一个红框，标出重叠区域
    # 框的左上角 x = stitch_point_x, y = 0
    # 框的宽度 = overlap, 高度 = final_h
    rect = patches.Rectangle((stitch_point_x, 0), overlap, final_h, 
                             linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    # ax1.add_patch(rect)
    # ax1.text(stitch_point_x, -10, "Stitch Seam", color='red', fontsize=12, fontweight='bold')

    # --- 提取重复部分的细节进行对比 ---
    # 我们只看中间高度的一段，方便观察
    h_crop_start = int(h1 * 0.4)
    h_crop_end = int(h1 * 0.6)
    
    # 提取 Img1 的重叠部分 (最右侧)
    overlap_img1 = img1[h_crop_start:h_crop_end, w1-overlap:w1]
    
    # 提取 Img2 的重叠部分 (最左侧，注意要处理 dy 偏移)
    # Img2 的对应部分应该是在 [0 : overlap]
    # 但是高度上，Img1 的 y 对应 Img2 的 y+dy
    img2_y_start = h_crop_start + dy
    img2_y_end = h_crop_end + dy
    
    # 边界保护
    img2_y_start = max(0, img2_y_start)
    img2_y_end = min(h2, img2_y_end)
    
    if img2_y_end > img2_y_start:
        overlap_img2 = img2[img2_y_start:img2_y_end, 0:overlap]
    else:
        overlap_img2 = np.zeros_like(overlap_img1)

    # --- 子图2: 左侧重复部分 (P1 Tail) ---
    ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=1)
    if overlap_img1.size > 0:
        ax2.imshow(cv2.cvtColor(overlap_img1, cv2.COLOR_BGR2RGB))
    ax2.set_title("Overlap Part from P1 (Tail)")
    ax2.axis('off')

    # --- 子图3: 右侧重复部分 (P2 Head) ---
    ax3 = plt.subplot2grid((2, 4), (1, 1), colspan=1)
    if overlap_img2.size > 0:
        ax3.imshow(cv2.cvtColor(overlap_img2, cv2.COLOR_BGR2RGB))
    ax3.set_title("Overlap Part from P2 (Head)")
    ax3.axis('off')
    
    # --- 子图4: 融合对比 (Check Alignment) ---
    # 将两张图叠加显示，如果对齐完美，图像应该清晰；如果错位，会模糊
    ax4 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
    
    # 确保尺寸一致才能叠加
    h_min = min(overlap_img1.shape[0], overlap_img2.shape[0])
    w_min = min(overlap_img1.shape[1], overlap_img2.shape[1])
    
    if h_min > 0 and w_min > 0:
        crop1 = overlap_img1[:h_min, :w_min]
        crop2 = overlap_img2[:h_min, :w_min]
        # 50% 透明度叠加
        blended = cv2.addWeighted(crop1, 0.5, crop2, 0.5, 0)
        ax4.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        ax4.set_title("Blend Check (Should be clear)")
    else:
        ax4.text(0.5, 0.5, "Size Mismatch or Error", ha='center')
    
    ax4.axis('off')

    plt.tight_layout()
    plt.show()



# ================= 主程序 =================
# ================= 主程序 =================
if __name__ == "__main__":
    target_dir = r"D:\Codes\PI Inkjet\connecttest"
    
    if os.path.exists(target_dir):
        files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith('.bmp')])
        if len(files) >= 2:
            p1 = os.path.join(target_dir, files[0])
            p2 = os.path.join(target_dir, files[1])
            
            # ★★★ 关键修改在这里 ★★★
            stitch_force_negative_with_vis(
                p1, p2, 
                expected_overlap=30,  # 你的期望重叠
                
                # 修改前: dy_limits=(-25, 2)  <- 范围太大，导致跑到了 -25
                # 修改后:
                dy_limits=(-15, -2)   
                # 解释: 
                # -15: 最小位移限制 (你说过不超过 -15)
                # -2 : 最大位移限制 (确保它一定是负的，且有一定的容错)
            )