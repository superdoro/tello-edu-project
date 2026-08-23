import cv2
import numpy as np

def generate_a4_markers(target_ids=[0, 1, 2, 3], marker_size=400):
    """
    生成指定的 ArUco 標籤，並將它們排版成 2x2 陣列輸出一張大圖方便列印。
    :param target_ids: 想要生成的數字 ID 清單 (建議給 4 個，剛好排成 2x2)
    :param marker_size: 單個標籤的基礎解析度
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    print("========================================")
    print("🖨️ 準備生成 ArUco 標籤集合圖檔 (A4 排版)...")
    print("========================================")

    # 用來收集產生好的標籤圖片
    marker_images = []

    for i in target_ids:
        # 生成標籤核心
        marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
        try:
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, i, marker_size)
        except AttributeError:
            marker_img = cv2.aruco.drawMarker(aruco_dict, i, marker_size)
            
        # 建立外圍白邊與裁切線 (留白 20%)
        padded_size = int(marker_size * 1.3)
        final_img = np.ones((padded_size, padded_size), dtype=np.uint8) * 255 
        
        cv2.rectangle(final_img, (0, 0), (padded_size-1, padded_size-1), (200, 200, 200), 2)
        
        offset = (padded_size - marker_size) // 2
        final_img[offset:offset+marker_size, offset:offset+marker_size] = marker_img
        
        cv2.putText(final_img, f"ID: {i} (DICT_4X4)", (20, padded_size - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                    
        marker_images.append(final_img)
        print(f"✅ 生成 ID: {i} 完成")

    # ==========================================
    # 影像排版 (Grid Layout)
    # ==========================================
    # 確保我們有剛好 4 張圖來排 2x2。如果不足，用空白圖片補齊
    while len(marker_images) < 4:
        blank = np.ones((padded_size, padded_size), dtype=np.uint8) * 255
        marker_images.append(blank)
        
    # 上排：圖0 + 圖1 (水平拼接)
    row1 = cv2.hconcat([marker_images[0], marker_images[1]])
    # 下排：圖2 + 圖3 (水平拼接)
    row2 = cv2.hconcat([marker_images[2], marker_images[3]])
    
    # 結合上下排 (垂直拼接)
    final_layout = cv2.vconcat([row1, row2])
    
    # 存檔
    output_filename = "all_markers_A4.png"
    cv2.imwrite(output_filename, final_layout)
    
    print("========================================")
    print(f"🎉 成功輸出組合圖檔: {output_filename}")
    print("💡 提示: 請將此圖檔以 A4 滿版列印，剪下後貼於氣球上即可使用。")
    print("========================================")

if __name__ == "__main__":
    # 生成 0, 1, 2, 3 號氣球標籤，剛好組合成一張圖
    generate_a4_markers(target_ids=[0, 1, 2, 3])