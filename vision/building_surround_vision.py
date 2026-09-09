import multiprocessing as mp
import cv2
import time
import math
import numpy as np
import orb_slam3
from vision.base import VisionProcessor, VisionData
from vision.fluid_explorer_vision import DepthExplorerVision

# ==========================================
# 獨立的 SLAM 背景工作進程 (Worker)
# 必須放在最外層，避免 multiprocessing 產生 Pickling Error
# ==========================================
def slam_worker(frame_queue, pose_queue, vocab_path, settings_path, mode="MAPPING"):
    # 啟動 ORB-SLAM3 (最後一個參數 True 代表開啟 3D 雲點視窗)
    slam = orb_slam3.ORB_SLAM3(vocab_path, settings_path, "MONOCULAR", True)
    
    if mode == "LOCALIZATION":
        print("[SLAM] 啟動純定位模式 (Localization Mode) - CPU 負載已釋放！")
        # 呼叫底層 C++ 關閉建圖執行緒，只進行畫面比對
        slam.ActivateLocalizationMode()
    else:
        print("[SLAM] 啟動建圖模式 (Mapping Mode) - 準備收集環境特徵...")

    start_time = time.time()

    while True:
        # 1. 取得最新影像 (若無則阻塞等待)
        frame = frame_queue.get()
        if frame is None:  # 收到毒藥丸 (Poison Pill) 結束信號
            break

        # 2. 執行 SLAM 運算
        timestamp = time.time() - start_time
        pose = slam.TrackMonocular(frame, timestamp)

        # 3. 解析座標並送回主進程
        if pose is not None and len(pose) > 0:
            pose_matrix = np.array(pose).reshape(4, 4)
            x, y, z = pose_matrix[0, 3], pose_matrix[1, 3], pose_matrix[2, 3]
            R = pose_matrix[0:3, 0:3]
            yaw = math.degrees(math.atan2(R[2,0], R[2,2]))

            # 確保佇列淨空，永遠只回傳最熱騰騰的座標
            while not pose_queue.empty():
                pose_queue.get_nowait()
            pose_queue.put((x, y, z, yaw))

    # ⚠️ 極度關鍵：必須正常呼叫 Shutdown，C++ 核心才會把地圖寫入硬碟
    print("[SLAM] 正在儲存地圖至硬碟，請稍候...")
    slam.Shutdown()
    print("[SLAM] 地圖儲存完成！")


# ==========================================
# 複合視覺轉接器 (整合 GPU YOLO 與 多進程 SLAM)
# ==========================================
class BuildingSurroundVision(VisionProcessor):
    # 💡 這裡新增了 run_mode 參數，方便直接從主程式切換
    def __init__(self, run_mode="MAPPING"):
        print("========================================")
        print(f"啟動終極架構：GPU YOLO + 多進程非同步 SLAM (目前模式: {run_mode})")
        print("========================================")

        self.depth_vision = DepthExplorerVision()

        # 強制使用 spawn 模式，獲取乾淨的進程上下文
        ctx = mp.get_context('spawn')

        self.frame_queue = ctx.Queue(maxsize=1)
        self.pose_queue = ctx.Queue(maxsize=1)

        # 啟動背景 SLAM 進程，並把 run_mode 傳給 slam_worker
        self.slam_process = ctx.Process(
            target=slam_worker,
            args=(self.frame_queue, self.pose_queue, "Vocabulary/ORBvoc.txt", "tello.yaml", run_mode)
        )
        self.slam_process.start()

        self.frame_count = 0
        self.latest_slam_data = {"x": 0.0, "z": 0.0, "yaw": 0.0, "tracked": False}

    def process_frame(self, frame) -> VisionData:
        self.frame_count += 1

        # 1. 主進程全速執行 YOLO
        depth_data = self.depth_vision.process_frame(frame)

        # 2. 跳幀機制：每 2 幀才送一張給 SLAM
        if self.frame_count % 2 == 0:
            if self.frame_queue.empty():
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frame_queue.put(gray_frame)

        # 3. 讀取 SLAM 最新計算結果
        if not self.pose_queue.empty():
            x, y, z, yaw = self.pose_queue.get_nowait()
            self.latest_slam_data = {"x": x, "z": z, "yaw": yaw, "tracked": True}

        # 4. 資料整併
        depth_data.x = self.latest_slam_data["x"]
        depth_data.z = self.latest_slam_data["z"]
        depth_data.yaw = self.latest_slam_data["yaw"]
        depth_data.tracking_state = 1 if self.latest_slam_data["tracked"] else 0

        # UI 視覺化疊加
        status = "TRACKING" if self.latest_slam_data["tracked"] else "LOST/INIT"
        color = (0, 255, 0) if self.latest_slam_data["tracked"] else (0, 0, 255)
        cv2.putText(depth_data.annotated_frame,
                    f"SLAM [{status}] X:{depth_data.x:.2f} Z:{depth_data.z:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return depth_data

    def shutdown(self):
        print("正在安全關閉 SLAM 背景進程...")
        self.frame_queue.put(None)
        self.slam_process.join()