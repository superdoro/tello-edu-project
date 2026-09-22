import atexit
import csv
import os
import queue
import threading
import time
from datetime import datetime

import cv2


class VideoRecorder:
    """
    實驗錄影器 (V 鍵開關)。

    每次錄影會在 recordings/ 下產生三個同名檔案：
        <時間>_<模式>_raw.mp4        乾淨畫面，沒有任何疊字 —— 拿來標註、訓練 YOLO
        <時間>_<模式>_annotated.mp4  與螢幕上看到的一樣 (偵測框、深度、狀態) —— 拿來找問題
        <時間>_<模式>_log.csv        每一格的飛行指令與狀態 —— 對照影片看「那一刻它為什麼這樣飛」

    raw 錄的是縮放後的 720x480，也就是 YOLO 推論時實際看到的畫面。
    原生 960x720 是 4:3、縮放後是 3:2，物體會被水平拉寬；用原生畫面訓練的模型
    上機後看到的比例不一樣，所以訓練資料要跟推論畫面一致。

    兩個設計重點：
    1. 絕不阻塞飛控迴圈。編碼與寫檔都在背景執行緒，主迴圈只做 put_nowait，
       佇列滿了就丟格 —— 丟一格影片沒關係，主迴圈卡住會讓 Tello 超過 15 秒
       收不到指令而自動降落。
    2. 影片以固定 FPS 依「實際時間」排列。主迴圈的速度會隨模式變動 (15~60 fps)，
       如果每格照收照寫，回放速度就會忽快忽慢，沒辦法拿來對照實際發生的時間。
       這裡依時間戳決定每個輸出格要放哪張畫面：迴圈快就跳格、慢就重複，
       回放永遠是真實時間。
    """

    OUT_DIR = "recordings"
    FPS = 30
    QUEUE_MAX = 30          # 約 1 秒的緩衝，正常情況下幾乎不會滿
    MAX_GAP_SEC = 2.0       # 單一畫面最多重複填補的時間 (例如起飛指令阻塞時)，避免檔案暴增

    LOG_FIELDS = ["t", "elapsed", "mode", "state", "detected",
                  "lr", "fb", "ud", "yv", "battery", "height"]

    def __init__(self):
        self.recording = False
        self.start_time = 0.0
        self.base_name = ""
        self._queue = None
        self._thread = None
        self.dropped = 0
        atexit.register(self.stop)   # 程式正常結束或例外離開時也會收尾，避免 mp4 損毀

    # ==========================================
    # 對外
    # ==========================================
    def toggle(self, mode_name=""):
        if self.recording:
            self.stop()
        else:
            self.start(mode_name)

    def start(self, mode_name=""):
        if self.recording:
            return
        os.makedirs(self.OUT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_mode = "".join(c if c.isalnum() else "_" for c in mode_name).strip("_") or "NA"
        self.base_name = os.path.join(self.OUT_DIR, f"{stamp}_{safe_mode}")

        self._queue = queue.Queue(maxsize=self.QUEUE_MAX)
        self.dropped = 0
        self.start_time = time.time()
        self.recording = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"[錄影] 開始 -> {self.base_name}_*.mp4")

    def stop(self):
        if not self.recording:
            return
        self.recording = False
        self._queue.put(None)                    # 結束訊號 (阻塞式放入，確保一定送得到)
        self._thread.join(timeout=5.0)           # 等背景把剩下的畫面寫完、檔案正確關閉
        dur = time.time() - self.start_time
        msg = f"[錄影] 停止，共 {dur:.1f} 秒 -> {self.base_name}_raw.mp4 / _annotated.mp4 / _log.csv"
        if self.dropped:
            msg += f" (佇列滿丟棄 {self.dropped} 格)"
        print(msg)

    def write(self, raw, annotated, log_row):
        """
        主迴圈每格呼叫一次。絕對不會阻塞：佇列滿了就丟棄這一格。

        :param raw: 沒有任何疊字的畫面 (呼叫端必須傳入複本，之後會在背景執行緒讀取)
        :param annotated: 有疊字的畫面 (同上，必須是複本)
        :param log_row: dict，欄位見 LOG_FIELDS
        """
        if not self.recording:
            return
        try:
            self._queue.put_nowait((time.time(), raw, annotated, log_row))
        except queue.Full:
            self.dropped += 1

    def elapsed(self):
        return time.time() - self.start_time if self.recording else 0.0

    def draw_indicator(self, frame):
        """在畫面上畫出紅色 REC 標示。只畫在顯示用的畫面上，不會進到錄影檔裡。"""
        if not self.recording:
            return
        sec = int(self.elapsed())
        text = f"REC {sec // 60:02d}:{sec % 60:02d}"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
        # 放在右上角電量正下方，跟左側的模式名稱、深度讀數都不重疊
        x = frame.shape[1] - tw - 10
        cv2.putText(frame, text, (x, 62), font, scale, (0, 0, 255), thick)
        # 每半秒閃一次，眼角餘光就看得到正在錄影
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (x - 14, 55), 7, (0, 0, 255), -1)

    # ==========================================
    # 背景執行緒
    # ==========================================
    def _worker(self):
        raw_w = ann_w = None
        csv_f = open(f"{self.base_name}_log.csv", "w", newline="", encoding="utf-8")
        log = csv.DictWriter(csv_f, fieldnames=self.LOG_FIELDS, extrasaction="ignore")
        log.writeheader()

        dt = 1.0 / self.FPS
        next_t = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                t, raw, ann, row = item

                # 影片尺寸以第一張畫面為準
                if raw_w is None:
                    h, w = raw.shape[:2]
                    raw_w = cv2.VideoWriter(f"{self.base_name}_raw.mp4", fourcc, self.FPS, (w, h))
                    ann_w = cv2.VideoWriter(f"{self.base_name}_annotated.mp4", fourcc, self.FPS, (w, h))
                    next_t = t

                # CSV 每格都記，不受影片的固定 FPS 影響，保留最完整的時間解析度
                row = dict(row)
                row["t"] = f"{t:.3f}"
                row["elapsed"] = f"{t - self.start_time:.3f}"
                log.writerow(row)

                # 依實際時間填入固定 FPS 的時間軸
                if t - next_t > self.MAX_GAP_SEC:
                    next_t = t - self.MAX_GAP_SEC   # 長時間卡住只補最後一段，避免檔案暴增
                while next_t <= t:
                    raw_w.write(raw)
                    ann_w.write(ann)
                    next_t += dt
        finally:
            if raw_w is not None:
                raw_w.release()
                ann_w.release()
            csv_f.close()
