import os
import time
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from vision.base import VisionProcessor, VisionData
from utils.dark_region import DarkRegionAnalyzer
from utils.blank_region import BlankSurfaceAnalyzer
from utils.white_strip import WhiteStripVerifier, WHITE, UNKNOWN, NOT_WHITE

# 由框寬估距離：dist_cm = STRIP_WIDTH_CM * FOCAL_PX / w_px。
# FOCAL_PX 是 720x480 影像的水平焦距 (約 461px)，紙條實寬約 3cm。
# 精度隨距離改善 —— 1m 處 w=14px，±1px 就是 ±7cm；35cm 處 w=39px，±1px 只有 ±0.9cm。
# 剛好在最需要準確的近距離最準。
STRIP_WIDTH_CM = 3.0
FOCAL_PX = 461.0
DIST_K = STRIP_WIDTH_CM * FOCAL_PX          # 1383


class _Slot:
    """
    一個被追蹤中的候選目標，以及它的白色驗證投票紀錄。

    存在的理由：白色驗證必須是「跨格累積」的決定。單格判斷會有 1~5% 的誤差，
    而盯著同一面牆看兩秒就有幾十次半獨立的機會，遲早會矇中一次。
    把票記在 slot 上，並且讓被否決的 slot 留下冷卻期，才真正擋得住。
    """
    __slots__ = ("id", "cx", "cy", "state", "votes", "rejected_until", "last_seen",
                 "last_verdict", "last_blob_aspect")

    def __init__(self, sid, cx, cy, now):
        self.id = sid
        self.cx = cx
        self.cy = cy
        self.state = "PENDING"
        self.votes = deque()            # [(時間, 判定)]，UNKNOWN 不會進來
        self.rejected_until = 0.0
        self.last_seen = now
        self.last_verdict = UNKNOWN
        self.last_blob_aspect = 0.0


class TissueDetector(VisionProcessor):
    """
    衛生紙條偵測器：YOLO 找目標 -> 白色驗證 -> 輸出鎖定的那一條，另外供應深度避障。

    輸出分成兩組互不相干的資料：

        data.tissue                        -> 給 TissueChargeControl 對準與逼近
        data.depth_L/C/R / depth_ahead
        data.depth_unreliable / dark_*     -> 給 FluidExploreControl 漫遊避障

    第二組欄位刻意與 fluid_explorer_vision.py 完全同名同單位 (公分)，
    因為 TissueChargeControl 的搜尋階段是直接把指令委派給 FluidExploreControl 的 ——
    同一份視覺資料要能無縫餵進去，那邊調好的 300/250/150 也才能直接沿用。

    ==========================================================
    三道關卡，順序有意義
    ==========================================================
    1. 信心門檻   擋不掉白牆誤判 (實測誤判 0.62 vs 真紙條 0.66)，只擋雜訊
    2. 形狀關卡   誤判的框又寬又矮、紙條又細又長，用長寬比分開
    3. 白色驗證   門檻化 + 分割，確認框裡真的是一條白色的東西 (utils/white_strip.py)

    第 3 關是跨格投票的：連續 VOTE_WINDOW 格裡有 VOTE_MIN_WHITE 格判定為白才算確認，
    被否決的位置則進入冷卻期不再考慮。實測 (0923 錄影，誤判 n=335 / 真紙條 n=803)：

        單格判斷        保留真目標 95%   誤判被確認 2%
        5 取 3 投票     保留真目標 97%   誤判被確認 0%   <-- 採用
        5 取 4 投票     保留真目標 91%   誤判被確認 0%

    獨立場次 (0922 錄影，n=2575，目視確認全為真紙條) 的保留率 99%，泛化沒問題。

    ==========================================================
    確認之後就不再重驗
    ==========================================================
    白色驗證只用來決定「要不要鎖定這一條」。一旦 CONFIRMED 就是終態，
    因為逼近到最後紙條會脹滿整個框，crop 裡沒有背景可以對比，分割必然失效 ——
    這時候還逐格重驗，只會在最後關頭把已經追了很久的目標丟掉。
    飛手需要反悔的話有 R 鍵 (reset_target)。

    ==========================================================
    已鎖定的目標用比較鬆的形狀門檻
    ==========================================================
    取得目標時要嚴格 (擋白牆)，但鎖定之後要寬鬆 —— 逼近時框會變寬、上下緣被畫面
    切掉，長寬比必然崩潰。用同一套嚴格門檻的話，會在最需要持續修正的最後
    幾十公分把目標擋掉。放寬之後閉迴路追蹤可以從約 9.6cm 延伸到約 4.3cm。

    按 F 鍵可以把被擋掉的框也畫出來，方便現場確認門檻對不對。
    """

    def __init__(self,
                 yolo_path="model/yolo26/runs/detect/yolo_tissue_collect/best_seg.pt",
                 depth_model_path="model/yolo26/runs/detect/yolo_depth_collect/yolo26n-depth.pt",
                 tissue_class_id=0, conf_threshold=0.5,
                 min_aspect=3.0, max_width_ratio=0.20,
                 min_aspect_locked=1.2, max_width_ratio_locked=0.45,
                 depth_band=(0.35, 0.45),
                 verifier=None, now_fn=time.time):
        """
        :param yolo_path: 衛生紙條偵測模型
        :param depth_model_path: 深度模型，供漫遊避障使用
        :param tissue_class_id: 紙條在模型中的類別編號
        :param conf_threshold: 信心門檻 (抓不到就調低)
                               ⚠️ 調高「不能」解決白牆誤判 —— 實測誤判框與真紙條的
                               信心值幾乎一樣 (0.62 vs 0.66，與長寬比的相關性只有 0.026)
        :param min_aspect: 取得目標時的最小長寬比 (高/寬)。白牆誤判的框又寬又矮，
                           紙條又細又長。實測 678 個框的長寬比在 3.0~3.5 有明顯的谷。
                           用長寬比而不是寬度，是因為它與距離無關。
        :param max_width_ratio: 取得目標時的寬度上限 (佔畫面比例)，擋超大框的保險
        :param min_aspect_locked / max_width_ratio_locked:
                           已鎖定目標專用的寬鬆門檻 (理由見類別說明)。
                           0.45 x 720 = 324px -> 估距 1383/324 = 4.3cm，
                           閉迴路追蹤的極限從 9.6cm 延伸到 4.3cm。
        :param depth_band: 深度取樣的水平帶，與 fluid_explorer_vision.py 一致
        :param verifier: 自訂的 WhiteStripVerifier (離線掃參數時用)，None 就用預設值
        :param now_fn: 取得目前時間的函式。離線重播錄影時傳入依影格時間戳推進的
                       假時鐘，投票的時效判斷才會與實飛一致 —— 否則影格跑得比
                       即時快很多，VOTE_MAX_AGE 永遠不會到期，測出來的行為是假的。
        """
        print("========================================")
        print("[系統訊息] 啟動衛生紙條獵手視覺 (YOLO 偵測 + 白色驗證 + 深度避障)...")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"[GPU 加速啟動] 模型已載入至 {torch.cuda.get_device_name(0)}！")
        else:
            print("[警告] 未偵測到 CUDA，推論將使用 CPU。")

        # 先確認檔案存在再交給 YOLO()。
        # 直接把不存在的路徑丟進去，ultralytics 可能會當成模型名稱跑去網路上下載 ——
        # 飛行時連的是 Tello 熱點、沒有外網，那個同步下載會卡住主迴圈，
        # Tello 超過約 15 秒收不到指令就會自動降落。
        self.model = None
        if os.path.exists(yolo_path):
            self.model = YOLO(yolo_path)
            self.model.to(self.device)
            print(f"[系統訊息] 紙條模型: {yolo_path}")
        else:
            print(f"[警告] 找不到紙條模型 {yolo_path}")
            print("        -> 本模式只會漫遊避障，不會偵測也不會逼近。")
            print("        -> 請用 model/yolo26/train_yoly.py 訓練後放到該路徑。")

        self.depth_model = None
        if depth_model_path and os.path.exists(depth_model_path):
            self.depth_model = YOLO(depth_model_path)
            self.depth_model.to(self.device)
        elif depth_model_path:
            print(f"[警告] 找不到深度模型 {depth_model_path} -> 漫遊將沒有避障能力。")
        print("========================================")

        self.tissue_class_id = tissue_class_id
        self.conf_threshold = conf_threshold
        self.min_aspect = min_aspect
        self.max_width_ratio = max_width_ratio
        self.min_aspect_locked = min_aspect_locked
        self.max_width_ratio_locked = max_width_ratio_locked
        self.depth_band = depth_band

        # 黑區分析：純黑牆面會讓深度模型誤判成「很遠」，漫遊時要知道深度不可信。
        # 與 FLUID EXPLORER 用同一個分析器，行為一致。
        self.dark_analyzer = DarkRegionAnalyzer()
        # 白色無紋理區分析：純白柱子/牆面一樣會讓深度模型誤判成很遠 (撞過柱子)
        self.blank_analyzer = BlankSurfaceAnalyzer()

        # 白色驗證器。參數是用 tests/replay_tissue.py 在 recordings/ 上掃出來的。
        self.verifier = verifier if verifier is not None else WhiteStripVerifier()
        self.MAX_VERIFY_PER_FRAME = 4   # 每格最多驗幾個框 (取最大的)，限制最壞情況耗時

        # F 鍵切換：連被擋掉的偵測也畫出來，方便現場調門檻
        self.show_all = False

        # ---------- 投票與黑名單 ----------
        self.VOTE_WINDOW = 5        # 投票視窗 (格)
        self.VOTE_MIN_WHITE = 3     # 視窗內至少幾格判定為白才確認
        # 視窗內有這麼多格判定為「不是白的」就直接否決。
        # 5-3+1=3：已經有 3 格不是白的，剩下最多 2 格為白，數學上不可能再確認了，
        # 早點否決只是讓黑名單早點生效，不會改變結論。
        self.VOTE_REJECT_MIN = self.VOTE_WINDOW - self.VOTE_MIN_WHITE + 1
        self.VOTE_MAX_AGE = 0.6     # 超過這麼久的票作廢 (秒)。
                                    # 用時間而不是純格數，是因為沒有 CUDA 時迴圈只有
                                    # 約 15Hz，5 格會變成 330ms，那時畫面早就轉過去了。
        self.REJECT_COOLDOWN = 5.0  # 被否決的位置多久之內不再考慮 (秒)
        self.SLOT_TTL = 1.5         # 多久沒再看到就把 slot 丟掉 (秒)

        # ---------- 目標黏著 ----------
        # 場地裡通常不只一條紙條。如果每一格都選「面積最大」的那條，兩條大小接近時
        # 選擇會在兩條之間來回跳 —— 實測錄影裡 389 格跳了 33 次 (368 -> 710 -> 325 -> 671)，
        # 無人機的轉向就跟著在 ±60 之間甩來甩去，永遠對不準。
        # 改成：已經鎖定一條就優先選離它最近的，只有它消失夠久才重新挑最大的。
        self.lock_cx = None
        self.lock_miss = 0
        self.LOCK_GATE = 120       # 與上一格目標水平距離在這個範圍內才算同一條
        # 連續這麼多格找不到鎖定的那條就放棄重挑。
        # 刻意很短 (約 0.1 秒)：這裡只負責蓋掉 YOLO 的單格閃爍。
        # 舊版設 15 格，加上行為層 0.5 秒的記憶，最壞情況會用一筆已經舊了 0.3 秒的
        # 觀測開迴路飛 0.82 秒 —— 實測錄影裡偏航有 35% 的時間頂在 ±60 就是這樣來的。
        # 撐久一點的工作交給行為層會衰減的記憶。
        self.LOCK_MISS_MAX = 5

        self._slots = []
        self._next_slot_id = 1
        self._now = now_fn

    # ==========================================
    # 對外
    # ==========================================
    def toggle_tracking_mode(self):
        """F 鍵：切換「顯示被擋掉的偵測」(灰色=信心不足，橘色=形狀不符)"""
        self.show_all = not self.show_all
        print(f"[視覺系統] 低信心偵測顯示：{'開啟' if self.show_all else '關閉'}")

    def reset(self):
        """切換飛行模式時清掉黑區分析的平滑狀態、目標鎖定與所有投票"""
        self.dark_analyzer.reset()
        self.lock_cx = None
        self.lock_miss = 0
        self._slots = []

    def reset_target(self):
        """
        R 鍵：放棄目前鎖定的那一條，並把它列入黑名單一段時間。

        需要這個是因為 CONFIRMED 是終態 —— 驗證通過之後就不再重驗，
        萬一鎖錯了 (或那一條搆不到)，飛手要有辦法說「不對，換一條」。
        """
        if self.lock_cx is not None:
            now = self._now()
            for s in self._slots:
                if abs(s.cx - self.lock_cx) <= self.LOCK_GATE:
                    s.state = "REJECTED"
                    s.rejected_until = now + self.REJECT_COOLDOWN
            print(f"[視覺系統] 已放棄目前目標，{self.REJECT_COOLDOWN:.0f} 秒內不再鎖定該位置")
        self.lock_cx = None
        self.lock_miss = 0

    # ==========================================
    # 投票與 slot 管理
    # ==========================================
    def _match_slot(self, cx, cy, now):
        """把一個框配到最近的 slot，配不到就開一個新的 (沿用 LOCK_GATE 的關聯規則)"""
        best, best_d = None, None
        for s in self._slots:
            d = abs(s.cx - cx)
            if d <= self.LOCK_GATE and (best_d is None or d < best_d):
                best, best_d = s, d
        if best is None:
            best = _Slot(self._next_slot_id, cx, cy, now)
            self._next_slot_id += 1
            self._slots.append(best)
        best.cx, best.cy, best.last_seen = cx, cy, now
        return best

    def _cast_vote(self, slot, verdict, now):
        """
        把一格的判定記進 slot 並更新狀態。

        UNKNOWN 不記票 —— 這是驗證器三值設計的重點：分割器說「我不知道」的時候，
        既有的判定就原封不動保留，不會因為近距離分割失效而把目標丟掉。
        """
        slot.last_verdict = verdict
        if verdict == UNKNOWN:
            return
        slot.votes.append((now, verdict))
        while slot.votes and (now - slot.votes[0][0] > self.VOTE_MAX_AGE
                              or len(slot.votes) > self.VOTE_WINDOW):
            slot.votes.popleft()

        n_white = sum(1 for _, v in slot.votes if v == WHITE)
        n_not = sum(1 for _, v in slot.votes if v == NOT_WHITE)
        if n_not >= self.VOTE_REJECT_MIN:
            slot.state = "REJECTED"
            slot.rejected_until = now + self.REJECT_COOLDOWN
        elif len(slot.votes) >= self.VOTE_WINDOW and n_white >= self.VOTE_MIN_WHITE:
            slot.state = "CONFIRMED"

    def _expire_slots(self, now):
        keep = []
        for s in self._slots:
            if s.state == "REJECTED":
                # 被否決的 slot 要撐過整個冷卻期，即使這段時間看不到它 ——
                # 否則誤判一閃一閃的時候黑名單根本不會生效
                if now < s.rejected_until:
                    keep.append(s)
            elif now - s.last_seen <= self.SLOT_TTL:
                keep.append(s)
        self._slots = keep

    def _pick(self, candidates):
        """
        從候選中選出要追的那一條 (黏著策略，理由見 __init__)。
        傳進來的只會是已經通過白色驗證的候選。
        """
        if self.lock_cx is not None:
            near = [c for c in candidates if abs(c["cx"] - self.lock_cx) <= self.LOCK_GATE]
            if near:
                best = min(near, key=lambda c: abs(c["cx"] - self.lock_cx))
                self.lock_cx, self.lock_miss = best["cx"], 0
                return best
            # 鎖定的那條這一格不見了：先撐幾格 (可能只是單格漏偵測)，不要馬上換目標
            self.lock_miss += 1
            if self.lock_miss <= self.LOCK_MISS_MAX:
                return None
        # 沒有鎖定，或鎖定的那條消失太久 -> 重新挑面積最大 (最近) 的
        best = max(candidates, key=lambda c: c["area"])
        self.lock_cx, self.lock_miss = best["cx"], 0
        return best

    # ==========================================
    # 深度 (供漫遊避障)
    # ==========================================
    def _depth_map(self, frame):
        """跑 YOLO depth 並取出與原圖同尺寸的深度圖，失敗時回傳 None"""
        if self.depth_model is None:
            return None

        res = self.depth_model(frame, verbose=False, device=self.device)[0]
        depth_map = None
        try:
            if hasattr(res, 'depth') and res.depth is not None:
                depth_map = res.depth.data.cpu().numpy().squeeze() * 100.0
            elif hasattr(res, 'masks') and res.masks is not None:
                depth_map = res.masks.data[0].cpu().numpy().squeeze() * 100.0
        except Exception:
            return None

        if depth_map is None or not isinstance(depth_map, np.ndarray) or depth_map.ndim < 2:
            return None

        h_img, w_img = frame.shape[:2]
        if depth_map.shape[:2] != (h_img, w_img):
            depth_map = cv2.resize(depth_map, (w_img, h_img))
        return depth_map

    # ==========================================
    def process_frame(self, frame) -> VisionData:
        now = self._now()
        data = VisionData(is_detected=False, annotated_frame=frame)
        data.tissue = None

        # 預設「前方無限遠」：深度模型缺席時，漫遊會退化成直線前進而不是卡住不動
        data.depth_L = data.depth_C = data.depth_R = data.depth_ahead = 999.0
        data.n_confirmed = data.n_pending = data.n_rejected = 0
        data.n_low_conf = data.n_bad_shape = data.n_seg_unknown = 0

        annotated_frame = frame.copy()
        h_img, w_img = frame.shape[:2]
        w3 = w_img // 3

        # ---------- 黑區分析 (必須在深度判讀之前) ----------
        dark_info = self.dark_analyzer.analyze(frame)
        data.dark_ratio = dark_info["ratio"]
        data.dark_L, data.dark_C, data.dark_R = dark_info["L"], dark_info["C"], dark_info["R"]
        data.depth_unreliable = dark_info["is_dark"]
        self.dark_analyzer.draw(annotated_frame, dark_info, warn_org=(10, 100))

        # ---------- 深度：左/中/右三段 (比照 fluid_explorer_vision) ----------
        depth_map = self._depth_map(frame)
        dy1 = dy2 = None
        if depth_map is not None:
            dy1 = int(h_img * self.depth_band[0])
            dy2 = int(h_img * self.depth_band[1])
            if dy2 > dy1:
                data.depth_L = float(np.mean(depth_map[dy1:dy2, 0:w3]))
                data.depth_C = float(np.mean(depth_map[dy1:dy2, w3:2 * w3]))
                data.depth_R = float(np.mean(depth_map[dy1:dy2, 2 * w3:w_img]))
                data.depth_ahead = data.depth_C

                cv2.rectangle(annotated_frame, (0, dy1), (w3, dy2), (255, 200, 0), 1)
                cv2.rectangle(annotated_frame, (w3, dy1), (2 * w3, dy2), (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (2 * w3, dy1), (w_img, dy2), (255, 200, 0), 1)
                cv2.putText(annotated_frame, f"L: {int(data.depth_L)}cm", (10, dy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                color_c = (0, 255, 0) if data.depth_C > 150 else (0, 0, 255)
                cv2.putText(annotated_frame, f"C: {int(data.depth_C)}cm", (w3 + 10, dy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_c, 2)
                cv2.putText(annotated_frame, f"R: {int(data.depth_R)}cm", (2 * w3 + 10, dy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                dy1 = dy2 = None

        # ---------- 白色無紋理區：深度模型看不到的白柱子/白牆，覆寫成近距離 ----------
        # 放在深度之後：只會把 depth_L/C/R 往近的方向改，漫遊引擎照原本的邏輯轉開或後退。
        # depth_ahead (TRACK 用) 不覆寫 —— 逼近紙條時紙條本身就是白的。
        blank_info = self.blank_analyzer.analyze(frame)
        self.blank_analyzer.apply(data, blank_info)
        self.blank_analyzer.draw(annotated_frame, blank_info)

        # ---------- YOLO 偵測紙條 ----------
        best = None
        if self.model is not None:
            self._expire_slots(now)
            results = self.model(frame, verbose=False, device=self.device)
            shaped = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    if int(box.cls[0]) != self.tissue_class_id:
                        continue
                    conf = float(box.conf[0])
                    cx, cy, w, h = [int(v) for v in box.xywh[0].tolist()]
                    if w <= 0 or h <= 0:
                        continue
                    if conf < self.conf_threshold:
                        data.n_low_conf += 1
                        if self.show_all:   # 除錯用：畫出被信心門檻擋掉的框
                            self._draw_box(annotated_frame, cx, cy, w, h, (120, 120, 120),
                                           f"{conf:.2f}", 1)
                        continue

                    # 形狀關卡：擋掉白牆誤判。
                    # 已經鎖定的那條用比較鬆的門檻 —— 逼近時框會變寬、上下緣被切掉，
                    # 長寬比必然崩潰，用同一套嚴格門檻會在最後關頭把目標擋掉。
                    locked_near = (self.lock_cx is not None
                                   and abs(cx - self.lock_cx) <= self.LOCK_GATE)
                    min_asp = self.min_aspect_locked if locked_near else self.min_aspect
                    max_wr = self.max_width_ratio_locked if locked_near else self.max_width_ratio
                    aspect = h / max(w, 1e-6)
                    if aspect < min_asp or w > max_wr * w_img:
                        data.n_bad_shape += 1
                        if self.show_all:      # 除錯用：畫出被形狀擋掉的框
                            self._draw_box(annotated_frame, cx, cy, w, h, (0, 165, 255),
                                           f"R{aspect:.1f}", 2)
                        continue

                    shaped.append({
                        "cx": cx, "cy": cy, "w": w, "h": h,
                        "area": w * h, "conf": conf, "aspect": aspect,
                        "dist_cm": DIST_K / max(w, 1),
                        "top_y": cy - h / 2.0, "bottom_y": cy + h / 2.0,
                        "top_clipped": cy - h / 2.0 <= 1,
                        "bottom_clipped": cy + h / 2.0 >= h_img - 1,
                    })

            # ---------- 白色驗證 ----------
            # 只驗還沒定案的 slot。已確認的不重驗 (理由見類別說明)，
            # 已否決的在冷卻期內也不必再驗。每格最多驗 MAX_VERIFY_PER_FRAME 個。
            shaped.sort(key=lambda c: -c["area"])
            n_verified = 0
            for c in shaped:
                slot = self._match_slot(c["cx"], c["cy"], now)
                c["slot"] = slot
                if slot.state == "PENDING" and n_verified < self.MAX_VERIFY_PER_FRAME:
                    info = self.verifier.analyze(frame, (c["cx"], c["cy"], c["w"], c["h"]))
                    n_verified += 1
                    c["wsinfo"] = info
                    slot.last_blob_aspect = info["blob_aspect"]
                    self._cast_vote(slot, info["verdict"], now)
                    if info["verdict"] == UNKNOWN:
                        data.n_seg_unknown += 1

            for s in self._slots:
                if s.state == "CONFIRMED":
                    data.n_confirmed += 1
                elif s.state == "REJECTED":
                    data.n_rejected += 1
                else:
                    data.n_pending += 1

            # ---------- 只有通過驗證的候選才進得了選擇池 ----------
            # 這一行同時實現了「不是純白就去找別的目標」：誤判根本不會成為候選，
            # 黏著策略自然會去挑別條。而被否決的位置有冷卻期，不會下一秒又被重新評估。
            eligible = [c for c in shaped if c["slot"].state == "CONFIRMED"]
            best = self._pick(eligible) if eligible else None
            if not eligible and self.lock_cx is not None:
                # 整格都沒有合格候選也要累計，否則鎖定永遠不會過期
                self.lock_miss += 1
                if self.lock_miss > self.LOCK_MISS_MAX:
                    self.lock_cx = None

            self._draw_candidates(annotated_frame, shaped, best)

            if best:
                # 把驗證結果一起帶出去，飛行紀錄才有辦法事後判斷「為什麼追這一條」
                slot = best["slot"]
                best["slot_id"] = slot.id
                best["white"] = 1
                best["votes"] = (sum(1 for _, v in slot.votes if v == WHITE), len(slot.votes))
                best["blob_aspect"] = round(slot.last_blob_aspect, 2)

                data.tissue = best
                data.is_detected = True
                data.cx, data.cy = best["cx"], best["cy"]
                data.w, data.h = best["w"], best["h"]

        # ---------- 前方淨空距離 (排除目標本身佔掉的欄位) ----------
        # 深度取樣帶是畫面中央那條，而對準好的紙條正好就落在裡面 ——
        # 近距離時 depth_C 量到的是紙條自己，避牆會在每次逆近時誤觸發。
        # 把目標框的欄位挖掉再取中央三分之一的平均，量到的才是「紙條後面還有多遠」。
        if depth_map is not None and dy1 is not None:
            data.depth_ahead = self._clear_depth(depth_map, dy1, dy2, w3, w_img, best,
                                                 data.depth_C)

        # ---------- 狀態列 ----------
        cv2.putText(annotated_frame, self._status_text(data, best), (10, h_img - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        data.annotated_frame = annotated_frame
        return data

    # ==========================================
    # 內部：深度與繪圖
    # ==========================================
    @staticmethod
    def _clear_depth(depth_map, y1, y2, w3, w_img, best, fallback):
        """中央三分之一的深度平均，但排除目標框佔掉的欄位"""
        cols = np.zeros(w_img, dtype=bool)
        cols[w3:2 * w3] = True
        if best is not None:
            bx0 = max(0, best["cx"] - best["w"] // 2)
            bx1 = min(w_img, best["cx"] + best["w"] // 2 + 1)
            cols[bx0:bx1] = False
        if not cols.any():
            # 紙條已經寬到蓋滿整個中央帶 -> 沒有「旁邊」可以量，退回原本的中央深度
            return fallback
        return float(np.mean(depth_map[y1:y2][:, cols]))

    @staticmethod
    def _draw_box(img, cx, cy, w, h, colour, label, thick):
        bx, by = cx - w // 2, cy - h // 2
        cv2.rectangle(img, (bx, by), (bx + w, by + h), colour, thick)
        if label:
            cv2.putText(img, label, (bx, max(12, by - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    def _draw_candidates(self, img, shaped, best):
        """
        把每個候選依狀態塗上顏色。這是飛行資訊不是除錯資訊 ——
        飛手看一眼就知道無人機為什麼追這條、為什麼不追那條。

            綠(粗)  已確認且鎖定 = 正在追的目標
            青      已確認但沒被選中
            黃      投票中
            洋紅    已否決 (不是白的)，冷卻中
            暗藍    分割器這一格棄權
        """
        now = self._now()
        for c in shaped:
            slot = c.get("slot")
            if slot is None:
                continue
            is_best = best is not None and slot is best.get("slot")
            if slot.state == "CONFIRMED":
                colour = (0, 255, 0) if is_best else (255, 255, 0)
                thick = 3 if is_best else 1
                label = (f"TISSUE {c['conf']:.2f} d{c['dist_cm']:.0f} a{c['aspect']:.1f}"
                         if is_best else f"d{c['dist_cm']:.0f}")
            elif slot.state == "REJECTED":
                colour, thick = (255, 0, 255), 2
                info = c.get("wsinfo")
                why = info["reason"] if info and info.get("reason") else ""
                label = f"NOTWHITE {why} {max(0.0, slot.rejected_until - now):.1f}s"
            elif slot.last_verdict == UNKNOWN and "wsinfo" in c:
                colour, thick = (180, 120, 0), 1
                label = "SEG?"
            else:
                colour, thick = (0, 255, 255), 2
                n_white = sum(1 for _, v in slot.votes if v == WHITE)
                label = f"W {n_white}/{self.VOTE_WINDOW}"

            self._draw_box(img, c["cx"], c["cy"], c["w"], c["h"], colour, label, thick)
            if "wsinfo" in c:
                self.verifier.draw(img, c["wsinfo"])

        if best is not None:
            cv2.drawMarker(img, (best["cx"], best["cy"]), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            if self.show_all and "wsinfo" in best:
                # 現場調 sat_max / blob_aspect_min 時最有用的畫面：直接看到分割結果
                self.verifier.draw_mask_pip(img, best["wsinfo"])

    def _status_text(self, data, best):
        if self.model is None:
            return "NO TISSUE MODEL (roam only)"
        s = f"TISSUE {'1' if best else '0'}/{data.n_confirmed}"
        if best:
            s += f"  d{best['dist_cm']:.0f}cm  a{best['aspect']:.1f}"
        if data.n_pending:
            s += f"  vote {data.n_pending}"
        if data.n_rejected:
            s += f"  rej {data.n_rejected}"
        if data.n_seg_unknown:
            s += f"  seg? {data.n_seg_unknown}"
        if data.n_low_conf:
            s += f"  low-conf {data.n_low_conf}"
        if data.n_bad_shape:
            s += f"  bad-shape {data.n_bad_shape}"
        if self.show_all:
            s += "  [SHOW ALL]"
        return s
