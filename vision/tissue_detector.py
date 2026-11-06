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
from utils.colour_strip import ColourStripVerifier

# 目標顏色 (C 鍵切換)。投票時 WHITE / NOT_WHITE 代表「符合 / 不符合目前的目標」，
# 不論目標是白色還是彩色 (見 utils/white_strip.py 的 MATCH / NO_MATCH)。
TARGET_WHITE = "white"
TARGET_COLOUR = "colour"
_TARGET_TAG = {TARGET_WHITE: "WHITE", TARGET_COLOUR: "COLOR"}

# 因為「顏色不符」被否決的理由 —— 這些是真的紙條、只是不是目前要撞的顏色，
# 要輸出到 data.avoid 讓飛行邏輯閃開 (白色模式：col / colour；彩色模式：white)。
# 形狀不符、多色 (畫、海報)、飛手按 R 放棄的都不在這裡。
# 只憑「不夠白 / 不夠彩」而沒有另一個驗證器正面確認的 (col? / white?) 也不在這裡 ——
# 那多半是牆角、柱子、門框被 YOLO 框到 (10/19 20:46 錄影一直被這些「有色紙條」嚇退)，見 _verify。
AVOID_REASONS = {"col", "colour", "white"}

# 由紙條厚度估距離：dist_cm = STRIP_WIDTH_CM * FOCAL_PX / 厚度_px (見 TissueDetector._strip_thickness)。
# FOCAL_PX 是 720x480 影像的水平焦距 (約 461px)，紙條實寬約 3cm。
# 精度隨距離改善 —— 1m 處厚度 14px，±1px 就是 ±7cm；35cm 處 39px，±1px 只有 ±0.9cm。
# 剛好在最需要準確的近距離最準。
# ⚠️ 不用框寬：紙條斜掛、晃動時框會寬好幾倍 (10/20 22:20 錄影：框寬估 16~22cm，實際約 1 公尺)。
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
                 "last_verdict", "last_blob_aspect", "colour_watch", "reject_reason", "thick",
                 "revoked")

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
        # 白色模式確認之後，持續記錄「這一格看不看得出彩色條紋」(見 TissueDetector._colour_watch)
        self.colour_watch = deque(maxlen=5)
        self.reject_reason = ""          # 被否決的理由 (見 AVOID_REASONS)
        self.thick = deque(maxlen=5)     # [(時間, 紙條厚度 px)]，估距取最近幾格的中位數
        self.revoked = False             # 確認過、後來看到彩色條紋才撤銷的 (飛行邏輯可能正在衝它)


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

    ==========================================================
    撞白色或撞彩色 (C 鍵切換)
    ==========================================================
    第 3 關的驗證器可以換：白色用 WhiteStripVerifier (亮 + 不飽和 + 細長)，
    彩色用 ColourStripVerifier (飽和 + 單一色相 + 細長)。兩者回傳同樣的三值判定，
    投票、黑名單、鎖定的邏輯完全共用。切換時清掉所有投票與鎖定，用新的標準重新判斷。

    ==========================================================
    不論顏色先靠近，夠近才判斷 (JUDGE_DIST_CM)
    ==========================================================
    遠處的紙條只有幾個像素寬，彩色條紋被壓縮雜訊糊掉，白的被判成有色、有色的被判成白的
    (10/19 實飛：遠處判錯一次就一直被「有色紙條」嚇退)。所以估距超過 JUDGE_DIST_CM 時不投票，
    當成「還沒判斷」的接近目標輸出 (data.tissue["judged"] = False)，
    飛行邏輯先飛過去，夠近時停下來看清楚再決定：是目標就衝，不是就轉開。

    ==========================================================
    估距：紙條輪廓的厚度 (_strip_thickness)
    ==========================================================
    紙條實寬 3cm 已知，量出它在畫面上「垂直於紙條方向」有幾個像素就能換算距離。
    試過的其他量法 (10/20 錄影) 都不行：
        框寬       斜掛、晃動時框寬好幾倍 (19:17：2~3 公尺外估成 70~115cm；
                   22:20：約 1 公尺估成 16~22cm，觸發了對空氣的 COAST)
        輪廓面積   取決於 YOLO 切出多長的一段：同一條紙條在「整條」與「只切到上半段」之間
                   跳 3 倍 (22:20 23.9~24.7 秒)；畫面邊緣的大框一出現就是 5 萬
        深度模型   看不到細紙條，量到的是後面的牆
    厚度不受切出來的長度、斜掛影響，逼近時平滑遞減 (22:20 52.6~55.0 秒 97 -> 46cm，
    與實際前進量一致)。每格量到的厚度再取同一個 slot 最近幾格的中位數，擋掉單格輪廓暴衝。

    ==========================================================
    模型類別
    ==========================================================
    目前用的 best.pt 只有一個類別 (tissue)。試過的 paper_two_class 模型有兩個類別
    (paper_plain / paper_red_marked)，兩個都是衛生紙條 —— 所以預設把模型的所有類別都收下
    (tissue_class_ids=None)，換模型不用改程式；顏色一律由驗證器在近距離判斷。
    類別名稱記在候選的 "cls" 欄位 (畫面標籤、飛行紀錄)，目前不參與判斷。
    """

    def __init__(self,
                 yolo_path="model/yolo26/runs/detect/yolo_tissue_collect/best_v3.pt",
                 depth_model_path="model/yolo26/runs/detect/yolo_depth_collect/yolo26n-depth.pt",
                 tissue_class_ids=None, conf_threshold=0.5,
                 min_aspect=3.0, max_width_ratio=0.20,
                 min_aspect_locked=1.2, max_width_ratio_locked=0.45,
                 depth_band=(0.35, 0.45),
                 verifier=None, now_fn=time.time, target=TARGET_WHITE,
                 yolo_imgsz=960, use_native=True):
        """
        :param yolo_path: 衛生紙條偵測模型
        :param depth_model_path: 深度模型，供漫遊避障使用
        :param tissue_class_ids: 要收下的類別編號 (int 或一組)，None = 模型的所有類別都是紙條
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
        :param target: 一開始要撞的顏色，TARGET_WHITE 或 TARGET_COLOUR (飛行中用 C 鍵切換)
        :param yolo_imgsz: YOLO 推論的輸入尺寸。960 = Tello 原生寬度，不放大也不縮小
        :param use_native: True 時 YOLO 吃 Tello 的原生畫面 (見 set_native_frame)
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

        if tissue_class_ids is None:
            self.tissue_class_ids = None
        elif isinstance(tissue_class_ids, int):
            self.tissue_class_ids = {tissue_class_ids}
        else:
            self.tissue_class_ids = set(tissue_class_ids)
        names = getattr(self.model, "names", None) or {}
        self.class_names = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        if self.class_names:
            used = [n for i, n in self.class_names.items()
                    if self.tissue_class_ids is None or i in self.tissue_class_ids]
            print(f"[系統訊息] 紙條類別: {', '.join(used)}")
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
        self.white_verifier = verifier if verifier is not None else WhiteStripVerifier()
        # 彩色驗證器。⚠️ 門檻還沒有用實際錄影校正過 (見 utils/colour_strip.py)
        self.colour_verifier = ColourStripVerifier()
        if target not in _TARGET_TAG:
            raise ValueError(f"target 只能是 {list(_TARGET_TAG)}，收到 {target!r}")
        self.target = target
        self.MAX_VERIFY_PER_FRAME = 4   # 每格最多驗幾個框 (取最近的)，限制最壞情況耗時
        # 夠近才判斷 (見類別說明)。80cm：舊版「輪廓面積 2500」在 10/20 22:20 錄影裡大約就是這個距離。
        # 已經開始投票的 slot，估距跳到 JUDGE_KEEP_CM 以內 (紙條扭轉、輪廓跳動) 也繼續驗；
        # 跳更遠就暫停投票 —— 單格輪廓暴衝不會讓遠處的紙條一路投下去。
        self.JUDGE_DIST_CM = 80
        self.JUDGE_KEEP_CM = 100
        self.THICK_WINDOW_SEC = 0.4     # 估距取同一個 slot 這段時間內厚度的中位數
        # 碰到畫面左右邊緣、一出現估距就這麼近的框不當紙條 (已鎖定的那條除外)：
        # 10/20 22:20 錄影 13.2 / 31.7 秒，邊緣的大框估距 7~8cm，一出現就停下來判斷再 DODGE。
        # 真的紙條會先在遠處被看到、一路追近，不會在畫面邊緣突然冒出來就貼在眼前
        self.EDGE_NEAR_CM = 25

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
        self.COLOUR_REVOKE_MIN = 3  # 白色模式：已確認的目標最近 5 格有幾格看到彩色條紋就撤銷
        # 顏色不符的紙條確定不是目標，否決要撐得比一般久 (而且只要還看得到就一直延長)：
        # 冷卻到期後會重新投票，那 5 格左右的空窗裡它不在閃避清單上
        self.AVOID_COOLDOWN = 30.0
        # 碰到畫面左右邊緣的框 (見 EDGE_NEAR_CM)
        self.AVOID_EDGE_PX = 6
        # 畫面兩旁的不該撞紙條不理：畫面分三等分，中心落在左右兩份的不列入閃避
        # (不閃、不繞、不出橫幅)。10/20 22:47 錄影一直被畫面兩旁的條紋紙條觸發 AVOID。
        # 1/3 寬 = 離中央 120px，60cm 處約 16cm，比機身半寬 (12cm) + 紙條半寬還寬；
        # 40cm 以內的才可能擦到護罩 (紙條很軟，不會出事)
        self.AVOID_SIDE_RATIO = 1.0 / 3.0
        self.SLOT_TTL = 1.5         # 多久沒再看到就把 slot 丟掉 (秒)
        # 被否決的 slot 是用「畫面上的水平位置」配對的，無人機一轉頭，所有紙條都會掃過整個畫面。
        # 10/19 20:46 錄影：否決看不到也保留 30 秒，經過那個舊位置的白紙條就被配進去、
        # 直接繼承「有色、要閃避」—— 整趟一條白紙條都沒確認過 (n_conf 全程 0)，一直被嚇退。
        # 所以：看不到超過 REJECT_TTL 就丟掉；剛剛沒看到 (超過 REJECT_GAP_SEC) 的只配很近的框。
        self.REJECT_TTL = 3.0       # 被否決的 slot 看不到多久就丟掉 (秒)，閃爍的誤判在這段時間內照樣擋
        self.REJECT_GAP_SEC = 0.25  # 被否決的 slot 這麼久沒看到，就只配 REJECT_STALE_GATE 以內的框
        self.REJECT_STALE_GATE = 40

        # ---------- 目標黏著 ----------
        # 場地裡通常不只一條紙條。如果每一格都選「面積最大」的那條，兩條大小接近時
        # 選擇會在兩條之間來回跳 —— 實測錄影裡 389 格跳了 33 次 (368 -> 710 -> 325 -> 671)，
        # 無人機的轉向就跟著在 ±60 之間甩來甩去，永遠對不準。
        # 改成：已經鎖定一條就優先選離它最近的，只有它消失夠久才重新挑最大的。
        self.lock_cx = None
        self.lock_miss = 0
        self.lock_judged = False   # 鎖定的是已確認的目標 (True) 還是還沒判斷的接近目標
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

        # ---------- YOLO 的輸入解析度 ----------
        # Tello 原生畫面是 960x720，主迴圈會先縮成 720x480 (比例也從 4:3 壓成 3:2) 給整條管線用。
        # 遠處的紙條只有幾個像素寬，縮小那一步損失最大；而模型 (yolo26s-seg，1280 訓練)
        # 預設會把 720x480 再放大到 1280 —— 丟掉的細節補不回來，只是變慢。
        # 10/19 錄影實測 (720x480 輸入)：640 / 960 / 1280 偵測到紙條的格數 25% / 22% / 22%，
        # 每格 10 / 18 / 33ms —— 放大沒有幫助。
        # 所以改成：有原生畫面就讓 YOLO 直接吃原生畫面 (imgsz 960 = 原生寬度)，
        # 偵測框與輪廓再換算回 720x480 座標，後面的驗證、追蹤、避障都不用改。
        self.yolo_imgsz = yolo_imgsz
        self.use_native = use_native
        self._native = None

    # ==========================================
    # 對外
    # ==========================================
    @property
    def verifier(self):
        """目前目標顏色對應的驗證器"""
        return self.colour_verifier if self.target == TARGET_COLOUR else self.white_verifier

    @property
    def target_tag(self):
        return _TARGET_TAG[self.target]

    def toggle_target(self):
        """
        C 鍵：切換要撞白色還是彩色的紙條。

        清掉所有投票、黑名單與鎖定 —— 舊的票是用另一套標準投的
        (白色模式否決掉的正好是彩色目標)，留著會讓新目標被舊的黑名單擋住。
        """
        self.target = TARGET_COLOUR if self.target == TARGET_WHITE else TARGET_WHITE
        self._slots = []
        self.lock_cx = None
        self.lock_miss = 0
        self.lock_judged = False
        name = "彩色" if self.target == TARGET_COLOUR else "全白"
        print(f"[視覺系統] 目標切換為：{name}的衛生紙條 ({self.target_tag})")

    def toggle_tracking_mode(self):
        """F 鍵：切換「顯示被擋掉的偵測」(灰色=信心不足，橘色=形狀不符)"""
        self.show_all = not self.show_all
        print(f"[視覺系統] 低信心偵測顯示：{'開啟' if self.show_all else '關閉'}")

    def set_native_frame(self, frame):
        """
        主迴圈在 process_frame 之前呼叫，交給這一格的 Tello 原生畫面 (縮小之前)。
        只用一次：process_frame 用完就清掉，避免下一格沒給時誤用舊畫面。
        """
        self._native = frame

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
                    s.reject_reason = "manual"
            print(f"[視覺系統] 已放棄目前目標，{self.REJECT_COOLDOWN:.0f} 秒內不再鎖定該位置")
        self.lock_cx = None
        self.lock_miss = 0

    # ==========================================
    # 投票與 slot 管理
    # ==========================================
    def _verify(self, frame, c):
        """
        用目前目標顏色的驗證器判斷一個候選框。

        白色模式多一道檢查：紙條上看得出彩色條紋 -> 擋掉。
        白色驗證器只看「亮的那一半夠不夠白」，白色衛生紙畫了幾段紫色條紋
        (10/13 錄影) 大部分還是白的，照樣 5/5 票通過；深色紙條掛在白牆前也會騙過它
        (亮的那一半是紙條兩側的白牆)。

        「要閃避」需要正面證據：白色模式下白色驗證器說「不夠白 (col)」，
        還要彩色驗證器在分割輪廓裡真的看到彩色條紋才算有色紙條，否則理由改成 col?
        (照樣不撞，但不閃)；彩色模式下說「是白的 (white)」也要白色驗證器確認。

        顏色一律只看紙條本身的像素 (YOLO 分割輪廓 c["poly"])，見 utils/colour_strip.py。
        """
        box = (c["cx"], c["cy"], c["w"], c["h"])
        poly = c.get("poly")
        if self.target == TARGET_COLOUR:
            info = self.colour_verifier.analyze(frame, box, poly)
            if info["verdict"] == NOT_WHITE and info["reason"] == "white":
                if self.white_verifier.analyze(frame, box)["verdict"] != WHITE:
                    info["reason"] = "white?"
            return info
        info = self.white_verifier.analyze(frame, box)
        if info["verdict"] == WHITE:
            cinfo = self.colour_verifier.features(frame, box, poly)
            if self.colour_verifier.has_colour(cinfo):
                info["verdict"], info["reason"] = NOT_WHITE, "colour"
        elif info["verdict"] == NOT_WHITE and info["reason"] == "col":
            cinfo = self.colour_verifier.features(frame, box, poly)
            if not self.colour_verifier.has_colour(cinfo):
                info["reason"] = "col?"
        elif info["verdict"] == NOT_WHITE:
            # 形狀不像紙條 (a1.7 這類)：近距離時條紋把白色切成好幾段，白色團塊變短就會這樣
            # (10/20 22:47 錄影 6 條條紋紙條都是這個理由被否決，沒進閃避清單)。
            # 紙條輪廓裡看得到彩色就當成有色紙條
            cinfo = self.colour_verifier.features(frame, box, poly)
            if self.colour_verifier.has_colour(cinfo):
                info["reason"] = "colour"
        return info

    def _colour_watch(self, frame, candidates, now):
        """
        白色模式：已確認的目標每一格都再看一次有沒有彩色條紋，連續看到就取消。

        確認是終態 (理由見類別說明)，而紙條在遠處時條紋又細又糊，可能在還看不清楚的時候
        就被確認成白的 —— 靠近之後條紋清楚了，不撤銷的話照樣衝過去撞。
        撤銷只會讓無人機放棄這一條，不會讓它去撞什麼，所以可以放心做。
        5 格裡有 COLOUR_REVOKE_MIN 格看到條紋才撤銷，避免單格雜訊把真的白紙條丟掉。

        :return: 撤銷後仍然合格的候選
        """
        keep = []
        for c in candidates:
            slot = c["slot"]
            cinfo = self.colour_verifier.features(frame, (c["cx"], c["cy"], c["w"], c["h"]),
                                                  c.get("poly"))
            if cinfo.get("ok"):                     # 太小看不清楚的格不算票
                slot.colour_watch.append(self.colour_verifier.has_colour(cinfo))
            if sum(slot.colour_watch) >= self.COLOUR_REVOKE_MIN:
                slot.state = "REJECTED"
                slot.rejected_until = now + self.AVOID_COOLDOWN
                slot.reject_reason = "colour"
                slot.revoked = True
                if self.lock_cx is not None and abs(slot.cx - self.lock_cx) <= self.LOCK_GATE:
                    self.lock_cx, self.lock_miss = None, 0
                print("[視覺系統] 已確認的紙條看到彩色條紋 -> 取消鎖定，改為閃避 (白色模式)")
                continue
            keep.append(c)
        return keep

    def _match_slot(self, cx, cy, now, taken=()):
        """
        把一個框配到最近的 slot，配不到就開一個新的 (沿用 LOCK_GATE 的關聯規則)。

        taken 是這一格已經配過的 slot id：同一格裡一個 slot 只能配一個框。
        否則兩條水平距離不到 LOCK_GATE 的紙條會併成同一個 slot ——
        一條該撞、一條不該撞時，判定就混在一起了。
        """
        best, best_d = None, None
        for s in self._slots:
            if s.id in taken:
                continue
            d = abs(s.cx - cx)
            gate = self.LOCK_GATE
            if s.state == "REJECTED" and now - s.last_seen > self.REJECT_GAP_SEC:
                gate = self.REJECT_STALE_GATE        # 見 __init__ 的 REJECT_TTL 說明
            if d <= gate and (best_d is None or d < best_d):
                best, best_d = s, d
        if best is None:
            best = _Slot(self._next_slot_id, cx, cy, now)
            self._next_slot_id += 1
            self._slots.append(best)
        best.cx, best.cy, best.last_seen = cx, cy, now
        return best

    def _cast_vote(self, slot, verdict, now, saw_colour=False):
        """
        把一格的判定記進 slot 並更新狀態。

        UNKNOWN 不記票 —— 這是驗證器三值設計的重點：分割器說「我不知道」的時候，
        既有的判定就原封不動保留，不會因為近距離分割失效而把目標丟掉。

        saw_colour：這一格在紙條輪廓裡看到彩色 (白色模式)。視窗裡只要有一格看到彩色，
        就不能確認成白的 —— 白紙條在校正資料裡從來沒有被當成彩色 (0/31)，條紋紙條則是
        有幾格看得到、有幾格看不到 (淡紫色條紋轉到側面、被模糊掉)。舊規則「5 格裡 3 格白」
        會把 3 白 2 彩的條紋紙條確認成白的。多等幾格，彩色到 VOTE_REJECT_MIN 格就否決。
        """
        slot.last_verdict = verdict
        if verdict == UNKNOWN:
            return
        slot.votes.append((now, verdict, saw_colour))
        while slot.votes and (now - slot.votes[0][0] > self.VOTE_MAX_AGE
                              or len(slot.votes) > self.VOTE_WINDOW):
            slot.votes.popleft()

        n_white = sum(1 for _, v, _c in slot.votes if v == WHITE)
        n_not = sum(1 for _, v, _c in slot.votes if v == NOT_WHITE)
        n_colour = sum(1 for _, _v, c in slot.votes if c)
        if n_not >= self.VOTE_REJECT_MIN:
            slot.state = "REJECTED"
            slot.rejected_until = now + self.REJECT_COOLDOWN
        elif (len(slot.votes) >= self.VOTE_WINDOW and n_white >= self.VOTE_MIN_WHITE
              and n_colour == 0):
            slot.state = "CONFIRMED"

    def _expire_slots(self, now):
        keep = []
        for s in self._slots:
            if s.state == "REJECTED":
                # 被否決的 slot 要撐過冷卻期，短暫看不到 (REJECT_TTL 內) 也一樣 ——
                # 否則誤判一閃一閃的時候黑名單根本不會生效。但看不到太久就丟掉，
                # 不然轉頭之後別的紙條掃過這個位置會被誤配進來 (見 __init__)
                if now < s.rejected_until and now - s.last_seen <= self.REJECT_TTL:
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
        # 沒有鎖定，或鎖定的那條消失太久 -> 重新挑最近的 (估距用厚度，見類別說明)
        best = min(candidates, key=lambda c: c["dist_cm"])
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
        data.avoid = []          # 顏色不符、要閃開的紙條 (見 AVOID_REASONS)

        # 預設「前方無限遠」：深度模型缺席時，漫遊會退化成直線前進而不是卡住不動
        data.depth_L = data.depth_C = data.depth_R = data.depth_ahead = 999.0
        data.n_confirmed = data.n_pending = data.n_rejected = 0
        data.n_low_conf = data.n_bad_shape = data.n_seg_unknown = data.n_far = data.n_edge = 0

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
            # 有原生畫面就讓 YOLO 吃原生畫面，結果再乘上 (sx, sy) 換回 720x480 座標
            native, self._native = self._native, None
            yolo_in = frame
            if (self.use_native and native is not None and native.ndim == 3
                    and native.shape[1] > w_img):
                yolo_in = native
            sx, sy = w_img / yolo_in.shape[1], h_img / yolo_in.shape[0]
            data.yolo_input = f"{yolo_in.shape[1]}x{yolo_in.shape[0]}"
            results = self.model(yolo_in, imgsz=self.yolo_imgsz, verbose=False, device=self.device)
            shaped = []
            if results and results[0].boxes is not None:
                # 分割輪廓 (整張影像座標)：顏色判斷只看紙條本身的像素
                masks = getattr(results[0], "masks", None)
                polys = masks.xy if masks is not None else None
                # 厚度用 YOLO 輸入畫面 (原生 960x720) 的輪廓量：像素是正方形、解析度也比較高
                polys_in = polys
                if polys is not None and (sx != 1.0 or sy != 1.0):
                    polys = [np.asarray(p, dtype=np.float32) * np.array([sx, sy], np.float32)
                             for p in polys]
                for i, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls[0])
                    if self.tissue_class_ids is not None and cls_id not in self.tissue_class_ids:
                        continue
                    conf = float(box.conf[0])
                    bx, by, bw, bh = box.xywh[0].tolist()
                    cx, cy, w, h = int(bx * sx), int(by * sy), int(bw * sx), int(bh * sy)
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
                    # 還沒判斷的接近目標不放寬：遠處用寬鬆門檻會把牆角、柱子也收進來
                    locked_near = (self.lock_cx is not None and self.lock_judged
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

                    poly = polys[i] if polys is not None and i < len(polys) else None
                    poly_in = polys_in[i] if polys_in is not None and i < len(polys_in) else None
                    thick = self._strip_thickness(poly_in, sx)
                    if thick is None:
                        thick = float(w)                 # 沒有輪廓 (不是分割模型)：退回框寬
                    dist_cm = DIST_K / max(thick, 1e-3)

                    # 邊緣誤判：碰到畫面左右邊緣、一出現就貼在眼前的大框 (見 EDGE_NEAR_CM)
                    locked_any = (self.lock_cx is not None
                                  and abs(cx - self.lock_cx) <= self.LOCK_GATE)
                    at_edge = (cx - w / 2.0 <= self.AVOID_EDGE_PX
                               or cx + w / 2.0 >= w_img - self.AVOID_EDGE_PX)
                    if at_edge and dist_cm < self.EDGE_NEAR_CM and not locked_any:
                        data.n_edge += 1
                        if self.show_all:
                            self._draw_box(annotated_frame, cx, cy, w, h, (0, 165, 255),
                                           f"EDGE d{dist_cm:.0f}", 2)
                        continue

                    shaped.append({
                        "cx": cx, "cy": cy, "w": w, "h": h,
                        "area": w * h, "conf": conf, "aspect": aspect,
                        "strip_px": thick, "dist_cm": dist_cm,   # 單格的值，配到 slot 後換成中位數
                        "top_y": cy - h / 2.0, "bottom_y": cy + h / 2.0,
                        "top_clipped": cy - h / 2.0 <= 1,
                        "bottom_clipped": cy + h / 2.0 >= h_img - 1,
                        "poly": poly,
                        "cls": self.class_names.get(cls_id, str(cls_id)),
                    })

            # ---------- 白色驗證 ----------
            # 只驗還沒定案的 slot。已確認的不重驗 (理由見類別說明)，
            # 已否決的在冷卻期內也不必再驗。每格最多驗 MAX_VERIFY_PER_FRAME 個。
            # 還太遠的不驗 (JUDGE_DIST_CM)：不論顏色先讓飛行邏輯飛近，夠近了再判斷。
            shaped.sort(key=lambda c: c["dist_cm"])
            n_verified = 0
            taken = set()
            for c in shaped:
                slot = self._match_slot(c["cx"], c["cy"], now, taken)
                taken.add(slot.id)
                c["slot"] = slot
                # 估距取這個 slot 最近幾格厚度的中位數：擋掉單格輪廓暴衝 / 只切到一小段
                slot.thick.append((now, c["strip_px"]))
                recent = [v for t, v in slot.thick if now - t <= self.THICK_WINDOW_SEC]
                c["strip_px"] = float(np.median(recent))
                c["dist_cm"] = DIST_K / max(c["strip_px"], 1e-3)
                judging = bool(slot.votes) and now - slot.votes[-1][0] <= self.VOTE_MAX_AGE
                limit = self.JUDGE_KEEP_CM if judging else self.JUDGE_DIST_CM
                c["far"] = slot.state == "PENDING" and c["dist_cm"] > limit
                # 被畫面左右邊緣切掉一截的紙條先不判斷：看不到的那一截可能正好是條紋
                # (10/20 22:47 錄影兩條碰邊的條紋紙條被確認成白的)。飛行邏輯會先轉機頭對準，
                # 轉進畫面裡再判斷，不會多花時間
                cut_off = (c["cx"] - c["w"] / 2.0 <= self.AVOID_EDGE_PX
                           or c["cx"] + c["w"] / 2.0 >= w_img - self.AVOID_EDGE_PX)
                if c["far"]:
                    data.n_far += 1
                elif slot.state == "PENDING" and cut_off:
                    pass
                elif slot.state == "PENDING" and n_verified < self.MAX_VERIFY_PER_FRAME:
                    info = self._verify(frame, c)
                    n_verified += 1
                    c["wsinfo"] = info
                    slot.last_blob_aspect = info["blob_aspect"]
                    saw_colour = (self.target == TARGET_WHITE and info["verdict"] == NOT_WHITE
                                  and info["reason"] in ("colour", "col"))
                    self._cast_vote(slot, info["verdict"], now, saw_colour)
                    if slot.state == "REJECTED":
                        slot.reject_reason = info["reason"]
                        if info["reason"] in AVOID_REASONS:
                            slot.rejected_until = now + self.AVOID_COOLDOWN
                            kind = "非白" if self.target == TARGET_WHITE else "白色 (不是目標)"
                            print(f"[視覺系統] 認出{kind}的紙條 (約 {c['dist_cm']:.0f}cm) -> 列入閃避")
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
            if self.target == TARGET_WHITE and eligible:
                eligible = self._colour_watch(frame, eligible, now)
            judged = bool(eligible)
            if not eligible:
                # 沒有已確認的目標 -> 挑一條還沒判斷的 (面積還小的、或正在投票的) 飛過去看清楚
                eligible = [c for c in shaped if c["slot"].state == "PENDING"]
                if (eligible and self.lock_judged and self.lock_cx is not None
                        and self.lock_miss < self.LOCK_MISS_MAX):
                    # 鎖定的已確認目標只是這一格沒抓到 (YOLO 單格閃爍)：先撐幾格，不要馬上改追
                    # 還沒判斷的紙條 —— 不論顏色先接近之後，畫面裡幾乎總有還沒判斷的紙條，
                    # 不撐的話機頭會在兩條之間甩 (10/20 錄影重播 8 次)
                    eligible = []
            if eligible and self.lock_judged != judged:
                self.lock_cx, self.lock_miss = None, 0     # 換了種類就重新挑，不要等舊鎖定過期
            best = self._pick(eligible) if eligible else None
            if best is not None:
                self.lock_judged = judged

            # ---------- 要閃避的紙條：顏色不符的真紙條 ----------
            # 深度模型幾乎看不到細細的紙條，不告訴飛行邏輯的話，漫遊會直接穿過去撞上
            for c in shaped:
                s = c["slot"]
                if s.state == "REJECTED" and s.reject_reason in AVOID_REASONS:
                    s.rejected_until = max(s.rejected_until, now + self.AVOID_COOLDOWN)
                    # 畫面兩旁 (含碰到邊緣的牆角柱子) 的不理，見 AVOID_SIDE_RATIO
                    if self._at_side(c["cx"], w_img):
                        c["side"] = True
                        continue
                    a = {k: c[k] for k in ("cx", "cy", "w", "h", "top_y", "bottom_y", "dist_cm",
                                           "strip_px")}
                    a["target"] = self.target_tag          # 讓飛行邏輯知道這是「非白」還是「白色」
                    a["revoked"] = s.revoked               # 剛剛還是鎖定的目標 (飛行邏輯不可以滑行過去)
                    data.avoid.append(a)
            if not eligible and self.lock_cx is not None:
                # 整格都沒有合格候選也要累計，否則鎖定永遠不會過期
                self.lock_miss += 1
                if self.lock_miss > self.LOCK_MISS_MAX:
                    self.lock_cx = None

            self._draw_candidates(annotated_frame, shaped, best)
            if data.avoid:
                self._draw_avoid_alert(annotated_frame, now)

            if best:
                # 把驗證結果一起帶出去，飛行紀錄才有辦法事後判斷「為什麼追這一條」
                slot = best["slot"]
                best["slot_id"] = slot.id
                best["judged"] = judged                # False = 還沒判斷，只能接近、不能衝
                best["far"] = bool(best.get("far"))    # True = 面積還不夠大 (還沒開始投票)
                best["strip_px"] = round(best["strip_px"], 1)          # 紙條厚度 (估距的依據)
                best["white"] = 1 if judged else 0     # 通過目前目標的驗證
                best["target"] = self.target_tag
                best["votes"] = (sum(1 for _, v, _c in slot.votes if v == WHITE), len(slot.votes))
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
    def _at_side(self, cx, w_img):
        """紙條中心在畫面左右兩旁 (AVOID_SIDE_RATIO)：不該撞的紙條在這裡就不理"""
        return cx < w_img * self.AVOID_SIDE_RATIO or cx > w_img * (1.0 - self.AVOID_SIDE_RATIO)

    @staticmethod
    def _strip_thickness(poly, scale_x=1.0):
        """
        紙條「垂直於紙條方向」的厚度，換算成 720x480 畫面的水平像素 (估距用，見類別說明)。

        在輪廓的外框範圍內畫出遮罩，做距離轉換 (每個像素到邊界的距離)：
        中線 (脊線，距離的局部最大值) 上的 2 x 距離就是那裡的厚度 —— 斜掛的紙條量到的也是
        垂直方向的厚度，不是水平寬度。取第 75 百分位：紙條扭轉時側面朝前的部分很薄，
        正面朝前的部分才是真的 3cm。

        :param poly: YOLO 輸入畫面座標的輪廓 (原生 960x720 時像素是正方形)
        :param scale_x: 輸入畫面 -> 720 寬的水平縮放 (原生 960 時是 0.75)
        :return: 厚度 (px)，輪廓太小或沒有輪廓時回 None
        """
        if poly is None or len(poly) < 3:
            return None
        p = np.asarray(poly, dtype=np.float32)
        x0, y0 = np.floor(p.min(axis=0)) - 2
        x1, y1 = np.ceil(p.max(axis=0)) + 2
        mw, mh = int(x1 - x0) + 1, int(y1 - y0) + 1
        if mw < 3 or mh < 3:
            return None
        m = np.zeros((mh, mw), np.uint8)
        cv2.fillPoly(m, [np.round(p - np.array([x0, y0], np.float32)).astype(np.int32)], 1)
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        ridge = (m > 0) & (dt >= 1.0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6)
        if not ridge.any():
            return None
        return float(np.percentile(2.0 * dt[ridge], 75)) * scale_x

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
            黃      投票中 (夠近、判斷中)
            白(細)  還太遠、還沒判斷。APPROACH = 正在飛過去看的那一條 (後面是估距)
            洋紅    已否決。AVOID = 顏色不符的紙條，飛行邏輯會閃開；side (細框) = 在畫面兩旁、不理；
                    NOT-... = 其他理由，冷卻中
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
                if slot.reject_reason in AVOID_REASONS:
                    thick = 4                       # 不該撞的紙條要一眼看得到
                info = c.get("wsinfo")
                why = info["reason"] if info and info.get("reason") else ""
                if slot.reject_reason in AVOID_REASONS and c.get("side"):
                    thick, label = 1, "side"        # 畫面兩旁的不理 (見 AVOID_SIDE_RATIO)
                elif slot.reject_reason in AVOID_REASONS:
                    label = f"AVOID d{c['dist_cm']:.0f}"
                else:
                    label = f"NOT-{self.target_tag} {why} {max(0.0, slot.rejected_until - now):.1f}s"
            elif c.get("far"):
                colour, thick = (255, 255, 255), 2 if is_best else 1
                d = f"{c['dist_cm']:.0f}cm"
                label = f"APPROACH {d}" if is_best else f"far {d}"
            elif slot.last_verdict == UNKNOWN and "wsinfo" in c:
                colour, thick = (180, 120, 0), 1
                label = "SEG?"
            else:
                colour, thick = (0, 255, 255), 2
                n_white = sum(1 for _, v, _c in slot.votes if v == WHITE)
                label = f"{self.target_tag[0]} {n_white}/{self.VOTE_WINDOW}"

            self._draw_box(img, c["cx"], c["cy"], c["w"], c["h"], colour, label, thick)
            if "wsinfo" in c:
                self.verifier.draw(img, c["wsinfo"])

        if best is not None:
            cv2.drawMarker(img, (best["cx"], best["cy"]), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            if self.show_all and "wsinfo" in best:
                # 現場調 sat_max / blob_aspect_min 時最有用的畫面：直接看到分割結果
                self.verifier.draw_mask_pip(img, best["wsinfo"])

    def _draw_avoid_alert(self, img, now):
        """
        看到不該撞的紙條時的畫面回饋：頂端大字橫幅 + 整個畫面閃爍的洋紅色外框。
        細框加小字在飛行中很容易漏看 (10/19 錄影)。
        """
        h, w = img.shape[:2]
        magenta = (255, 0, 255)
        if int(now * 4) % 2 == 0:                   # 每秒閃 2 次
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), magenta, 10)
        text = ("NON-WHITE STRIP - AVOID" if self.target == TARGET_WHITE
                else "WHITE STRIP (NOT TARGET) - AVOID")
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        x0, y0 = (w - tw) // 2 - 12, 42                # 在模式名稱下方，不蓋到左上角的字
        cv2.rectangle(img, (x0, y0), (x0 + tw + 24, y0 + th + 18), magenta, -1)
        cv2.putText(img, text, (x0 + 12, y0 + th + 8), font, scale, (255, 255, 255), thick)

    def _status_text(self, data, best):
        if self.model is None:
            return "NO TISSUE MODEL (roam only)"
        s = f"[{self.target_tag}] TISSUE {'1' if best else '0'}/{data.n_confirmed}"
        if best:
            s += f"  d{best['dist_cm']:.0f}cm  a{best['aspect']:.1f}"
            if not best.get("judged", True):
                s += "  (unjudged)"
        if data.n_far:
            s += f"  far {data.n_far}"
        if getattr(data, "avoid", None):
            s += f"  avoid {len(data.avoid)}"
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
        if data.n_edge:
            s += f"  edge {data.n_edge}"
        if self.show_all:
            s += "  [SHOW ALL]"
        return s
