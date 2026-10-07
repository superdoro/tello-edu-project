import cv2
import numpy as np


class BlankSurfaceAnalyzer:
    """
    白色無紋理區分析器 —— DarkRegionAnalyzer 的另一半。

    單張影像深度模型面對純白、沒有紋理的柱子或牆面時，和純黑牆一樣讀不出距離，
    而且通常誤判成「很遠」。2026-10-05 17:25 那趟就是這樣撞上柱子：
    柱子佔滿畫面左側、右緣已經侵入中央帶，深度卻讀 L 213~299cm、C 450~1000cm，
    漫遊引擎以 fb=50 直直飛了 2 秒撞上去。

    判斷方式是逐「欄」看：一欄在取樣帶內有 COL_BLANK_FRAC 以上的像素
    是「亮而且沒有紋理」，這一欄就算空白。再統計左/中/右三段各有多少比例的欄是空白。
    逐欄而不是逐像素，是因為柱子是直立的 —— 侵入中央帶時是整欄整欄地出現，
    而遠處白牆上方的天花板、下方的地板紋理會把逐像素的比例稀釋掉。

    「整欄」是畫面幾乎全高 (3%~97%)，不是中間一條帶子 —— 這是用來分遠近的幾何條件：
    無人機在 190cm 高時，一面素白的牆要從畫面頂端一路白到底端，必須在約 2.2m 以內；
    再遠的話畫面下方一定會露出地板 (磁磚有紋理) 或家具。17:40 那趟用中間帶子判斷時，
    2~4m 外的素白牆 (這個房間很多) 一直被當成近距離，漫遊會一直後退轉彎。
    改成全高後，柱子的判定完全不變，遠處的素白牆全部歸零。

    用 17:25 那趟錄影量到的數值 (中央三分之一被空白欄佔的比例)：
        撞柱子前 11.0~12.6s       L=1.00  C=0.15~0.29
        正常漫遊 4.4~8.6s         L,C <= 0.25，且左右從未同時 >= 0.9
        正對真牆 2.6~3.4s         C=0.80~0.93 (深度模型這時也有看到)
    """

    def __init__(self, roi_top=0.03, roi_bottom=0.97, tex_thresh=6.0, bright_thresh=90,
                 col_blank_frac=0.8, side_full=0.9, front_full=0.5, side_intrude=0.12,
                 near_cm=120.0):
        """
        :param roi_top / roi_bottom: 取樣範圍 (畫面高度比例)。幾乎全高，理由見類別說明
        :param tex_thresh: 15x15 視窗平均梯度低於此值算「沒有紋理」
        :param bright_thresh: 灰階高於此值才算「白」(暗的交給 DarkRegionAnalyzer)
        :param col_blank_frac: 一欄裡有這個比例以上的像素是空白，這一欄就算空白
        :param side_full: 左或右三分之一有這個比例以上的欄是空白 -> 那一側貼著白牆
        :param front_full: 中央三分之一有這個比例以上的欄是空白 -> 正前方是白牆
        :param side_intrude: 側邊貼著白牆時，中央被侵入這個比例就算擋在路徑上
        :param near_cm: 判定為障礙時，覆寫給漫遊引擎的深度 (低於漫遊的 CRITICAL_DIST 150
                        -> 一邊後退一邊轉開)
        """
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.tex_thresh = tex_thresh
        self.bright_thresh = bright_thresh
        self.col_blank_frac = col_blank_frac
        self.side_full = side_full
        self.front_full = front_full
        self.side_intrude = side_intrude
        self.near_cm = near_cm

    def analyze(self, frame):
        """
        :return: dict
                 L / C / R      : 三段各自的空白欄比例
                 near_L/C/R     : 該段是否判定為近距離白色障礙
                 cols           : 每一欄是否空白 (供 UI 疊圖)
                 roi            : (y1, y2)
        """
        h_img, w_img = frame.shape[:2]
        y1 = int(h_img * self.roi_top)
        y2 = int(h_img * self.roi_bottom)
        w3 = w_img // 3

        # 上下多取一點邊界，模糊與梯度才不會在取樣帶邊緣失真
        pad = 10
        ya, yb = max(0, y1 - pad), min(h_img, y2 + pad)
        g = cv2.GaussianBlur(cv2.cvtColor(frame[ya:yb], cv2.COLOR_BGR2GRAY), (5, 5), 0)
        mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        tex = cv2.blur(mag, (15, 15))
        blank = (tex < self.tex_thresh) & (g > self.bright_thresh)
        blank = blank[y1 - ya:y1 - ya + (y2 - y1)]

        cols = blank.mean(axis=0) > self.col_blank_frac if blank.size else np.zeros(w_img, bool)
        L = float(cols[:w3].mean())
        C = float(cols[w3:2 * w3].mean())
        R = float(cols[2 * w3:].mean())

        near_L = L >= self.side_full
        near_R = R >= self.side_full
        near_C = C >= self.front_full or ((near_L or near_R) and C >= self.side_intrude)

        return {"L": L, "C": C, "R": R,
                "near_L": near_L, "near_C": near_C, "near_R": near_R,
                "cols": cols, "roi": (y1, y2)}

    def apply(self, data, info):
        """把判定結果覆寫進 VisionData 的 depth_L/C/R (只會往近的方向改)"""
        data.blank_L, data.blank_C, data.blank_R = info["L"], info["C"], info["R"]
        data.blank_near_L, data.blank_near_C, data.blank_near_R = (
            info["near_L"], info["near_C"], info["near_R"])
        if info["near_L"]:
            data.depth_L = min(data.depth_L, self.near_cm)
        if info["near_C"]:
            data.depth_C = min(data.depth_C, self.near_cm)
        if info["near_R"]:
            data.depth_R = min(data.depth_R, self.near_cm)

    @staticmethod
    def draw(annotated_frame, info):
        """被判定為空白的欄在取樣帶上緣畫一條黃線；判定為障礙的那段標 WHITE WALL"""
        y1, _ = info["roi"]
        cols = info["cols"]
        xs = np.flatnonzero(cols)
        if xs.size:
            annotated_frame[max(0, y1 - 3):y1, xs] = (0, 255, 255)
        w3 = annotated_frame.shape[1] // 3
        for i, key in enumerate(("near_L", "near_C", "near_R")):
            if info[key]:
                cv2.putText(annotated_frame, "WHITE WALL", (i * w3 + 10, y1 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return annotated_frame
