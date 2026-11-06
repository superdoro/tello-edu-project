import csv
import os
import queue
import re
import threading
from collections import Counter
from datetime import datetime

import cv2


def normalize_plate(text):
    """比對用的正規化：大寫、去掉空白與 - 。ABC-1234 / abc 1234 / ABC1234 算同一個號碼"""
    if not text:
        return ""
    return re.sub(r"[\s\-_.]", "", str(text)).upper()


class PlateReader:
    """
    非同步 OCR + 投票。

    OCR 一次要幾十到幾百 ms，放在主迴圈裡跑，Tello 收不到指令的時間會超過看門狗門檻 (0.3 秒)。
    所以 OCR 在背景執行緒跑，主迴圈只做兩件不會阻塞的事 (同 VideoRecorder 的做法)：
        submit(crop)  背景閒著就把截圖交出去，忙就回 False (這一格不送，下一格再試)
        poll()        取回已經跑完的結果，同時計票

    世代編號：clear() 之後，上一個車牌還在跑的那一次 OCR 結果會被丟掉，
    不會混進新車牌的票裡 (換車牌、按 R 重讀時)。

    :param ocr: PlateOCRBackend (見 vision/plate_backends.py)
    :param min_conf: 信心低於這個的結果不計票
    :param out_dir: 讀到的車牌截圖與 plates.csv 存在這裡
    :param synchronous: True 時 submit 直接在呼叫端跑 OCR (離線測試、重播用，實飛不要開)
    """

    OUT_DIR = os.path.join("recordings", "plates")

    def __init__(self, ocr, min_conf=0.5, out_dir=None, synchronous=False):
        self.ocr = ocr
        self.min_conf = min_conf
        self.out_dir = out_dir or self.OUT_DIR
        self.synchronous = synchronous
        self._jobs = queue.Queue(maxsize=1)
        self._results = queue.Queue()
        self._busy = threading.Event()
        self._gen = 0
        self._lock = threading.Lock()
        self._thread = None
        self._warned = False
        self.reset_votes()

    # ==========================================
    # 主迴圈呼叫 (都不會阻塞)
    # ==========================================
    @property
    def busy(self):
        return self._busy.is_set()

    def submit(self, crop):
        """把截圖交給背景 OCR。背景還在跑上一張就回 False，這一張不送"""
        if crop is None or crop.size == 0 or self._busy.is_set():
            return False
        with self._lock:
            gen = self._gen
        self.n_submitted += 1
        if self.synchronous:
            self._run_one(gen, crop)
            return True
        self._ensure_thread()
        self._busy.set()
        try:
            self._jobs.put_nowait((gen, crop))
        except queue.Full:
            self._busy.clear()
            self.n_submitted -= 1
            return False
        return True

    def poll(self):
        """
        取回這段時間跑完的結果並計票。
        :return: 這次新取回的 [(原始文字 or None, 信心), ...]
        """
        out = []
        while True:
            try:
                gen, text, conf, crop = self._results.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                if gen != self._gen:
                    continue                       # 上一個車牌的結果，丟掉
            out.append((text, conf))
            self.last_text = text or ""
            key = normalize_plate(text)
            if key and conf >= self.min_conf:
                self.votes[key] += 1
                self.raw_forms.setdefault(key, Counter())[text] += 1
                # 每個號碼留最近一張讀出它的截圖，存檔用
                self.crops[key] = crop
        return out

    def clear(self):
        """換車牌 / 重讀：清掉票數，背景正在跑的那一次結果也作廢"""
        with self._lock:
            self._gen += 1
        self.reset_votes()

    def reset_votes(self):
        self.votes = Counter()          # 正規化號碼 -> 票數
        self.raw_forms = {}             # 正規化號碼 -> Counter(原始寫法)
        self.crops = {}
        self.last_text = ""
        self.n_submitted = 0

    def leader(self):
        """
        目前票數最多的號碼。
        :return: (顯示用的原始寫法, 票數)，沒有票時回 (None, 0)
        """
        if not self.votes:
            return None, 0
        key, n = self.votes.most_common(1)[0]
        return self.raw_forms[key].most_common(1)[0][0], n

    def votes_text(self):
        """log 用：ABC1234:2 XYZ999:1"""
        return " ".join(f"{k}:{n}" for k, n in self.votes.most_common(3))

    def save(self, text, uncertain=False):
        """
        存下讀到的車牌：<時間>_<號碼>.jpg 與 plates.csv。
        只在讀完時呼叫一次 (一張小截圖，幾 ms)，失敗也不影響飛行。
        """
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            key = normalize_plate(text)
            safe = "".join(c if c.isalnum() else "_" for c in key) or "NA"
            img_path = os.path.join(self.out_dir, f"{stamp}_{safe}.jpg")
            crop = self.crops.get(key)
            if crop is not None:
                cv2.imwrite(img_path, crop)
            csv_path = os.path.join(self.out_dir, "plates.csv")
            new = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "plate", "votes", "uncertain", "image"])
                w.writerow([stamp, text, self.votes_text(), int(uncertain),
                            os.path.basename(img_path) if crop is not None else ""])
            print(f"[車牌辨識] 已存檔 -> {img_path if crop is not None else csv_path}")
        except Exception as e:
            print(f"[車牌辨識] 存檔失敗：{e}")

    # ==========================================
    # 背景執行緒
    # ==========================================
    def _ensure_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _worker(self):
        while True:
            gen, crop = self._jobs.get()
            self._run_one(gen, crop)
            self._busy.clear()

    def _run_one(self, gen, crop):
        try:
            text, conf = self.ocr.read(crop)
            conf = float(conf or 0.0)
        except Exception as e:
            # 後端出錯不可以讓執行緒死掉 (之後的 submit 會永遠沒有結果)，當成讀不到
            if not self._warned:
                print(f"[車牌辨識] OCR 發生錯誤 (之後不再顯示)：{e!r}")
                self._warned = True
            text, conf = None, 0.0
        self._results.put((gen, text, conf, crop))
