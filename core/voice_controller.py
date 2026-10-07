import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time


class VoiceController:
    """
    離線語音指令 (Vosk + arecord)。

    為什麼一定要離線：飛行時電腦連的是 Tello 熱點，沒有外網。
    任何雲端語音服務 (Google / Azure) 在飛行中都叫不動，而且一旦卡在等待回應，
    主迴圈就會停擺 —— Tello 超過約 15 秒收不到指令會自動降落。

    為什麼用 arecord 而不是 sounddevice：sounddevice 需要系統層的 PortAudio，
    這台機器沒有、裝它要 sudo。arecord 是 alsa-utils 內建的，直接開子行程讀
    16kHz 單聲道 PCM 就好，零系統相依。

    為什麼不用 Vosk 的限定詞彙 (grammar)：中文小模型以「詞」為單位，
    指令裡的單字 (羅、飛、進) 不在詞彙表內，限定反而會整句認不出來。
    改成自由辨識再用文字比對，不依賴詞彙表。

    安全設計：
    1. 必須先講喚醒詞 (泰羅 / tello / 無人機) 才會執行，避免閒聊誤觸發。
    2. 同一個指令在 COOLDOWN_SEC 內只會送出一次，避免辨識抖動連續觸發。
    3. 辨識完全在背景執行緒，主迴圈只做 get_command() 取佇列，絕不阻塞。
    """

    MODEL_DIR = "model/vosk/vosk-model-small-cn-0.22"
    SAMPLE_RATE = 16000
    BLOCK = 4000          # 每次從 arecord 讀的位元組數 (約 0.125 秒)
    COOLDOWN_SEC = 1.5    # 同一個指令的最短間隔

    # ==========================================
    # 喚醒詞與指令表 —— 內容是用 TTS 實測這個模型「聽成什麼」之後決定的，
    # 不是照著想當然耳的寫法列的。兩個實測發現：
    #
    # 1. 模型輸出是「簡體」。所以每個指令都要有簡體寫法，只寫繁體永遠對不上。
    # 2. 自創的名字認不出來。原本 readme 用的「泰羅」被聽成「嗨哦」，完全不會觸發。
    #    下面這些喚醒詞是實測 100% 認得的，建議用「無人機」或「小白」。
    #    仍保留泰羅的幾種常見誤聽結果，想沿用舊口令的話還是有機會觸發。
    # ==========================================
    WAKE_WORDS = [
        "无人机", "無人機",      # 實測最穩，建議用這個
        "小白", "悟空",          # 實測 100% 正確，短、好念
        "飞机", "飛機", "阿飞", "阿飛",
        "tello",
        # 「泰羅」的常見誤聽結果 (盡力而為，不保證)
        "嗨哦", "海螺", "开罗", "泰罗", "泰羅", "太罗",
    ]

    # 比對方式是「關鍵詞出現在句子裡」，不需要整句一字不差。
    # 每一項都同時列繁體與模型實際輸出的簡體。
    #
    # 註記 <推薦> 的說法是用 utils/voice_bench.py 混入四旋翼噪音掃描出來的 ——
    # 同一個意思換個講法，抗噪能力差很多。整組換成推薦說法之後，
    # 10dB 訊噪比下的正確率從 80% 變成 100%。
    INTENT_MAP = {
        # 左右放在最前面：這是最怕聽錯的兩個指令，先比對最明確的說法。
        "向左": ["左转", "左轉",           # <推薦> 噪音下 3/3
                 "往左", "左边", "左邊", "向左", "向佐", "left"],
        "向右": ["往右",                   # <推薦> 噪音下 3/3
                 "右转", "右轉", "右边", "右邊", "向右", "right"],

        "起飛": ["起飞", "起飛", "takeoff", "take off"],
        "降落": ["降落", "着陆", "著陸", "land"],
        "前進": ["前进", "前進", "往前", "向前", "forward"],
        "後退": ["往后", "往後",           # <推薦>「後退」實測會被聽成「后错误」，10dB 就失效
                 "后退", "後退", "向后", "向後", "退后", "後面", "backward"],
        "上升": ["升高",                   # <推薦> 噪音下 3/3；「往上」會被聽成「网上」
                 "上升", "往上", "向上", "up"],
        "下降": ["下降", "降低", "往下", "向下", "down"],
        "轉向": ["转身", "轉身",           # <推薦> 噪音下 3/3
                 "转向", "轉向", "旋转", "turn"],
        "停止": ["停止", "停下", "悬停", "懸停", "煞车", "煞車", "stop", "hover"],
        "切換": ["切换", "切換", "换模式", "換模式", "switch"],
    }

    # ==========================================
    # 第二層：噪音容忍表
    #
    # 用 utils/voice_tolerance.py 蒐集「噪音下實際被聽成什麼」而來。
    # 語料顯示錯聽有固定模式，例如「後退」幾乎一定變成「后错误」、「往左」變「王所」。
    # 收進來就能救回那些講了非推薦說法的情況。
    #
    # 但這層很危險，因為錯聽結果常常是日常詞彙。「無人機是網上買的」裡有「网上」、
    # 「無人機有錯誤」裡有「错误」—— 照第一層那樣整句比對，這些都會誤觸發。
    # 所以第二層額外加兩個條件 (見 _handle)：
    #   1. 錯聽詞必須「緊接」在喚醒詞後面，不能出現在句子中間
    #      「無人機有錯誤」-> 尾巴是「有错误」，「错误」不在開頭 -> 不觸發
    #   2. 尾巴最多只能比錯聽詞長 TOLERANCE_SLACK 個字
    #      「無人機象棋比賽」-> 尾巴「象棋比赛」比「象棋」長 2 個字 -> 不觸發
    #
    # 另外刻意排除掉幾個雖然常出現、但本身是日常詞的錯聽結果：
    #   错误 / 效果 / 享受 / 行驶 / 网上 / 网下 / 象棋 / 新高
    # 它們就算加上上面兩個條件仍可能被正常對話命中 (實測「无人机象棋比赛」
    # 和「无人机新高度」都會誤觸發)，寧可少救幾次也不要誤觸發。
    # ==========================================
    TOLERANCE_SLACK = 1           # 尾巴最多比錯聽詞長幾個字
    TOLERANCE_MAP = {
        "後退": ["后错", "后配", "后后"],     # 說「後退」時的錯聽
        "向左": ["王所", "王府"],             # 說「往左」「向左」時的錯聽
        "向右": ["像有", "有家"],             # 說「向右」「右邊」時的錯聽
        "停止": ["形象"],                     # 說「停下」時的錯聽
    }

    def __init__(self, model_dir=None, device=None, enabled=True):
        """
        :param model_dir: Vosk 模型目錄，None 則用預設路徑
        :param device: arecord 的裝置字串 (例如 "plughw:1,0")，None 則自動偵測
        :param enabled: False 可完全停用 (不開執行緒、不佔麥克風)
        """
        self.available = False
        self.queue = queue.Queue()
        self.last_text = ""          # 最近一次辨識到的原始文字，供畫面顯示
        self.last_text_time = 0.0
        self._last_cmd = ""
        self._last_cmd_time = 0.0
        self._proc = None
        self._thread = None
        self._running = False

        if not enabled:
            print("[語音] 已停用")
            return

        model_dir = model_dir or self.MODEL_DIR
        if not os.path.isdir(model_dir):
            print(f"[語音] 找不到模型 {model_dir} -> 語音功能停用")
            print("        下載: https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip")
            return
        if shutil.which("arecord") is None:
            print("[語音] 找不到 arecord (alsa-utils) -> 語音功能停用")
            return

        self.device = device or self._find_device()
        if self.device is None:
            print("[語音] 找不到錄音裝置 -> 語音功能停用")
            return

        try:
            import vosk
            vosk.SetLogLevel(-1)          # 關掉 Kaldi 的大量 log，否則會洗版飛行訊息
            self._model = vosk.Model(model_dir)
            self._rec = vosk.KaldiRecognizer(self._model, self.SAMPLE_RATE)
        except Exception as e:
            print(f"[語音] Vosk 初始化失敗 ({e}) -> 語音功能停用")
            return

        self.available = True
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"[語音] 已啟動 (裝置 {self.device})，說「無人機 + 指令」，例如「無人機起飛」、「無人機左轉」")

    # ==========================================
    @staticmethod
    def _find_device():
        """從 arecord -l 找出第一張有錄音功能的卡，回傳 plughw:<card>,<dev>"""
        try:
            out = subprocess.run(["arecord", "-l"], capture_output=True, text=True,
                                 timeout=5).stdout
        except Exception:
            return None
        # 輸出會依系統語言變化，所以只抓數字：「... 1: Audio [...], ... 0: USB Audio ...」
        m = re.search(r"(\d+)\s*:\s*\S+.*?\[.*?\].*?(\d+)\s*:", out)
        if m:
            return f"plughw:{m.group(1)},{m.group(2)}"
        m = re.search(r"card\s+(\d+).*?device\s+(\d+)", out, re.I)
        return f"plughw:{m.group(1)},{m.group(2)}" if m else None

    def _worker(self):
        """背景：開 arecord 子行程，把 PCM 餵給 Vosk，辨識到指令就放進佇列"""
        cmd = ["arecord", "-D", self.device, "-f", "S16_LE",
               "-r", str(self.SAMPLE_RATE), "-c", "1", "-t", "raw", "-q"]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[語音] 無法啟動 arecord: {e}")
            self.available = False
            return

        while self._running:
            data = self._proc.stdout.read(self.BLOCK)
            if not data:
                break
            try:
                if self._rec.AcceptWaveform(data):
                    text = json.loads(self._rec.Result()).get("text", "")
                    self._handle(text)
            except Exception:
                continue        # 單次辨識失敗不該讓整個語音執行緒死掉

    def _handle(self, raw_text):
        """把辨識文字轉成標準指令。空白要去掉 —— 中文模型輸出是「无 人 机 起 飞」。"""
        text = raw_text.replace(" ", "").strip().lower()
        if not text:
            return
        self.last_text = text
        self.last_text_time = time.time()

        # 找出喚醒詞，同時記下它後面剩下什麼 (第二層要用)
        tail = None
        for w in sorted(self.WAKE_WORDS, key=len, reverse=True):
            i = text.find(w)
            if i >= 0:
                tail = text[i + len(w):]
                break
        if tail is None:
            return                      # 沒喚醒詞：當成閒聊，不執行

        # 第一層：安全詞。整句任何位置出現都算數。
        for std, keys in self.INTENT_MAP.items():
            if any(k in text for k in keys):
                self._emit(std, text)
                return

        # 第二層：噪音容忍詞。只有在「句子很短」而且「緊接在喚醒詞後面」時才算，
        # 否則日常對話裡的同名詞會誤觸發 (理由見 TOLERANCE_MAP 的說明)。
        for std, keys in self.TOLERANCE_MAP.items():
            if any(tail.startswith(k) and len(tail) <= len(k) + self.TOLERANCE_SLACK
                   for k in keys):
                self._emit(std, text, tolerated=True)
                return

    def _emit(self, std, text, tolerated=False):
        """送出指令，並擋掉冷卻時間內的重複觸發"""
        now = time.time()
        if std == self._last_cmd and now - self._last_cmd_time < self.COOLDOWN_SEC:
            return
        self._last_cmd, self._last_cmd_time = std, now
        self.queue.put(std)
        print(f"[語音] 「{text}」-> {std}" + ("  (噪音容忍)" if tolerated else ""))

    # ==========================================
    # 對外
    # ==========================================
    def get_command(self) -> str:
        """主迴圈每格呼叫。沒有新指令就回空字串，絕不阻塞。"""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return ""

    def recent_text(self, within=2.0) -> str:
        """最近聽到的原始文字 (供畫面顯示，讓飛手知道麥克風有在收音)"""
        if self.available and time.time() - self.last_text_time < within:
            return self.last_text
        return ""

    def teardown(self):
        self._running = False
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)
