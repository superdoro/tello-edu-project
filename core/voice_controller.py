import os
import time
import pythonnet

# ==========================================
# 1. 強制指定使用 Windows 傳統的 .NET Framework
# ==========================================
try:
    pythonnet.load("netfx")
except Exception:
    pass

import clr 

class VoiceController:
    def __init__(self):
        self.current_command = ""
        self.is_running = True
        self.last_trigger_time = 0  # 用於抗噪防連發機制
        
        # ==========================================
        # 🔥 意圖分組映射表 (Intent Grouping)
        # ==========================================
        # 這種結構具備極高的擴充性！
        # Key: 系統標準指令 (會傳給 manual_control.py)
        # Value: 觸發該指令的所有可能口語/同義句/多國語言清單
        self.intent_map = {
            # "起飛": [
            #     "tello please take off", "泰羅請起飛", "無人機起飛"
            # ],
            # "降落": [
            #     "tello please land", "泰羅請降落", "安全降落"
            # ],
            "前進": [
                "tello go forward", "泰羅向前飛", "往前飛"
            ],
            "後退": [
                "tello go backward", "泰羅向後退", "往後退"
            ],
            "向左": [
                "tello go left", "泰羅向左飛", "往左飛"
            ],
            "向右": [
                "tello go right", "泰羅向右飛", "往右飛"
            ],
            "上升": [
                "tello go up", "泰羅向上升", "往上升"
            ],
            "下降": [
                "tello go down", "泰羅向下降", "往下降"
            ],
            "轉向": [
                "tello turn around", "泰羅請轉向", "轉個身"
            ]
        }

        # 系統初始化時，自動將意圖表反轉為「扁平查詢表」，提升比對速度 (O(1))
        # 結果會像這樣：{"泰羅向前飛": "前進", "往前飛": "前進", ...}
        self.phrase_to_cmd = {}
        for standard_cmd, phrases in self.intent_map.items():
            for phrase in phrases:
                self.phrase_to_cmd[phrase] = standard_cmd
        
        try:
            # ==========================================
            # 直接指定 Windows 系統底層的 DLL 絕對路徑
            # ==========================================
            dll_path_64 = r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\WPF\System.Speech.dll"
            dll_path_32 = r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\WPF\System.Speech.dll"
            
            if os.path.exists(dll_path_64):
                clr.AddReference(dll_path_64)
                print("   -> 成功載入 64 位元 System.Speech.dll")
            elif os.path.exists(dll_path_32):
                clr.AddReference(dll_path_32)
                print("   -> 成功載入 32 位元 System.Speech.dll")
            else:
                clr.AddReference("System.Speech")
                
            from System.Speech.Recognition import (
                SpeechRecognitionEngine, 
                Choices, 
                GrammarBuilder, 
                Grammar, 
                RecognizeMode
            )
            
            # 3. 初始化辨識引擎
            self.recognizer = SpeechRecognitionEngine()
            
            # 4. 將展開後的「所有語音長句」加入辨識清單
            choices = Choices()
            for phrase in self.phrase_to_cmd.keys():
                choices.Add(phrase)
            
            grammar = Grammar(GrammarBuilder(choices))
            self.recognizer.LoadGrammar(grammar)
            
            # 5. 設定麥克風並啟動
            self.recognizer.SetInputToDefaultAudioDevice()
            
            def on_speech_recognized(sender, e):
                # ==========================================
                # 🔥 軟體抗噪處理邏輯
                # ==========================================
                if e.Result.Confidence > 0.4:
                    current_time = time.time()
                    
                    # 2. 防連發冷卻 (Debounce)：限制每 1 秒內只能接收一次指令
                    if current_time - self.last_trigger_time > 1.0:
                        recognized_text = e.Result.Text
                        
                        # 3. 透過扁平查詢表，將長句映射為標準指令
                        if recognized_text in self.phrase_to_cmd:
                            # 這裡取出的永遠會是乾淨的 "前進"、"向左" 等標準詞彙
                            self.current_command = self.phrase_to_cmd[recognized_text]
                            self.last_trigger_time = current_time
                            
                            # 印出狀態方便除錯
                            print(f"🎤 [語音] 聽到: '{recognized_text}' -> 轉換為意圖: '{self.current_command}' (信心度: {e.Result.Confidence:.2f})")
                    
            self.recognizer.SpeechRecognized += on_speech_recognized
            self.recognizer.RecognizeAsync(RecognizeMode.Multiple)
            
            print("✅ [系統訊息] 意圖導向語音監聽已啟動！")
            
        except Exception as e:
            print(f"❌ [錯誤] Windows 語音初始化失敗: {e}")
            self.is_running = False

    def get_command(self) -> str:
        cmd = self.current_command
        self.current_command = "" 
        return cmd

    def teardown(self):
        self.is_running = False
        if hasattr(self, 'recognizer'):
            self.recognizer.RecognizeAsyncCancel()
            self.recognizer.Dispose()