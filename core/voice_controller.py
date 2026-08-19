import os
import time
import threading
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
        # 長音節指令對映表 (Mapping)
        # ==========================================
        # 提示：若您的 Windows 系統預設為中文，對純英文的辨識度可能會稍弱。
        # 因此我們同時放入「英文長句」與「中文長句」，讓系統都能聽懂，並統一轉換為短指令。
        self.command_map = {
            "tello please take off": "起飛",
            "泰羅請起飛": "起飛",
            
            "tello please land": "降落",
            "泰羅請降落": "降落",
            
            "tello go forward": "前進",
            "泰羅向前飛": "前進",
            
            "tello go backward": "後退",
            "泰羅向後退": "後退",
            
            "tello go left": "向左",
            "泰羅向左飛": "向左",
            
            "tello go right": "向右",
            "泰羅向右飛": "向右",
            
            "tello go up": "上升",
            "泰羅向上升": "上升",
            
            "tello go down": "下降",
            "泰羅向下降": "下降",
            
            "tello turn around": "轉向",
            "泰羅請轉向": "轉向"
        }
        
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
            
            # 4. 將對映表中的「長音節句子」加入辨識清單
            choices = Choices()
            for long_cmd in self.command_map.keys():
                choices.Add(long_cmd)
            
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
                    if current_time - self.last_trigger_time > 1:
                        recognized_text = e.Result.Text
                        
                        # 3. 將聽到的長句子，轉換為系統期待的短指令
                        if recognized_text in self.command_map:
                            self.current_command = self.command_map[recognized_text]
                            self.last_trigger_time = current_time
                            
                            # 印出狀態方便除錯
                            print(f"🎤 [語音] 聽到: '{recognized_text}' -> 轉換為: '{self.current_command}' (信心度: {e.Result.Confidence:.2f})")
                    
            self.recognizer.SpeechRecognized += on_speech_recognized
            self.recognizer.RecognizeAsync(RecognizeMode.Multiple)
            
            print("✅ [系統訊息] 長音節抗噪語音監聽已啟動！")
            
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