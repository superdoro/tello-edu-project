import os
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
        
        try:
            # ==========================================
            # 2. 終極解法：直接指定 Windows 系統底層的 DLL 絕對路徑
            # ==========================================
            # 優先尋找 64 位元路徑 (Framework64)
            dll_path_64 = r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\WPF\System.Speech.dll"
            # 備用尋找 32 位元路徑 (Framework)
            dll_path_32 = r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\WPF\System.Speech.dll"
            
            if os.path.exists(dll_path_64):
                clr.AddReference(dll_path_64)
                print("   -> 成功載入 64 位元 System.Speech.dll")
            elif os.path.exists(dll_path_32):
                clr.AddReference(dll_path_32)
                print("   -> 成功載入 32 位元 System.Speech.dll")
            else:
                # 如果都找不到，嘗試最後的預設呼叫
                clr.AddReference("System.Speech")
                
            # 載入模組
            from System.Speech.Recognition import (
                SpeechRecognitionEngine, 
                Choices, 
                GrammarBuilder, 
                Grammar, 
                RecognizeMode
            )
            
            # 3. 初始化辨識引擎
            self.recognizer = SpeechRecognitionEngine()
            
            # 4. 建立嚴格的關鍵字清單
            choices = Choices()
            commands = [
                "降落", 
                "前進", "後退", "向左", "向右", "上升", "下降", "轉向",
            ]
            for cmd in commands:
                choices.Add(cmd)
            
            grammar = Grammar(GrammarBuilder(choices))
            self.recognizer.LoadGrammar(grammar)
            
            # 5. 設定麥克風並啟動
            self.recognizer.SetInputToDefaultAudioDevice()
            
            def on_speech_recognized(sender, e):
                # 信心度大於 0.7 才視為有效指令
                if e.Result.Confidence > 0.7:
                    self.current_command = e.Result.Text
                    
            self.recognizer.SpeechRecognized += on_speech_recognized
            self.recognizer.RecognizeAsync(RecognizeMode.Multiple)
            
            print("✅ [系統訊息] Windows 內建超高準確度離線語音監聽已啟動！")
            
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