import os
import time
import json  # 新增：用於輸出 JSON
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
        #  意圖分組映射表 (Intent Grouping)
        # ==========================================
        self.intent_map = {
            "起飛": [
                "tello please take off", "泰羅請起飛", "無人機起飛"
            ],
            "降落": [
                "tello please land", "泰羅請降落", "安全降落"
            ],
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

        # 將意圖表反轉為「扁平查詢表」
        self.phrase_to_cmd = {}
        for standard_cmd, phrases in self.intent_map.items():
            for phrase in phrases:
                self.phrase_to_cmd[phrase] = standard_cmd
        
        try:
            # 直接指定 Windows 系統底層的 DLL 絕對路徑
            dll_path_64 = r"C:\Windows/Microsoft.NET/Framework64/v4.0.30319/WPF/System.Speech.dll"
            dll_path_32 = r"C:\Windows/Microsoft.NET/Framework/v4.0.30319/WPF/System.Speech.dll"
            
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
            
            # 初始化辨識引擎
            self.recognizer = SpeechRecognitionEngine()
            
            choices = Choices()
            for phrase in self.phrase_to_cmd.keys():
                choices.Add(phrase)
            
            grammar = Grammar(GrammarBuilder(choices))
            self.recognizer.LoadGrammar(grammar)
            self.recognizer.SetInputToDefaultAudioDevice()
            
            def on_speech_recognized(sender, e):
                # ==========================================
                # 🔥 軟體抗噪處理邏輯與 JSON 儲存
                # ==========================================
                if e.Result.Confidence > 0.4:
                    current_time = time.time()
                    
                    # 防連發冷卻 (Debounce)
                    if current_time - self.last_trigger_time > 1.0:
                        recognized_text = e.Result.Text
                        
                        if recognized_text in self.phrase_to_cmd:
                            self.current_command = self.phrase_to_cmd[recognized_text]
                            self.last_trigger_time = current_time
                            
                            # 1. 終端機印出除錯訊息
                            print(f"🎤 [語音] 聽到: '{recognized_text}' -> 轉換為意圖: '{self.current_command}' (信心度: {e.Result.Confidence:.2f})")
                            
                            # 2. 打包成 JSON 字典
                            log_data = {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "raw_text": recognized_text,
                                "intent": self.current_command,
                                "confidence": round(e.Result.Confidence, 2)
                            }
                            
                            # 3. 附加寫入 (Append) 檔案中，確保不覆蓋舊資料
                            # 使用 JSON Lines (.jsonl) 格式，每一行都是一個獨立的 JSON 物件
                            with open("voice_test_log.json", "a", encoding="utf-8") as f:
                                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                    
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


# ==========================================
# 獨立測試區塊 (只會在直接執行此檔案時運行)
# ==========================================
if __name__ == "__main__":
    print("========================================")
    print("🚀 進入語音模組獨立測試模式 (按 Ctrl+C 結束)")
    print("檔案將儲存至: voice_test_log.json")
    print("========================================")
    
    # 建立語音控制器物件
    vc = VoiceController()
    
    try:
        # 建立一個無限迴圈，讓主程式保持運行以接收麥克風事件
        while vc.is_running:
            # 可以模擬主程式定期向它索取指令的行為 (雖然印出動作已在 Callback 中完成)
            cmd = vc.get_command()
            time.sleep(0.1) 
    except KeyboardInterrupt:
        print("\n🛑 測試手動結束。")
    finally:
        vc.teardown()