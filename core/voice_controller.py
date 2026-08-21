import os
import time
import json  
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
        self.last_trigger_time = 0  
        self.lock = threading.Lock() 
        
        # ==========================================
        # 🔥 喚醒詞與意圖狀態設定
        # ==========================================
        self.wake_words = ["tello", "泰羅", "無人機"]
        
        # 行動意圖分組映射表 (可以自由混搭帶有喚醒詞或不帶喚醒詞的句子)
        self.intent_map = {
            "起飛": ["tello please take off", "泰羅請起飛", "無人機起飛", "起飛"],
            "降落": ["tello please land", "泰羅請降落", "安全降落", "降落"],
            "前進": ["tello go forward", "泰羅向前飛", "往前飛", "前進"],
            "後退": ["tello go backward", "泰羅向後退", "往後退", "後退"],
            "向左": ["tello go left", "泰羅向左飛", "往左飛", "向左"],
            "向右": ["tello go right", "泰羅向右飛", "往右飛", "向右"],
            "上升": ["tello go up", "泰羅向上升", "往上升", "上升"],
            "下降": ["tello go down", "泰羅向下降", "往下降", "下降"],
            "轉向": ["tello turn around", "泰羅請轉向", "轉個身", "轉向"]
        }

        # 展開為扁平查詢表 {"泰羅向前飛": "前進", ...}
        self.phrase_to_cmd = {}
        for standard_cmd, phrases in self.intent_map.items():
            for phrase in phrases:
                self.phrase_to_cmd[phrase.lower()] = standard_cmd
        
        try:
            # 載入 Windows 語音底層 DLL
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
            from System.Globalization import CultureInfo

            # 設定語系
            try:
                culture = CultureInfo("zh-TW")
                self.recognizer = SpeechRecognitionEngine(culture)
                print("ℹ️ [系統訊息] 語音引擎已設定為 [繁體中文 (zh-TW)]")
            except Exception:
                self.recognizer = SpeechRecognitionEngine()
            
            # 將所有詞句加入辨識清單
            choices = Choices()
            for phrase in self.phrase_to_cmd.keys():
                choices.Add(phrase)
            
            gb = GrammarBuilder(choices)
            gb.Culture = self.recognizer.RecognizerInfo.Culture
            
            grammar = Grammar(gb)
            self.recognizer.LoadGrammar(grammar)
            self.recognizer.SetInputToDefaultAudioDevice()
            
            def on_speech_recognized(sender, e):
                with self.lock:
                    current_time = time.time()
                    
                    # 防連發冷卻 (Debounce)
                    if current_time - self.last_trigger_time < 1.0:
                        return

                    # ==========================================
                    # 🔥 One-shot 喚醒與意圖分析核心邏輯
                    # ==========================================
                    valid_match_found = False
                    
                    # e.Result.Alternates 包含了系統猜測的「所有可能結果」，並已自動由信心度高到低排序
                    for alt in e.Result.Alternates:
                        conf = alt.Confidence
                        text = alt.Text.lower()

                        print(f"[語音陣列] 聽到: '{text}' -> 意圖: '{self.current_command}' (信心度: {conf:.2f})")

                        if conf < 0.4:
                            break # 信心度太低

                        # 檢查這個句子是否包含任何一個「喚醒詞」
                        has_wake_word = any(w in text for w in self.wake_words)
                        
                        if has_wake_word and text in self.phrase_to_cmd:
                            # 找到了！因為 Alternates 是排序過的，所以這絕對是「符合條件且信心度最高」的結果
                            self.current_command = self.phrase_to_cmd[text]
                            self.last_trigger_time = current_time
                            valid_match_found = True
                            
                            print(f"✨ [指令鎖定] 聽到: '{text}' -> 意圖: '{self.current_command}' (信心度: {conf:.2f})")
                            
                            # 儲存 JSON Log
                            log_data = {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "raw_text": text,
                                "intent": self.current_command,
                                "confidence": round(conf, 2)
                            }
                            with open("voice_test_log.json", "a", encoding="utf-8") as f:
                                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                                f.flush()
                                
                            break # 找到最高信心的正確指令後，跳出迴圈

            self.recognizer.SpeechRecognized += on_speech_recognized
            self.recognizer.RecognizeAsync(RecognizeMode.Multiple)
            
            print("✅ [系統訊息] One-shot 連續指令語音監聽已啟動！")
            
        except Exception as e:
            print(f"❌ [錯誤] Windows 語音初始化失敗: {e}")
            self.is_running = False

    def get_command(self) -> str:
        with self.lock:
            cmd = self.current_command
            self.current_command = "" 
            return cmd

    def teardown(self):
        self.is_running = False
        if hasattr(self, 'recognizer'):
            self.recognizer.RecognizeAsyncCancel()
            self.recognizer.Dispose()


# ==========================================
# 獨立測試區塊
# ==========================================
if __name__ == "__main__":
    print("========================================")
    print("🚀 進入【One-shot 連續指令】語音模組測試 (按 Ctrl+C 結束)")
    print("💡 測試方式：請直接說出「泰羅向前飛」或「Tello please take off」")
    print("檔案將儲存至: voice_test_log.json")
    print("========================================")
    
    vc = VoiceController()
    
    try:
        while vc.is_running:
            cmd = vc.get_command()
            if cmd:
                print(f"🤖 [無人機主控] 收到安全指令，準備發送給飛控: 【{cmd}】\n")
            time.sleep(0.1) 
    except KeyboardInterrupt:
        print("\n🛑 測試手動結束。")
    finally:
        vc.teardown()