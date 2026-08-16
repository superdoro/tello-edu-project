from core.tello_app import TelloApp

if __name__ == '__main__':
    print("📍 [檢查點 1] 啟動 Tello 控制系統...")
    
    try:
        print("📍 [檢查點 2] 準備初始化 TelloApp...")
        app = TelloApp()
        
        print("📍 [檢查點 3] TelloApp 初始化成功，準備連線與執行主迴圈...")
        app.run()
        
        print("📍 [檢查點 4] 系統已完全關閉。")
        
    except Exception as e:
        print(f"❌ 發生 Python 錯誤: {e}")
        import traceback
        traceback.print_exc()