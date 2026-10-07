# Tello EDU 自主飛行專案

以 DJI Tello EDU 為平台的視覺自主飛行實驗專案。電腦端負責所有運算 (YOLO 偵測、單目深度、語音辨識)，
無人機只回傳影像、接收 rc 速度指令。除了鍵盤 / 語音手動操作，還有手勢跟隨、深度漫遊避障、
氣球獵手、無人機跟隨、衛生紙條衝撞等自動模式，用 `Z` 鍵依序切換。

## 飛行模式

| 模式 | 做什麼 | 視覺模組 | 使用的權重 |
|---|---|---|---|
| MANUAL CONTROL | 鍵盤 + 離線語音指令手動操作 | 無 | — |
| HAND TRACKER | 跟隨人 (胸腔鎖定 / 手部指揮)，可環繞 | `vision/pose_tracker.py` | `yolo_pose_collect/yolo26n-pose.pt` |
| FLUID EXPLORER | 單目深度漫遊避障 | `vision/fluid_explorer_vision.py` | `yolo_depth_collect/yolo26n-depth.pt` |
| BALLOON HUNT | 依編號搜尋並攻擊氣球 (ArUco 標籤補償) | `vision/balloon_detector_with_aruco.py` | `yolo_ballon_collect/best.pt` + 深度 |
| DRONE FOLLOW | 跟隨另一台無人機 | `vision/drone_detector.py` | `yolo_drone_collect/epoch30.pt` |
| TISSUE CHARGE | 漫遊找天花板垂下的衛生紙條並衝撞 | `vision/tissue_detector.py` | `yolo_tissue_collect/best_seg.pt` + 深度 |

權重路徑都在 `model/yolo26/runs/detect/` 底下，已經放在 repo 裡，clone 下來就能用 (見[模型權重](#模型權重))。

## 環境需求

- **無人機**：DJI Tello EDU (一般版 Tello 也能飛，但沒有測過)
- **電腦**：Linux (開發環境為 Ubuntu)，建議有 NVIDIA 顯示卡
  - 開發機：RTX 3050 Ti Laptop、CUDA 13.0、torch 2.14.0+cu130
  - 沒有 GPU 也能執行，但深度推論太慢，漫遊類的模式無法安全飛行
- **Python**：3.10 (conda 環境名稱 `UAV`)
- **選用**：麥克風 + `arecord` (alsa-utils，Ubuntu 通常已內建) 才能用語音指令

## 安裝

```bash
git clone git@github.com:superdoro/tello-edu-project.git
cd tello-edu-project

# 1. 建立 conda 環境 (含 CUDA 13.0 版的 torch)
conda env create -f environment.yml
conda activate UAV
```

顯示卡驅動不支援 CUDA 13 的話，把 `environment.yml` 裡的 `cu130` 改成對應版本 (例如 `cu126`)，
並把 `requirements.txt` 的 torch / torchvision 改成該版本有提供的版號。
想手動安裝的話，步驟寫在 `requirements.txt` 的檔頭。

```bash
# 2. (選用) 離線語音：下載 Vosk 中文小模型 (42MB，解開後 66MB，沒有放進 repo)
curl -LO https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip vosk-model-small-cn-0.22.zip -d model/vosk/

# 3. (選用) 身分辨識：在有網路的環境預先下載人臉模型權重
python utils/identity_warmup.py
```

第 3 步一定要在連 Tello 之前做：飛行時電腦連的是 Tello 的 Wi-Fi，沒有外網，
臨時下載權重會卡住主迴圈，Tello 約 15 秒收不到指令就會自動降落。

## 執行

1. 開啟 Tello 電源，電腦連上 `TELLO-XXXXXX` Wi-Fi
2. 啟動程式：

   ```bash
   conda activate UAV
   python main.py
   ```

3. 程式會連線、開啟影像串流，視窗出現畫面後按 `T` 起飛

## 操作

| 按鍵 | 動作 |
|---|---|
| `T` / `L` | 起飛 / 降落 |
| `Q` | 降落並結束程式 |
| `↑` `↓` `←` `→` | 前進 / 後退 / 左移 / 右移 |
| `W` / `S` | 上升 / 下降 |
| `A` / `D` | 機頭左轉 / 右轉 |
| `Z` | 切換飛行模式 (依上表順序循環) |
| `V` | 開始 / 停止錄影 (見[錄影與飛行紀錄](#錄影與飛行紀錄)) |
| `F` | 模式專用：HAND TRACKER 切換「胸腔鎖定 / 手部指揮」；TISSUE CHARGE 顯示被擋掉的偵測 |
| `R` | 模式專用：HAND TRACKER 循環鎖定畫面中的人；TISSUE CHARGE 放棄目前鎖定的紙條 |
| `O` | HAND TRACKER 環繞：關閉 → 逆時針 → 順時針 → 關閉 |

- 任何自動模式中按方向鍵都會立刻接管，放開後交還自動控制。
- HAND TRACKER 環繞時會自動拉近距離；切換飛行模式時會自動關閉環繞。

### 語音指令

離線辨識 (Vosk 中文小模型)，飛行時不需要網路。說「喚醒詞 + 指令」，例如「無人機起飛」。

**喚醒詞** (必須先講，避免閒聊誤觸發)

```
無人機   <- 建議用這個，實測最穩
小白 / 悟空 / 飛機 / 阿飛 / tello
```

⚠️ 舊版的「泰羅」實測會被聽成「嗨哦」—— 這個模型認不出自創的名字，請改用上面的喚醒詞。

**指令** (括號內是實測最抗噪的說法，噪音大時請優先用它)

```
起飛    起飛 / take off
降落    降落 / 著陸 / land
前進    前進 / 往前 / 向前
後退   （往後）/ 後退 / 向後       <- 「後退」會被聽成「后错误」，請說「往後」
向左   （左轉）/ 往左 / 左邊       <- 「往左」在噪音下會變成「王所」
向右   （往右）/ 右轉 / 右邊
上升   （升高）/ 上升              <- 「往上」會被聽成「网上」，不要用
下降    下降 / 降低 / 往下
轉向   （轉身）/ 轉向 / 旋轉
停止    停止 / 停下
切換    切換模式 / switch
```

**抗噪實測** (`python utils/voice_bench.py`，混入合成的四旋翼噪音)

```
訊噪比      正確率    認錯指令   誤觸發
乾淨        100%       0        0
20dB       100%       0        0
15dB       100%       0        0
10dB       100%       0        0
 5dB        69%       0        0
 0dB         0%       0        0
```

所有噪音等級下都**沒有認錯指令、也沒有誤觸發** —— 失效方式一律是「沒反應」，
不會把「左」聽成「右」。噪音太大時叫不動，但不會亂飛。

起飛、降落、切換模式會直接觸發對應動作；方向指令交給 ManualControl，維持約 5 秒後自動停止。
任何鍵盤輸入都會立刻蓋過語音。畫面左下角會顯示最近聽到的話，方便確認麥克風有在收音。
沒有麥克風、或沒下載模型時語音會自動停用，其餘功能不受影響。

## TISSUE CHARGE (衛生紙條衝撞) 使用說明

流程：`SEARCH` 漫遊找紙條 → `TRACK` 對準並逼近 → `COAST` 紙條貼到眼前看不見時再滑行一小段 → `BACKOFF` 退開 → 回到漫遊。

- **高度鎖定**：這個模式不會自動改變高度，只有按 `W` / `S` 才會升降。
  請先手動飛到紙條尖端附近的高度，再按 `Z` 切進這個模式。
  Tello 回報的高度讀值會隨起飛點、下方的桌櫃改變，不可靠，所以程式完全不使用它。
- **紅點 = 前進路線**：畫面上的紅點在畫面高度 75% 的位置 (Tello 鏡頭略朝下，往前飛時機身前傾，
  真正飛過去的那一列在畫面中心上方)。紙條有橫過紅點那一列才會往前衝；
  整條在上方或下方超過 1 秒就放棄換下一條 —— 一直看到「紙條整條在前進路線上方」就是高度太低，用 `W` 調高。
- **白牆判定**：純白、沒有紋理的柱子或牆面，單目深度模型會誤判成很遠。
  畫面中「從頂端一路白到底端」的區域會被當成近距離障礙 (畫面上標 `WHITE WALL`)，漫遊會轉開。
- `F` 顯示被擋掉的偵測 (灰色 = 信心不足、橘色 = 形狀不符)，現場調門檻時用；
  `R` 放棄目前鎖定的紙條並暫時列入黑名單。

## 錄影與飛行紀錄

按 `V` 開始 / 停止錄影，`recordings/` 底下會產生三個同名檔案 (資料夾不進版控)：

| 檔案 | 內容 |
|---|---|
| `<時間>_<模式>_raw.mp4` | 乾淨畫面 (720x480，與 YOLO 推論看到的一致)，可拿來標註訓練資料 |
| `<時間>_<模式>_annotated.mp4` | 與螢幕上看到的一樣 (偵測框、深度、狀態)，用來找問題 |
| `<時間>_<模式>_log.csv` | 每一格的模式、狀態、送出的 `lr/fb/ud/yv`、電量、高度讀值，以及各模式的診斷欄位 |

影片以固定 30fps、依實際時間排列，所以影格編號 ÷ 30 就是 CSV 的 `elapsed`，兩邊可以直接對照。
TISSUE CHARGE 的診斷欄位包含 `tw` (紙條框寬)、`bottom_y`、`cy`、`err_x/err_y` (十字與紅點的差距)、
`depth_ahead`、`blank` (白牆判定) 等，欄位清單見 `utils/video_recorder.py` 的 `LOG_FIELDS`。

## 模型權重

都放在 `model/yolo26/runs/detect/`：

| 權重 | 大小 | 用途 |
|---|---|---|
| `yolo_tissue_collect/best_seg.pt` | 22 MB | TISSUE CHARGE 紙條偵測 (分割) |
| `yolo_depth_collect/yolo26n-depth.pt` | 12 MB | 單目深度 (FLUID EXPLORER、BALLOON HUNT、TISSUE CHARGE) |
| `yolo_pose_collect/yolo26n-pose.pt` | 7.5 MB | HAND TRACKER 人體姿態 |
| `yolo_ballon_collect/best.pt` | 5.1 MB | BALLOON HUNT 氣球偵測 |
| `yolo_drone_collect/epoch30.pt` | 57 MB | DRONE FOLLOW 無人機偵測 |

repo 裡另外還有幾個目前程式沒有載入的權重 (`yolo_drone_collect/best.pt`、`yolo_pose_collect/yolo26s-pose.pt`、
`yolo_ballon_collect/best2.pt`、`yolo_tissue_collect/best.pt`、`yolo_digit_collect/best.pt`)，以及訓練用的基底 `model/yolo26/yolo26n.pt`。
訓練中途的 `epoch*.pt` checkpoint 不進版控 (`.gitignore`)，只有 DRONE FOLLOW 正在用的 `epoch30.pt` 例外。

重新訓練：`model/yolo26/train_yoly.py`。

## 測試與離線工具

不需要無人機就能跑：

```bash
# TISSUE CHARGE 狀態機、高度鎖定、白牆判定、指令看門狗的行為測試
python tests/test_tissue_charge.py

# 用錄影檔重播紙條偵測與追蹤邏輯，調門檻不必起飛
python tests/replay_tissue.py dump  recordings/<時間>_TISSUE_CHARGE_raw.mp4
python tests/replay_tissue.py sweep <dump.json> --fp 60-67 --true 8-20,75-85
python tests/replay_tissue.py fsm   recordings/<時間>_TISSUE_CHARGE_raw.mp4 --log recordings/<時間>_TISSUE_CHARGE_log.csv
```

| 工具 | 用途 |
|---|---|
| `utils/voice_bench.py` | 語音指令抗噪測試 (需要 `pip install gTTS av`，首次需要網路) |
| `utils/voice_tolerance.py` | 產生語音指令的噪音容忍表 |
| `utils/identity_bench.py` | 身分辨識的離線測試 |
| `utils/generate_aruco.py` | 產生 BALLOON HUNT 用的 ArUco 標籤 (A4 排版，成品是 `utils/all_markers_A4.png`) |

## 安全機制

- **指令看門狗** (`core/drone_controller.py`)：Tello 會一直執行最後收到的 rc 指令。
  主迴圈超過 0.3 秒沒送出新指令，背景執行緒會自動送一次懸停。
- **迴圈過慢診斷** (`core/tello_app.py`)：單一迴圈超過 0.3 秒時，終端機會印出 `[迴圈過慢]`
  與各階段 (輸入/語音、視覺、決策、顯示) 的耗時。
- **黑區 / 白牆判定** (`utils/dark_region.py`、`utils/blank_region.py`)：純黑或純白、沒有紋理的表面，
  單目深度模型讀不出距離，判定到就當成近距離障礙。
- 電量低時 Tello 的反應不穩定，請充飽電再飛；飛行時注意不要站在無人機的前進路線上。

## 專案結構

```
tello-edu-project/
├── main.py                       # 程式進入點 (啟動 TelloApp)
├── environment.yml               # conda 環境
├── requirements.txt              # pip 相依套件 (含安裝說明)
│
├── core/                         # 核心：主迴圈、硬體、輸入
│   ├── tello_app.py              # 主迴圈與模式清單 (TelloApp)
│   ├── drone_controller.py       # 封裝 djitellopy，含指令看門狗
│   ├── ui_controller.py          # Pygame 鍵盤輸入與畫面顯示
│   └── voice_controller.py       # Vosk 離線語音指令
│
├── vision/                       # 視覺處理 (VisionProcessor 實作)
│   ├── base.py                   # VisionProcessor 介面與 VisionData
│   ├── pose_tracker.py           # HAND TRACKER
│   ├── fluid_explorer_vision.py  # FLUID EXPLORER
│   ├── balloon_detector_with_aruco.py  # BALLOON HUNT
│   ├── drone_detector.py         # DRONE FOLLOW
│   ├── tissue_detector.py        # TISSUE CHARGE
│   ├── identity/                 # 身分辨識子系統 (人臉 + 體態)
│   └── unused module/            # 舊實驗模組，目前沒有使用
│
├── behaviors/                    # 飛行策略 (FlightBehavior 實作)
│   ├── base.py                   # FlightBehavior 介面
│   ├── manual_control.py         # MANUAL CONTROL (鍵盤 + 語音)
│   ├── pose_follow.py            # HAND TRACKER
│   ├── fluid_explore.py          # FLUID EXPLORER (TISSUE CHARGE 的漫遊也用它)
│   ├── balloon_hunt.py           # BALLOON HUNT
│   ├── drone_follow.py           # DRONE FOLLOW
│   ├── tissue_charge.py          # TISSUE CHARGE
│   └── unused module/            # 舊實驗模組，目前沒有使用
│
├── utils/                        # 共用工具與離線測試腳本
│   ├── pid_controller.py
│   ├── video_recorder.py         # V 鍵錄影
│   ├── dark_region.py            # 黑區判定
│   ├── blank_region.py           # 白牆判定
│   ├── white_strip.py            # 紙條白色驗證
│   └── ...                       # 語音 / 身分辨識測試、ArUco 產生器
│
├── model/
│   ├── yolo26/                   # YOLO 權重與訓練腳本
│   └── vosk/                     # 語音模型 (自行下載，不進版控)
│
├── tests/                        # 不需要無人機的測試與重播工具
├── recordings/                   # V 鍵錄影 (不進版控)
└── data/identity/                # 身分辨識特徵庫 (個人資料，不進版控)
```

## 架構設計

主迴圈 `TelloApp` 只負責調度，飛行邏輯拆成四層，新增功能時只需要新增對應的類別：

1. **硬體抽象層 `DroneController`**：專門負責與無人機通訊 (封裝 djitellopy)。換一台無人機只需要改這個類別。
2. **輸入與介面層 `UIController`**：處理鍵盤監聽與畫面顯示；改用搖桿或完整 GUI 只需要改這裡。語音由 `VoiceController` 補上鍵盤沒按的部分。
3. **視覺處理 `VisionProcessor`**：接收一張畫面，回傳 `VisionData` (目標位置、深度等)。
4. **飛行策略 `FlightBehavior`**：根據使用者輸入與視覺結果，算出下一步的 `(lr, fb, ud, yv)`。

每個飛行模式是一組 `(FlightBehavior, VisionProcessor)`，登記在 `core/tello_app.py` 的 `self.modes` 清單裡。

```mermaid
classDiagram
    class TelloApp {
        - drone: DroneController
        - ui: UIController
        - recorder: VideoRecorder
        - modes: list
        + run()
        + toggle_mode()
    }
    class DroneController {
        + connect()
        + takeoff()
        + land()
        + get_video_frame()
        + send_movement(lr, fb, ud, yv)
    }
    class UIController {
        - voice: VoiceController
        + get_input() UserInput
        + display_frame(frame)
    }
    class VisionProcessor {
        <<Interface>>
        + process_frame(frame) VisionData
    }
    class FlightBehavior {
        <<Interface>>
        + calculate_command(user_input, vision_data) tuple
    }

    TelloApp *-- DroneController
    TelloApp *-- UIController
    TelloApp *-- VideoRecorder
    TelloApp o-- VisionProcessor : 目前模式
    TelloApp o-- FlightBehavior : 目前模式
    UIController *-- VoiceController

    VisionProcessor <|.. BodyPoseTracker
    VisionProcessor <|.. DepthExplorerVision
    VisionProcessor <|.. BalloonDetector
    VisionProcessor <|.. DroneDetector
    VisionProcessor <|.. TissueDetector

    FlightBehavior <|.. ManualControl
    FlightBehavior <|.. BodyFollowControl
    FlightBehavior <|.. FluidExploreControl
    FlightBehavior <|.. BalloonHuntControl
    FlightBehavior <|.. DroneFollowControl
    FlightBehavior <|.. TissueChargeControl
    TissueChargeControl *-- FluidExploreControl : 漫遊
```

## 待辦與參考資源

待辦：

- [ ] 身體/手部追蹤需要鎖定目標的功能
    * [ ] 人臉辨識
    * [ ] 切換鎖定
- [ ] 空戰功能?
- [ ] 語音系統

參考資源：

* [An Autonomous Tracking and Landing Method for Unmanned Aerial Vehicles Based on Visual Navigation](https://www.mdpi.com/2504-446X/7/12/703) (可用於測試無人機自動駕駛的標籤，具備順序性，且容易辨識)
* 自動導航可嘗試 [光流](https://ieeexplore.ieee.org/document/10964683)
* 以下缺乏設備，暫時不考慮：EGO-Planner、GAAS (Generic Autonomous Aviation System)、Vision-Based Drone Obstacle Avoidance by DRL
