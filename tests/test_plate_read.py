"""
PLATE READER (車牌辨識) 的行為測試 —— 不需要無人機、不需要錄影，也不載入深度模型。

重點：什麼時候才算停穩、停穩前絕對不讀；OCR 不會卡住主迴圈；投票與正規化；
跟丟 / 漂離 / R 鍵之後狀態對不對；接近時各軸的方向對不對 (方向錯一次就撞上去了)。
最後用一個簡單的模擬世界跑閉迴路，確認從遠處一路走到讀出號碼。

用假時鐘驅動，所有跟時間有關的狀態轉換都能精確測到。

    python tests/test_plate_read.py
"""

import math
import os
import sys
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behaviors.plate_read import PlateReadControl
from vision.base import VisionData
from vision.plate_backends import (PlateDetectorBackend, PlateOCRBackend, ArucoPlateDetector,
                                   ArucoIdOCR, sharpness)
from vision.plate_detector import PlateVision
from vision.plate_reader import PlateReader, normalize_plate

NO_INPUT = SimpleNamespace(lr=0, fb=0, ud=0, yv=0, voice_command=None)
F_H, F_V = PlateVision.F_H, PlateVision.F_V
OUT_DIR = tempfile.mkdtemp(prefix="plate_test_")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def tick(self, dt=0.05):
        self.t += dt


class SeqOCR(PlateOCRBackend):
    """依序回傳預先排好的結果，用完之後都是讀不到"""

    def __init__(self, seq=()):
        self.seq = list(seq)
        self.calls = 0

    def read(self, crop):
        self.calls += 1
        return self.seq.pop(0) if self.seq else (None, 0.0)


class FakeDepth:
    """代替 DepthExplorerVision：四周開闊、深度可信"""

    def process_frame(self, frame):
        d = VisionData(is_detected=True, annotated_frame=frame.copy())
        d.depth_L = d.depth_C = d.depth_R = 999.0
        d.dark_ratio = d.dark_L = d.dark_C = d.dark_R = 0.0
        d.depth_unreliable = False
        return d


class FakeDetector(PlateDetectorBackend):
    def __init__(self, frames):
        """frames: 每一格要回傳的偵測結果 (list of list)；用完就重複最後一格"""
        self.frames = list(frames)
        self.seen = []

    def detect(self, frame):
        self.seen.append(frame)
        out = self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]
        return [dict(d) for d in out]


def make_plate(cx=360.0, cy=240.0, dist=70.0, hits=10, sharp=100.0):
    w, h = 32.0 * F_H / dist, 15.0 * F_V / dist
    return {"cx": cx, "cy": cy, "w": w, "h": h, "x1": cx - w / 2, "y1": cy - h / 2,
            "x2": cx + w / 2, "y2": cy + h / 2, "conf": 0.9, "dist_cm": dist, "hits": hits,
            "aspect": 2.13, "crop": np.full((40, 90, 3), 128, np.uint8), "sharp": sharp}


def plate_vision(**kw):
    d = VisionData(is_detected=True)
    d.plate = make_plate(**kw)
    d.depth_L = d.depth_C = d.depth_R = 999.0
    d.depth_unreliable = False
    return d


def blank_vision(depth_unreliable=False):
    d = VisionData(is_detected=False)
    d.plate = None
    d.depth_L = d.depth_C = d.depth_R = 999.0
    d.depth_unreliable = depth_unreliable
    d.dark_ratio = 0.5 if depth_unreliable else 0.0
    d.dark_L = d.dark_C = d.dark_R = d.dark_ratio
    return d


def make_behavior(clk, seq=(), reader=True):
    r = PlateReader(SeqOCR(seq), out_dir=OUT_DIR, synchronous=True) if reader else None
    return PlateReadControl(reader=r, now_fn=clk)


def engage(b, clk, **kw):
    """從 SEARCH 走正常路徑鎖定車牌 (會設好 engage_since)"""
    out = b.calculate_command(NO_INPUT, plate_vision(**kw))
    clk.tick()
    return out


def run(b, clk, vision_fn, seconds, dt=0.05):
    """跑 seconds 秒，vision_fn(i) 給第 i 格的 vision_data。回傳 [(時間, 指令, 狀態)]"""
    out = []
    n = int(round(seconds / dt))
    for i in range(n):
        cmd = b.calculate_command(NO_INPUT, vision_fn(i))
        out.append((clk.t, cmd, b.state))
        clk.tick(dt)
    return out


results = []


def check(name, cond, detail=""):
    results.append((name, cond, detail))
    print(f"{'PASS' if cond else '**FAIL**':10s} {name}" + (f"   {detail}" if detail else ""))


# ==========================================
# 飛行邏輯
# ==========================================
def t_manual_override():
    clk = Clock()
    b = make_behavior(clk)
    engage(b, clk)
    key = SimpleNamespace(lr=40, fb=0, ud=0, yv=0, voice_command=None)
    out = b.calculate_command(key, plate_vision())
    check("按方向鍵 -> 直接送飛手的指令", out == (40, 0, 0, 0), f"{out}")
    check("按方向鍵 -> 回到 SEARCH", b.state == "SEARCH", b.state)


def t_search_roams_altitude_locked():
    clk = Clock()
    b = make_behavior(clk)
    out = run(b, clk, lambda i: blank_vision(), 0.5)
    check("SEARCH 開闊 -> 委派漫遊往前飛",
          all(c[1] == b.explorer.FORWARD_SPEED for _, c, _ in out), f"{out[-1][1]}")
    out = run(b, clk, lambda i: blank_vision(depth_unreliable=True), 1.0)
    check("黑區脫困時漫遊在轉 (確實有委派)", all(c[3] != 0 for _, c, _ in out), f"{out[-1][1]}")
    check("SEARCH 高度鎖定 (黑區脫困的爬升被擋掉)", all(c[2] == 0 for _, c, _ in out),
          f"ud={[c[2] for _, c, _ in out][:5]}")


def t_min_hits():
    clk = Clock()
    b = make_behavior(clk)
    b.calculate_command(NO_INPUT, plate_vision(hits=2, dist=200))
    check("偵測不到 3 格 -> 不接近", b.state == "SEARCH", b.state)
    b.calculate_command(NO_INPUT, plate_vision(hits=3, dist=200))
    check("連續看到 3 格 -> APPROACH", b.state == "APPROACH", b.state)


def _approach_cmd(**kw):
    clk = Clock()
    b = make_behavior(clk)
    return engage(b, clk, **kw), b


def t_approach_directions():
    (lr, fb, ud, yv), b = _approach_cmd(dist=200)
    check("遠 -> 往前 (遠距離速度)", fb == b.FB_FAR, f"fb={fb}")
    (_, fb, _, _), b = _approach_cmd(dist=100)
    check("近 (FAR_CM 以內) -> 放慢", 0 < fb <= b.FB_NEAR, f"fb={fb}")
    (_, fb, _, _), _ = _approach_cmd(dist=50)
    check("太近 -> 後退", fb < 0, f"fb={fb}")
    (_, fb, _, _), _ = _approach_cmd(dist=73)
    check("在距離死區內 -> 不前後動", fb == 0, f"fb={fb}")
    (_, _, _, yv), _ = _approach_cmd(cx=460, dist=200)
    check("車牌偏右 -> 往右轉 (yv > 0)", yv > 0, f"yv={yv}")
    (_, _, _, yv), _ = _approach_cmd(cx=260, dist=200)
    check("車牌偏左 -> 往左轉 (yv < 0)", yv < 0, f"yv={yv}")
    (_, _, ud, _), _ = _approach_cmd(cy=330, dist=200)
    check("車牌在下方 -> 下降 (ud < 0)", ud < 0, f"ud={ud}")
    (_, _, ud, _), _ = _approach_cmd(cy=150, dist=200)
    check("車牌在上方 -> 上升 (ud > 0)", ud > 0, f"ud={ud}")
    (_, _, ud, _), b = _approach_cmd(cy=470, dist=200)
    check("升降限速", abs(ud) <= b.UD_LIMIT, f"ud={ud}")
    (_, fb_off, _, _), b = _approach_cmd(cx=360 + 90, dist=200)
    check("偏太多 (> 2 倍對準容差) -> 先不前進，只轉", fb_off == 0, f"fb={fb_off}")

    # 目標距離被調得太近：估距 < MIN_DIST_CM 一律不前進
    clk = Clock()
    b = make_behavior(clk)
    b.TARGET_DIST_CM = 25
    _, fb, _, _ = engage(b, clk, dist=35)
    check("估距 < MIN_DIST_CM -> 不前進 (擋設定錯誤)", fb <= 0, f"fb={fb}")


def t_hold_then_read():
    clk = Clock()
    b = make_behavior(clk, seq=[("ABC-1234", 0.9)] * 5)
    t0 = clk.t
    out = run(b, clk, lambda i: plate_vision(), 3.0)
    t_read = next((t for t, _, s in out if s == "READ"), None)
    check("停穩不到 HOLD_SEC 不讀",
          all(s != "READ" for t, _, s in out if t < t0 + b.HOLD_SEC), "")
    check("停穩滿 HOLD_SEC 才開始讀", t_read is not None and t_read - t0 >= b.HOLD_SEC,
          f"READ 在 {t_read - t0:.2f}s" if t_read is not None else "沒進 READ")
    check("同號碼 2 票 -> DONE", b.state == "DONE" and b.result_text == "ABC-1234",
          f"{b.state} {b.result_text}")
    check("OCR 只送了 2 次 (收集 COLLECT_SEC 才送)", b.reader.ocr.calls == 2,
          f"calls={b.reader.ocr.calls}")
    check("DONE 顯示號碼", b.alert_text() == "PLATE ABC-1234", f"{b.alert_text()}")
    files = os.listdir(OUT_DIR)
    check("存下截圖與 plates.csv",
          "plates.csv" in files and any(f.endswith("_ABC1234.jpg") for f in files), f"{files}")
    out = run(b, clk, lambda i: plate_vision(), 2.0)
    check("DONE 不會自己飛走", all(s == "DONE" for _, _, s in out) and
          all(c == (0, 0, 0, 0) for _, c, _ in out), f"{out[-1]}")


def t_jitter_never_reads():
    clk = Clock()
    b = make_behavior(clk, seq=[("ABC1234", 0.9)] * 5)
    out = run(b, clk, lambda i: plate_vision(cx=360 + (10 if i % 2 else -10)), 4.0)
    check("框一直晃 (±10px) -> 永遠不進 READ", all(s != "READ" for _, _, s in out),
          f"states={sorted(set(s for _, _, s in out))}")
    check("沒送過 OCR", b.reader.ocr.calls == 0, f"calls={b.reader.ocr.calls}")


def _read_until_done(seq, seconds=7.0):
    clk = Clock()
    b = make_behavior(clk, seq=seq)
    run(b, clk, lambda i: plate_vision(), seconds)
    return b


def t_votes():
    check("正規化：ABC-1234 / abc 1234 / ABC1234 同一個",
          normalize_plate("ABC-1234") == normalize_plate("abc 1234") == "ABC1234", "")
    b = _read_until_done([("ABC-1234", 0.9), ("abc 1234", 0.8)])
    check("不同寫法算同一票 -> DONE，不是不確定",
          b.state == "DONE" and normalize_plate(b.result_text) == "ABC1234"
          and not b.result_uncertain, f"{b.result_text} uncertain={b.result_uncertain}")
    b = _read_until_done([("XYZ999", 0.9)])
    check("逾時只有 1 票 -> 取它但標 ?", b.result_text == "XYZ999" and b.result_uncertain,
          f"{b.alert_text()}")
    check("不確定時畫面加 ?", b.alert_text() == "PLATE XYZ999 ?", f"{b.alert_text()}")
    b = _read_until_done([("LOW111", 0.3), ("LOW111", 0.3)])
    check("信心不足的結果不計票 -> 讀取失敗",
          b.state == "DONE" and b.result_text is None, f"{b.result_text}")
    check("失敗時畫面提示按 R", b.alert_text() == "READ FAILED - PRESS R", f"{b.alert_text()}")


def t_drift_and_loss():
    clk = Clock()
    b = make_behavior(clk, seq=[("ABC1234", 0.9)])
    run(b, clk, lambda i: plate_vision(), 1.8)
    n_votes = sum(b.reader.votes.values())
    check("讀到第一票 (READ 中)", b.state == "READ" and n_votes == 1, f"{b.state} votes={n_votes}")
    run(b, clk, lambda i: plate_vision(cx=360 + 70), 0.1)
    check("讀取中漂離 -> 退回停穩 / 接近", b.state in ("HOLD", "APPROACH"), b.state)
    check("漂離時票保留", sum(b.reader.votes.values()) == 1, f"{dict(b.reader.votes)}")
    run(b, clk, lambda i: blank_vision(), b.LOST_SEC + 0.2)
    check("跟丟超過 LOST_SEC -> SEARCH", b.state == "SEARCH", b.state)
    check("回 SEARCH 清掉票", sum(b.reader.votes.values()) == 0, f"{dict(b.reader.votes)}")


def t_memory_decay():
    clk = Clock()
    b = make_behavior(clk)
    _, _, _, yv0 = engage(b, clk, cx=460, dist=200)
    prev = b.pid_yv.prev_error
    clk.tick(0.1)
    _, _, _, yv1 = b.calculate_command(NO_INPUT, blank_vision())
    check("單格漏偵測 -> 指令衰減 (不是歸零也不是照舊)", 0 < yv1 < yv0, f"yv {yv0} -> {yv1}")
    check("記憶期間不碰 PID", b.pid_yv.prev_error == prev, f"{prev} -> {b.pid_yv.prev_error}")
    clk.tick(0.3)
    out = b.calculate_command(NO_INPUT, blank_vision())
    check("記憶過期但還沒跟丟 -> 懸停", out == (0, 0, 0, 0) and b.state == "APPROACH",
          f"{out} {b.state}")


def t_approach_timeout():
    clk = Clock()
    b = make_behavior(clk)
    # 車牌一直偏在旁邊 (模擬怎麼轉都對不準)
    out = run(b, clk, lambda i: plate_vision(cx=360 + 45), b.APPROACH_MAX_SEC + 0.5)
    check("停不穩超過 APPROACH_MAX_SEC -> 放棄回 SEARCH", b.state == "SEARCH", b.state)
    check("放棄後 IGNORE_SEC 內不鎖定", b.ignore_until > clk.t, f"{b.ignore_until - clk.t:.1f}s")
    run(b, clk, lambda i: plate_vision(), 1.0)
    check("冷卻期間看到車牌也不接近", b.state == "SEARCH", b.state)


def t_r_key():
    b = _read_until_done([("ABC1234", 0.9)] * 2, seconds=3.0)
    clk = b._now
    check("先讀完", b.state == "DONE", b.state)
    b.reset_target()
    check("DONE 按 R -> 重讀 (HOLD，票清空)",
          b.state == "HOLD" and not b.reader.votes and b.result_text is None,
          f"{b.state} {dict(b.reader.votes)}")
    b.reader.ocr.seq = [("NEW5678", 0.9)] * 2
    run(b, clk, lambda i: plate_vision(), 3.0)
    check("重讀得到新結果", b.state == "DONE" and b.result_text == "NEW5678", f"{b.result_text}")

    clk = Clock()
    b = make_behavior(clk)
    engage(b, clk, dist=200)
    b.reset_target()
    check("APPROACH 按 R -> 放棄 + 冷卻", b.state == "SEARCH" and b.ignore_until > clk.t, b.state)

    # 讀完之後飛手接手 -> 冷卻，不會一放手又鎖回剛剛那個
    b = _read_until_done([("ABC1234", 0.9)] * 2, seconds=3.0)
    clk = b._now
    key = SimpleNamespace(lr=0, fb=-40, ud=0, yv=0, voice_command=None)
    b.calculate_command(key, plate_vision())
    clk.tick()
    b.calculate_command(NO_INPUT, plate_vision())
    check("DONE 後飛手接手 -> 冷卻期間不重新接近", b.state == "SEARCH", b.state)


def t_tello_app_routes_r_key():
    from core.tello_app import TelloApp
    clk = Clock()
    b = make_behavior(clk)
    engage(b, clk, dist=200)
    TelloApp.reset_tracking_target(SimpleNamespace(vision=None, behavior=b))
    check("tello_app 的 R 鍵會通知 behavior", b.state == "SEARCH", b.state)
    called = []
    vis = SimpleNamespace(reset_target=lambda: called.append(1))
    TelloApp.reset_tracking_target(SimpleNamespace(vision=vis, behavior=object()))
    check("vision 的 reset_target 照舊會呼叫 (TISSUE CHARGE)", called == [1], f"{called}")


def t_log_fields():
    from utils.video_recorder import VideoRecorder
    clk = Clock()
    b = make_behavior(clk, seq=[("ABC1234", 0.9)])
    run(b, clk, lambda i: plate_vision(), 1.8)
    keys = set(b.log_fields())
    missing = keys - set(VideoRecorder.LOG_FIELDS)
    check("log_fields 的欄位都在 LOG_FIELDS 裡", not missing and keys, f"缺 {missing}")
    check("log 記到票數", b.log_fields().get("ocr_votes") == "ABC1234:1", f"{b.log_fields()}")


def _world(seconds=20.0, dt=0.05, theta=10.0, dz=-25.0, dist=250.0, seq=None):
    """
    簡單的模擬世界：車牌在機頭右邊 theta 度、低 dz 公分、距離 dist 公分。
    rc 指令照「yv 1 = 每秒 1 度、ud / fb 1 = 每秒 1 公分」移動，延遲一格才生效。
    """
    clk = Clock()
    b = make_behavior(clk, seq=seq if seq is not None else [("ID2", 1.0)] * 10)
    px_per_rad = F_H
    pending = (0, 0, 0, 0)
    trace = []
    for _ in range(int(seconds / dt)):
        _, fb, ud, yv = pending
        theta -= yv * dt
        dz -= ud * dt
        dist -= fb * dt
        cx = 360 + px_per_rad * math.tan(math.radians(theta))
        cy = 240 - F_V * dz / dist
        pending = b.calculate_command(NO_INPUT, plate_vision(cx=cx, cy=cy, dist=dist))
        trace.append((clk.t, b.state, dist, cx, cy, pending))
        clk.tick(dt)
        if b.state == "DONE":
            break
    return b, trace


def t_closed_loop():
    b, trace = _world()
    t_done = trace[-1][0] - trace[0][0]
    check("模擬：從 250cm 外一路走到讀出號碼", b.state == "DONE" and b.result_text == "ID2",
          f"{b.state} {b.result_text} 用了 {t_done:.1f}s")
    _, _, dist, cx, cy, _ = trace[-1]
    check("模擬：停在目標距離、畫面中央", abs(dist - b.TARGET_DIST_CM) <= 0.15 * b.TARGET_DIST_CM
          and abs(cx - 360) <= 30 and abs(cy - 240) <= 30,
          f"dist={dist:.0f} cx={cx:.0f} cy={cy:.0f}")
    min_dist = min(d for _, _, d, _, _, _ in trace)
    check("模擬：過程中沒有衝過頭太多", min_dist >= b.TARGET_DIST_CM * 0.85, f"最近 {min_dist:.0f}cm")
    reads = [t for t, s, *_ in trace if s == "READ"]
    stable = [t for t, s, *_ in trace if s == "HOLD"]
    check("模擬：讀之前有先停穩", stable and reads and min(stable) < min(reads), "")


# ==========================================
# OCR 背景執行緒
# ==========================================
class SlowOCR(PlateOCRBackend):
    def __init__(self, delay=0.2, fail_first=False):
        self.delay, self.fail_first, self.calls = delay, fail_first, 0

    def read(self, crop):
        self.calls += 1
        time.sleep(self.delay)
        if self.fail_first and self.calls == 1:
            raise RuntimeError("模擬 OCR 當掉")
        return f"OK{self.calls}", 0.9


def _wait_idle(r, timeout=3.0):
    t_end = time.time() + timeout
    while r.busy and time.time() < t_end:
        time.sleep(0.01)
    time.sleep(0.02)


def t_reader_thread():
    crop = np.full((40, 90, 3), 128, np.uint8)
    r = PlateReader(SlowOCR(0.2), out_dir=OUT_DIR)
    t0 = time.time()
    ok = r.submit(crop)
    dt = time.time() - t0
    check("慢 OCR 時 submit 不阻塞 (< 5ms)", ok and dt < 0.005, f"{dt * 1000:.1f}ms")
    check("背景忙的時候 submit 回 False", r.submit(crop) is False, "")
    _wait_idle(r)
    got = r.poll()
    check("跑完之後 poll 拿得到結果並計票", got == [("OK1", 0.9)] and r.votes["OK1"] == 1, f"{got}")

    r = PlateReader(SlowOCR(0.05, fail_first=True), out_dir=OUT_DIR)
    r.submit(crop)
    _wait_idle(r)
    first = r.poll()
    r.submit(crop)
    _wait_idle(r)
    second = r.poll()
    check("OCR 丟例外不會讓執行緒死掉", first == [(None, 0.0)] and second == [("OK2", 0.9)],
          f"{first} {second}")

    r = PlateReader(SlowOCR(0.2), out_dir=OUT_DIR)
    r.submit(crop)
    r.clear()
    _wait_idle(r)
    got = r.poll()
    check("clear() 之後舊車牌的結果作廢", got == [] and not r.votes, f"{got}")


# ==========================================
# 視覺管線
# ==========================================
def _frames(native_fill=(255, 0, 0)):
    native = np.zeros((720, 960, 3), np.uint8)
    native[:] = native_fill
    frame = cv2.resize(native, (720, 480), interpolation=cv2.INTER_AREA)
    return native, frame


def t_plate_vision_native():
    det = FakeDetector([[{"box": (480, 360, 600, 420), "conf": 0.9}]])
    pv = PlateVision(detector=det, depth=FakeDepth())
    native, frame = _frames()
    pv.set_native_frame(native)
    d = pv.process_frame(frame)
    p = d.plate
    check("偵測吃的是原生畫面", det.seen[0].shape[:2] == (720, 960), f"{det.seen[0].shape}")
    check("座標換回 720x480", abs(p["cx"] - 405) < 0.5 and abs(p["cy"] - 260) < 0.5
          and abs(p["w"] - 90) < 0.5 and abs(p["h"] - 40) < 0.5,
          f"cx={p['cx']:.1f} cy={p['cy']:.1f} w={p['w']:.1f} h={p['h']:.1f}")
    exp = min(32 * F_H / 90, 15 * F_V / 40)
    check("估距取寬、高換算中較近的", abs(p["dist_cm"] - exp) < 0.5, f"{p['dist_cm']:.1f} vs {exp:.1f}")
    check("截圖從原生畫面切、四周多留 10%", p["crop"].shape[:2] == (72, 144), f"{p['crop'].shape}")
    check("交給後端的是 BGR (串流 RGB 轉過)",
          list(det.seen[0][0, 0]) == [0, 0, 255] and list(p["crop"][0, 0]) == [0, 0, 255],
          f"{det.seen[0][0, 0]}")
    check("深度欄位照樣帶出去 (給漫遊用)", d.depth_C == 999.0 and d.depth_unreliable is False, "")

    det = FakeDetector([[{"box": (480, 360, 600, 420), "conf": 0.9}]])
    pv = PlateVision(detector=det, depth=FakeDepth(), to_bgr=False)
    pv.set_native_frame(native)
    pv.process_frame(frame)
    check("to_bgr=False 時不轉", list(det.seen[0][0, 0]) == [255, 0, 0], f"{det.seen[0][0, 0]}")

    det = FakeDetector([[{"box": (480, 360, 600, 420), "conf": 0.3}]])
    pv = PlateVision(detector=det, depth=FakeDepth())
    d = pv.process_frame(frame)
    check("信心不足的不收", d.plate is None and d.n_low_conf == 1, "")

    pv = PlateVision(detector=None, depth=FakeDepth())
    d = pv.process_frame(frame)
    check("沒有偵測器 -> 不偵測，狀態列 NO PLATE MODEL",
          d.plate is None and pv._status_text(d, None).startswith("NO PLATE MODEL"), "")


def t_plate_vision_lock():
    a = {"box": (100, 300, 220, 360), "conf": 0.9}      # 小的，先出現
    big = {"box": (700, 300, 900, 400), "conf": 0.9}    # 大的，後出現
    det = FakeDetector([[a], [a, big]] + [[big]] * 8)
    pv = PlateVision(detector=det, depth=FakeDepth())
    native, frame = _frames()
    picks = []
    for _ in range(9):
        pv.set_native_frame(native)
        p = pv.process_frame(frame).plate
        picks.append(None if p is None else ("A" if p["cx"] < 360 else "B"))
    check("鎖定後不會跳去更大的車牌", picks[:2] == ["A", "A"], f"{picks}")
    check("鎖定的不見了先撐幾格 (不馬上換)", picks[2:7] == [None] * 5, f"{picks}")
    check("撐過 LOCK_MISS_MAX 才換", picks[7] == "B", f"{picks}")
    check("hits 從換的那格重新算", pv.hits == 2, f"hits={pv.hits}")


def t_aruco_stand_in():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    native = np.full((720, 960, 3), 255, np.uint8)
    native[300:420, 400:520] = cv2.aruco.generateImageMarker(d, 2, 120)[..., None]
    frame = cv2.resize(native, (720, 480), interpolation=cv2.INTER_AREA)
    pv = PlateVision(detector=ArucoPlateDetector(plate_size_cm=8.0), depth=FakeDepth())
    pv.set_native_frame(native)
    p = pv.process_frame(frame).plate
    check("ArUco 替身偵測得到", p is not None and p.get("marker_id") == 2, f"{p and p.get('marker_id')}")
    if p is None:
        return
    check("ArUco 框的位置正確", abs(p["cx"] - 460 * 0.75) < 2 and abs(p["w"] - 120 * 0.75) < 3,
          f"cx={p['cx']:.1f} w={p['w']:.1f}")
    ocr = ArucoIdOCR()
    check("ArUco 替身 OCR 從截圖讀出 ID", ocr.read(p["crop"]) == ("ID2", 1.0), f"{ocr.read(p['crop'])}")
    # 轉機頭時的水平動態模糊 (ArUco 對輕微的糊很耐，15px 以上才讀不到)
    k = np.zeros((21, 21), np.float32)
    k[10] = 1.0 / 21
    blurred = cv2.filter2D(p["crop"], -1, k)
    check("動態模糊的截圖讀不到", ocr.read(blurred) == (None, 0.0), f"{ocr.read(blurred)}")
    half = p["crop"][:, :p["crop"].shape[1] // 2]
    check("切掉一半的截圖讀不到", ocr.read(half) == (None, 0.0), f"{ocr.read(half)}")
    check("清晰度：糊掉的比較低", sharpness(blurred) < sharpness(p["crop"]),
          f"{sharpness(blurred):.0f} < {sharpness(p['crop']):.0f}")


if __name__ == "__main__":
    for fn in (t_manual_override, t_search_roams_altitude_locked, t_min_hits,
               t_approach_directions, t_hold_then_read, t_jitter_never_reads, t_votes,
               t_drift_and_loss, t_memory_decay, t_approach_timeout, t_r_key,
               t_tello_app_routes_r_key, t_log_fields, t_closed_loop,
               t_reader_thread, t_plate_vision_native, t_plate_vision_lock, t_aruco_stand_in):
        print(f"\n--- {fn.__name__} ---")
        fn()
    bad = [n for n, ok, _ in results if not ok]
    print("\n" + "=" * 60)
    print(f"{len(results) - len(bad)}/{len(results)} 通過"
          + (f"，未通過: {bad}" if bad else "，全部通過"))
    sys.exit(1 if bad else 0)
