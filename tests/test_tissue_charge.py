"""
TissueChargeControl 的行為測試 —— 不需要無人機，也不需要錄影。

重點在那些「實飛才會發現、但發現時已經撞上去了」的性質：
瞄準點會不會發散 (之前撞過天花板)、高度上限擋不擋得住、
跟丟之後會不會失控、深度避牆有沒有真的擋住前進。

用假時鐘驅動，所有跟時間有關的狀態轉換都能精確測到。

    python tests/test_tissue_charge.py
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behaviors.tissue_charge import TissueChargeControl
from vision.base import VisionData

NO_INPUT = SimpleNamespace(lr=0, fb=0, ud=0, yv=0, voice_command=None)

# 垂直焦距 = 水平焦距 x (Fv/Fh)。縮放是非等向的，這個比值就是控制器裡的修正項。
F_H = 461.0
F_V = F_H * ((480.0 / 720.0) / (720.0 / 960.0))     # 約 410


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def tick(self, dt=0.02):
        self.t += dt


def make_vision(cx=360, w=20, top_y=0.0, bottom_y=300.0, height=100,
                depth_ahead=999.0, depth_unreliable=False):
    d = VisionData(is_detected=True)
    d.tissue = {
        "cx": cx, "cy": (top_y + bottom_y) / 2.0, "w": w, "h": bottom_y - top_y,
        "area": w * (bottom_y - top_y), "conf": 0.8,
        "aspect": (bottom_y - top_y) / max(w, 1),
        "dist_cm": 1383.0 / max(w, 1),
        "top_y": top_y, "bottom_y": bottom_y,
        "top_clipped": top_y <= 1, "bottom_clipped": bottom_y >= 479,
    }
    d.height = height
    d.depth_ahead = depth_ahead
    d.depth_unreliable = depth_unreliable
    return d


def blank_vision(height=100, depth_L=999.0, depth_C=999.0, depth_R=999.0):
    """沒偵測到紙條的一格。深度預設四周開闊"""
    d = VisionData(is_detected=False)
    d.tissue = None
    d.height = height
    d.depth_L, d.depth_C, d.depth_R = depth_L, depth_C, depth_R
    d.depth_ahead = depth_C
    d.depth_unreliable = False
    return d


results = []


def check(name, cond, detail=""):
    results.append((name, cond, detail))
    print(f"{'PASS' if cond else '**FAIL**':10s} {name}" + (f"   {detail}" if detail else ""))


# ==========================================
def t_aim_point_is_75_percent_height():
    """瞄準點 (畫面上的紅點) 在 75% 高度，畫面水平中央"""
    b = TissueChargeControl()
    check("瞄準點在 (360, 120)", b.aim_point() == (360, 120), f"aim_point={b.aim_point()}")

    # 紙條只橫過畫面中心、沒有橫過瞄準列 -> 不前進；橫過瞄準列 -> 前進
    clk = Clock()
    b1 = TissueChargeControl(now_fn=clk); b1.change_state("TRACK")
    _, fb_low, _, _ = b1.calculate_command(NO_INPUT, make_vision(top_y=180.0, bottom_y=400.0))
    b2 = TissueChargeControl(now_fn=clk); b2.change_state("TRACK")
    _, fb_hit, _, _ = b2.calculate_command(NO_INPUT, make_vision(top_y=0.0, bottom_y=240.0))
    check("紙條在瞄準列下方 -> 不前進", fb_low == 0, f"fb={fb_low}")
    check("紙條橫過瞄準列 -> 前進", fb_hit > 0, f"fb={fb_hit}")


def t_altitude_locked():
    """
    高度鎖定：不管哪個狀態、紙條在畫面哪裡、高度讀值多少，自動指令的 ud 都是 0。
    只有飛手按 W/S 可以改變高度。
    """
    clk = Clock()
    cases = [
        ("SEARCH 開闊", None, blank_vision(height=120)),
        ("SEARCH 讀值異常", None, blank_vision(height=-500)),
        ("TRACK 紙條在頭頂上", "TRACK", make_vision(top_y=0.0, bottom_y=100.0)),
        ("TRACK 紙條在下方", "TRACK", make_vision(top_y=200.0, bottom_y=450.0)),
        ("TRACK 紙條橫過路線", "TRACK", make_vision(top_y=0.0, bottom_y=300.0)),
        ("COAST", "COAST", blank_vision()),
        ("BACKOFF", "BACKOFF", blank_vision()),
    ]
    for name, state, v in cases:
        b = TissueChargeControl(now_fn=clk)
        if state:
            b.change_state(state)
        uds = []
        for _ in range(10):
            uds.append(b.calculate_command(NO_INPUT, v)[2])
            clk.tick(0.05)
        check(f"{name} -> 不動高度", all(u == 0 for u in uds), f"ud={uds}")

    # 黑區脫困原本會送 DARK_CLIMB 爬升
    b2 = TissueChargeControl(now_fn=clk)
    v = blank_vision()
    v.depth_unreliable = True
    _, _, ud, _ = b2.calculate_command(NO_INPUT, v)
    check("黑區脫困 -> 不爬升", ud == 0, f"ud={ud}")

    up = SimpleNamespace(lr=0, fb=0, ud=40, yv=0, voice_command=None)
    _, _, ud, _ = b2.calculate_command(up, blank_vision())
    check("飛手按 W/S -> 照送", ud == 40, f"ud={ud}")


def t_off_path_gives_up():
    """紙條沒有橫過前進路線 (在路線上方或下方) -> 高度不能動，撞不到，很快放棄換下一條"""
    clk = Clock()
    for name, v in (("上方", make_vision(top_y=0.0, bottom_y=60.0)),
                    ("下方", make_vision(top_y=200.0, bottom_y=450.0))):
        b = TissueChargeControl(now_fn=clk)
        b.calculate_command(NO_INPUT, v)
        t0 = clk.t
        fbs = []
        while b.state == "TRACK" and clk.t - t0 < 20:
            clk.tick(0.05)
            fbs.append(b.calculate_command(NO_INPUT, v)[1])
        took = clk.t - t0
        check(f"紙條在路線{name} -> 很快放棄",
              b.state == "SEARCH" and took <= b.UNREACHABLE_SEC + 0.2 and b.ignore_until > clk.t,
              f"{took:.1f} 秒後 state={b.state}")
        check(f"紙條在路線{name} -> 不往前衝", max(fbs[:-1] or [0]) == 0, f"fb 最大 {max(fbs[:-1] or [0])}")


def t_depth_blocks_forward():
    """前方太近時只轉不前進 —— 使用者確認要的避牆行為"""
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    b.change_state("TRACK")
    near = make_vision(cx=420, bottom_y=300.0, depth_ahead=40.0)
    far = make_vision(cx=420, bottom_y=300.0, depth_ahead=400.0)
    _, fb_n, _, yv_n = b.calculate_command(NO_INPUT, near)
    b2 = TissueChargeControl(now_fn=clk); b2.change_state("TRACK")
    _, fb_f, _, yv_f = b2.calculate_command(NO_INPUT, far)
    check("前方太近 -> 不前進", fb_n == 0, f"fb={fb_n} (深度 40cm)")
    check("前方太近 -> 轉向照常修正", yv_n != 0, f"yv={yv_n}")
    check("前方淨空 -> 正常前進", fb_f > 0, f"fb={fb_f} (深度 400cm)")

    # 深度不可信 (暗處) 時不否決，否則無人機永遠不會前進
    b3 = TissueChargeControl(now_fn=clk); b3.change_state("TRACK")
    dark = make_vision(cx=420, bottom_y=300.0, depth_ahead=40.0, depth_unreliable=True)
    _, fb_d, _, _ = b3.calculate_command(NO_INPUT, dark)
    check("深度不可信時不否決前進", fb_d > 0, f"fb={fb_d}")


def t_proportional_forward():
    """前進速度要隨對準程度連續變化，不是二元開關"""
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    b.change_state("TRACK")
    speeds = []
    for cx in (360, 390, 420, 460, 520):
        bb = TissueChargeControl(now_fn=clk); bb.change_state("TRACK")
        _, fb, _, _ = bb.calculate_command(NO_INPUT, make_vision(cx=cx, bottom_y=300.0))
        speeds.append(fb)
    check("對準時前進最快", speeds[0] == max(speeds), f"fb 序列 {speeds}")
    check("偏差越大前進越慢", all(a >= b_ for a, b_ in zip(speeds, speeds[1:])),
          f"fb 序列 {speeds}")
    check("偏差夠大時完全不前進", speeds[-1] == 0, f"fb={speeds[-1]}")


def t_not_crossing_path_blocks_forward():
    """紙條沒有橫過前進路線時絕不前進 (否則會從上面或下面穿過去)"""
    clk = Clock()
    b = TissueChargeControl(now_fn=clk); b.change_state("TRACK")
    above = make_vision(cx=360, top_y=0.0, bottom_y=100.0)      # 整條在頭頂上
    _, fb, _, _ = b.calculate_command(NO_INPUT, above)
    check("紙條在頭頂上 -> 不前進", fb == 0, f"fb={fb}")


def t_memory_decays_and_spares_pid():
    """記憶期間要讓指令衰減，而且不可以再呼叫 PID (不然 prev_error 會被污染)"""
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    b.change_state("TRACK")
    b.calculate_command(NO_INPUT, make_vision(cx=460, bottom_y=300.0))
    prev_err_before = b.pid_yv.prev_error
    first = b.last_yv

    outs = []
    for _ in range(10):                     # 0.02 x 10 = 0.2 秒 < MEMORY_DURATION
        clk.tick(0.02)
        outs.append(b.calculate_command(NO_INPUT, blank_vision())[3])
    check("記憶期間指令單調衰減",
          all(abs(a) >= abs(b_) for a, b_ in zip(outs, outs[1:])) and abs(outs[-1]) < abs(first),
          f"yv {first} -> {outs[-1]}")
    check("記憶期間不碰 PID 狀態", b.pid_yv.prev_error == prev_err_before,
          f"prev_error {prev_err_before} -> {b.pid_yv.prev_error}")
    check("記憶期間狀態不變", b.state == "TRACK", f"state={b.state}")


def t_coast_then_backoff():
    """近距離跟丟 -> 短滑行 -> 退開 -> 回到搜尋"""
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    b.change_state("TRACK")
    # 先對準且夠近，讓它真的在前進
    for _ in range(3):
        b.calculate_command(NO_INPUT, make_vision(cx=360, w=46, bottom_y=300.0))  # 估距 30cm
        clk.tick(0.02)
    check("逼近時確實在前進", b.last_fb > 0, f"fb={b.last_fb}, 估距={b.last_dist_cm:.0f}cm")

    clk.tick(b.MEMORY_DURATION + 0.01)      # 讓記憶過期
    _, fb, _, yv = b.calculate_command(NO_INPUT, blank_vision())
    check("近距離跟丟 -> 進入 COAST", b.state == "COAST", f"state={b.state}")
    check("COAST 用追蹤速度不是衝刺速度", fb == b.COAST_FB, f"fb={fb}")

    coast_fbs = []
    t_coast = clk.t
    for _ in range(int(b.COAST_SEC / 0.02) + 20):    # 依設定的滑行時間等，多留 0.4 秒
        clk.tick(0.02)
        cmd = b.calculate_command(NO_INPUT, blank_vision())
        if b.state != "COAST":
            break
        coast_fbs.append(cmd[1])
    coast_time = clk.t - t_coast
    check("COAST 期間偏航歸零", all(c == 0 for c in [cmd[3]]), f"yv={cmd[3]}")
    check("COAST 期間速度固定為追蹤速度", all(f == b.COAST_FB for f in coast_fbs),
          f"fb={set(coast_fbs)}")
    check("COAST 結束後退開", b.state == "BACKOFF", f"state={b.state}")
    check("COAST 滑行時間符合 COAST_SEC", abs(coast_time - b.COAST_SEC) <= 0.05,
          f"滑行 {coast_time:.2f}s (設定 {b.COAST_SEC}s，約 {b.COAST_FB * b.COAST_SEC:.0f} 單位)")

    for _ in range(60):
        clk.tick(0.02)
        cmd = b.calculate_command(NO_INPUT, blank_vision())
        if b.state == "BACKOFF":
            check_back = cmd[1]
        if b.state == "SEARCH":
            break
    check("BACKOFF 是往後退", check_back < 0, f"fb={check_back}")
    check("退完回到搜尋並冷卻", b.state == "SEARCH" and b.ignore_until > clk.t,
          f"state={b.state}")
    check("剪斷次數有累計", b.cut_count == 1, f"cut_count={b.cut_count}")


def t_far_loss_goes_straight_to_search():
    """遠距離跟丟就直接回去重找，不滑行 —— 滑行只是為了補完看不見的最後幾公分"""
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    b.change_state("TRACK")
    for _ in range(3):
        b.calculate_command(NO_INPUT, make_vision(cx=360, w=10, bottom_y=300.0))  # 估距 138cm
        clk.tick(0.02)
    clk.tick(b.MEMORY_DURATION + 0.01)
    b.calculate_command(NO_INPUT, blank_vision())
    check("遠距離跟丟 -> 直接回搜尋不滑行", b.state == "SEARCH",
          f"state={b.state}, 估距={b.last_dist_cm:.0f}cm")


def t_long_search_does_not_kill_track():
    """
    漫遊很久才找到紙條時，TRACK 不可以因為沿用漫遊的計時而立刻放棄。

    這是從舊版一路帶過來的 bug：time_in_state 在迴圈開頭就算好了，
    SEARCH -> TRACK 的同一格用的還是漫遊的經過時間，只要漫遊超過
    TRACK_MAX_SEC (15 秒)，第一格就會直接觸發「追太久」而放棄。
    紙條稀疏的場地漫遊超過 15 秒是常態，等於整個模式失效。
    """
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    # 漫遊 30 秒 (超過 TRACK_MAX_SEC) 之後才看到紙條
    b.state_start_time = clk.t
    clk.tick(30.0)
    b.calculate_command(NO_INPUT, make_vision(cx=360, bottom_y=300.0))
    check("漫遊很久後仍能進入追蹤", b.state == "TRACK",
          f"漫遊 30 秒後 state={b.state}")
    check("不會立刻觸發放棄冷卻", b.ignore_until <= clk.t,
          f"ignore_until - now = {b.ignore_until - clk.t:.1f}s")


def t_hit_state_is_gone():
    """確認舊的盲衝狀態與參數真的被移除了"""
    b = TissueChargeControl()
    gone = [n for n in ("HIT_FB", "HIT_SEC", "was_advancing") if hasattr(b, n)]
    check("盲衝相關的屬性已全部移除", not gone, f"殘留: {gone}")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "behaviors", "tissue_charge.py")).read()
    check("程式碼裡沒有 HIT 狀態", '"HIT"' not in src)


def t_log_fields():
    """CSV 診斷欄位要真的有值，而且欄名要在 LOG_FIELDS 裡"""
    from utils.video_recorder import VideoRecorder
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    b.change_state("TRACK")
    b.calculate_command(NO_INPUT, make_vision(cx=400, w=30, bottom_y=310.0, depth_ahead=250.0))
    lg = b.log_fields()
    need = {"tw", "bottom_y", "dist", "cy", "err_x", "err_y", "ax", "blocked", "depth_ahead"}
    check("診斷欄位齊全", need <= set(lg), f"缺少 {need - set(lg)}")
    unknown = set(lg) - set(VideoRecorder.LOG_FIELDS)
    check("欄名都在 LOG_FIELDS 裡", not unknown, f"未登記: {unknown}")


def t_charge_not_abandoned_when_crossing():
    """
    19:34 那趟兩次：已爬到上限、紙條橫過路線 (十字比紅點高 30px)、正往前衝 ->
    1 秒後判定「搆不到」放棄，沒撞到。紙條橫過路線就要一直衝下去。
    """
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    b.change_state("TRACK")
    fbs, uds = [], []
    for _ in range(60):                  # 3 秒，遠超過 UNREACHABLE_SEC
        _, fb, ud, _ = b.calculate_command(NO_INPUT, make_vision(w=11, top_y=0.0, bottom_y=180.0))
        fbs.append(fb); uds.append(ud)
        clk.tick(0.05)
    check("紙條橫過路線 -> 不放棄", b.state == "TRACK", f"3 秒後 state={b.state}")
    check("紙條橫過路線 -> 一直往前衝", min(fbs[5:]) > 0, f"fb 最小 {min(fbs[5:])}")
    check("衝撞中高度不動", all(u == 0 for u in uds), f"ud 範圍 {min(uds)}..{max(uds)}")

    # 十字在紅點下方 (紙條大半在路線下方，但仍橫過) -> 一樣不下降
    b2 = TissueChargeControl(now_fn=clk); b2.change_state("TRACK")
    _, fb, ud, _ = b2.calculate_command(NO_INPUT, make_vision(top_y=60.0, bottom_y=460.0))
    check("十字在紅點下方但紙條橫過路線 -> 照衝、不下降", ud == 0 and fb > 0, f"ud={ud} fb={fb}")


def t_blank_surface_pillar():
    """
    17:25 撞柱子的重現：左側 40% 是純白無紋理的柱子，右緣侵入中央帶。
    深度模型讀成很遠，分析器要把左側與中央覆寫成近距離。
    """
    import numpy as np
    from utils.blank_region import BlankSurfaceAnalyzer
    rng = np.random.default_rng(0)
    img = rng.integers(40, 220, size=(480, 720, 3), dtype=np.uint8)   # 有紋理的背景
    img[:, :315] = 200                                                 # 白柱子，右緣在 x=315
    an = BlankSurfaceAnalyzer()
    info = an.analyze(img)
    d = SimpleNamespace(depth_L=700.0, depth_C=700.0, depth_R=700.0)
    an.apply(d, info)
    check("白柱子在左側 -> 左側判定為近", info["near_L"], f"L={info['L']:.2f}")
    check("白柱子侵入中央帶 -> 前方判定為近", info["near_C"], f"C={info['C']:.2f}")
    check("右側有紋理 -> 不受影響", not info["near_R"] and d.depth_R == 700.0,
          f"R={info['R']:.2f}")
    check("深度只往近的方向覆寫", d.depth_C == an.near_cm and d.depth_L == an.near_cm,
          f"L/C = {d.depth_L}/{d.depth_C}")

    plain = rng.integers(40, 220, size=(480, 720, 3), dtype=np.uint8)
    info2 = an.analyze(plain)
    check("全畫面有紋理 -> 不觸發", not (info2["near_L"] or info2["near_C"] or info2["near_R"]),
          f"L/C/R={info2['L']:.2f}/{info2['C']:.2f}/{info2['R']:.2f}")


def t_search_delegates_to_explorer():
    """
    SEARCH 的水平指令完全由漫遊避障決定。白柱子在左前方 (TissueDetector 把左側與中央
    覆寫成 120cm) -> 漫遊要往右轉開並後退，不能像 17:25 那趟直直撞上去。
    """
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    lr, fb, ud, yv = b.calculate_command(
        NO_INPUT, blank_vision(depth_L=120.0, depth_C=120.0, depth_R=600.0))
    lr, fb, ud, yv = b.calculate_command(
        NO_INPUT, blank_vision(depth_L=120.0, depth_C=120.0, depth_R=600.0))
    check("左前方白柱子 -> 往右轉開", yv > 0, f"yv={yv}")
    check("左前方白柱子 -> 不往前衝", fb <= 0, f"fb={fb}")
    check("漫遊不動高度", ud == 0, f"ud={ud}")


def t_watchdog_stops_drone_on_stall():
    """主迴圈卡住時看門狗要送懸停 (16:47 那趟卡 2.56 秒，fb=50 直直撞牆)"""
    import time as _t
    from core.drone_controller import DroneController

    sent = []
    dc = DroneController.__new__(DroneController)
    import threading
    dc.drone = SimpleNamespace(send_rc_control=lambda *a: sent.append(a))
    dc.is_connected = True
    dc.WATCHDOG_SEC = 0.3
    dc._rc_lock = threading.Lock()
    dc._last_rc = (0, 0, 0, 0)
    dc._last_rc_time = _t.time()
    th = threading.Thread(target=dc._watchdog, daemon=True)
    th.start()

    dc.send_movement(0, 50, 0, 0)
    _t.sleep(0.15)
    check("迴圈正常時看門狗不插手", sent == [(0, 50, 0, 0)], f"sent={sent}")
    _t.sleep(0.4)                       # 模擬主迴圈卡住
    check("卡住超過門檻 -> 送出懸停", sent[-1] == (0, 0, 0, 0), f"sent={sent}")
    n = len(sent)
    _t.sleep(0.3)
    check("懸停只送一次", len(sent) == n, f"sent={sent}")
    dc.is_connected = False
    th.join(timeout=1)


if __name__ == "__main__":
    for fn in (t_aim_point_is_75_percent_height, t_altitude_locked, t_off_path_gives_up,
               t_depth_blocks_forward, t_proportional_forward,
               t_not_crossing_path_blocks_forward, t_memory_decays_and_spares_pid,
               t_coast_then_backoff, t_far_loss_goes_straight_to_search,
               t_long_search_does_not_kill_track,
               t_hit_state_is_gone, t_log_fields,
               t_charge_not_abandoned_when_crossing,
               t_blank_surface_pillar, t_search_delegates_to_explorer,
               t_watchdog_stops_drone_on_stall):
        print(f"\n--- {fn.__name__} ---")
        fn()
    bad = [n for n, ok, _ in results if not ok]
    print("\n" + "=" * 60)
    print(f"{len(results) - len(bad)}/{len(results)} 通過"
          + (f"，未通過: {bad}" if bad else "，全部通過"))
    sys.exit(1 if bad else 0)
