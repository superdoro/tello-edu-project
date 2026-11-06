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
    覆寫成 120cm) -> 漫遊要往右轉開，不能像 17:25 那趟直直撞上去；也不後退 (10/19 改)。
    """
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    lr, fb, ud, yv = b.calculate_command(
        NO_INPUT, blank_vision(depth_L=120.0, depth_C=120.0, depth_R=600.0))
    lr, fb, ud, yv = b.calculate_command(
        NO_INPUT, blank_vision(depth_L=120.0, depth_C=120.0, depth_R=600.0))
    check("左前方白柱子 -> 往右轉開", yv > 0, f"yv={yv}")
    check("左前方白柱子 -> 原地轉 (不往前衝、不後退)", fb == 0, f"fb={fb}")
    check("漫遊不動高度", ud == 0, f"ud={ud}")


def _explore_run(ex, clk, depths, seconds, dt=0.05):
    """depths: (L, C, R) 或 lambda t -> (L, C, R)"""
    import io, contextlib
    out = []
    t0 = clk.t
    for _ in range(int(round(seconds / dt))):
        L, C, R = depths(clk.t - t0) if callable(depths) else depths
        with contextlib.redirect_stdout(io.StringIO()):
            out.append((ex.state, ex.calculate_command(NO_INPUT, blank_vision(depth_L=L, depth_C=C, depth_R=R))))
        clk.tick(dt)
    return out


def t_explorer_random_turn():
    """
    漫遊避障 (10/19 改)：遇到障礙不後退，往比較空的一側轉一個隨機角度 (45~120 度) 再前進。
    """
    import random
    from behaviors.fluid_explore import FluidExploreControl
    dt = 0.05

    # 任何深度組合都不後退
    clk = Clock()
    ex = FluidExploreControl(now_fn=clk, rng=random.Random(1))
    rnd = random.Random(7)
    seq = [(rnd.uniform(40, 800), rnd.uniform(40, 800), rnd.uniform(40, 800)) for _ in range(40)]
    out = _explore_run(ex, clk, lambda t: seq[min(int(t / 0.5), len(seq) - 1)], 20.0)
    check("漫遊 20 秒隨機障礙 -> 從不後退", all(c[1] >= 0 for _, c in out),
          f"最小 fb={min(c[1] for _, c in out)}")

    # 右邊比較空 -> 往右轉；轉的角度落在 45~120 度
    angles = []
    for seed in range(12):
        clk = Clock()
        ex = FluidExploreControl(now_fn=clk, rng=random.Random(seed))
        out = _explore_run(ex, clk, lambda t: (100, 200, 600) if t < 0.1 else (100, 900, 600), 3.0)
        turn = [c for s_, c in out if s_ == "TURN"]
        angles.append(sum(c[3] for c in turn) * dt)
        if seed == 0:
            check("右邊比較空 -> 往右轉", turn and all(c[3] > 0 for c in turn[:-1]),     # 最後一格是轉完回直線
                  f"yv={set(c[3] for c in turn)}")
            check("前方還有 2m -> 邊走邊轉 (弧線)", turn and turn[0][1] == ex.TURN_ARC_FB, f"fb={turn[0][1]}")
            check("轉完前方開闊 -> 回到直線前進", out[-1][0] == "FORWARD" and out[-1][1][1] == ex.FORWARD_SPEED,
                  f"{out[-1]}")
    check("轉的角度在 45~120 度之間", all(45 - 4 <= a <= 120 + 4 for a in angles),
          f"{[round(a) for a in angles]}")
    check("每次轉的角度不一樣 (隨機)", max(angles) - min(angles) > 20, f"{[round(a) for a in angles]}")

    # 左右差不多 -> 方向隨機，兩邊都會出現
    dirs = set()
    for seed in range(12):
        clk = Clock()
        ex = FluidExploreControl(now_fn=clk, rng=random.Random(seed))
        out = _explore_run(ex, clk, (400, 200, 420), 0.1)
        dirs |= {1 if c[3] > 0 else -1 for s_, c in out if s_ == "TURN" and c[3]}
    check("左右一樣空 -> 方向隨機 (左右都會出現)", dirs == {-1, 1}, f"方向 {dirs}")

    # 太近 -> 原地轉；轉完還擋著 -> 同方向再轉一次 (不左右來回)
    clk = Clock()
    ex = FluidExploreControl(now_fn=clk, rng=random.Random(3))
    out = _explore_run(ex, clk, (600, 100, 150), 6.0)
    turn = [c for s_, c in out if s_ == "TURN"]
    check("前方 1m 內 -> 原地轉 (fb=0)", turn and all(c[1] == 0 for c in turn), f"fb={set(c[1] for c in turn)}")
    check("一直擋著 -> 同一個方向持續轉", turn and all(c[3] < 0 for c in turn) and len(turn) == len(out) - 1,
          f"yv={set(c[3] for c in turn)}")

    # 偏好方向 (紙條模式剛往右避開有色紙條) -> 就算左邊比較空也往右
    clk = Clock()
    ex = FluidExploreControl(now_fn=clk, rng=random.Random(0))
    ex.prefer_side(1, 5.0)
    out = _explore_run(ex, clk, (900, 200, 300), 0.1)
    check("偏好方向有效期內 -> 照偏好方向轉", any(c[3] > 0 for _, c in out), f"{out}")
    clk.tick(6.0)
    ex.change_state("FORWARD")
    out = _explore_run(ex, clk, (900, 200, 300), 0.1)
    check("偏好方向過期 -> 回到往空的一側轉", any(c[3] < 0 for _, c in out), f"{out}")


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


# ==========================================
# 撞白色 / 撞彩色 (C 鍵)
# ==========================================
PURPLE = (200, 70, 120)        # BGR，接近 10/13 錄影裡紙條上的紫色條紋 (OpenCV 色相約 130)


def _scene(strip_bgr=(245, 245, 245), bands=(), band_bgr=PURPLE, x=350, w=18, top=10, bottom=250,
           bg=None, seed=0):
    """
    合成畫面：淺灰牆 (帶一點雜訊) 前面掛一條紙條。
    預設是已經夠近、可以判斷顏色的紙條 (厚度 18px -> 估距 1383/18 = 77cm < JUDGE_DIST_CM)。
    bands: 紙條上的彩色條紋，每段是 (起點比例, 終點比例)，沿紙條由上往下量
    bg: 框範圍內的背景顏色 (模擬紙條後面是彩色的牆 / 櫃子)
    回傳 (畫面, YOLO 框, 分割輪廓)
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(175, 6, size=(480, 720, 3)), 0, 255).astype(np.uint8)
    if bg is not None:
        img[top - 5:bottom + 5, x - 8:x + w + 8] = bg
    img[top:bottom, x:x + w] = strip_bgr
    length = bottom - top
    for a, b in bands:
        img[top + int(a * length):top + int(b * length), x:x + w] = band_bgr
    box = (x + w // 2, (top + bottom) // 2, w + 8, length + 10)       # YOLO 框比紙條略大
    poly = np.array([[x, top], [x + w, top], [x + w, bottom], [x, bottom]], dtype=np.float32)
    return img, box, poly


def _cand(box, poly):
    return dict(cx=box[0], cy=box[1], w=box[2], h=box[3], poly=poly)


def _detectors():
    """不載入任何模型的偵測器 (白色模式、彩色模式各一)，只用來呼叫它的驗證流程"""
    import io, contextlib
    from vision.tissue_detector import TissueDetector, TARGET_WHITE, TARGET_COLOUR
    with contextlib.redirect_stdout(io.StringIO()):
        return tuple(TissueDetector(yolo_path="/nonexistent.pt", depth_model_path=None, target=t)
                     for t in (TARGET_WHITE, TARGET_COLOUR))


STRIPED = ((0.10, 0.20), (0.45, 0.55), (0.80, 0.90))     # 三段紫色條紋，共佔 30%


def t_colour_verifiers():
    """
    偵測器實際使用的驗證流程對各種合成紙條的判定。
    一定要透過偵測器的 _verify 測：白色模式的「有彩色條紋就擋」是在那裡加的。
    """
    from utils.white_strip import MATCH, NO_MATCH, UNKNOWN
    det_w, det_c = _detectors()
    cases = [
        # 名稱,                          畫面參數,                                    白色模式, 彩色模式
        ("純白紙條",                     dict(),                                       MATCH,    NO_MATCH),
        ("白紙條 + 三段紫色條紋 (實際場地)", dict(bands=STRIPED),                     NO_MATCH, MATCH),
        ("白紙條 + 一段紫色條紋 (15%)",  dict(bands=((0.45, 0.60),)),                NO_MATCH, MATCH),
        ("整條紅色紙條",                 dict(strip_bgr=(40, 40, 210)),                NO_MATCH, MATCH),
        ("整條深藍色紙條 (比牆暗)",      dict(strip_bgr=(150, 60, 20)),                NO_MATCH, MATCH),
        ("紅藍兩色 (像畫/海報)",         dict(strip_bgr=(40, 40, 210),
                                              bands=((0.5, 1.0),), band_bgr=(210, 60, 20)), None, NO_MATCH),
        ("白紙條掛在藍色櫃子前",         dict(bg=(200, 90, 30)),                       None,     NO_MATCH),
    ]
    for name, kw, want_w, want_c in cases:
        img, box, poly = _scene(**kw)
        info_w = det_w._verify(img, _cand(box, poly))
        info_c = det_c._verify(img, _cand(box, poly))
        if want_w is not None:
            check(f"{name} -> 白色模式 {'接受' if want_w == MATCH else '擋掉'}",
                  info_w["verdict"] == want_w, f"verdict={info_w['verdict']} ({info_w['reason']})")
        check(f"{name} -> 彩色模式 {'接受' if want_c == MATCH else '擋掉'}",
              info_c["verdict"] == want_c,
              f"verdict={info_c['verdict']} ({info_c['reason']}) 彩色列 {info_c['coloured_rows']:.2f} "
              f"色相集中 {info_c['hue_conc']:.2f}")

    # 沒有分割輪廓 (偵測模型不是分割模型) 時退回看框中央，條紋紙條一樣要分得出來
    img, box, _ = _scene(bands=STRIPED)
    info = det_c._verify(img, _cand(box, None))
    check("沒有分割輪廓 -> 退回看框中央，仍認得出條紋", info["verdict"] == MATCH and info["source"] == "centre",
          f"verdict={info['verdict']} source={info['source']}")

    # 紙條太小 (太遠) -> 不下結論，不在看不清楚的時候投票
    img, box, poly = _scene(bands=STRIPED, w=2, top=100, bottom=118)
    info = det_c._verify(img, _cand(box, poly))
    check("紙條太小 -> 彩色模式不下結論", info["verdict"] == UNKNOWN, f"verdict={info['verdict']} ({info['reason']})")

    # 紅色的色相橫跨 0 度兩側 (OpenCV 的 2 與 177)，圓形統計要把它當成同一個顏色
    import cv2
    img, box, poly = _scene(strip_bgr=(40, 40, 210))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[160:230, 350:362, 0] = 177
    info = det_c._verify(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), _cand(box, poly))
    check("紅色跨 0 度仍算單一顏色", info["verdict"] == MATCH and info["hue_conc"] > 0.95,
          f"色相集中 {info['hue_conc']:.2f}")

    # 偏暖光源下的白紙條 (稍微泛黃) 不可以被當成彩色
    img, box, poly = _scene(strip_bgr=(200, 225, 235))
    info = det_c._verify(img, _cand(box, poly))
    check("泛黃的白紙條 -> 彩色模式擋掉", info["verdict"] == NO_MATCH, f"reason={info['reason']}")


def _fake_model(get_box_poly):
    """假的 YOLO 分割模型：每一格回傳 get_box_poly() 給的那一條紙條 (框 + 輪廓)"""
    class _T(list):
        def tolist(self):
            return list(self)

    def model(frame, **kw):
        box, poly = get_box_poly()
        cx, cy, w, h = box
        b = SimpleNamespace(cls=[0], conf=[0.9], xywh=[_T([cx, cy, w, h])])
        return [SimpleNamespace(boxes=[b], masks=SimpleNamespace(xy=[poly]))]
    return model


def _make_detector(target):
    import io, contextlib
    from vision.tissue_detector import TissueDetector
    clk = Clock()
    with contextlib.redirect_stdout(io.StringIO()):
        det = TissueDetector(yolo_path="/nonexistent.pt", depth_model_path=None,
                             now_fn=clk, target=target)
    return det, clk


def _run(det, clk, img, box, poly, frames=10):
    """用假模型把整條偵測流程 (驗證 -> 投票 -> 確認 -> 鎖定) 跑幾格"""
    import io, contextlib
    det.model = _fake_model(lambda: (box, poly))
    data = None
    for _ in range(frames):
        with contextlib.redirect_stdout(io.StringIO()):
            data = det.process_frame(img.copy())
        clk.tick(0.05)
    return data


def t_detector_target_colour():
    """整條偵測流程：白色模式不鎖彩色條紋紙條、彩色模式只鎖它；C 鍵切換會清掉舊的投票"""
    import io, contextlib
    from vision.tissue_detector import TARGET_WHITE, TARGET_COLOUR
    white = _scene()
    striped = _scene(bands=STRIPED)

    for target, scene, want, name in (
            (TARGET_WHITE, white, True, "白色模式 -> 鎖定純白紙條"),
            (TARGET_WHITE, striped, False, "白色模式 -> 不鎖定有紫色條紋的紙條 (10/13 實飛撞的就是它)"),
            (TARGET_COLOUR, striped, True, "彩色模式 -> 鎖定有紫色條紋的紙條"),
            (TARGET_COLOUR, white, False, "彩色模式 -> 不鎖定純白紙條")):
        det, clk = _make_detector(target)
        d = _run(det, clk, *scene)
        locked = d.tissue is not None
        check(name, locked == want, f"tissue={'有' if locked else '無'} rejected={d.n_rejected}")
        if locked:
            check(f"{name} -> 紀錄的目標顏色正確", d.tissue.get("target") == det.target_tag,
                  f"target={d.tissue.get('target')}")

    # 遠處看起來是白的 (條紋還看不清楚) -> 白色模式確認並鎖定；
    # 靠近之後看到條紋 -> 要撤銷，不可以因為「確認是終態」就照樣衝過去
    det, clk = _make_detector(TARGET_WHITE)
    d = _run(det, clk, *white)
    check("遠處看起來是白的 -> 先確認鎖定", d.tissue is not None, f"tissue={'有' if d.tissue else '無'}")
    d = _run(det, clk, *striped, frames=6)
    check("靠近後看到條紋 -> 撤銷鎖定", d.tissue is None and d.n_rejected >= 1,
          f"tissue={'有' if d.tissue else '無'} rejected={d.n_rejected}")

    # 真的白紙條偶爾一格量到雜訊，不可以被撤銷
    det, clk = _make_detector(TARGET_WHITE)
    d = _run(det, clk, *white)
    d = _run(det, clk, *striped, frames=1)
    d = _run(det, clk, *white, frames=8)
    check("單格雜訊不會撤銷白紙條", d.tissue is not None, f"tissue={'有' if d.tissue else '無'}")

    # 白色模式先把條紋紙條列入黑名單，按 C 切換後要能馬上重新評估並鎖定它
    det, clk = _make_detector(TARGET_WHITE)
    _run(det, clk, *striped)
    with contextlib.redirect_stdout(io.StringIO()):
        det.toggle_target()
    check("C 鍵切換 -> 換成彩色驗證器、清掉投票與鎖定",
          det.target == TARGET_COLOUR and det.verifier is det.colour_verifier
          and not det._slots and det.lock_cx is None,
          f"target={det.target} slots={len(det._slots)}")
    d = _run(det, clk, *striped)
    check("切換後不被舊的黑名單擋住 -> 鎖定條紋紙條", d.tissue is not None,
          f"tissue={'有' if d.tissue else '無'}")
    with contextlib.redirect_stdout(io.StringIO()):
        det.toggle_target()
    check("再按一次 C -> 回到白色", det.target == TARGET_WHITE and det.verifier is det.white_verifier,
          f"target={det.target}")

    try:
        _make_detector("purple")
        check("不合法的 target 要報錯", False, "沒有報錯")
    except ValueError:
        check("不合法的 target 要報錯", True)


# ==========================================
# 閃避不該撞的紙條
# ==========================================
def _two_strips(x_white, x_striped):
    """一條純白、一條有紫色條紋的紙條掛在同一個畫面"""
    img, box_w, poly_w = _scene(x=x_white)
    img2, box_s, poly_s = _scene(x=x_striped, bands=STRIPED)
    img[:, x_striped:x_striped + 18] = img2[:, x_striped:x_striped + 18]
    return img, [(box_w, poly_w), (box_s, poly_s)]


def _run_multi(det, clk, img, strips, frames=10, cls=0):
    """假模型一次回傳多條紙條 (cls：模型類別編號，兩類模型的 1 = paper_red_marked)"""
    import io, contextlib

    class _T(list):
        def tolist(self):
            return list(self)

    def model(frame, **kw):
        boxes = [SimpleNamespace(cls=[cls], conf=[0.9], xywh=[_T(list(b))]) for b, _ in strips]
        return [SimpleNamespace(boxes=boxes, masks=SimpleNamespace(xy=[p for _, p in strips]))]
    det.model = model
    data = None
    for _ in range(frames):
        with contextlib.redirect_stdout(io.StringIO()):
            data = det.process_frame(img.copy())
        clk.tick(0.05)
    return data


def t_detector_avoid_list():
    """顏色不符的紙條要放進 data.avoid；該撞的那條才是 data.tissue"""
    from vision.tissue_detector import TARGET_WHITE, TARGET_COLOUR
    img, strips = _two_strips(250, 450)                 # 兩條都在畫面中間那 1/3 (兩旁的不理)
    for target, want_hit, want_avoid in ((TARGET_WHITE, 259, 459), (TARGET_COLOUR, 459, 259)):
        det, clk = _make_detector(target)
        d = _run_multi(det, clk, img, strips)
        hit = d.tissue["cx"] if d.tissue else None
        avoid = [a["cx"] for a in d.avoid]
        tag = "白色" if target == TARGET_WHITE else "彩色"
        check(f"{tag}模式 -> 撞對的那條", hit == want_hit, f"tissue cx={hit} (應為 {want_hit})")
        check(f"{tag}模式 -> 另一條放進閃避清單", avoid == [want_avoid], f"avoid={avoid}")

    # 兩條紙條水平只差 80px (小於 LOCK_GATE 120)：不可以併成同一個 slot
    img, strips = _two_strips(330, 410)
    det, clk = _make_detector(TARGET_WHITE)
    d = _run_multi(det, clk, img, strips)
    check("兩條很近的紙條 -> 分開判斷", d.tissue is not None and d.tissue["cx"] == 339
          and [a["cx"] for a in d.avoid] == [419],
          f"tissue={d.tissue['cx'] if d.tissue else None} avoid={[a['cx'] for a in d.avoid]}")

    # 只要還看得到，就一直在閃避清單上 (不會因為 5 秒冷卻到期而出現空窗)
    det, clk = _make_detector(TARGET_WHITE)
    gaps = 0
    for _ in range(160):                          # 8 秒
        d = _run_multi(det, clk, img, strips, frames=1)
        gaps += not d.avoid
    check("看得到就一直閃避 (8 秒內沒有空窗)", gaps <= 10, f"不在清單上的格數 {gaps} (前幾格是投票期)")


def _avoid_vision(strips, **kw):
    v = blank_vision(**kw)
    v.avoid = strips
    return v


def _strip(cx, dist, top=0.0, bottom=300.0):
    w = 1383.0 / dist
    return {"cx": cx, "cy": (top + bottom) / 2, "w": w, "h": bottom - top,
            "top_y": top, "bottom_y": bottom, "dist_cm": dist}


def t_search_avoids_strips():
    """漫遊時：擋在前進路線上的不該撞紙條要繞開；不在中間的 (旁邊、頭頂上方) 不用理"""
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    lr, fb, ud, yv = b.calculate_command(NO_INPUT, _avoid_vision([]))
    check("沒有要閃的紙條 -> 直直漫遊", yv == 0 and fb > 0, f"fb={fb} yv={yv}")

    b = TissueChargeControl(now_fn=clk)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        v = _avoid_vision([_strip(360, 200)])
        lr, fb, ud, yv = b.calculate_command(NO_INPUT, v)
    check("正前方 2m 有不該撞的紙條 -> 平移繞開 (不轉、不後退)",
          lr != 0 and yv == 0 and fb == 0 and b.state == "DODGE",
          f"lr={lr} fb={fb} yv={yv} state={b.state}")

    b = TissueChargeControl(now_fn=clk)
    v = _avoid_vision([_strip(150, 180)])          # 左側 180cm，不在機身會掃到的範圍
    lr, fb, ud, yv = b.calculate_command(NO_INPUT, v)
    check("側邊不擋路的紙條 -> 不理它、繼續漫遊", b.state == "SEARCH" and fb > 0 and yv == 0,
          f"state={b.state} fb={fb} yv={yv}")

    b = TissueChargeControl(now_fn=clk)
    v = _avoid_vision([_strip(360, 200, top=0.0, bottom=40.0)])
    lr, fb, ud, yv = b.calculate_command(NO_INPUT, v)
    check("紙條整條在頭頂上方 -> 不用閃", yv == 0 and fb > 0 and v.depth_C == 999.0,
          f"fb={fb} yv={yv}")

    b = TissueChargeControl(now_fn=clk)
    v = _avoid_vision([_strip(80, 400)])           # 遠處左側：不在中間
    lr, fb, ud, yv = b.calculate_command(NO_INPUT, v)
    check("不在中間的紙條 -> 完全不理 (不改深度、直直漫遊)",
          v.depth_L == 999.0 and v.depth_C == 999.0 and v.depth_R == 999.0
          and b.state == "SEARCH" and fb > 0 and yv == 0 and lr == 0,
          f"L={v.depth_L:.0f} C={v.depth_C:.0f} R={v.depth_R:.0f} state={b.state} cmd={(lr, fb, yv)}")
    check("閃避距離寫進 log", b.log_fields().get("avoid") == 400, f"avoid={b.log_fields().get('avoid')}")


def t_track_blocked_by_strip():
    """
    追目標時：非白紙條擋在通往白紙條的路線上 -> 先用視差判斷前後 (不再用寬度推算的距離)；
    不擋路的 (在旁邊、在頭頂上方) 照常衝。
    """
    clk = Clock()

    def tgt(strips):
        v = make_vision(cx=360, w=10, bottom_y=300.0)          # 目標估距 138cm
        v.avoid = strips
        return v

    import io, contextlib
    for name, st in (("看起來比較近", _strip(365, 70)), ("看起來比較遠", _strip(360, 300))):
        b = TissueChargeControl(now_fn=clk); b.change_state("TRACK")
        with contextlib.redirect_stdout(io.StringIO()):
            lr, fb, _, _ = b.calculate_command(NO_INPUT, tgt([st]))
        check(f"擋路的非白紙條 ({name}) -> 停下做視差探測", b.state == "PROBE" and fb == 0 and lr != 0
              and b.ignore_until <= clk.t, f"state={b.state} lr={lr} fb={fb}")

    for name, st in (("在旁邊", _strip(600, 80)), ("在頭頂上方", _strip(360, 80, top=0.0, bottom=40.0))):
        b = TissueChargeControl(now_fn=clk); b.change_state("TRACK")
        _, fb, _, _ = b.calculate_command(NO_INPUT, tgt([st]))
        check(f"不該撞的紙條{name} -> 照常衝", fb > 0 and b.state == "TRACK", f"fb={fb} state={b.state}")



def _dodge_run(b, clk, vision_fn, seconds, dt=0.05):
    out = []
    for _ in range(int(round(seconds / dt))):
        out.append((b.state, b.calculate_command(NO_INPUT, vision_fn())))
        clk.tick(dt)
    return out


def t_dodge_non_target():
    """
    正前方 (機身會掃到) 有不該撞的紙條：往空曠的一側平移，紙條離開機身會掃到的範圍就交還漫遊、
    照原方向繼續往前 (從旁邊繞過去)；不在中間的不理。
    兩側都很近、或平移太久繞不開才原地轉。任何情況都不後退。
    """
    import io, contextlib, random
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    strip_left = _strip(330, 180)                  # 正前方略偏左 180cm (機身會掃到)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        out = _dodge_run(b, clk, lambda: _avoid_vision([strip_left]), 0.05)
    check("正前方有不該撞的紙條 -> 進入 DODGE", b.state == "DODGE" and b.dodge_mode == "slide",
          f"state={b.state} mode={b.dodge_mode}")
    check("終端機有提示", "平移繞過去" in log.getvalue(), log.getvalue().strip())
    check("畫面中央有大字提示", b.alert_text() is not None, f"alert={b.alert_text()}")
    check("兩側一樣空、紙條在左 -> 往右平移 (機頭不動、不前進不後退)",
          out[0][1][0] > 0 and out[0][1][1] == 0 and out[0][1][3] == 0, f"cmd={out[0][1]}")

    # 紙條在畫面上往左移 (無人機往右平移)，離開機身會掃到的範圍 (+ 遲滯) 就交還漫遊
    clear = b._dodge_clear_px(strip_left)
    with contextlib.redirect_stdout(io.StringIO()):
        out = _dodge_run(b, clk, lambda: _avoid_vision([_strip(360 - clear + 3, 180)]), 0.1)
        check("還差一點 -> 繼續平移", b.state == "DODGE" and out[-1][1][0] > 0, f"state={b.state}")
        out = _dodge_run(b, clk, lambda: _avoid_vision([_strip(360 - clear - 1, 180)]), 0.05)
    check("紙條離開機身會掃到的範圍 -> 交還漫遊", b.state == "SEARCH" and b.alert_text() is None,
          f"state={b.state}")
    with contextlib.redirect_stdout(io.StringIO()):
        out = _dodge_run(b, clk, lambda: _avoid_vision([_strip(360 - clear - 1, 180)]), 0.5)
    check("繞開之後照原方向往前 (紙條還看得到也不再閃)", all(s_ == "SEARCH" and c[1] > 0 and c[3] == 0
                                                         for s_, c in out), f"{set(out)}")

    # 往空曠的一側：右邊近、左邊空 -> 就算紙條偏左也往左平移
    b = TissueChargeControl(now_fn=clk)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        out = _dodge_run(b, clk, lambda: _avoid_vision([strip_left], depth_L=400, depth_R=200), 0.05)
    check("左邊比較空 -> 往左平移", out[0][1][0] < 0 and out[0][1][3] == 0 and "比較空" in log.getvalue(),
          f"cmd={out[0][1]}")

    # 紙條一直擋著 (平移太久繞不開) -> 改成原地往同一側轉
    b = TissueChargeControl(now_fn=clk)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        out = _dodge_run(b, clk, lambda: _avoid_vision([strip_left]), b.DODGE_SLIDE_MAX_SEC + 0.3)
    slide = [c for s_, c in out if c[0] != 0]
    turn = [c for s_, c in out if c[3] != 0]
    check("平移太久繞不開 -> 改原地往同一側轉", slide and turn and all(c[3] > 0 for c in turn)
          and len(slide) * 0.05 >= b.DODGE_SLIDE_MAX_SEC - 0.05 and "繞不開" in log.getvalue(),
          f"平移 {len(slide) * 0.05:.2f}s 轉 {len(turn)} 格")

    # 平移中紙條看不到了 -> 至少平移 DODGE_BLIND_SEC 才交還
    b = TissueChargeControl(now_fn=clk)
    with contextlib.redirect_stdout(io.StringIO()):
        _dodge_run(b, clk, lambda: _avoid_vision([strip_left]), 0.05)
        out = _dodge_run(b, clk, lambda: _avoid_vision([]), 2.0)
    slid = sum(1 for s_, c in out if s_ == "DODGE" and c[0] > 0) * 0.05 + 0.05
    check("紙條看不到 -> 平移約 DODGE_BLIND_SEC 再交還", b.state == "SEARCH"
          and b.DODGE_BLIND_SEC - 0.05 <= slid <= b.DODGE_BLIND_SEC + 0.15, f"平移 {slid:.2f}s")

    # 兩側都很近 (平移過去也是撞牆) -> 原地往比較空的一側轉 35~75 度，轉完有冷卻
    b = TissueChargeControl(now_fn=clk)
    b._rng = random.Random(0)
    with contextlib.redirect_stdout(io.StringIO()):
        out = _dodge_run(b, clk, lambda: _avoid_vision([strip_left], depth_L=100, depth_R=140), 2.0)
    turn = [c for s_, c in out if s_ == "DODGE" and c[3] != 0]
    deg = sum(c[3] for c in turn) * 0.05
    check("兩側都很近 -> 原地轉、不平移", turn and all(c[0] == 0 for _, c in out), f"lr={set(c[0] for _, c in out)}")
    check("往比較空的一側 (右) 轉 35~75 度", all(c[3] > 0 for c in turn) and 35 - 4 <= deg <= 75 + 4,
          f"轉了約 {deg:.0f} 度")
    check("轉完回到漫遊、之後轉彎優先往同一側",
          out[-1][0] == "SEARCH" and b.explorer.bias_dir == 1 and b.explorer.bias_until > clk.t,
          f"state={out[-1][0]} bias={b.explorer.bias_dir}")
    with contextlib.redirect_stdout(io.StringIO()):
        out = _dodge_run(b, clk, lambda: _avoid_vision([strip_left], depth_L=100, depth_R=140), 0.5)
    check("原地轉之後冷卻期內不重複觸發", all(s_ != "DODGE" for s_, _ in out), f"states={set(s_ for s_, _ in out)}")
    degs = []
    for seed in range(8):
        b = TissueChargeControl(now_fn=clk)
        b._rng = random.Random(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            out = _dodge_run(b, clk, lambda: _avoid_vision([strip_left], depth_L=100, depth_R=140), 1.6)
        degs.append(sum(c[3] for s_, c in out if s_ == "DODGE") * 0.05)
    check("原地轉的角度每次不一樣 (隨機)", max(degs) - min(degs) > 10, f"{[round(d) for d in degs]}")

    # 很近 (60cm) 紙條在右 -> 往左平移，一樣不後退
    b = TissueChargeControl(now_fn=clk)
    with contextlib.redirect_stdout(io.StringIO()):
        out = _dodge_run(b, clk, lambda: _avoid_vision([_strip(400, 60)]), 1.5)
    check("很近、紙條在右 -> 往左平移", out[0][1][0] < 0 and out[0][1][3] == 0, f"cmd={out[0][1]}")
    check("很近也不後退", all(c[1] >= 0 for _, c in out), f"fb={set(c[1] for _, c in out)}")

    # 不擋路的不觸發：側邊、在頭頂上方、或比 DODGE_DIST 遠
    for name, st in (("在側邊", _strip(150, 180)), ("在頭頂上方", _strip(360, 150, top=0.0, bottom=40.0)),
                     ("很遠", _strip(360, 400))):
        b = TissueChargeControl(now_fn=clk)
        with contextlib.redirect_stdout(io.StringIO()):
            b.calculate_command(NO_INPUT, _avoid_vision([st]))
        check(f"不該撞的紙條{name} -> 不避開", b.state != "DODGE", f"state={b.state}")

    # 避開途中飛手按鍵 -> 立刻交還
    b = TissueChargeControl(now_fn=clk)
    with contextlib.redirect_stdout(io.StringIO()):
        b.calculate_command(NO_INPUT, _avoid_vision([strip_left]))
        cmd = b.calculate_command(SimpleNamespace(lr=0, fb=-30, ud=0, yv=0, voice_command=None),
                                  _avoid_vision([strip_left]))
    check("避開途中按鍵 -> 飛手接管", cmd == (0, -30, 0, 0) and b.state == "SEARCH", f"cmd={cmd}")


def t_detector_avoid_feedback():
    """畫面回饋：看到不該撞的紙條時，頂端有洋紅色大橫幅"""
    import numpy as np
    from vision.tissue_detector import TARGET_WHITE
    img, strips = _two_strips(200, 450)
    det, clk = _make_detector(TARGET_WHITE)
    d = _run_multi(det, clk, img, strips)
    banner = d.annotated_frame[42:75, 200:520]
    magenta = np.all(banner == (255, 0, 255), axis=2).mean()
    check("看到非白紙條 -> 頂端出現洋紅色橫幅", d.avoid and magenta > 0.3, f"橫幅洋紅比例 {magenta:.2f}")
    check("閃避清單帶有目標顏色", d.avoid and d.avoid[0].get("target") == "WHITE",
          f"target={d.avoid[0].get('target') if d.avoid else None}")
    d2 = _run_multi(*_make_detector(TARGET_WHITE), _scene()[0], [_scene()[1:]])
    banner2 = d2.annotated_frame[42:75, 200:520]
    check("只有白紙條 -> 沒有橫幅", np.all(banner2 == (255, 0, 255), axis=2).mean() < 0.05, "")



def t_detector_avoid_needs_evidence():
    """
    只有真的看得出顏色的才列入閃避 (10/19 20:46 錄影：牆角、柱子被當成有色紙條，一直被嚇退)；
    碰到畫面左右邊緣的框也不列入。
    """
    from vision.tissue_detector import TARGET_WHITE, TARGET_COLOUR
    PINK = (180, 120, 250)       # 整條粉紅色紙條
    GREY, YELLOW = (150, 150, 150), (40, 220, 250)

    # 只有背景有顏色：灰色的東西掛在亮黃色牆前 —— 白色驗證器說「不夠白」(亮的那一半是牆)，
    # 紙條輪廓裡卻沒有顏色 -> 沒有正面證據，不撞也不閃
    for target in (TARGET_WHITE, TARGET_COLOUR):
        img, box, poly = _scene(strip_bgr=GREY, bg=YELLOW)
        det, clk = _make_detector(target)
        d = _run_multi(det, clk, img, [(box, poly)])
        tag = "白色" if target == TARGET_WHITE else "彩色"
        check(f"{tag}模式：只有背景有顏色 -> 不撞也不閃", d.tissue is None and not d.avoid,
              f"tissue={d.tissue is not None} avoid={len(d.avoid)}")

    # 飽和度約 60 的米色：10/20 校正之後算有顏色 (淡紫色麥克筆條紋也只有 40~60；
    # 實際錄影裡的牆角、門框 coloured_rows 最多 0.06，照樣算沒顏色)
    img, box, poly = _scene(strip_bgr=(180, 210, 235))
    det, clk = _make_detector(TARGET_WHITE)
    d = _run_multi(det, clk, img, [(box, poly)])
    check("白色模式：飽和度約 60 的米色紙條 -> 不撞、列入閃避", d.tissue is None and len(d.avoid) == 1,
          f"tissue={d.tissue is not None} avoid={len(d.avoid)}")

    img, box, poly = _scene(strip_bgr=PINK)
    det, clk = _make_detector(TARGET_WHITE)
    d = _run_multi(det, clk, img, [(box, poly)])
    check("白色模式：整條粉紅色紙條 -> 列入閃避", d.tissue is None and len(d.avoid) == 1,
          f"tissue={d.tissue is not None} avoid={len(d.avoid)}")

    img, box, poly = _scene(x=0, bands=STRIPED)
    det, clk = _make_detector(TARGET_WHITE)
    d = _run_multi(det, clk, img, [(box, poly)])
    check("條紋紙條碰到畫面左緣 -> 不列入閃避", not d.avoid, f"avoid={len(d.avoid)}")
    img, box, poly = _scene(x=708, bands=STRIPED)
    det, clk = _make_detector(TARGET_WHITE)
    d = _run_multi(det, clk, img, [(box, poly)])
    check("條紋紙條碰到畫面右緣 -> 不列入閃避", not d.avoid, f"avoid={len(d.avoid)}")


def t_detector_side_and_colour_rules():
    """
    10/20 22:47 錄影：
      畫面兩旁 (左右各 1/3) 的不該撞紙條不理 —— 不進閃避清單、沒有橫幅；
      近距離條紋把白色切斷 (形狀理由 a1.x 否決) 的條紋紙條要認成有色、列入閃避；
      投票視窗裡只要有一格看到彩色，就不能確認成白的 (3 白 2 彩的條紋紙條以前會被確認)。
    """
    import numpy as np
    from vision.tissue_detector import TARGET_WHITE
    for x, where, want in ((90, "左邊 1/3", 0), (560, "右邊 1/3", 0), (350, "中間 1/3", 1)):
        img, box, poly = _scene(x=x, bands=STRIPED)
        det, clk = _make_detector(TARGET_WHITE)
        d = _run_multi(det, clk, img, [(box, poly)])
        banner = d.annotated_frame[42:75, 200:520]
        magenta = np.all(banner == (255, 0, 255), axis=2).mean()
        check(f"條紋紙條在畫面{where} -> {'列入閃避' if want else '不理 (不閃、沒有橫幅)'}",
              len(d.avoid) == want and (magenta > 0.3) == bool(want) and d.tissue is None,
              f"avoid={len(d.avoid)} 橫幅={magenta:.2f} tissue={d.tissue is not None}")

    # 五段條紋：白色被切成很短的幾段 -> 白色驗證器用形狀理由否決 -> 看得到彩色就認成有色紙條
    many = tuple((a, a + 0.1) for a in (0.0, 0.2, 0.4, 0.6, 0.8))
    img, box, poly = _scene(bands=many)
    det, clk = _make_detector(TARGET_WHITE)
    info = det._verify(img, _cand(box, poly))
    d = _run_multi(det, clk, img, [(box, poly)])
    check("條紋把白色切斷 (形狀不像) 的紙條 -> 理由是 colour、列入閃避",
          info["reason"] == "colour" and len(d.avoid) == 1, f"reason={info['reason']} avoid={len(d.avoid)}")

    # 一格白、一格看得到條紋輪流出現 -> 以前 5 格裡 3 白就確認，現在不確認、最後否決
    white, striped = _scene(), _scene(bands=STRIPED)
    det, clk = _make_detector(TARGET_WHITE)
    states = []
    for i in range(10):
        img = (white if i % 2 == 0 else striped)[0]          # 同一個位置，畫面輪流
        _run_multi(det, clk, img, [(white[1], white[2])], frames=1)
        states.append(det._slots[0].state)
    check("白、條紋輪流出現 -> 不會被確認成白的", "CONFIRMED" not in states and states[-1] == "REJECTED",
          f"states={states}")


def t_revoked_target_no_coast():
    """
    鎖定的白紙條靠近後看到條紋被撤銷 -> 不可以當成「貼到眼前」滑行過去 (10/20 22:47 錄影滑進條紋紙條)；
    在正前方就平移繞開。單純跟丟 (沒有撤銷) 照舊滑行。
    """
    import io, contextlib
    clk = Clock()
    for revoked, want in ((True, "DODGE"), (False, "COAST")):
        b = TissueChargeControl(now_fn=clk)
        _tc_run(b, clk, lambda: _judging_vision(dist=45, judged=True), 0.2)    # 正在衝、估距 45cm
        a = dict(_strip(360, 45), target="WHITE", revoked=revoked)
        v = _avoid_vision([a] if revoked else [])
        with contextlib.redirect_stdout(io.StringIO()) as log:
            out = [b.calculate_command(NO_INPUT, v)]
            clk.tick(0.05)
            for _ in range(int(b.MEMORY_DURATION / 0.05) + 2):
                out.append(b.calculate_command(NO_INPUT, _avoid_vision([a] if revoked else [])))
                clk.tick(0.05)
        name = "目標被撤銷 (看到條紋)" if revoked else "單純跟丟 (貼到眼前)"
        check(f"{name} -> {'平移繞開，不滑行' if revoked else '照舊滑行'}",
              b.state == want and (not revoked or "看到彩色條紋" in log.getvalue()),
              f"state={b.state} {log.getvalue().strip()[-50:]}")


def t_detector_rejected_slot_not_inherited():
    """
    10/19 20:46 錄影：被否決的有色紙條看不到之後還留在原位置 30 秒，
    無人機轉頭時經過那個位置的白紙條被配進去、繼承「要閃避」，整趟一條白紙條都沒確認過。
    """
    from vision.tissue_detector import TARGET_WHITE
    det, clk = _make_detector(TARGET_WHITE)
    img_s, box_s, poly_s = _scene(x=450, bands=STRIPED)
    d = _run_multi(det, clk, img_s, [(box_s, poly_s)])
    check("條紋紙條 -> 列入閃避", len(d.avoid) == 1, f"avoid={len(d.avoid)}")

    # 轉頭：條紋紙條離開畫面 0.5 秒，然後一條白紙條掃到差不多的位置 (差 70px，舊的閘門是 120px)
    img_e, _, _ = _scene(x=450)
    _run_multi(det, clk, img_e, [], frames=10)
    img_w, box_w, poly_w = _scene(x=520)
    d = _run_multi(det, clk, img_w, [(box_w, poly_w)])
    check("轉頭後經過舊位置的白紙條 -> 不繼承閃避，照樣確認成目標",
          d.tissue is not None and not d.avoid,
          f"tissue={d.tissue is not None} avoid={len(d.avoid)}")

    # 看不到超過 REJECT_TTL -> 否決的 slot 丟掉
    det, clk = _make_detector(TARGET_WHITE)
    _run_multi(det, clk, img_s, [(box_s, poly_s)])
    _run_multi(det, clk, img_e, [], frames=int(det.REJECT_TTL / 0.05) + 2)
    check("否決的紙條看不到超過 REJECT_TTL -> 丟掉", not any(s.state == "REJECTED" for s in det._slots),
          f"slots={[(s.state, s.reject_reason) for s in det._slots]}")

    # 短暫閃爍 (原地、看不到 0.5 秒) -> 黑名單照樣有效
    det, clk = _make_detector(TARGET_WHITE)
    _run_multi(det, clk, img_s, [(box_s, poly_s)])
    _run_multi(det, clk, img_e, [], frames=10)
    d = _run_multi(det, clk, img_s, [(box_s, poly_s)], frames=1)
    check("原地短暫看不到再出現 -> 馬上又在閃避清單上", len(d.avoid) == 1, f"avoid={len(d.avoid)}")


# ==========================================
# 兩類模型、近距離才判斷
# ==========================================
def t_detector_two_classes():
    """paper_two_class 模型：paper_plain (0) 與 paper_red_marked (1) 都是紙條，兩類都要收"""
    import io, contextlib
    from vision.tissue_detector import TissueDetector, TARGET_WHITE
    img, box, poly = _scene()
    det, clk = _make_detector(TARGET_WHITE)
    det.class_names = {0: "paper_plain", 1: "paper_red_marked"}
    d = _run_multi(det, clk, img, [(box, poly)], cls=1)
    check("class 1 (paper_red_marked) 的白紙條 -> 照樣驗證、鎖定", d.tissue is not None and d.tissue["judged"],
          f"tissue={d.tissue is not None}")
    check("類別名稱帶出去 (畫面 / log)", d.tissue is not None and d.tissue.get("cls") == "paper_red_marked",
          f"cls={d.tissue.get('cls') if d.tissue else None}")

    clk = Clock()
    with contextlib.redirect_stdout(io.StringIO()):
        det = TissueDetector(yolo_path="/nonexistent.pt", depth_model_path=None, now_fn=clk,
                             tissue_class_ids=0)
    d = _run_multi(det, clk, img, [(box, poly)], cls=1)
    check("指定只收 class 0 -> class 1 不收", d.tissue is None, f"tissue={d.tissue is not None}")

    import os
    path = "model/yolo26/runs/detect/yolo_tissue_collect/best.pt"
    if os.path.exists(path):
        clk = Clock()
        with contextlib.redirect_stdout(io.StringIO()) as log:
            det = TissueDetector(depth_model_path=None, now_fn=clk)
        check("預設載入 best.pt (單類別 tissue)、所有類別都收",
              "best.pt" in log.getvalue() and det.tissue_class_ids is None
              and list(det.class_names.values()) == ["tissue"], f"classes={det.class_names}")


def _shrink(box, poly, keep):
    """只留紙條上段 keep 比例 (框與輪廓一起縮)：YOLO 只切到一段紙條。厚度不變 = 距離不變"""
    import numpy as np
    x0, y0 = float(poly[:, 0].min()), float(poly[:, 1].min())
    x1, y1 = float(poly[:, 0].max()), float(poly[:, 1].max())
    y1 = y0 + (y1 - y0) * keep
    p = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    return (box[0], int((y0 + y1) / 2), box[2], int(y1 - y0 + 10)), p


def _thin(box, poly, px):
    """把紙條變細 = 變遠 (框與輪廓一起，中心不動)。輪廓 px 寬 -> 厚度約 px+2 -> 估距約 1383/(px+2)"""
    import numpy as np
    xc = float(poly[:, 0].min() + poly[:, 0].max()) / 2.0
    y0, y1 = float(poly[:, 1].min()), float(poly[:, 1].max())
    p = np.array([[xc - px / 2, y0], [xc + px / 2, y0], [xc + px / 2, y1], [xc - px / 2, y1]],
                 dtype=np.float32)
    return (box[0], box[1], int(px + 8), box[3]), p


def t_detector_judges_close_only():
    """
    不論顏色先接近：估距 (紙條厚度) 還超過 JUDGE_DIST_CM 就不投票 (顏色常判錯)，
    當成還沒判斷的接近目標；夠近才判斷。
    """
    import numpy as np
    from vision.tissue_detector import TARGET_WHITE
    for name, bands in (("白紙條", ()), ("條紋紙條", STRIPED)):
        img, box, poly = _scene(bands=bands)
        det, clk = _make_detector(TARGET_WHITE)
        d = _run_multi(det, clk, img, [_thin(box, poly, 6)])
        slot = det._slots[0] if det._slots else None
        check(f"遠處的{name} -> 不投票", slot is not None and not slot.votes and slot.state == "PENDING",
              f"votes={len(slot.votes) if slot else None} state={slot.state if slot else None}")
        check(f"遠處的{name} -> 當成還沒判斷的接近目標", d.tissue is not None and not d.tissue["judged"]
              and d.tissue["far"] and not d.avoid,
              f"tissue={d.tissue and (d.tissue['judged'], d.tissue['far'])} avoid={len(d.avoid)}")
        check(f"遠處的{name} -> 估距用紙條厚度", d.tissue is not None and 150 <= d.tissue["dist_cm"] <= 200,
              f"dist={d.tissue and round(d.tissue['dist_cm'])} strip_px={d.tissue and d.tissue['strip_px']}")

        d = _run_multi(det, clk, img, [(box, poly)])         # 飛近了
        if not bands:
            check("夠近的白紙條 -> 確認成目標", d.tissue is not None and d.tissue["judged"],
                  f"tissue={d.tissue and d.tissue['judged']}")
        else:
            check("夠近的條紋紙條 -> 列入閃避", d.tissue is None and len(d.avoid) == 1,
                  f"tissue={d.tissue is not None} avoid={len(d.avoid)}")

    # 10/20 錄影：扭轉、斜掛的遠處紙條框很寬 (框寬 20px，用框寬估只有 69cm)，
    # 但輪廓很細 -> 估距用厚度，還很遠，不判斷
    img, box, _ = _scene(bands=STRIPED)
    thin = np.array([[357, 10], [361, 10], [361, 250], [357, 250]], dtype=np.float32)
    det, clk = _make_detector(TARGET_WHITE)
    d = _run_multi(det, clk, img, [(box, thin)])
    check("框很寬但輪廓很細的條紋紙條 -> 不判斷、先接近",
          d.tissue is not None and not d.tissue["judged"] and not d.avoid and not det._slots[0].votes,
          f"tissue={d.tissue and d.tissue['judged']} avoid={len(d.avoid)} votes={len(det._slots[0].votes)}")

    # 已經開始投票的，估距跳遠一點 (紙條扭轉，還在 JUDGE_KEEP_CM 以內) 也繼續驗
    img, box, poly = _scene(bands=STRIPED)
    det, clk = _make_detector(TARGET_WHITE)
    _run_multi(det, clk, img, [(box, poly)], frames=2)
    d = _run_multi(det, clk, img, [_thin(box, poly, 14)], frames=8)
    check("投票中估距跳到 80~100cm -> 繼續判斷完", len(d.avoid) == 1, f"avoid={len(d.avoid)}")
    det, clk = _make_detector(TARGET_WHITE)
    _run_multi(det, clk, img, [_thin(box, poly, 14)], frames=8)
    check("還沒開始投票、估距 80~100cm -> 不開始判斷", not det._slots[0].votes,
          f"votes={len(det._slots[0].votes)}")

    # 單格輪廓暴衝讓遠處的紙條開始投票，之後又變回很細 -> 不可以一路投下去
    det, clk = _make_detector(TARGET_WHITE)
    _run_multi(det, clk, img, [(box, poly)], frames=1)
    d = _run_multi(det, clk, img, [_thin(box, poly, 6)], frames=20)
    check("單格暴衝之後變回很遠 -> 停止投票，還是接近目標",
          d.tissue is not None and not d.tissue["judged"] and d.tissue["far"] and not d.avoid,
          f"tissue={d.tissue and (d.tissue['judged'], d.tissue['far'])} avoid={len(d.avoid)}")
    # 反過來：遠處的紙條單格輪廓變粗 -> 中位數擋掉，連一票都不投
    det, clk = _make_detector(TARGET_WHITE)
    _run_multi(det, clk, img, [_thin(box, poly, 6)], frames=5)
    _run_multi(det, clk, img, [(box, poly)], frames=1)
    check("遠處的紙條單格輪廓變粗 -> 中位數擋掉，不投票", not det._slots[0].votes,
          f"votes={len(det._slots[0].votes)}")


def t_detector_thickness_distance():
    """
    估距用紙條厚度 (10/20 22:20 錄影)：
    YOLO 只切到一段紙條時面積跳 3 倍，厚度不變；斜掛時框寬好幾倍，厚度不變；
    畫面邊緣一出現就貼在眼前的大框不當紙條 (已鎖定的那條除外)。
    """
    import math
    import numpy as np
    from vision.tissue_detector import TARGET_WHITE
    img, box, poly = _scene()
    det, clk = _make_detector(TARGET_WHITE)
    full = _run_multi(det, clk, img, [(box, poly)], frames=3).tissue["dist_cm"]
    det, clk = _make_detector(TARGET_WHITE)
    part = _run_multi(det, clk, img, [_shrink(box, poly, 0.35)], frames=3).tissue["dist_cm"]
    check("只切到上面 35% 的紙條 -> 估距跟整條一樣", abs(part - full) / full < 0.1,
          f"整條 {full:.0f}cm  只切到一段 {part:.0f}cm")

    # 斜 10 度：框寬變成 3 倍多，厚度不變
    a = math.radians(10)
    cx, cy, half_w, half_h = 359.0, 130.0, 9.0, 120.0
    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    tilt = np.array([[cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)]
                     for x, y in corners], dtype=np.float32)
    bw = int(tilt[:, 0].max() - tilt[:, 0].min())
    bh = int(tilt[:, 1].max() - tilt[:, 1].min())
    det, clk = _make_detector(TARGET_WHITE)
    d = _run_multi(det, clk, img, [((int(cx), int(cy), bw, bh), tilt)], frames=3)
    check("斜掛的紙條 -> 估距跟正掛的差不多 (框寬估的會差好幾倍)",
          d.tissue is not None and abs(d.tissue["dist_cm"] - full) / full < 0.15,
          f"正掛 {full:.0f}cm  斜掛 {d.tissue and round(d.tissue['dist_cm'])}cm  框寬估 {1383 / bw:.0f}cm")

    # 畫面右緣、一出現就貼在眼前 (厚度 60px -> 23cm) 的大框 -> 不當紙條
    edge_poly = np.array([[660, 10], [719, 10], [719, 250], [660, 250]], dtype=np.float32)
    edge_box = (690, 130, 60, 250)
    det, clk = _make_detector(TARGET_WHITE)
    d = _run_multi(det, clk, img, [(edge_box, edge_poly)], frames=3)
    check("畫面邊緣一出現就貼在眼前的大框 -> 不當紙條", d.tissue is None and d.n_edge >= 1,
          f"tissue={d.tissue is not None} edge={d.n_edge}")
    # 一路追近的那條長大到碰邊 -> 照樣追
    img_r, box_r, poly_r = _scene(x=665)
    det, clk = _make_detector(TARGET_WHITE)
    d = _run_multi(det, clk, img_r, [(box_r, poly_r)])
    locked = d.tissue is not None and d.tissue["judged"]
    d = _run_multi(det, clk, img_r, [(edge_box, edge_poly)], frames=4)   # 中位數要幾格才跟上
    check("已鎖定的紙條近到碰邊 -> 照樣追", locked and d.tissue is not None and d.tissue["dist_cm"] < 30,
          f"鎖定={locked} tissue={d.tissue and round(d.tissue['dist_cm'])}")


def t_detector_holds_confirmed_lock():
    """
    已確認的目標只是單格沒抓到 (YOLO 閃爍)，畫面裡又有還沒判斷的紙條：
    先撐 LOCK_MISS_MAX 格，不要馬上改追那條 (10/20 錄影重播：機頭在兩條之間甩了 8 次)
    """
    from vision.tissue_detector import TARGET_WHITE
    img, strips = _two_strips(350, 150)
    white, small = strips[0], _thin(*strips[1], 6)
    det, clk = _make_detector(TARGET_WHITE)
    d = _run_multi(det, clk, img, [white, small])
    check("白紙條確認、另一條還很遠", d.tissue is not None and d.tissue["judged"] and d.tissue["cx"] == 359,
          f"tissue={d.tissue and (d.tissue['cx'], d.tissue['judged'])}")
    d = _run_multi(det, clk, img, [small], frames=1)
    check("確認的目標單格沒抓到 -> 不改追還沒判斷的那條", d.tissue is None,
          f"tissue={d.tissue and (d.tissue['cx'], d.tissue['judged'])}")
    d = _run_multi(det, clk, img, [white, small], frames=1)
    check("目標回來 -> 照樣追", d.tissue is not None and d.tissue["cx"] == 359,
          f"tissue={d.tissue and d.tissue['cx']}")
    d = _run_multi(det, clk, img, [small], frames=det.LOCK_MISS_MAX + 1)
    check("目標真的不見了 (超過 LOCK_MISS_MAX 格) -> 改去接近還沒判斷的那條",
          d.tissue is not None and not d.tissue["judged"] and d.tissue["cx"] == 159,
          f"tissue={d.tissue and (d.tissue['cx'], d.tissue['judged'])}")


def _judging_vision(cx=360, dist=110, judged=False, far=False, avoid=(), box_w=None):
    """估距 dist (紙條厚度換算)；box_w 給了就讓框寬跟厚度不一樣 (斜掛的紙條框很寬)"""
    strip_px = 1383.0 / dist
    v = make_vision(cx=cx, w=box_w if box_w else strip_px, bottom_y=300.0)
    v.tissue.update({"judged": judged, "far": far, "target": "WHITE",
                     "strip_px": strip_px, "dist_cm": dist})
    v.avoid = list(avoid)
    return v


def _tc_run(b, clk, vision_fn, seconds, dt=0.05):
    import io, contextlib
    out = []
    with contextlib.redirect_stdout(io.StringIO()) as log:
        for _ in range(int(round(seconds / dt))):
            out.append((b.state, b.calculate_command(NO_INPUT, vision_fn())))
            clk.tick(dt)
    return out, log.getvalue()


def t_approach_then_judge():
    """
    還沒判斷的紙條：遠處照常飛過去、快到時放慢、進判斷距離停住看清楚；
    是目標就衝，不是就轉開；看太久判斷不出來或判斷中跟丟也轉開。絕不對還沒判斷的紙條滑行。
    """
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    out, _ = _tc_run(b, clk, lambda: _judging_vision(dist=300, far=True), 0.1)
    check("遠處還沒判斷的紙條 -> 飛過去 (巡航速度)", b.state == "TRACK" and out[-1][1][1] == b.FB_FAR,
          f"state={b.state} fb={out[-1][1][1]} mode={b.get_mode()}")
    check("狀態列顯示 APPROACH 與估距", "APPROACH d300cm" in b.get_mode(), b.get_mode())
    out, _ = _tc_run(b, clk, lambda: _judging_vision(dist=300, far=True, box_w=60), 0.1)
    check("框很寬 (斜掛)、但厚度估距還很遠 -> 照巡航速度接近", out[-1][1][1] == b.FB_FAR,
          f"fb={out[-1][1][1]}")
    out, _ = _tc_run(b, clk, lambda: _judging_vision(dist=110, far=True), 0.1)
    check("快到判斷距離 (JUDGE_SLOW_CM 以內) -> 放慢", out[-1][1][1] == b.FB_NEAR, f"fb={out[-1][1][1]}")
    out, _ = _tc_run(b, clk, lambda: _judging_vision(cx=400, dist=75), 0.5)
    check("到判斷距離 -> 停住 (不前進) 但繼續轉機頭對準", all(c[1] == 0 for _, c in out) and out[-1][1][3] > 0,
          f"fb={set(c[1] for _, c in out)} yv={out[-1][1][3]}")
    check("畫面提示 CHECKING TARGET", b.alert_text() == "CHECKING TARGET", f"alert={b.alert_text()}")
    out, log = _tc_run(b, clk, lambda: _judging_vision(dist=110, judged=True), 0.2)
    check("判定是目標 -> 衝", out[-1][1][1] > 0 and b.state == "TRACK" and "是目標" in log,
          f"fb={out[-1][1][1]} state={b.state}")

    # 判定不是目標：它從目標變成閃避清單上、正前方的紙條 -> 轉開，判斷開始後完全沒有前進
    b = TissueChargeControl(now_fn=clk)
    out, _ = _tc_run(b, clk, lambda: _judging_vision(dist=110), 0.4)
    v_rej = lambda: _avoid_vision([dict(_strip(365, 110), target="WHITE")])
    out2, log = _tc_run(b, clk, v_rej, 0.1)
    check("近距離判定不是目標 -> 轉開", b.state == "DODGE" and "近距離看清楚" in log, f"state={b.state} {log.strip()}")
    out3, _ = _tc_run(b, clk, lambda: blank_vision(), 2.0)
    check("判斷開始之後完全沒有前進 (直到交還漫遊)",
          all(c[1] <= 0 for st, c in out + out2 + out3 if st != "SEARCH"),
          f"fb={set(c[1] for st, c in out + out2 + out3 if st != 'SEARCH')}")
    check("轉開之後暫時不接近還沒判斷的紙條", b.approach_ignore_until > clk.t, "")
    out, _ = _tc_run(b, clk, lambda: _judging_vision(dist=300, far=True), 0.5)
    check("冷卻中看到還沒判斷的紙條 -> 繼續漫遊", all(st == "SEARCH" for st, _ in out),
          f"states={set(st for st, _ in out)}")
    out, _ = _tc_run(b, clk, lambda: _judging_vision(dist=110, judged=True), 0.1)
    check("冷卻中看到已確認的目標 -> 照樣追", b.state == "TRACK", f"state={b.state}")

    # 看太久判斷不出來 -> 轉開
    b = TissueChargeControl(now_fn=clk)
    out, log = _tc_run(b, clk, lambda: _judging_vision(dist=110), b.JUDGE_MAX_SEC + 0.3)
    check("判斷太久沒結果 -> 轉開", "DODGE" in [st for st, _ in out] and "判斷不出來" in log,
          f"states={sorted(set(st for st, _ in out))}")

    # 判斷中跟丟：先停住等，等不到就往遠離它的一側平移繞開 (紙條在右 -> 往左)
    b = TissueChargeControl(now_fn=clk)
    _tc_run(b, clk, lambda: _judging_vision(cx=400, dist=110), 0.3)
    out, log = _tc_run(b, clk, lambda: blank_vision(), 1.0)
    early = [c for st, c in out[:int(b.JUDGE_LOST_SEC / 0.05) - 1]]
    check("判斷中短暫跟丟 -> 停住等 (不前進)", all(c[1] == 0 for c in early), f"{set(early)}")
    dodge = [c for st, c in out if st == "DODGE" and c[0] != 0]
    check("等不到、紙條在正前方 -> 往遠離它的一側平移繞開",
          "跟丟" in log and dodge and all(c[0] < 0 and c[3] == 0 for c in dodge),
          f"cmd={set(dodge)} {log.strip()[-60:]}")

    # 判斷中跟丟、但紙條不在中間 (機身掃不到) -> 不用閃，直接回漫遊
    b = TissueChargeControl(now_fn=clk)
    _tc_run(b, clk, lambda: _judging_vision(cx=560, dist=110), 0.3)
    out, log = _tc_run(b, clk, lambda: blank_vision(), 1.0)
    check("等不到、紙條不在中間 -> 不閃、回漫遊", all(st != "DODGE" for st, _ in out) and b.state == "SEARCH"
          and "不在正前方" in log and b.approach_ignore_until > clk.t,
          f"states={set(st for st, _ in out)} {log.strip()[-50:]}")

    # 還沒判斷的紙條絕不滑行 (就算剛剛在前進、估距又很近)
    b = TissueChargeControl(now_fn=clk)
    out, _ = _tc_run(b, clk, lambda: _judging_vision(dist=55, far=True), 0.2)
    out2, _ = _tc_run(b, clk, lambda: blank_vision(), 1.0)
    check("還沒判斷的紙條跟丟 -> 不滑行", out[-1][1][1] > 0 and all(st != "COAST" for st, _ in out2),
          f"跟丟前 fb={out[-1][1][1]} states={set(st for st, _ in out2)}")

    # 斜掛的紙條框很寬 (框寬估 23cm)，但厚度估距 110cm -> 跟丟時不滑行 (10/20 22:20 62.8 秒對空氣滑行)
    b = TissueChargeControl(now_fn=clk)
    out, _ = _tc_run(b, clk, lambda: _judging_vision(dist=110, judged=True, box_w=60), 0.2)
    out2, _ = _tc_run(b, clk, lambda: blank_vision(), 1.0)
    check("框很寬但厚度估距 110cm 的目標跟丟 -> 不滑行", out[-1][1][1] > 0 and all(st != "COAST" for st, _ in out2),
          f"跟丟前 fb={out[-1][1][1]} states={set(st for st, _ in out2)}")

    # 機身會掃到的範圍、繞開的餘裕都用紙條厚度當比例尺，不用框寬
    b = TissueChargeControl(now_fn=clk)
    diag = {"cx": 300, "w": 60, "strip_px": 1383.0 / 150}       # 斜掛：框寬 60px，厚度估 150cm
    straight = {"cx": 300, "w": 1383.0 / 150}
    check("斜掛紙條的機身範圍 = 正掛的 (不被框寬放大)",
          abs(b._clear_px(diag) - b._clear_px(straight)) < 0.01 and b._clear_px(diag) < 60,
          f"斜掛 {b._clear_px(diag):.0f}px 正掛 {b._clear_px(straight):.0f}px")

    # 滑行中畫面裡出現還沒判斷的別條紙條 (後面遠處的) -> 不打斷滑行；確認的目標回來才回 TRACK
    for name, vis, want in (("還沒判斷的別條紙條", lambda: _judging_vision(cx=200, dist=300, far=True), "COAST"),
                            ("確認的目標又出現", lambda: _judging_vision(dist=40, judged=True), "TRACK")):
        b = TissueChargeControl(now_fn=clk)
        _tc_run(b, clk, lambda: _judging_vision(dist=40, judged=True), 0.1)
        _tc_run(b, clk, lambda: blank_vision(), b.MEMORY_DURATION + 0.1)
        coasting = b.state == "COAST"
        out, _ = _tc_run(b, clk, vis, 0.1)
        check(f"滑行中看到{name} -> {want}", coasting and out[-1][0] == want and b.state == want,
              f"滑行={coasting} state={b.state}")


# ==========================================
# YOLO 吃 Tello 原生畫面 (960x720)
# ==========================================
def t_native_resolution_yolo():
    """
    有原生畫面時 YOLO 吃原生畫面 (imgsz 960)，框與輪廓換算回 720x480；
    沒給原生畫面時退回用 720x480 (離線重播舊錄影)。
    """
    import io, contextlib, cv2
    import numpy as np
    from vision.tissue_detector import TARGET_WHITE

    # 原生畫面：淺灰牆 + 一條白紙條 (原生座標 x=480~496, y=100~340)
    rng = np.random.default_rng(0)
    native = np.clip(rng.normal(175, 6, size=(720, 960, 3)), 0, 255).astype(np.uint8)
    native[100:340, 480:496] = 245
    small = cv2.resize(native, (720, 480), interpolation=cv2.INTER_AREA)
    nbox = (488, 220, 24, 250)
    npoly = np.array([[480, 100], [496, 100], [496, 340], [480, 340]], np.float32)

    seen = []

    class _T(list):
        def tolist(self):
            return list(self)

    def model(frame, **kw):
        seen.append((frame.shape[:2], kw.get("imgsz")))
        b = SimpleNamespace(cls=[0], conf=[0.9], xywh=[_T(list(nbox))])
        return [SimpleNamespace(boxes=[b], masks=SimpleNamespace(xy=[npoly]))]

    det, clk = _make_detector(TARGET_WHITE)
    det.model = model
    for _ in range(10):
        det.set_native_frame(native)
        with contextlib.redirect_stdout(io.StringIO()):
            d = det.process_frame(small.copy())
        clk.tick(0.05)
    check("YOLO 吃的是原生畫面、imgsz 960", seen[-1] == ((720, 960), 960), f"{seen[-1]}")
    check("輸入解析度記錄下來", getattr(d, "yolo_input", "") == "960x720", f"{getattr(d, 'yolo_input', '')}")
    t = d.tissue
    check("框換算回 720x480 座標", t is not None and (t["cx"], t["cy"], t["w"]) == (366, 146, 18),
          f"cx/cy/w = {(t['cx'], t['cy'], t['w']) if t else None} (應為 366/146/18)")
    # 紙條原生寬 16px = 720 畫面 12px -> 1383/12 = 115cm (框故意比紙條寬，框寬估的會是 77cm)
    check("估距用原生畫面量的紙條厚度 (不是框寬)", t is not None and abs(t["dist_cm"] - 115) / 115 < 0.12,
          f"dist={t['dist_cm'] if t else None:.1f} (紙條 115cm，框寬估 77cm)")
    check("白紙條照樣通過驗證並鎖定", t is not None and t.get("target") == "WHITE", "")

    # 下一格沒有給原生畫面 -> 用 720x480，不可以沿用上一格的原生畫面
    with contextlib.redirect_stdout(io.StringIO()):
        d = det.process_frame(small.copy())
    check("沒給原生畫面 -> 退回 720x480", seen[-1][0] == (480, 720) and d.yolo_input == "720x480",
          f"{seen[-1]}")


def t_recorder_native_video():
    """錄影時有原生畫面就多存一支 _native.mp4 (960x720)，沒有就不產生"""
    import os, tempfile, time as _t, cv2
    import numpy as np
    from utils.video_recorder import VideoRecorder
    tmp = tempfile.mkdtemp(dir="/tmp/claude-1000/-home-lin-Desktop-UAV-project-tello-edu-project/c444c350-d858-46b8-9a9c-c1fe1e04b59e/scratchpad")
    for with_native in (True, False):
        rec = VideoRecorder()
        rec.OUT_DIR = tmp
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            rec.start("TEST_NATIVE" if with_native else "TEST_PLAIN")
            for i in range(10):
                small = np.full((480, 720, 3), i * 20, np.uint8)
                nat = np.full((720, 960, 3), i * 20, np.uint8) if with_native else None
                rec.write(small, small.copy(), {"mode": "T"}, native=nat)
                _t.sleep(0.03)
            rec.stop()
        path = rec.base_name + "_native.mp4"
        if with_native:
            cap = cv2.VideoCapture(path)
            size = (int(cap.get(3)), int(cap.get(4)))
            check("有原生畫面 -> 產生 960x720 的 _native.mp4", os.path.exists(path) and size == (960, 720),
                  f"exists={os.path.exists(path)} size={size}")
        else:
            check("沒有原生畫面 -> 不產生 _native.mp4", not os.path.exists(path), "")


def t_video_stream_config():
    """連線時設定 720P / 30fps / 5Mbps；韌體不支援某一項也不可以中斷連線"""
    import io, contextlib
    from djitellopy import tello as _tello
    from core.drone_controller import DroneController
    calls = []

    def ok(name):
        return lambda v: calls.append((name, v))

    def fail(v):
        raise Exception("error")

    dc = DroneController.__new__(DroneController)
    dc.drone = SimpleNamespace(set_video_resolution=ok("res"), set_video_fps=ok("fps"),
                               set_video_bitrate=ok("bitrate"))
    with contextlib.redirect_stdout(io.StringIO()):
        dc._configure_video()
    check("設定解析度 / 幀率 / 位元率",
          calls == [("res", _tello.Tello.RESOLUTION_720P), ("fps", _tello.Tello.FPS_30),
                    ("bitrate", _tello.Tello.BITRATE_5MBPS)], f"{calls}")
    calls.clear()
    dc.drone.set_video_bitrate = fail
    with contextlib.redirect_stdout(io.StringIO()) as out:
        dc._configure_video()
    check("韌體不支援 -> 印警告、繼續", len(calls) == 2 and "警告" in out.getvalue(), out.getvalue().strip())




# ==========================================
# 視差判斷前後：2D 閉迴路模擬
# ==========================================
def _parallax_world(target, avoid, seconds=14.0, yaw_drift=0.0, dt=0.05):
    """
    俯視的 2D 世界：無人機在 (y 橫向, z 前方)，機頭角 psi (右轉為正)。
    target / avoid 是紙條的 (橫向, 前方) 位置 (公分)，紙條寬 3cm、從天花板垂到路線下方。
    回傳 (經過的狀態, 有沒有剪到白紙條, 離非白紙條最近的距離, 視差結論)
    """
    import io, contextlib, math
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    b.change_state("TRACK")
    y = z = psi = 0.0
    states, cut, min_gap = [], False, 1e9

    def project(X, Z):
        dx, dz = X - y, Z - z
        f = dz * math.cos(psi) + dx * math.sin(psi)
        r = dx * math.cos(psi) - dz * math.sin(psi)
        if f < 25:                       # 太近 / 已經在後面：看不到
            return None
        cx = 360 + 461.0 * r / f
        if not 0 <= cx < 720:
            return None
        return cx, 1383.0 / f

    out = io.StringIO()
    for _ in range(int(seconds / dt)):
        t_img, a_img = project(*target), project(*avoid)
        if t_img:
            v = make_vision(cx=int(t_img[0]), w=max(1, int(round(t_img[1]))), bottom_y=300.0)
        else:
            v = blank_vision()
        v.avoid = ([{"cx": a_img[0], "cy": 150.0, "w": a_img[1], "h": 300.0, "top_y": 0.0,
                     "bottom_y": 300.0, "dist_cm": 1383.0 / a_img[1], "target": "WHITE"}]
                   if a_img else [])
        with contextlib.redirect_stdout(out):
            lr, fb, ud, yv = b.calculate_command(NO_INPUT, v)
        if not states or states[-1] != b.state:
            states.append(b.state)
        psi += math.radians(yv + yaw_drift) * dt
        z_prev, y_prev = z, y
        y += (fb * math.sin(psi) + lr * math.cos(psi)) * dt
        z += (fb * math.cos(psi) - lr * math.sin(psi)) * dt
        # 剪到白紙條：穿過白紙條那個深度時，橫向在 8cm 以內
        if (z_prev - target[1]) * (z - target[1]) <= 0 and abs(y - target[0]) < 8:
            cut = True
        min_gap = min(min_gap, math.hypot(y - avoid[0], z - avoid[1]))
        clk.tick(dt)
        if cut and b.state == "SEARCH":
            break
    verdicts = [line.split("視差：")[1].split(" ->")[0] for line in out.getvalue().splitlines()
                if "視差：" in line]
    return states, cut, min_gap, verdicts


def t_parallax_closed_loop():
    """白紙條與非白紙條排成一直線時，靠視差分前後：在前面就繞路、在後面就直接衝"""
    BODY = 12 + 1.5                  # 機身半寬 + 紙條半寬：小於這個就是擦到
    cases = [
        # 名稱,                        白紙條,     非白紙條,   要不要繞路
        ("非白紙條在前面 (擋路)",       (0, 220),  (0, 110),   True),
        ("非白紙條在後面、隔很遠",       (0, 120),  (0, 280),   False),
        ("非白紙條在後面、只隔 40cm",    (0, 150),  (0, 190),   True),
        ("非白紙條在前面、稍微偏一點",   (10, 230), (-5, 120),  True),
    ]
    for name, target, avoid, want_detour in cases:
        states, cut, gap, verdicts = _parallax_world(target, avoid)
        detoured = "DETOUR" in states
        check(f"{name} -> 有做視差探測", "PROBE" in states, f"states={states}")
        check(f"{name} -> {'繞路' if want_detour else '直接衝'}", detoured == want_detour,
              f"視差結論 {verdicts} states={states}")
        check(f"{name} -> 剪到白紙條", cut, f"states={states}")
        check(f"{name} -> 沒擦到非白紙條", gap > BODY, f"最近 {gap:.1f}cm")
        check(f"{name} -> 沒有放棄白紙條", "DODGE" not in states, f"states={states}")

    # 平移時機頭偏移 (每秒 6 度) -> 兩條一起移動的量會在相減時抵消，判斷照樣正確
    states, cut, gap, verdicts = _parallax_world((0, 220), (0, 110), yaw_drift=6.0)
    check("機頭偏移時 -> 仍判斷非白紙條在前面", verdicts and "在前面" in verdicts[0],
          f"視差結論 {verdicts}")
    check("機頭偏移時 -> 繞路後剪到白紙條、沒擦到非白紙條", cut and gap > BODY, f"cut={cut} 最近 {gap:.1f}cm")

    # 沒有擋路 (非白紙條在旁邊) -> 不探測，直接衝
    states, cut, gap, _ = _parallax_world((0, 200), (90, 150))
    check("非白紙條在旁邊 -> 不做視差探測、直接剪", "PROBE" not in states and cut, f"states={states}")


def _dodge_world(avoid, depth_L=999.0, depth_R=999.0, seconds=8.0, dt=0.05):
    """
    俯視的 2D 世界：漫遊直直往前，正前方掛一條不該撞的紙條 avoid = (橫向, 前方) 公分。
    回傳 (經過的狀態, 經過紙條那個深度時的橫向距離, 用掉的偏航總量)
    """
    import io, contextlib, math
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    y = z = psi = 0.0
    states, pass_gap, yaw_used = [], None, 0.0
    for _ in range(int(seconds / dt)):
        dx, dz = avoid[0] - y, avoid[1] - z
        f = dz * math.cos(psi) + dx * math.sin(psi)
        r = dx * math.cos(psi) - dz * math.sin(psi)
        v = blank_vision(depth_L=depth_L, depth_R=depth_R)
        cx = 360 + 461.0 * r / f if f > 25 else -1
        v.avoid = ([{"cx": cx, "cy": 150.0, "w": 1383.0 / f, "h": 300.0, "top_y": 0.0,
                     "bottom_y": 300.0, "dist_cm": f, "target": "WHITE"}] if 0 <= cx < 720 else [])
        with contextlib.redirect_stdout(io.StringIO()):
            lr, fb, ud, yv = b.calculate_command(NO_INPUT, v)
        if not states or states[-1] != b.state:
            states.append(b.state)
        yaw_used += abs(yv) * dt
        psi += math.radians(yv) * dt
        z_prev = z
        y += (fb * math.sin(psi) + lr * math.cos(psi)) * dt
        z += (fb * math.cos(psi) - lr * math.sin(psi)) * dt
        if pass_gap is None and (z_prev - avoid[1]) * (z - avoid[1]) <= 0:
            pass_gap = abs(y - avoid[0])
        clk.tick(dt)
    return states, pass_gap, yaw_used


def t_dodge_slide_closed_loop():
    """閉迴路：在中間的紙條平移繞過去 (機身不會掃到、不轉機頭)；不在中間的直接飛過去"""
    BODY = 12 + 1.5                  # 機身半寬 + 紙條半寬：小於這個就是擦到
    cases = [
        # 名稱,                              紙條,        深度 L / R,      要不要閃
        ("正中間 150cm",                     (0, 150),   (999, 999),     True),
        ("稍微偏左 120cm",                   (-5, 120),  (999, 999),     True),
        ("稍微偏左、但左邊比較空 (往左繞)",    (-5, 150),  (400, 200),     True),
        ("正中間 240cm",                     (0, 240),   (999, 999),     True),
        ("近在眼前 60cm、稍微偏右",           (3, 60),    (999, 999),     True),
        ("在旁邊 (不在中間)",                 (40, 150),  (999, 999),     False),
    ]
    for name, avoid, (dl, dr), want in cases:
        states, gap, yaw = _dodge_world(avoid, dl, dr)
        check(f"{name} -> {'平移繞開' if want else '不理'}", ("DODGE" in states) == want, f"states={states}")
        # 繞開的要多留 DODGE_MARGIN_CM (8cm) 給 Tello 的飄移，扣 1cm 給模擬的離散誤差
        need = BODY + (TissueChargeControl().DODGE_MARGIN_CM - 1 if want else 0)
        check(f"{name} -> 從旁邊飛過去、沒擦到 (留 {need - BODY:.0f}cm 餘裕)", gap is not None and gap > need,
              f"經過時橫向距離 {gap if gap is None else round(gap, 1)}cm")
        check(f"{name} -> 機頭沒轉", yaw == 0, f"偏航 {yaw:.0f}")


def t_search_waits_for_pending_vote():
    """漫遊時畫面裡有紙條正在投票 (可能是白紙條)，非白紙條又不是近在眼前 -> 先停住等，不急著避開"""
    import io, contextlib
    clk = Clock()
    b = TissueChargeControl(now_fn=clk)
    out = []
    for _ in range(20):                      # 1 秒
        v = _avoid_vision([_strip(340, 180)])
        v.n_pending = 1
        with contextlib.redirect_stdout(io.StringIO()):
            out.append((b.state, b.calculate_command(NO_INPUT, v)))
        clk.tick(0.05)
    first_dodge = next((i for i, (s_, _) in enumerate(out) if s_ == "DODGE"), None)
    check("有紙條在投票 -> 先停住等", out[0][1] == (0, 0, 0, 0) and out[0][0] == "SEARCH", f"{out[0]}")
    check("等太久 (0.6 秒) 才避開", first_dodge is not None and first_dodge * 0.05 >= b.PENDING_WAIT_SEC - 0.05,
          f"第 {first_dodge} 格開始避開")

    b = TissueChargeControl(now_fn=clk)
    v = _avoid_vision([_strip(340, 60)])     # 近在眼前 -> 不等
    v.n_pending = 1
    with contextlib.redirect_stdout(io.StringIO()):
        b.calculate_command(NO_INPUT, v)
    check("非白紙條近在眼前 -> 不等，直接避開", b.state == "DODGE", f"state={b.state}")



# ==========================================
# 影像凍結保護
# ==========================================
def t_video_freeze_guard():
    """
    10/19 20:21 / 20:27 兩趟：影像串流有 35% 的時間凍結，凍結期間照舊畫面盲飛撞上東西。
    畫面超過 0.3 秒沒更新 -> 自動模式懸停；手動模式與飛手按鍵不攔。
    """
    import io, contextlib, time as _t
    import numpy as np
    from core.drone_controller import DroneController
    with contextlib.redirect_stdout(io.StringIO()):
        from core.tello_app import stale_video_guard, FRAME_STALE_SEC

    # 畫面新鮮度：同一個陣列物件 = 沒有新畫面
    dc = DroneController.__new__(DroneController)
    dc.is_connected = True
    dc._last_frame_obj, dc._last_frame_time = None, _t.time()
    holder = SimpleNamespace(frame=np.zeros((720, 960, 3), np.uint8))
    dc.drone = SimpleNamespace(get_frame_read=lambda: holder)
    dc.get_video_frame()
    _t.sleep(0.15)
    dc.get_video_frame()                       # 同一張
    age_same = dc.frame_age()
    holder.frame = np.zeros((720, 960, 3), np.uint8)
    dc.get_video_frame()                       # 新的一張
    age_new = dc.frame_age()
    check("同一張畫面 -> 畫面年齡持續增加", age_same >= 0.14, f"{age_same:.2f}s")
    check("新畫面 -> 畫面年齡歸零", age_new < 0.02, f"{age_new:.3f}s")

    cmd = (0, 50, 0, 30)
    out, frozen = stale_video_guard(cmd, NO_INPUT, 0.5, manual_mode=False)
    check("自動模式、畫面凍結 0.5 秒 -> 懸停", out == (0, 0, 0, 0) and frozen, f"{out}")
    out, frozen = stale_video_guard(cmd, NO_INPUT, 0.1, manual_mode=False)
    check("畫面正常 -> 照送", out == cmd and not frozen, f"{out}")
    out, frozen = stale_video_guard(cmd, NO_INPUT, 2.0, manual_mode=True)
    check("手動模式 -> 不攔", out == cmd and not frozen, f"{out}")
    key = SimpleNamespace(lr=0, fb=40, ud=0, yv=0, voice_command=None)
    out, frozen = stale_video_guard((0, 40, 0, 0), key, 2.0, manual_mode=False)
    check("飛手正在按鍵 -> 不攔", out == (0, 40, 0, 0) and not frozen, f"{out}")
    check("凍結門檻 0.3 秒", FRAME_STALE_SEC == 0.3, f"{FRAME_STALE_SEC}")

    check("預設不修改影像串流設定 (5Mbps 會讓畫面凍結)", DroneController.CONFIGURE_VIDEO is False,
          f"CONFIGURE_VIDEO={DroneController.CONFIGURE_VIDEO}")



if __name__ == "__main__":
    for fn in (t_aim_point_is_75_percent_height, t_altitude_locked, t_off_path_gives_up,
               t_depth_blocks_forward, t_proportional_forward,
               t_not_crossing_path_blocks_forward, t_memory_decays_and_spares_pid,
               t_coast_then_backoff, t_far_loss_goes_straight_to_search,
               t_long_search_does_not_kill_track,
               t_hit_state_is_gone, t_log_fields,
               t_charge_not_abandoned_when_crossing,
               t_blank_surface_pillar, t_search_delegates_to_explorer, t_explorer_random_turn,
               t_watchdog_stops_drone_on_stall,
               t_colour_verifiers, t_detector_target_colour,
               t_detector_avoid_list, t_search_avoids_strips, t_track_blocked_by_strip,
               t_dodge_non_target, t_detector_avoid_feedback, t_detector_avoid_needs_evidence,
               t_detector_rejected_slot_not_inherited, t_detector_side_and_colour_rules,
               t_revoked_target_no_coast,
               t_detector_two_classes, t_detector_judges_close_only, t_detector_thickness_distance,
               t_detector_holds_confirmed_lock, t_approach_then_judge,
               t_native_resolution_yolo, t_recorder_native_video, t_video_stream_config,
               t_parallax_closed_loop, t_dodge_slide_closed_loop, t_search_waits_for_pending_vote,
               t_video_freeze_guard):
        print(f"\n--- {fn.__name__} ---")
        fn()
    bad = [n for n, ok, _ in results if not ok]
    print("\n" + "=" * 60)
    print(f"{len(results) - len(bad)}/{len(results)} 通過"
          + (f"，未通過: {bad}" if bad else "，全部通過"))
    sys.exit(1 if bad else 0)
