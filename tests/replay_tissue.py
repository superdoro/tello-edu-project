"""
離線重播工具：用錄影檔驗證紙條偵測與追蹤邏輯，不必起飛。

為什麼要這個：每次改門檻都實飛驗證，成本高、可重現性差，而且危險的參數
(例如爬升上限) 在空中才發現不對就來不及了。錄影檔裡已經有真實場景的
誤判與真目標，門檻該設多少可以直接量出來。

流程分兩段，因為影像處理很慢、套門檻很快：

    dump   跑一次 YOLO + 白色驗證器，把每個框的所有原始特徵倒成 json
           (這段慢，約 1~2 分鐘)
    sweep  在倒出來的特徵上掃門檻，即時出結果
           (這段瞬間完成，可以反覆掃)

用法：
    python tests/replay_tissue.py dump  recordings/20260923_232845_TISSUE_CHARGE_raw.mp4
    python tests/replay_tissue.py sheet <dump.json> --times 61,62,63
    python tests/replay_tissue.py sweep <dump.json> --fp 60-67 --true 8-20,75-85
    python tests/replay_tissue.py track recordings/<時間>_TISSUE_CHARGE_raw.mp4 --from 50 --to 56
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from types import SimpleNamespace

from utils.white_strip import WhiteStripVerifier, WHITE, UNKNOWN, NOT_WHITE

# 錄影是以固定 30fps 的時間軸寫出的 (見 utils/video_recorder.py)，
# 所以影格編號除以 30 就是 CSV 裡的 elapsed，兩邊可以直接對起來。
FPS = 30.0

# 掃描用的參數網格。pad_ratio 與 sat_max 會改變影像處理本身，
# 所以必須在 dump 階段就把每個組合都算過一遍；其餘門檻可以事後重算。
PAD_GRID = [0.0, 0.3, 0.6]
SAT_GRID = [50, 80, 120, 255]
# 實測發現 sat_max 是主導參數 (擋掉誤判的比例：255->0%, 120->41%, 80->67%, 50->97%)，
# 而 pad_ratio>0 有害 (保留真目標從 86% 掉到 64%)，所以細掃時把網格改成這兩組。


def parse_ranges(text):
    """'60-67,75-85' -> [(60.0, 67.0), (75.0, 85.0)]"""
    out = []
    if not text:
        return out
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-")
        out.append((float(a), float(b)))
    return out


def in_ranges(t, ranges):
    return any(a <= t <= b for a, b in ranges)


# ==========================================
# dump：跑 YOLO + 白色驗證器，倒出所有原始特徵
# ==========================================
def cmd_dump(args):
    from ultralytics import YOLO
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(args.model)
    model.to(device)

    # 每個 (pad_ratio, sat_max) 組合各一個驗證器 —— 這兩個參數會改變影像處理，
    # 不能事後重算，必須在這裡就全部算過
    pad_grid = args.pad_grid if args.pad_grid else PAD_GRID
    sat_grid = args.sat_grid if args.sat_grid else SAT_GRID
    verifiers = {(p, s): WhiteStripVerifier(pad_ratio=p, sat_max=s)
                 for p in pad_grid for s in sat_grid}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"打不開影片 {args.video}")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"影片 {args.video}  共 {total} 格  裝置 {device}")

    rows = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / FPS

        res = model(frame, verbose=False, device=device)[0]
        if res.boxes is not None:
            for box in res.boxes:
                if args.cls >= 0 and int(box.cls[0]) != args.cls:
                    continue
                conf = float(box.conf[0])
                if conf < args.conf:
                    continue
                cx, cy, w, h = [int(v) for v in box.xywh[0].tolist()]
                if w <= 0 or h <= 0:
                    continue

                row = {
                    "frame": idx, "t": round(t, 3),
                    "cx": cx, "cy": cy, "w": w, "h": h,
                    "conf": round(conf, 4),
                    "aspect": round(h / max(w, 1e-6), 4),
                    "feat": {},
                }
                # 每個參數組合各算一次特徵。mask 不存 (太大且事後用不到)。
                for (p, s), v in verifiers.items():
                    f = v.features(frame, (cx, cy, w, h))
                    row["feat"][f"{p}_{s}"] = {
                        "ok": f["ok"], "reason": f["reason"],
                        "otsu_sep": round(f["otsu_sep"], 2),
                        "colour_frac": round(f["colour_frac"], 4),
                        "blob_aspect": round(f["blob_aspect"], 3),
                        "blob_area_frac": round(f["blob_area_frac"], 4),
                        "blob_w_ratio": round(f["blob_w_ratio"], 3),
                        "med_s": round(f["med_s"], 1),
                        "med_v": round(f["med_v"], 1),
                    }
                rows.append(row)

        idx += 1
        if idx % 100 == 0:
            print(f"\r  {idx}/{total}  已收集 {len(rows)} 個框", end="", flush=True)
    cap.release()
    print(f"\r  {idx}/{total}  已收集 {len(rows)} 個框")

    out = {
        "video": args.video, "model": args.model, "fps": FPS,
        "n_frames": idx, "conf": args.conf,
        "pad_grid": pad_grid, "sat_grid": sat_grid,
        "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"已寫出 {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")
    return 0


# ==========================================
# sheet：把候選框的 crop 拼成一張接觸表，用來人工標真值
# ==========================================
def cmd_sheet(args):
    dump = json.load(open(args.dump))
    rows = dump["rows"]

    if args.times:
        want = [float(x) for x in args.times.split(",")]
        picked = []
        for wt in want:
            near = min(rows, key=lambda r: abs(r["t"] - wt))
            picked.append(near)
    else:
        # 沿整段影片均勻取樣
        step = max(1, len(rows) // args.n)
        picked = rows[::step][:args.n]

    cap = cv2.VideoCapture(dump["video"])
    cell_w, cell_h = 90, 200
    cols = min(len(picked), 12)
    grid_rows = (len(picked) + cols - 1) // cols
    sheet = np.full((grid_rows * (cell_h + 22), cols * (cell_w + 6), 3), 40, np.uint8)

    for i, r in enumerate(picked):
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, frame = cap.read()
        if not ok:
            continue
        x0 = max(0, r["cx"] - r["w"] // 2); x1 = min(frame.shape[1], r["cx"] + r["w"] // 2)
        y0 = max(0, r["cy"] - r["h"] // 2); y1 = min(frame.shape[0], r["cy"] + r["h"] // 2)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (cell_w, cell_h))
        gr, gc = i // cols, i % cols
        py, px = gr * (cell_h + 22), gc * (cell_w + 6)
        sheet[py:py + cell_h, px:px + cell_w] = crop
        cv2.putText(sheet, f"{r['t']:.1f}s", (px + 2, py + cell_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cap.release()
    cv2.imwrite(args.out, sheet)
    print(f"已寫出接觸表 {args.out}  ({len(picked)} 個框)")
    return 0


# ==========================================
# sweep：在倒出的特徵上掃門檻
# ==========================================
def vote_confirm(seq, window, need):
    """
    對一串判定做滑動投票，回傳「有多少格被確認」。
    UNKNOWN 不記入投票 —— 這是驗證器三值設計的重點。
    """
    buf = []
    confirmed = 0
    for v in seq:
        if v != UNKNOWN:
            buf.append(v)
            if len(buf) > window:
                buf.pop(0)
        if len(buf) == window and buf.count(WHITE) >= need:
            confirmed += 1
    return confirmed


def cmd_sweep(args):
    dump = json.load(open(args.dump))
    rows = dump["rows"]
    fp_ranges = parse_ranges(args.fp)
    true_ranges = parse_ranges(args.true)

    # 先套形狀關卡 —— 白色驗證是它的補充，不是取代，所以兩者要疊著看
    def shape_ok(r):
        return r["aspect"] >= args.min_aspect and r["w"] <= args.max_width_ratio * 720

    fp = [r for r in rows if in_ranges(r["t"], fp_ranges) and shape_ok(r)]
    tp = [r for r in rows if in_ranges(r["t"], true_ranges) and shape_ok(r)]
    print(f"通過形狀關卡 (aspect>={args.min_aspect}, w<={args.max_width_ratio*720:.0f}px) 的框：")
    print(f"  誤判區間  n={len(fp)}")
    print(f"  真目標區間 n={len(tp)}")
    if not fp or not tp:
        print("兩邊都要有樣本才能掃，請調整 --fp / --true")
        return 1

    v = WhiteStripVerifier()

    def evaluate(pad, sat, sep, asp, area, colf):
        key = f"{pad}_{sat}"
        def verdicts(group):
            out = []
            for r in group:
                f = dict(r["feat"][key])
                f["blob_box"] = None if f["blob_area_frac"] <= 0 else (0, 0, 1, 1)
                vd, _ = v.verdict_from(f, otsu_min_sep=sep, blob_aspect_min=asp,
                                       min_blob_area_frac=area, colour_frac_min=colf)
                out.append(vd)
            return out
        return verdicts(tp), verdicts(fp)

    # ---- 單格規則 ----
    print("\n" + "=" * 78)
    print("單格規則 (保留真目標 % / 擋掉誤判 %)  —— UNKNOWN 算作「保留」，因為它不下結論")
    print("=" * 78)
    print(f"{'pad':>4} {'sat':>4} {'sep':>4} {'aspect':>7} | {'保留真目標':>10} {'擋掉誤判':>9} "
          f"| {'真目標 UNK':>10} {'誤判 UNK':>9}")
    best = None
    for pad in dump["pad_grid"]:
        for sat in dump["sat_grid"]:
            for sep in args.sep_grid:
                for asp in args.aspect_grid:
                    tv, fv = evaluate(pad, sat, sep, asp, args.min_area, args.colour_frac)
                    keep = sum(1 for x in tv if x != NOT_WHITE) / len(tv)
                    rej = sum(1 for x in fv if x == NOT_WHITE) / len(fv)
                    t_unk = sum(1 for x in tv if x == UNKNOWN) / len(tv)
                    f_unk = sum(1 for x in fv if x == UNKNOWN) / len(fv)
                    score = keep + rej
                    if best is None or score > best[0]:
                        best = (score, pad, sat, sep, asp, keep, rej)
                    if args.verbose or (pad == 0.0 and sat == 80 and sep in (0, 20)):
                        print(f"{pad:>4} {sat:>4} {sep:>4} {asp:>7.1f} | "
                              f"{keep*100:>9.0f}% {rej*100:>8.0f}% | "
                              f"{t_unk*100:>9.0f}% {f_unk*100:>8.0f}%")
    print(f"\n綜合最佳： pad={best[1]} sat={best[2]} otsu_min_sep={best[3]} "
          f"blob_aspect_min={best[4]}  -> 保留 {best[5]*100:.0f}% / 擋掉 {best[6]*100:.0f}%")

    # ---- 時序投票 ----
    print("\n" + "=" * 78)
    print(f"時序投票 (pad={args.pad} sat={args.sat} sep={args.sep} aspect={args.aspect})")
    print("  被確認的格數比例：真目標越高越好，誤判越低越好")
    print("=" * 78)
    tv, fv = evaluate(args.pad, args.sat, args.sep, args.aspect, args.min_area, args.colour_frac)
    print(f"{'投票':>10} | {'真目標確認':>11} {'誤判確認':>10}")
    for window, need in [(1, 1), (3, 2), (5, 3), (5, 4), (7, 5)]:
        tc = vote_confirm(tv, window, need) / len(tv)
        fc = vote_confirm(fv, window, need) / len(fv)
        tag = f"{window}-of-{need}"
        print(f"{tag:>10} | {tc*100:>10.0f}% {fc*100:>9.0f}%")

    # ---- UNKNOWN 隨距離 (框寬) 的變化 -> 近距離是否優雅退化 ----
    print("\n" + "=" * 78)
    print("真目標的 UNKNOWN 比例 vs 框寬 (框越寬代表越近)")
    print("  近距離 UNKNOWN 升高是預期且正確的：分割器該在看不準時棄權，而不是誤判")
    print("=" * 78)
    key = f"{args.pad}_{args.sat}"
    bins = [(0, 20), (20, 40), (40, 60), (60, 100), (100, 200), (200, 999)]
    print(f"{'框寬(px)':>12} {'估距(cm)':>10} {'n':>5} {'UNKNOWN':>9} {'NOT_WHITE':>10}")
    for lo, hi in bins:
        grp = [r for r in tp if lo <= r["w"] < hi]
        if not grp:
            continue
        vs = []
        for r in grp:
            f = dict(r["feat"][key])
            f["blob_box"] = None if f["blob_area_frac"] <= 0 else (0, 0, 1, 1)
            vd, _ = v.verdict_from(f, otsu_min_sep=args.sep, blob_aspect_min=args.aspect,
                                   min_blob_area_frac=args.min_area,
                                   colour_frac_min=args.colour_frac)
            vs.append(vd)
        mid = (lo + min(hi, 300)) / 2
        print(f"{lo:>5}-{hi:<6} {1383/max(mid,1):>9.0f} {len(grp):>5} "
              f"{sum(1 for x in vs if x==UNKNOWN)/len(vs)*100:>8.0f}% "
              f"{sum(1 for x in vs if x==NOT_WHITE)/len(vs)*100:>9.0f}%")
    return 0


# ==========================================
# detect：跑完整的 TissueDetector (含投票與黑名單)，看實際會鎖定什麼
# ==========================================
def cmd_detect(args):
    """
    這是整合測試 —— 前面的 sweep 只驗證了白色驗證器本身，
    這裡驗證 slot 投票、黑名單、黏著鎖定串起來之後的實際行為。
    """
    from vision.tissue_detector import TissueDetector

    clock = {"t": 0.0}
    det = TissueDetector(depth_model_path=None, now_fn=lambda: clock["t"])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"打不開影片 {args.video}")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        clock["t"] = idx / FPS          # 假時鐘：依影格時間戳推進，投票時效才會與實飛一致
        d = det.process_frame(frame)
        t = d.tissue
        rows.append({
            "t": clock["t"],
            "locked": t is not None,
            "cx": t["cx"] if t else -1,
            "w": t["w"] if t else 0,
            "dist": t["dist_cm"] if t else 0,
            "n_conf": d.n_confirmed, "n_pend": d.n_pending, "n_rej": d.n_rejected,
            "n_shape": d.n_bad_shape,
        })
        idx += 1
        if idx % 200 == 0:
            print(f"\r  {idx}/{total}", end="", flush=True)
    cap.release()
    print(f"\r  {idx}/{total}")

    fp_ranges = parse_ranges(args.fp)
    true_ranges = parse_ranges(args.true)

    def report(name, ranges):
        g = [r for r in rows if in_ranges(r["t"], ranges)] if ranges else rows
        if not g:
            return
        lock = sum(1 for r in g if r["locked"]) / len(g)
        rej = sum(1 for r in g if r["n_rej"] > 0) / len(g)
        print(f"  {name:<22} n={len(g):<5} 鎖定 {lock*100:>3.0f}%   "
              f"有被否決的位置 {rej*100:>3.0f}%")

    print("\n" + "=" * 70)
    print("整合結果 (投票 + 黑名單 + 黏著鎖定)")
    print("=" * 70)
    report("全部影格", None)
    if true_ranges:
        report("真目標區間 (應該高)", true_ranges)
    if fp_ranges:
        report("誤判區間 (應該 0)", fp_ranges)

    # 目標跳動：黏著策略有沒有在兩條之間來回甩
    jumps = 0
    prev = None
    for r in rows:
        if r["locked"]:
            if prev is not None and abs(r["cx"] - prev) > det.LOCK_GATE:
                jumps += 1
            prev = r["cx"]
    n_lock = sum(1 for r in rows if r["locked"])
    print(f"\n  鎖定總格數 {n_lock} / {len(rows)} ({n_lock/len(rows)*100:.0f}%)，"
          f"目標跳動 {jumps} 次")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            wr = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"  逐格結果已寫出 {args.csv}")
    return 0



# ==========================================
# fsm：偵測器 + 行為層一起重播，跟實際飛行紀錄比對
# ==========================================
def cmd_fsm(args):
    """
    把新的 TissueDetector 與 TissueChargeControl 串起來跑同一段錄影，
    看新的控制律「面對同樣的畫面會送出什麼指令」，再跟當時實際送出的比對。

    ⚠️ 這是開迴路重播：影片是舊控制律飛出來的，新控制律如果真的在飛，
    看到的畫面會不一樣。所以它不是在預測新版飛起來會怎樣，
    而是在回答「同樣的輸入下，指令分布有沒有變健康」——
    偏航飽和、爬升頂到上限、完全不前進這三個病徵都可以這樣量。
    """
    import csv as _csv
    from vision.tissue_detector import TissueDetector
    from behaviors.tissue_charge import TissueChargeControl

    clock = {"t": 0.0}
    now_fn = lambda: clock["t"]
    det = TissueDetector(depth_model_path=args.depth_model or None, now_fn=now_fn)
    beh = TissueChargeControl(now_fn=now_fn)
    beh.state_start_time = 0.0

    # 實際飛行紀錄裡的高度，用 elapsed 對回來 —— 高度上限要用真實數據才測得準
    heights = {}
    old = {}
    if args.log:
        for r in _csv.DictReader(open(args.log)):
            try:
                e = round(float(r["elapsed"]), 1)
                heights[e] = int(float(r.get("height") or 0))
                old[e] = r
            except (ValueError, TypeError):
                continue

    user = SimpleNamespace(lr=0, fb=0, ud=0, yv=0, voice_command=None)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"打不開影片 {args.video}")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        clock["t"] = idx / FPS
        d = det.process_frame(frame)
        d.height = heights.get(round(clock["t"], 1), 100)
        cmd = beh.calculate_command(user, d)
        lg = beh.log_fields()
        out.append({
            "t": clock["t"], "state": beh.state,
            "lr": cmd[0], "fb": cmd[1], "ud": cmd[2], "yv": cmd[3],
            "locked": int(d.tissue is not None), "height": d.height,
            "dist": lg.get("dist", ""), "ax": lg.get("ax", ""),
            "cy": lg.get("cy", ""), "err_y": lg.get("err_y", ""),
            "blocked": lg.get("blocked", ""),
        })
        idx += 1
        if idx % 200 == 0:
            print(f"\r  {idx}/{total}", end="", flush=True)
    cap.release()
    print(f"\r  {idx}/{total}")

    # ---- 狀態時間軸 ----
    trans = []
    for a, b in zip(out, out[1:]):
        if a["state"] != b["state"]:
            trans.append((round(b["t"], 1), a["state"], b["state"]))
    from collections import Counter
    print("\n" + "=" * 70)
    print("狀態時間軸")
    print("=" * 70)
    print("  各狀態格數:", dict(Counter(r["state"] for r in out)))
    print("  轉換次數:", len(trans), " ", dict(Counter(f"{a}->{b}" for _, a, b in trans)))

    # ---- 控制指令的健康度，跟舊紀錄比 ----
    trk = [r for r in out if r["state"] == "TRACK"]
    print("\n" + "=" * 70)
    print("TRACK 期間的指令分布 (新版重播 vs 當時實際送出的)")
    print("=" * 70)
    if trk:
        adv = sum(1 for r in trk if r["fb"] > 0) / len(trk)
        udmax = sum(1 for r in trk if r["ud"] >= beh.CLIMB_MAX) / len(trk)
        yvsat = sum(1 for r in trk if abs(r["yv"]) >= beh.pid_yv.limit) / len(trk)
        print(f"  新版 TRACK n={len(trk)}   前進中 {adv*100:.0f}%   "
              f"爬升頂到上限 {udmax*100:.0f}%   偏航飽和 {yvsat*100:.0f}%")
    if old:
        o = [r for r in old.values() if r.get("state") == "TRACK"]
        if o:
            def num(r, k):
                try:
                    return float(r[k])
                except (ValueError, TypeError, KeyError):
                    return 0.0
            adv = sum(1 for r in o if num(r, "fb") > 0) / len(o)
            udmax = sum(1 for r in o if num(r, "ud") >= 30) / len(o)
            yvsat = sum(1 for r in o if abs(num(r, "yv")) >= 60) / len(o)
            print(f"  舊版 TRACK n={len(o)}   前進中 {adv*100:.0f}%   "
                  f"爬升頂到上限 {udmax*100:.0f}%   偏航飽和 {yvsat*100:.0f}%")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = _csv.DictWriter(f, fieldnames=list(out[0].keys()))
            wr.writeheader()
            wr.writerows(out)
        print(f"\n  逐格結果已寫出 {args.csv}")
    return 0


# ==========================================
# track：逐格看鎖定紙條的估距 (紙條厚度) 穩不穩
# ==========================================
def cmd_track(args):
    """
    整個 TissueDetector 重播一段錄影。同名的 _native.mp4 在的話一起餵 (跟實飛一樣讓 YOLO 吃原生畫面)。
    逐格印出鎖定紙條的估距 (紙條厚度)、對照用的框寬估距、還沒判斷 / 判斷中，
    以及被 EDGE 擋掉的框；最後統計估距的逐格跳動，和每次開始判斷顏色時的距離。
    """
    import csv as _csv
    import io
    import contextlib
    from vision.tissue_detector import TissueDetector, DIST_K

    clock = {"t": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        det = TissueDetector(depth_model_path=None, now_fn=lambda: clock["t"])
    native_path = args.native
    if not native_path and args.video.endswith("_raw.mp4"):
        native_path = args.video[:-len("_raw.mp4")] + "_native.mp4"
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"打不開影片 {args.video}")
        return 1
    capn = cv2.VideoCapture(native_path) if native_path and os.path.exists(native_path) else None
    print(f"影片 {args.video}" + (f"\n原生 {native_path}" if capn else "\n(沒有原生畫面，YOLO 吃 720x480)"))

    rows = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if capn is not None:
            okn, fn = capn.read()
            if okn:
                det.set_native_frame(fn)
        clock["t"] = idx / FPS
        with contextlib.redirect_stdout(io.StringIO()):
            d = det.process_frame(frame)
        t = d.tissue
        rows.append({
            "t": round(clock["t"], 2), "edge": d.n_edge,
            "cx": t["cx"] if t else "", "dist": round(t["dist_cm"], 1) if t else "",
            "strip_px": t["strip_px"] if t else "", "box_w": t["w"] if t else "",
            "box_dist": round(DIST_K / max(t["w"], 1), 1) if t else "",
            "far": int(t["far"]) if t else "", "judged": int(t["judged"]) if t else "",
        })
        idx += 1
        if idx % 200 == 0:
            print(f"\r  {idx} 格", end="", flush=True)
    print(f"\r  共 {idx} 格")

    print(f"\n{'t':>6} {'cx':>4} {'估距':>6} {'厚度':>5} {'框寬':>4} {'框寬估距':>8}  狀態")
    for r in rows:
        if not (args.t_from <= r["t"] <= args.t_to) or (r["dist"] == "" and not r["edge"]):
            continue
        if r["dist"] == "":
            print(f"{r['t']:6.2f}  EDGE 擋掉 {r['edge']} 個")
            continue
        state = "已確認" if r["judged"] else ("接近中" if r["far"] else "判斷中")
        edge = f"  (EDGE 擋掉 {r['edge']})" if r["edge"] else ""
        print(f"{r['t']:6.2f} {r['cx']:4d} {r['dist']:6.0f} {r['strip_px']:5.1f} {r['box_w']:4d} "
              f"{r['box_dist']:8.0f}  {state}{edge}")

    # ---- 統計：同一條紙條 (相鄰格、cx 差 20px 內) 的估距逐格跳動，厚度 vs 框寬 ----
    def jumps(key):
        out = []
        for a, b in zip(rows, rows[1:]):
            if a["dist"] != "" and b["dist"] != "" and abs(a["cx"] - b["cx"]) <= 20:
                out.append(abs(b[key] - a[key]) / max(a[key], 1e-6))
        return sorted(out)
    print("\n" + "=" * 70)
    for key, name in (("dist", "厚度估距"), ("box_dist", "框寬估距")):
        j = jumps(key)
        if j:
            print(f"  {name} 逐格跳動  中位數 {j[len(j) // 2]:.1%}  p90 {j[int(len(j) * 0.9)]:.1%}  "
                  f"超過 30% 的格 {sum(x > 0.3 for x in j) / len(j):.1%}  (n={len(j)})")
    starts = [b for a, b in zip(rows, rows[1:])
              if b["dist"] != "" and b["far"] == 0 and b["judged"] == 0 and a["far"] != 0]
    print(f"  開始判斷顏色 {len(starts)} 次，當時估距: "
          + ", ".join(f"{r['t']:.1f}s {r['dist']:.0f}cm" for r in starts[:20]))
    print(f"  EDGE 擋掉的格數: {sum(1 for r in rows if r['edge'])}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n  逐格結果已寫出 {args.csv}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="跑 YOLO + 白色驗證器，倒出原始特徵")
    d.add_argument("video")
    d.add_argument("--out", default="/tmp/tissue_dump.json")
    d.add_argument("--model", default="model/yolo26/runs/detect/yolo_tissue_collect/best.pt")
    d.add_argument("--cls", type=int, default=-1, help="只收這個類別，-1 = 全部 (預設)")
    d.add_argument("--conf", type=float, default=0.5)
    d.add_argument("--pad-grid", type=float, nargs="+", default=None)
    d.add_argument("--sat-grid", type=int, nargs="+", default=None)
    d.set_defaults(func=cmd_dump)

    s = sub.add_parser("sheet", help="把候選框拼成接觸表，用來人工標真值")
    s.add_argument("dump")
    s.add_argument("--out", default="/tmp/tissue_sheet.png")
    s.add_argument("--times", default="", help="要取樣的時間點，逗號分隔 (秒)")
    s.add_argument("--n", type=int, default=24)
    s.set_defaults(func=cmd_sheet)

    w = sub.add_parser("sweep", help="在倒出的特徵上掃門檻")
    w.add_argument("dump")
    w.add_argument("--fp", default="", help="誤判時間區間，如 60-67")
    w.add_argument("--true", default="", help="真目標時間區間，如 8-20,75-85")
    w.add_argument("--min-aspect", type=float, default=3.0)
    w.add_argument("--max-width-ratio", type=float, default=0.20)
    w.add_argument("--pad", type=float, default=0.0)
    w.add_argument("--sat", type=int, default=80)
    w.add_argument("--sep", type=float, default=20.0)
    w.add_argument("--aspect", type=float, default=4.0)
    w.add_argument("--min-area", type=float, default=0.02)
    w.add_argument("--colour-frac", type=float, default=0.15)
    w.add_argument("--sep-grid", type=float, nargs="+", default=[0, 10, 20, 30])
    w.add_argument("--aspect-grid", type=float, nargs="+", default=[3.0, 4.0, 5.0])
    w.add_argument("--verbose", action="store_true")
    w.set_defaults(func=cmd_sweep)

    t = sub.add_parser("detect", help="跑完整的 TissueDetector，看實際會鎖定什麼")
    t.add_argument("video")
    t.add_argument("--fp", default="")
    t.add_argument("--true", default="")
    t.add_argument("--csv", default="")
    t.set_defaults(func=cmd_detect)

    m = sub.add_parser("fsm", help="偵測器 + 行為層一起重播，跟實際飛行紀錄比對")
    m.add_argument("video")
    m.add_argument("--log", default="", help="同一次飛行的 _log.csv，用來取高度與舊指令")
    m.add_argument("--depth-model", default="")
    m.add_argument("--csv", default="")
    m.set_defaults(func=cmd_fsm)

    k = sub.add_parser("track", help="逐格看鎖定紙條的估距 (紙條厚度) 與 EDGE 過濾")
    k.add_argument("video", help="_raw.mp4；同名的 _native.mp4 會自動一起用")
    k.add_argument("--native", default="", help="原生畫面影片 (預設找同名的 _native.mp4)")
    k.add_argument("--from", dest="t_from", type=float, default=0.0, help="只印這段時間 (秒)")
    k.add_argument("--to", dest="t_to", type=float, default=1e9)
    k.add_argument("--csv", default="")
    k.set_defaults(func=cmd_track)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

