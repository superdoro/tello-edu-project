"""
model/yolo26/autolabel_tissue.py 的測試 —— 不需要錄影、YOLO 或 SAM (用假的 segmenter)。

重點是那些「錯了也看不出來、但會把資料集教壞」的性質：
軌跡有沒有串錯、插值對不對、有沒有把沒標到的紙條教成背景、輪廓轉成 YOLO 格式之後還是同一個形狀。

    python tests/test_autolabel.py
"""

import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("autolabel_tissue",
                                               os.path.join(ROOT, "model/yolo26/autolabel_tissue.py"))
al = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(al)

W, H = 960, 720

results = []


def check(name, cond, detail=""):
    results.append((name, cond, detail))
    print(f"{'PASS' if cond else '**FAIL**':10s} {name}" + (f"   {detail}" if detail else ""))


def strip_poly(cx, top=100, bottom=400, w=12):
    return [[cx - w / 2, top], [cx + w / 2, top], [cx + w / 2, bottom], [cx - w / 2, bottom]]


def det(cx, conf, top=100, bottom=400, w=12, poly=True):
    return {"box": (cx - w / 2 - 2, top - 3, cx + w / 2 + 2, bottom + 3), "conf": conf,
            "poly": strip_poly(cx, top, bottom, w) if poly else None}


def blob_det(cx, conf, top=300, bottom=380, w=90):
    """又寬又矮的誤抓 (牆角、畫框)"""
    return {"box": (cx - w / 2, top, cx + w / 2, bottom), "conf": conf,
            "poly": [[cx - w / 2, top], [cx + w / 2, top], [cx + w / 2, bottom], [cx - w / 2, bottom]]}


# ==========================================
def t_tracking():
    # 轉頭時每格移動 40px (比紙條寬得多，IoU = 0)，還是同一條
    frames = [[det(100 + 40 * f, 0.8)] for f in range(15)]
    tr = al.track_detections(frames)
    check("每格移動 40px 的紙條 -> 同一條軌跡", len(tr) == 1, f"軌跡 {len(tr)} 條")

    # 兩條紙條同向移動、相隔 150px -> 兩條軌跡，不會互換
    frames = [[det(200 + 20 * f, 0.8), det(350 + 20 * f, 0.8)] for f in range(15)]
    tr = al.track_detections(frames)
    ok = len(tr) == 2 and all(len({round(d["box"][0] - 20 * f) for f, d in t.items()}) == 1 for t in tr.values())
    check("兩條紙條一起移動 -> 兩條軌跡、不互換", ok, f"軌跡 {len(tr)} 條")

    # 中間漏抓 5 格 (等速) -> 還是同一條
    frames = [[det(300 + 10 * f, 0.8)] if not 5 <= f < 10 else [] for f in range(15)]
    tr = al.track_detections(frames)
    check("中間 5 格沒抓到 -> 同一條軌跡", len(tr) == 1, f"軌跡 {len(tr)} 條")

    # 上下沒重疊 (另一個高度的東西) -> 不串在一起
    frames = [[det(300, 0.8)], [det(305, 0.8, top=500, bottom=700)]]
    tr = al.track_detections(frames)
    check("上下沒重疊 -> 不同軌跡", len(tr) == 2, f"軌跡 {len(tr)} 條")


def t_fill_and_classify():
    d = {0: det(300, 0.8), 4: det(340, 0.8)}
    filled = al.fill_gaps(d)
    mid = filled.get(2)
    check("中間空 3 格 -> 線性插值補框", mid is not None and mid.get("interp") and abs(al.cx_of(mid["box"]) - 320) < 1e-6,
          f"cx={al.cx_of(mid['box']) if mid else None}")
    d2 = {0: det(300, 0.8), 30: det(340, 0.8)}
    check("空太久 (> MAX_GAP) -> 不插值", len(al.fill_gaps(d2)) == 2, "")

    true_tr = {f: det(300, 0.8 if f % 2 == 0 else 0.3) for f in range(12)}
    check("夠多高信心、細長 -> 真紙條", al.classify_track(true_tr, W) == "true", al.classify_track(true_tr, W))
    fp_tr = {f: blob_det(500, 0.7) for f in range(4)}
    check("高信心但又寬又矮 -> 誤抓", al.classify_track(fp_tr, W) == "fp", al.classify_track(fp_tr, W))
    weak = {f: det(300, 0.3) for f in range(20)}
    check("全程低信心 -> 不確定", al.classify_track(weak, W) == "ambig", al.classify_track(weak, W))
    short = {f: det(300, 0.8) for f in range(3)}
    check("太短 -> 不確定 (不敢當真紙條)", al.classify_track(short, W) == "ambig", al.classify_track(short, W))
    flat = {f: {"box": (300, 400, 420, 412), "conf": 0.8, "poly": None} for f in range(12)}
    check("橫的細長物體 (筆電散熱孔) -> 誤抓，不是紙條", al.classify_track(flat, W) == "fp", al.classify_track(flat, W))
    edge = {f: det(W - 5, 0.8) for f in range(12)}
    check("貼著畫面邊緣 (柱子邊) -> 不確定", al.classify_track(edge, W) == "ambig", al.classify_track(edge, W))


def _video(n=40):
    """一條真紙條 (第 10~14 格低信心、15~19 格沒抓到)，第 30 格以後多一個誤抓"""
    frames = []
    for f in range(n):
        dets = []
        if f < 10 or f >= 20:
            dets.append(det(300 + f, 0.85))
        elif f < 15:
            dets.append(det(300 + f, 0.3))
        if f >= 30:
            dets.append(blob_det(700, 0.7))
        frames.append(dets)
    return frames


def t_candidates():
    frames = _video()
    tr = al.track_detections(frames)
    cls = {t: al.classify_track(d, W) for t, d in tr.items()}
    cands = al.plan_candidates(frames, tr, cls, W, H)
    kinds = {c["frame"]: c["kind"] for c in cands}
    check("低信心的格 -> 漏抓", all(kinds.get(f) == "missed" for f in range(10, 15)), "")
    check("沒抓到的格 (插值) -> 漏抓", all(kinds.get(f) == "missed" for f in range(15, 20)), "")
    check("抓得很好的格 -> 錨點", kinds.get(5) == "anchor", f"{kinds.get(5)}")
    check("有誤抓的格 -> 誤抓", kinds.get(32) == "fp", f"{kinds.get(32)}")
    c32 = next(c for c in cands if c["frame"] == 32)
    check("誤抓的格裡，真紙條照樣標成 tissue、誤抓標成背景",
          sorted(i["cls"] for i in c32["instances"]) == ["bg", "tissue"], f"{[i['cls'] for i in c32['instances']]}")

    # 畫面裡有一個不確定的框 (信心中等、只出現幾格) -> 整格不收，避免漏標的紙條被教成背景
    frames = _video()
    for f in (12, 13):
        frames[f].append(det(600, 0.4))
    tr = al.track_detections(frames)
    cls = {t: al.classify_track(d, W) for t, d in tr.items()}
    kinds = {c["frame"]: c["kind"] for c in al.plan_candidates(frames, tr, cls, W, H)}
    check("有不確定的框 -> 整格不收", 12 not in kinds and 13 not in kinds and 11 in kinds, "")

    # 抽樣：每段每 STRIDE 格一張；錨點不超過失敗樣本的 ANCHOR_RATIO
    frames = [[det(300, 0.85 if f % 40 < 20 else 0.3)] for f in range(400)]
    tr = al.track_detections(frames)
    cls = {t: al.classify_track(d, W) for t, d in tr.items()}
    cands = al.plan_candidates(frames, tr, cls, W, H)
    picked = al.subsample(cands, max_per_track=1000)
    missed = [c["frame"] for c in picked if c["kind"] == "missed"]
    anchors = [c for c in picked if c["kind"] == "anchor"]
    check("同一段失敗區間每 STRIDE 格取一張", missed[:4] == [20, 25, 30, 35], f"{missed[:6]}")
    check("錨點不超過失敗樣本的 ANCHOR_RATIO", len(anchors) <= al.ANCHOR_RATIO * len(missed) + 1e-9,
          f"錨點 {len(anchors)} / 失敗 {len(missed)}")
    picked = al.subsample(cands, max_per_track=5)
    check("每條軌跡最多 max_per_track 張", sum(c["kind"] == "missed" for c in picked) == 5, "")


def t_masks():
    poly = strip_poly(300, w=14)
    m = al.poly_to_mask(poly, W, H)
    back = al.poly_to_mask(al.mask_to_polygon(m), W, H)
    iou = (m & back).sum() / max((m | back).sum(), 1)
    check("遮罩 -> 輪廓 -> 遮罩，形狀不變", iou > 0.9, f"IoU {iou:.3f}")

    line = al.yolo_line(poly, W, H)
    nums = [float(v) for v in line.split()[1:]]
    xs, ys = np.array(nums[0::2]) * W, np.array(nums[1::2]) * H
    check("YOLO-seg 一行：類別 0、正規化座標還原回像素", line.startswith("0 ")
          and np.allclose(xs, [p[0] for p in poly], atol=1e-3) and np.allclose(ys, [p[1] for p in poly], atol=1e-3), "")

    box = (291, 97, 309, 403)
    check("細長、位置對 -> 合格", al.validate_mask(m, box, 14) is not None, "")
    blob = al.poly_to_mask([[250, 150], [350, 150], [350, 250], [250, 250]], W, H)
    check("團狀 -> 不合格", al.validate_mask(blob, box, 14) is None, "")
    wide = al.poly_to_mask(strip_poly(300, w=60), W, H)
    check("比軌跡寬太多 -> 不合格", al.validate_mask(wide, box, 14) is None, "")
    off = al.poly_to_mask(strip_poly(600, w=14), W, H)
    check("位置不對 -> 不合格", al.validate_mask(off, box, 14) is None, "")
    flat = al.poly_to_mask([[240, 245], [360, 245], [360, 255], [240, 255]], W, H)
    check("橫的細長遮罩 -> 不合格", al.validate_mask(flat, (230, 230, 370, 270), 14) is None, "")
    # 紙條連旁邊的背景切成不規則的一團 (10/19 20:46 f000295)
    lump = al.poly_to_mask([[294, 100], [306, 100], [306, 400], [294, 400], [270, 330], [262, 200], [280, 140]], W, H)
    check("紙條連背景切成一團 -> 不合格", al.validate_mask(lump, box, 40) is None, "")

    def click(x, y):
        s_ = {"instances": [{"poly": strip_poly(300), "box": [290, 97, 310, 403]},
                            {"poly": None, "box": [600, 300, 700, 380]}]}
        return al._hit_instance(s_, x, y)
    check("review：點輪廓 / 點框 / 點空白", (click(302, 250), click(650, 340), click(450, 50)) == (0, 1, None),
          f"{(click(302, 250), click(650, 340), click(450, 50))}")


def t_resolve_polygons():
    frame = np.zeros((H, W, 3), np.uint8)
    hi = dict(det(300, 0.9), tid=1)
    lo = dict(det(500, 0.2, poly=False), tid=2)
    cand = {"frame": 0, "kind": "missed",
            "instances": [{"cls": "tissue", "tid": 1, "det": hi}, {"cls": "tissue", "tid": 2, "det": lo}]}
    calls = []

    def fake_sam(img, boxes):
        calls.append(boxes)
        return [al.poly_to_mask(strip_poly(500), W, H) for _ in boxes]

    r = al.resolve_polygons(frame, cand, fake_sam, {1: 16, 2: 16})
    check("高信心用 YOLO 輪廓、低信心交給 SAM", r is not None and [x["src"] for x in r] == ["yolo", "sam"]
          and len(calls) == 1 and len(calls[0]) == 1, f"{[x['src'] for x in r] if r else None}")

    def bad_sam(img, boxes):
        return [al.poly_to_mask([[400, 300], [600, 300], [600, 400], [400, 400]], W, H) for _ in boxes]

    check("SAM 輪廓不合格、也沒有 YOLO 輪廓 -> 整格不收", al.resolve_polygons(frame, cand, bad_sam, {1: 16, 2: 16}) is None, "")
    lo2 = dict(det(500, 0.2), tid=2)
    cand2 = {"frame": 0, "kind": "missed", "instances": [{"cls": "tissue", "tid": 2, "det": lo2}]}
    r = al.resolve_polygons(frame, cand2, bad_sam, {2: 16})
    check("SAM 不合格但 YOLO 低信心輪廓合格 -> 用 YOLO 的", r is not None and r[0]["src"] == "yolo-low", "")


def t_flying_frames():
    rows = [{"elapsed": f"{t / 10:.2f}", "height": "150" if t < 50 else "0"} for t in range(100)]
    rows = [r for r in rows if not 70 <= float(r["elapsed"]) * 10 < 90] + [{"elapsed": "x", "height": "?"}]
    fly = al.flying_frames(rows, 330, 30.0)                 # 11 秒的影片，紀錄只到 9.9 秒
    check("高度 > 0 的時候 -> 在飛", all(fly[:140]), "")
    check("高度 0 (落地、被拿著) -> 不收", not any(fly[160:200]), "")
    check("紀錄超出時間範圍 -> 不收", not any(fly[320:]), "")
    check("壞掉的紀錄行不影響", len(fly) == 330, "")

    frames = _video()
    tr = al.track_detections(frames)
    cls = {t: al.classify_track(d, W) for t, d in tr.items()}
    fly = [f < 20 for f in range(len(frames))]
    kinds = {c["frame"] for c in al.plan_candidates(frames, tr, cls, W, H, fly)}
    check("沒在飛的格不當樣本", kinds and max(kinds) < 20, f"最後一格 {max(kinds) if kinds else None}")


def t_review_and_export():
    with tempfile.TemporaryDirectory() as tmp:
        staging = os.path.join(tmp, "staging")
        os.makedirs(os.path.join(staging, "vid", "images"))
        import cv2
        samples = []
        for k, (t, kind) in enumerate([(1.0, "missed"), (2.0, "fp"), (25.0, "missed"), (40.0, "anchor")]):
            sid = f"vid_f{k:06d}"
            cv2.imwrite(os.path.join(staging, "vid", "images", sid + ".jpg"), np.zeros((H, W, 3), np.uint8))
            insts = [{"cls": "tissue", "tid": 1, "box": [290, 97, 310, 403], "conf": 0.3, "interp": False,
                      "poly": strip_poly(300), "src": "sam"},
                     {"cls": "bg", "tid": 2, "box": [600, 300, 700, 380], "conf": 0.7, "interp": False,
                      "poly": None, "src": ""}]
            samples.append({"id": sid, "video": "vid", "frame": k, "t": t, "kind": kind, "size": [W, H],
                            "instances": insts})
        with open(os.path.join(staging, "vid", "meta.jsonl"), "w") as fh:
            for s in samples:
                fh.write(json.dumps(s) + "\n")
        review = {"vid_f000000": {"keep": True, "cls": ["tissue", "bg"]},
                  "vid_f000001": {"keep": True, "cls": ["bg", "bg"]},        # 點掉 -> 變成負樣本
                  "vid_f000002": {"keep": False, "cls": ["tissue", "bg"]},
                  "vid_f000003": {"keep": True, "cls": ["tissue", "bg"]}}
        al.save_review(os.path.join(tmp, "review.json"), review)

        out = os.path.join(tmp, "ds")
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            al.main(["export", "--staging", staging, "--out", out])
        files = {}
        for sp in ("train", "val"):
            d = os.path.join(out, "labels", sp)
            if os.path.isdir(d):
                for name in os.listdir(d):
                    files[name[:-4]] = open(os.path.join(d, name)).read().strip()
        check("丟掉的樣本不匯出", "vid_f000002" not in files and len(files) == 3, f"{sorted(files)}")
        check("點成背景的實例不出現在 label (變成負樣本)", files.get("vid_f000001") == "", repr(files.get("vid_f000001")))
        check("tissue 實例照樣匯出、背景不標", files.get("vid_f000000", "").count("\n") == 0
              and files.get("vid_f000000", "").startswith("0 "), "")
        yaml = open(os.path.join(out, "data.yaml")).read()
        check("data.yaml：單類別 tissue、train / val 路徑", "0: tissue" in yaml and "train: images/train" in yaml
              and "val: images/val" in yaml, "")
        both = all(os.path.isdir(os.path.join(out, "images", sp)) for sp in ("train", "val"))
        check("train / val 都有東西", both, "")
        check("切分是決定性的", al.split_of(samples[0]) == al.split_of(dict(samples[0])), "")

        with contextlib.redirect_stdout(io.StringIO()):
            try:
                al.main(["export", "--staging", staging, "--out", out])
                refused = False
            except SystemExit:
                refused = True
        check("已經有資料集 -> 沒加 --overwrite 不覆蓋", refused, "")


if __name__ == "__main__":
    for fn in (t_tracking, t_fill_and_classify, t_candidates, t_masks, t_resolve_polygons, t_flying_frames,
               t_review_and_export):
        print(f"\n--- {fn.__name__} ---")
        fn()
    bad = [n for n, ok, _ in results if not ok]
    print("\n" + "=" * 60)
    print(f"{len(results) - len(bad)}/{len(results)} 通過" + (f"，未通過: {bad}" if bad else "，全部通過"))
    sys.exit(1 if bad else 0)
