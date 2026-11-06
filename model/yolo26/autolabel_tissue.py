"""
從錄影裡「辨識失敗」的畫面自動標註 YOLO-seg 資料集 (單類別 tissue)，拿來微調 best.pt。

best.pt 是用合成資料訓練的，在實際場地常常漏抓 (同一條紙條前後幾格抓得到、中間抓不到或信心很低)，
也會把牆角、窗簾之類的東西當成紙條。這兩種畫面正是模型最需要學的。

流程分三步：

    mine    跑 best.pt + 追蹤，挑出漏抓 / 誤抓的畫面，自動產生分割標註 (低信心或沒抓到的用 SAM 補輪廓)
    review  OpenCV 視窗快速人工檢查：留 / 丟，滑鼠點實例切換 tissue <-> 背景
    export  把留下來的樣本匯出成 YOLO-seg 資料集 (train / val + data.yaml)

用法：
    python model/yolo26/autolabel_tissue.py mine   recordings/<時間>_TISSUE_CHARGE_native.mp4 [...]
    python model/yolo26/autolabel_tissue.py sheet  (隨機抽幾張畫成接觸表，先看標註品質)
    python model/yolo26/autolabel_tissue.py review
    python model/yolo26/autolabel_tissue.py export --out datasets/tissue_real

==========================================================
怎麼判斷「辨識失敗」
==========================================================
每一格都跑 best.pt (conf 0.1，比實飛的 0.5 低很多)，再把框串成軌跡：

    真紙條   至少 TRUE_MIN_HIGH 格信心 >= 0.5、總長 >= TRUE_MIN_LEN 格、框是直的細長條 (高/寬 >= 3)
             -> 同一條軌跡裡信心 < 0.5 或沒抓到 (中間空 <= MAX_GAP 格，用前後插值補框) 的格 = 漏抓
    誤抓     至少 FP_MIN_HIGH 格信心 >= 0.5，但形狀不像紙條 (又寬又矮、橫的、框太寬，與偵測器的形狀關卡一致)
             -> 信心 >= 0.5 的格 = 誤抓，整個框當背景 (不標)
    不確定   其他，以及大部分時間貼著畫面左右邊緣的 (多半是柱子、門框的邊，真紙條在邊緣也只露出一截)。
             畫面裡只要有不確定的框信心 >= AMBIG_CONF，整格就不收 ——
             一張圖的標註必須把每一條紙條都標到，漏標的紙條會被教成背景。

同一趟的 _log.csv 在旁邊就會自動讀，只收「無人機確實在飛」(高度 > MIN_FLY_HEIGHT_CM) 的畫面：
10/19 20:42 那趟 45 秒後就落地了，之後是被人拿著、倒著拿、放在桌上的畫面，標了只會教壞模型。

另外抽一些「本來就抓得很好」的畫面當錨點 (ANCHOR_RATIO)，微調之後原本會的才不會退步。

==========================================================
輪廓
==========================================================
高信心的框直接用 YOLO 自己的分割輪廓。低信心或插值出來的框交給 SAM (以框當提示)，
再檢查輪廓是不是細長的一條 (validate_mask)；不合格就改用 YOLO 的低信心輪廓，兩個都不合格就丟掉這格。
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL = os.path.join(ROOT, "model/yolo26/runs/detect/yolo_tissue_collect/best.pt")
DEFAULT_SAM = os.path.join(ROOT, "model/sam/sam2.1_b.pt")
DEFAULT_STAGING = os.path.join(ROOT, "datasets/tissue_autolabel/staging")

HIGH_CONF = 0.5        # 偵測器實飛的信心門檻 (TissueDetector.conf_threshold)
LOW_CONF = 0.1         # 挖低信心偵測用
AMBIG_CONF = 0.25      # 不確定的框信心超過這個，整格不收

TRACK_GATE_PX = 60     # 追蹤：預測位置與偵測的水平距離上限 (960 寬畫面)
TRACK_V_OVERLAP = 0.5  # 追蹤：上下重疊至少佔較短那個框的這麼多
MAX_GAP = 15           # 軌跡中間空掉這麼多格以內 (0.5 秒) 才插值補框

TRUE_MIN_HIGH = 5
TRUE_MIN_LEN = 10
TRUE_MIN_ASPECT = 3.0  # 框的高/寬，與 TissueDetector 的 min_aspect 一致 (紙條是直的，橫的細長物體不算)
FP_MIN_HIGH = 2
FP_MAX_ASPECT = 2.0
MAX_WIDTH_RATIO = 0.20 # 與 TissueDetector 的 max_width_ratio 一致
EDGE_PX = 8            # 框離畫面左右邊緣這麼近算「貼邊」
EDGE_MAX_FRAC = 0.5    # 高信心的格有超過這個比例貼邊 -> 不確定
MIN_FLY_HEIGHT_CM = 10 # 飛行紀錄的高度 <= 這個 -> 沒在飛 (落地、被拿著)
LOG_MAX_DT = 0.5       # 影格時間找不到這麼近的紀錄 -> 當成不知道，不收

STRIDE = 5             # 同一段失敗區間每幾格取一格 (連續畫面幾乎一樣)
MAX_PER_TRACK = 20     # 每條軌跡最多貢獻幾張
ANCHOR_EVERY = 15
ANCHOR_RATIO = 0.2     # 錨點最多佔失敗樣本數的這個比例

MASK_MIN_AREA = 30
MASK_MIN_ASPECT = 3.0
MASK_MIN_V_ASPECT = 1.5   # 遮罩外框的高/寬：紙條是直的 (容許斜約 30 度)
MASK_MAX_BOX_WIDTH = 1.5  # 遮罩寬度不能超過這一格框寬的這個倍數 (SAM 有時把紙條連背景切成一大塊)
MASK_MIN_FILL = 0.45      # 遮罩面積 / 最小外接矩形面積：紙條是實心的一條，不規則的一團不算
MASK_MAX_WIDTH_FACTOR = 2.5
MASK_MIN_IOU = 0.3

VAL_RATIO = 0.2
SPLIT_BLOCK_SEC = 10.0 # 切 train / val 時以「影片 + 10 秒區段」為單位，幾乎一樣的畫面不會分到兩邊


# ==========================================
# 幾何小工具
# ==========================================
def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def v_overlap(a, b):
    """兩個框上下重疊的長度 / 較短那個框的高度"""
    inter = min(a[3], b[3]) - max(a[1], b[1])
    short = min(a[3] - a[1], b[3] - b[1])
    return inter / short if short > 0 else 0.0


def cx_of(box):
    return (box[0] + box[2]) / 2.0


def box_aspect(box):
    """框的高/寬 (直的紙條 >= 3，橫的細長物體 < 1)"""
    x0, y0, x1, y1 = box
    return (y1 - y0) / max(x1 - x0, 1e-6)


def touches_side(box, frame_w, px=EDGE_PX):
    return box[0] <= px or box[2] >= frame_w - px


# ==========================================
# 追蹤
# ==========================================
def track_detections(dets_by_frame, gate=TRACK_GATE_PX, max_gap=MAX_GAP):
    """
    把每一格的偵測串成軌跡，每個偵測寫上 "tid"。

    不用 ByteTrack：它靠 IoU 配對，細長的框在無人機轉頭時 (一格移動幾十像素) IoU 幾乎是 0。
    紙條是直的、主要左右移動，所以用「等速度預測的 cx 差」配對，再要求上下有重疊。
    每格只有幾條，貪婪配對 (成本由小到大) 就夠了。

    :param dets_by_frame: list，第 i 個元素是第 i 格的偵測 [{"box": (x0,y0,x1,y1), "conf", "poly"}]
    :return: {tid: {frame: det}}
    """
    tracks = {}       # tid -> {"last": frame, "box": box, "vx": float}
    out = {}
    next_id = 1
    for f, dets in enumerate(dets_by_frame):
        active = {t: s for t, s in tracks.items() if f - s["last"] <= max_gap + 1}
        pairs = []
        for t, s in active.items():
            dt = f - s["last"]
            pred = cx_of(s["box"]) + s["vx"] * dt
            g = gate * (1.0 + 0.5 * (dt - 1))       # 空越久，容許偏差越大
            for j, d in enumerate(dets):
                cost = abs(cx_of(d["box"]) - pred)
                if cost <= g and v_overlap(s["box"], d["box"]) >= TRACK_V_OVERLAP:
                    pairs.append((cost, t, j))
        pairs.sort()
        used_t, used_d = set(), set()
        for cost, t, j in pairs:
            if t in used_t or j in used_d:
                continue
            used_t.add(t)
            used_d.add(j)
            s = tracks[t]
            d = dets[j]
            dt = f - s["last"]
            v_new = (cx_of(d["box"]) - cx_of(s["box"])) / dt
            s["vx"] = 0.5 * s["vx"] + 0.5 * v_new
            s["last"], s["box"] = f, d["box"]
            d["tid"] = t
            out[t][f] = d
        for j, d in enumerate(dets):
            if j in used_d:
                continue
            tracks[next_id] = {"last": f, "box": d["box"], "vx": 0.0}
            d["tid"] = next_id
            out[next_id] = {f: d}
            next_id += 1
    return out


def classify_track(dets, frame_w):
    """
    :param dets: {frame: det}
    :return: "true" / "fp" / "ambig"
    """
    vals = list(dets.values())
    high = [d for d in vals if d["conf"] >= HIGH_CONF]
    if not high:
        return "ambig"
    aspect = float(np.median([box_aspect(d["box"]) for d in high]))
    wr = float(np.median([(d["box"][2] - d["box"][0]) / frame_w for d in high]))
    edge = sum(touches_side(d["box"], frame_w) for d in high) / len(high)
    if edge > EDGE_MAX_FRAC:
        return "ambig"
    if (len(high) >= TRUE_MIN_HIGH and len(vals) >= TRUE_MIN_LEN
            and aspect >= TRUE_MIN_ASPECT and wr <= MAX_WIDTH_RATIO):
        return "true"
    if len(high) >= FP_MIN_HIGH and (aspect < FP_MAX_ASPECT or wr > MAX_WIDTH_RATIO):
        return "fp"
    return "ambig"


def fill_gaps(dets, max_gap=MAX_GAP):
    """
    真紙條軌跡中間沒抓到的格，用前後兩格線性插值補框 (conf 0、沒有輪廓、interp=True)。
    :return: 新的 {frame: det}，包含原本的與補上的
    """
    frames = sorted(dets)
    out = dict(dets)
    for f0, f1 in zip(frames, frames[1:]):
        gap = f1 - f0 - 1
        if gap <= 0 or gap > max_gap:
            continue
        b0, b1 = np.array(dets[f0]["box"], float), np.array(dets[f1]["box"], float)
        for f in range(f0 + 1, f1):
            k = (f - f0) / (f1 - f0)
            out[f] = {"box": tuple((b0 + (b1 - b0) * k).tolist()), "conf": 0.0, "poly": None,
                      "interp": True, "tid": dets[f0].get("tid")}
    return out


# ==========================================
# 挑樣本
# ==========================================
def plan_candidates(dets_by_frame, tracks, classes, frame_w, frame_h, flying=None):
    """
    逐格決定這一格能不能當樣本、是哪一種。
    :param flying: 每一格「無人機在飛」的 bool 序列 (flying_frames)，None = 不檢查

    :return: [{"frame", "kind", "instances": [{"cls": "tissue"/"bg", "det", "tid"}]}]
             kind: missed (有真紙條沒抓好) / fp (有誤抓) / anchor (真紙條全部抓得很好)
    """
    filled = {t: fill_gaps(d) for t, d in tracks.items() if classes[t] == "true"}
    by_frame = {}
    for t, d in filled.items():
        for f, det in d.items():
            by_frame.setdefault(f, []).append(("tissue", t, det))
    for t, d in tracks.items():
        for f, det in d.items():
            if classes[t] == "fp" and det["conf"] >= HIGH_CONF:
                by_frame.setdefault(f, []).append(("bg", t, det))
            elif classes[t] == "ambig" and det["conf"] >= AMBIG_CONF:
                by_frame.setdefault(f, []).append(("ambig", t, det))

    cands = []
    for f in sorted(by_frame):
        if flying is not None and not (f < len(flying) and flying[f]):
            continue
        items = by_frame[f]
        if any(c == "ambig" for c, _, _ in items):
            continue
        ok = True
        for c, _, det in items:
            if c != "tissue":
                continue
            x0, y0, x1, y1 = det["box"]
            cw = min(x1, frame_w) - max(x0, 0)
            if cw < 0.5 * (x1 - x0) or y1 <= 0 or y0 >= frame_h:   # 插值的框大半跑出畫面
                ok = False
        if not ok:
            continue
        tissue = [det for c, _, det in items if c == "tissue"]
        if any(det["conf"] < HIGH_CONF for det in tissue):
            kind = "missed"
        elif any(c == "bg" for c, _, _ in items):
            kind = "fp"
        elif tissue:
            kind = "anchor"
        else:
            continue
        cands.append({"frame": f, "kind": kind,
                      "instances": [{"cls": c, "tid": t, "det": det} for c, t, det in items]})
    return cands


def subsample(cands, stride=STRIDE, max_per_track=MAX_PER_TRACK, anchor_every=ANCHOR_EVERY,
              anchor_ratio=ANCHOR_RATIO):
    """
    連續的失敗畫面幾乎一樣：每段每 stride 格取一格，每條軌跡最多 max_per_track 張；
    錨點每 anchor_every 格取一格，最多佔失敗樣本數的 anchor_ratio。
    """
    picked = []
    per_track = {}
    for kind in ("missed", "fp"):
        seg_start = prev = None
        for c in (c for c in cands if c["kind"] == kind):
            f = c["frame"]
            if prev is None or f - prev > 1:
                seg_start = f
            prev = f
            if (f - seg_start) % stride:
                continue
            if kind == "missed":
                tids = [i["tid"] for i in c["instances"]
                        if i["cls"] == "tissue" and i["det"]["conf"] < HIGH_CONF]
            else:
                tids = [i["tid"] for i in c["instances"] if i["cls"] == "bg"]
            if all(per_track.get((kind, t), 0) >= max_per_track for t in tids):
                continue
            for t in tids:
                per_track[(kind, t)] = per_track.get((kind, t), 0) + 1
            picked.append(c)
    anchors = [c for c in cands if c["kind"] == "anchor" and c["frame"] % anchor_every == 0]
    limit = int(anchor_ratio * len(picked))
    if len(anchors) > limit:
        idx = np.linspace(0, len(anchors) - 1, limit).round().astype(int) if limit > 0 else []
        anchors = [anchors[i] for i in sorted(set(idx))]
    picked += anchors
    picked.sort(key=lambda c: c["frame"])
    return picked


# ==========================================
# 輪廓
# ==========================================
def largest_component(mask):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


def validate_mask(mask, expect_box, med_w):
    """
    SAM / YOLO 給的遮罩是不是「細長的一條紙條、而且就在預期的位置」。
    :return: 只留最大連通塊的遮罩，不合格回傳 None
    """
    comp = largest_component(mask)
    if comp is None or int(comp.sum()) < MASK_MIN_AREA:
        return None
    ys, xs = np.nonzero(comp)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    short, long_ = min(w, h), max(w, h)
    if short <= 0 or long_ / max(short, 1.0) < MASK_MIN_ASPECT:
        return None
    box_w = expect_box[2] - expect_box[0]
    if short > max(min(MASK_MAX_WIDTH_FACTOR * med_w, MASK_MAX_BOX_WIDTH * box_w), 8.0):
        return None
    if comp.sum() / max(w * h, 1.0) < MASK_MIN_FILL:
        return None
    bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    if box_aspect(bbox) < MASK_MIN_V_ASPECT:
        return None
    if box_iou(bbox, expect_box) < MASK_MIN_IOU:
        return None
    return comp


def mask_to_polygon(mask, eps=1.0):
    cnts, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    return c.astype(float).tolist() if len(c) >= 3 else None


def poly_to_mask(poly, w, h):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 1)
    return m


def yolo_line(poly, w, h, cls=0):
    """YOLO-seg 一行：cls x1 y1 x2 y2 ... (正規化到 0~1)"""
    p = np.asarray(poly, float)
    xs = np.clip(p[:, 0] / w, 0.0, 1.0)
    ys = np.clip(p[:, 1] / h, 0.0, 1.0)
    return f"{cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))


def expand_box(box, w, h, fx=0.3, fy=0.1):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    return (max(0.0, x0 - fx * bw), max(0.0, y0 - fy * bh), min(float(w), x1 + fx * bw), min(float(h), y1 + fy * bh))


def resolve_polygons(frame, cand, segmenter, med_w_by_tid):
    """
    把一個候選畫面的每個 tissue 實例變成輪廓。任何一個失敗就整格不收 (回傳 None)。
    :param segmenter: callable(frame, [box]) -> [mask 或 None]
    """
    h, w = frame.shape[:2]
    need_sam = []
    result = []
    for inst in cand["instances"]:
        det = inst["det"]
        r = {"cls": inst["cls"], "tid": inst["tid"], "box": [float(v) for v in det["box"]],
             "conf": round(float(det["conf"]), 3), "interp": bool(det.get("interp")), "poly": None, "src": ""}
        if inst["cls"] == "tissue":
            if det["conf"] >= HIGH_CONF and det.get("poly") is not None:
                r["poly"], r["src"] = [list(map(float, p)) for p in det["poly"]], "yolo"
            else:
                need_sam.append(len(result))
        result.append(r)

    if need_sam:
        boxes = [expand_box(result[i]["box"], w, h) for i in need_sam]
        masks = segmenter(frame, boxes) if segmenter is not None else [None] * len(boxes)
        for i, m in zip(need_sam, masks):
            r = result[i]
            med_w = med_w_by_tid.get(r["tid"], r["box"][2] - r["box"][0])
            good = validate_mask(m, r["box"], med_w) if m is not None else None
            if good is not None:
                r["poly"], r["src"] = mask_to_polygon(good), "sam"
            else:
                ypoly = cand["instances"][i]["det"].get("poly")
                if ypoly is not None and validate_mask(poly_to_mask(ypoly, w, h), r["box"], med_w) is not None:
                    r["poly"], r["src"] = [list(map(float, p)) for p in ypoly], "yolo-low"
            if r["poly"] is None:
                return None
    return result


class SamSegmenter:
    """ultralytics SAM，以框當提示。第一次會自動下載權重到 weights 指定的路徑"""

    def __init__(self, weights=DEFAULT_SAM):
        from ultralytics import SAM
        os.makedirs(os.path.dirname(weights) or ".", exist_ok=True)
        self.model = SAM(weights)

    def __call__(self, frame, boxes):
        r = self.model(frame, bboxes=[list(b) for b in boxes], verbose=False)[0]
        if r.masks is None:
            return [None] * len(boxes)
        data = r.masks.data.cpu().numpy() > 0.5
        out = [data[i] if i < len(data) else None for i in range(len(boxes))]
        return out


# ==========================================
# 飛行紀錄：只收在飛的畫面
# ==========================================
def flying_frames(log_rows, n_frames, fps):
    """
    每一格「無人機在飛」的 bool。錄影是以固定 fps 的時間軸寫出的 (utils/video_recorder.py)，
    影格編號 / fps 就是 CSV 的 elapsed。找不到 LOG_MAX_DT 內的紀錄 (紀錄斷掉) 一律當成不知道 -> False。

    :param log_rows: [{"elapsed": str/float, "height": str/int}]
    """
    ts, ok = [], []
    for r in log_rows:
        try:
            t = float(r["elapsed"])
            h = float(r["height"])
        except (KeyError, TypeError, ValueError):
            continue
        ts.append(t)
        ok.append(h > MIN_FLY_HEIGHT_CM)
    if not ts:
        return [False] * n_frames
    ts = np.asarray(ts)
    order = np.argsort(ts)
    ts, ok = ts[order], np.asarray(ok)[order]
    out = []
    for f in range(n_frames):
        t = f / fps
        k = int(np.searchsorted(ts, t))
        best = min((j for j in (k - 1, k) if 0 <= j < len(ts)), key=lambda j: abs(ts[j] - t))
        out.append(bool(ok[best]) and abs(ts[best] - t) <= LOG_MAX_DT)
    return out


def find_log(video):
    """同一趟的 _log.csv (與 _native.mp4 / _raw.mp4 放在一起)"""
    path = os.path.join(os.path.dirname(os.path.abspath(video)), _video_stem(video) + "_log.csv")
    return path if os.path.exists(path) else None


# ==========================================
# mine
# ==========================================
def _video_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    for suffix in ("_native", "_raw"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def detect_video(path, model_path, imgsz=960):
    """每一格跑 YOLO (conf LOW_CONF)，回傳 (dets_by_frame, fps, (w, h))"""
    from ultralytics import YOLO
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"打不開影片 {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dets_by_frame = []
    size = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        size = (frame.shape[1], frame.shape[0])
        r = model(frame, imgsz=imgsz, conf=LOW_CONF, verbose=False, device=device)[0]
        dets = []
        polys = r.masks.xy if r.masks is not None else None
        for i, b in enumerate(r.boxes or []):
            poly = polys[i] if polys is not None and i < len(polys) and len(polys[i]) >= 3 else None
            dets.append({"box": tuple(round(v, 1) for v in b.xyxy[0].tolist()),
                         "conf": round(float(b.conf[0]), 3),
                         "poly": np.round(poly, 1).tolist() if poly is not None else None})
        dets_by_frame.append(dets)
        if len(dets_by_frame) % 300 == 0:
            print(f"   偵測 {len(dets_by_frame)}/{total} 格")
    cap.release()
    return dets_by_frame, fps, size


def cmd_mine(args):
    segmenter = None
    for video in args.videos:
        stem = _video_stem(video)
        out_dir = os.path.join(args.staging, stem)
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        cache = os.path.join(out_dir, "cache.json")
        print(f"[{stem}]")
        if os.path.exists(cache) and not args.redetect:
            with open(cache) as fh:
                c = json.load(fh)
            dets_by_frame, fps, size = c["dets"], c["fps"], tuple(c["size"])
            for dets in dets_by_frame:
                for d in dets:
                    d["box"] = tuple(d["box"])
            print(f"   沿用偵測快取 ({len(dets_by_frame)} 格)")
        else:
            dets_by_frame, fps, size = detect_video(video, args.model, args.imgsz)
            with open(cache, "w") as fh:
                json.dump({"video": os.path.abspath(video), "fps": fps, "size": size,
                           "dets": dets_by_frame}, fh)
        w, h = size

        tracks = track_detections(dets_by_frame)
        classes = {t: classify_track(d, w) for t, d in tracks.items()}
        n_cls = {k: sum(1 for v in classes.values() if v == k) for k in ("true", "fp", "ambig")}
        print(f"   軌跡 {len(tracks)} 條：真紙條 {n_cls['true']}、誤抓 {n_cls['fp']}、不確定 {n_cls['ambig']}")
        med_w = {t: float(np.median([d["box"][2] - d["box"][0] for d in tracks[t].values()
                                     if d["conf"] >= HIGH_CONF] or [8.0]))
                 for t in tracks if classes[t] == "true"}

        flying = None
        log_path = None if args.no_log else (args.log or find_log(video))
        if log_path:
            import csv
            with open(log_path, newline="") as fh:
                flying = flying_frames(list(csv.DictReader(fh)), len(dets_by_frame), fps)
            print(f"   飛行紀錄 {os.path.basename(log_path)}：{sum(flying)}/{len(flying)} 格在飛")
        else:
            print("   [警告] 找不到同一趟的 _log.csv，無法排除落地、被拿著的畫面 (review 時請特別注意)")

        cands = plan_candidates(dets_by_frame, tracks, classes, w, h, flying)
        picked = subsample(cands, stride=args.stride, max_per_track=args.max_per_track)
        print(f"   候選畫面 {len(cands)} 格 -> 抽樣 {len(picked)} 格 "
              + str({k: sum(1 for c in picked if c['kind'] == k) for k in ('missed', 'fp', 'anchor')}))

        if segmenter is None and any(i["cls"] == "tissue" and i["det"]["conf"] < HIGH_CONF
                                     for c in picked for i in c["instances"]):
            segmenter = SamSegmenter(args.sam)

        img_dir = os.path.join(out_dir, "images")
        for name in os.listdir(img_dir):                 # 重跑：清掉上一次產生的圖 (都是這個工具寫的)
            if name.endswith(".jpg"):
                os.remove(os.path.join(img_dir, name))

        want = {c["frame"]: c for c in picked}
        cap = cv2.VideoCapture(video)
        kept, dropped = [], 0
        f = -1
        while want:
            ok, frame = cap.read()
            if not ok:
                break
            f += 1
            if f not in want:
                continue
            cand = want.pop(f)
            inst = resolve_polygons(frame, cand, segmenter, med_w)
            if inst is None:
                dropped += 1
                continue
            sid = f"{stem}_f{f:06d}"
            cv2.imwrite(os.path.join(out_dir, "images", sid + ".jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            kept.append({"id": sid, "video": stem, "frame": f, "t": round(f / fps, 2), "kind": cand["kind"],
                         "size": [w, h], "instances": inst})
        cap.release()
        with open(os.path.join(out_dir, "meta.jsonl"), "w") as fh:
            for m in kept:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        n_kind = {k: sum(1 for m in kept if m["kind"] == k) for k in ("missed", "fp", "anchor")}
        print(f"   寫出 {len(kept)} 張 {n_kind}，輪廓不合格丟掉 {dropped} 張 -> {out_dir}")


# ==========================================
# 樣本讀寫、畫圖 (review / sheet / export 共用)
# ==========================================
def load_samples(staging):
    samples = []
    if not os.path.isdir(staging):
        return samples
    for stem in sorted(os.listdir(staging)):
        meta = os.path.join(staging, stem, "meta.jsonl")
        if not os.path.exists(meta):
            continue
        with open(meta) as fh:
            for line in fh:
                if line.strip():
                    m = json.loads(line)
                    m["image"] = os.path.join(staging, stem, "images", m["id"] + ".jpg")
                    samples.append(m)
    return samples


def load_review(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_review(path, review):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(review, fh, ensure_ascii=False, indent=0)
    os.replace(tmp, path)


def instance_classes(sample, review):
    """review 裡點過的實例以 review 為準"""
    r = review.get(sample["id"], {})
    cls = r.get("cls")
    if cls and len(cls) == len(sample["instances"]):
        return list(cls)
    return [i["cls"] for i in sample["instances"]]


_SRC_COLOUR = {"yolo": (0, 255, 0), "sam": (0, 255, 255), "yolo-low": (255, 255, 0)}


def draw_sample(img, sample, classes, highlight=None):
    for k, (inst, cls) in enumerate(zip(sample["instances"], classes)):
        x0, y0, x1, y1 = [int(round(v)) for v in inst["box"]]
        thick = 3 if k == highlight else 2
        if cls == "tissue" and inst.get("poly"):
            pts = np.round(np.asarray(inst["poly"])).astype(np.int32)
            col = _SRC_COLOUR.get(inst.get("src"), (0, 255, 0))
            cv2.polylines(img, [pts], True, col, thick)
            label = f"{inst.get('src', '')} {inst['conf']:.2f}" + (" interp" if inst.get("interp") else "")
        elif cls == "tissue":                                  # 誤抓被點回 tissue，但沒有輪廓
            col = (0, 165, 255)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, thick)
            label = "NO MASK"
        else:
            col = (0, 0, 255)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, thick)
            label = f"BG {inst['conf']:.2f}"
        cv2.putText(img, label, (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return img


def label_lines(sample, classes):
    w, h = sample["size"]
    return [yolo_line(i["poly"], w, h) for i, c in zip(sample["instances"], classes)
            if c == "tissue" and i.get("poly")]


# ==========================================
# sheet：隨機抽幾張畫成接觸表
# ==========================================
def cmd_sheet(args):
    samples = load_samples(args.staging)
    if args.kind:
        samples = [s for s in samples if s["kind"] == args.kind]
    if not samples:
        raise SystemExit("沒有樣本，先跑 mine")
    random.Random(args.seed).shuffle(samples)
    tiles = []
    for s in samples[: args.n]:
        img = cv2.imread(s["image"])
        draw_sample(img, s, [i["cls"] for i in s["instances"]])
        cv2.putText(img, f"{s['kind']} {s['id']}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (480, 360)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [cv2.hconcat(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(args.out, cv2.vconcat(rows))
    print(f"接觸表 ({min(args.n, len(samples))} 張) -> {args.out}")


# ==========================================
# review：人工檢查
# ==========================================
KEY_LEFT = {81, 65361, 2424832, ord("a")}
KEY_RIGHT = {83, 65363, 2555904, ord("d"), ord(" ")}


def _hit_instance(sample, x, y, tol=6.0):
    """滑鼠點到哪個實例 (先找輪廓，再找框)"""
    best, best_d = None, None
    for k, inst in enumerate(sample["instances"]):
        if inst.get("poly"):
            d = cv2.pointPolygonTest(np.asarray(inst["poly"], np.float32), (float(x), float(y)), True)
            if d >= -tol and (best_d is None or -d < best_d):
                best, best_d = k, -d
    if best is not None:
        return best
    for k, inst in enumerate(sample["instances"]):
        x0, y0, x1, y1 = inst["box"]
        if x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol:
            return k
    return None


def cmd_review(args):
    samples = load_samples(args.staging)
    if args.kind:
        samples = [s for s in samples if s["kind"] == args.kind]
    if not samples:
        raise SystemExit("沒有樣本，先跑 mine")
    review_path = os.path.join(os.path.dirname(args.staging.rstrip("/")), "review.json")
    review = load_review(review_path)
    i = next((k for k, s in enumerate(samples) if s["id"] not in review), 0)
    state = {"cls": None}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        k = _hit_instance(samples[i], x, y)
        if k is not None:
            state["cls"][k] = "bg" if state["cls"][k] == "tissue" else "tissue"
            state["dirty"] = True

    win = "autolabel review"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    print("[d / → / 空白] 留   [x] 丟   [a / ←] 上一張   [q] 存檔離開   滑鼠點實例：tissue <-> 背景")
    while 0 <= i < len(samples):
        s = samples[i]
        state["cls"] = instance_classes(s, review)
        state["dirty"] = False
        base = cv2.imread(s["image"])
        while True:
            img = draw_sample(base.copy(), s, state["cls"])
            status = review.get(s["id"], {}).get("keep")
            status = "unreviewed" if status is None else ("KEEP" if status else "DROP")
            n_done = sum(1 for x in samples if x["id"] in review)
            cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(img, f"{i + 1}/{len(samples)} [{s['kind']}] {s['id']}  {status}  done {n_done}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(img, "d/->:keep  x:drop  a/<-:back  q:quit  click:tissue<->bg",
                        (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow(win, img)
            key = cv2.waitKeyEx(50)
            if key == -1:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    key = ord("q")
                else:
                    continue
            if key in KEY_RIGHT or key == ord("x"):
                review[s["id"]] = {"keep": key != ord("x"), "cls": state["cls"]}
                save_review(review_path, review)
                i += 1
                break
            if key in KEY_LEFT:
                if state["dirty"]:
                    review[s["id"]] = {"keep": review.get(s["id"], {}).get("keep", True), "cls": state["cls"]}
                    save_review(review_path, review)
                i = max(0, i - 1)
                break
            if key == ord("q"):
                save_review(review_path, review)
                cv2.destroyAllWindows()
                print(f"已存檔：看過 {sum(1 for x in samples if x['id'] in review)}/{len(samples)} 張 -> {review_path}")
                return
    save_review(review_path, review)
    cv2.destroyAllWindows()
    print(f"全部看完 ({len(samples)} 張) -> {review_path}")


# ==========================================
# export
# ==========================================
def split_of(sample, val_ratio=VAL_RATIO):
    block = f"{sample['video']}:{int(sample['t'] // SPLIT_BLOCK_SEC)}"
    hv = int(hashlib.md5(block.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if hv < val_ratio else "train"


def cmd_export(args):
    samples = load_samples(args.staging)
    review = load_review(os.path.join(os.path.dirname(args.staging.rstrip("/")), "review.json"))
    chosen = []
    for s in samples:
        r = review.get(s["id"])
        if r is None and not args.include_unreviewed:
            continue
        if r is not None and not r.get("keep", True):
            continue
        chosen.append(s)
    if not chosen:
        raise SystemExit("沒有可匯出的樣本 (還沒 review？可加 --include-unreviewed)")

    out = os.path.abspath(args.out)
    if os.path.exists(os.path.join(out, "data.yaml")):
        if not args.overwrite:
            raise SystemExit(f"{out} 已經有資料集，加 --overwrite 覆蓋")
        for sub in ("images", "labels"):
            shutil.rmtree(os.path.join(out, sub), ignore_errors=True)
        for name in os.listdir(out):
            if name.endswith(".cache"):
                os.remove(os.path.join(out, name))

    splits = {s["id"]: split_of(s, args.val_ratio) for s in chosen}
    if len(set(splits.values())) == 1 and len(chosen) > 1:     # 區段太少全落在同一邊 -> 最後一個區段當 val
        last = max(chosen, key=lambda s: (s["video"], s["t"]))
        blk = (last["video"], int(last["t"] // SPLIT_BLOCK_SEC))
        for s in chosen:
            if (s["video"], int(s["t"] // SPLIT_BLOCK_SEC)) == blk:
                splits[s["id"]] = "val"

    stats = {}
    for s in chosen:
        sp = splits[s["id"]]
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out, sub, sp), exist_ok=True)
        shutil.copy2(s["image"], os.path.join(out, "images", sp, s["id"] + ".jpg"))
        lines = label_lines(s, instance_classes(s, review))
        with open(os.path.join(out, "labels", sp, s["id"] + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        st = stats.setdefault(sp, {"images": 0, "instances": 0, "negatives": 0})
        st["images"] += 1
        st["instances"] += len(lines)
        st["negatives"] += not lines
        st[s["kind"]] = st.get(s["kind"], 0) + 1

    with open(os.path.join(out, "data.yaml"), "w") as fh:
        fh.write("# 由 model/yolo26/autolabel_tissue.py export 產生：實景的辨識失敗畫面 (漏抓 / 誤抓) + 錨點\n")
        fh.write(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: tissue\n")
    for sp in ("train", "val"):
        if sp in stats:
            print(f"   {sp}: {stats[sp]}")
    print(f"資料集 -> {out}/data.yaml")


# ==========================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine", help="挑出辨識失敗的畫面並自動標註")
    m.add_argument("videos", nargs="+", help="錄影的 _native.mp4 (960x720，與實飛 YOLO 的輸入一致)")
    m.add_argument("--staging", default=DEFAULT_STAGING)
    m.add_argument("--model", default=DEFAULT_MODEL)
    m.add_argument("--sam", default=DEFAULT_SAM, help="SAM 權重，第一次會自動下載")
    m.add_argument("--imgsz", type=int, default=960)
    m.add_argument("--stride", type=int, default=STRIDE)
    m.add_argument("--max-per-track", type=int, default=MAX_PER_TRACK)
    m.add_argument("--redetect", action="store_true", help="不用偵測快取，重新跑 YOLO")
    m.add_argument("--log", default=None, help="飛行紀錄 _log.csv (預設找影片旁邊同名的)；只能配一支影片")
    m.add_argument("--no-log", action="store_true", help="不看飛行紀錄 (不排除落地的畫面)")

    s = sub.add_parser("sheet", help="隨機抽幾張畫成接觸表")
    s.add_argument("--staging", default=DEFAULT_STAGING)
    s.add_argument("--out", default=os.path.join(os.path.dirname(DEFAULT_STAGING), "sheet.jpg"))
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--kind", choices=("missed", "fp", "anchor"))
    s.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("review", help="人工檢查 (OpenCV 視窗)")
    r.add_argument("--staging", default=DEFAULT_STAGING)
    r.add_argument("--kind", choices=("missed", "fp", "anchor"))

    e = sub.add_parser("export", help="匯出 YOLO-seg 資料集")
    e.add_argument("--staging", default=DEFAULT_STAGING)
    e.add_argument("--out", default=os.path.join(ROOT, "datasets/tissue_real"))
    e.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    e.add_argument("--include-unreviewed", action="store_true")
    e.add_argument("--overwrite", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "mine" and args.log and len(args.videos) > 1:
        ap.error("--log 只能配一支影片 (多支影片會自動找各自的 _log.csv)")
    {"mine": cmd_mine, "sheet": cmd_sheet, "review": cmd_review, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()
