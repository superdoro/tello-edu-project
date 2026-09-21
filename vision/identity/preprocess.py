"""
影像前處理：人臉對齊、人體裁切、品質量測。

這個模組只依賴 numpy 與 cv2，不碰 torch。所有「要不要用這張影像」的判斷都集中在這裡，
特徵抽取器 (embedders.py) 只負責把已經整理好的影像轉成向量。

為什麼品質判斷要獨立出來：單目人臉辨識最大的誤差來源不是模型，而是餵給模型的影像不合格
(太小、太側、太模糊、逆光)。這些情況下 embedding 不是「比較不準」而是「完全不可信」，
必須在進模型之前就擋掉，否則會把垃圾樣本收進特徵庫並持續污染後續比對。
"""

import math
import cv2
import numpy as np

# ==========================================
# 人臉對齊模板
# 眼線位於 38% 高度、瞳距佔 36% 寬度。
# 這是 MTCNN 人臉框的典型統計值，而 facenet-pytorch 的 VGGFace2 權重正是在
# MTCNN(image_size=160, margin=0) 的裁切上訓練的，模板越接近該分佈，embedding 越可信。
# ==========================================
FACE_OUT = 160
DST_EYE_Y = 61.0
DST_IOD = 58.0
DST_CX = 80.0
YAW_RECENTER = 0.25   # 依鼻子偏移微調裁切中心的阻尼係數 (1.0 = 完全對齊鼻子)

# ==========================================
# 人臉品質門檻
# 兩欄：比對 (match) 與登錄 (enroll)。登錄一律比比對嚴格，
# 因為比對錯了只影響這一格，登錄錯了會污染特徵庫並持續影響之後每一格。
# ==========================================
EYE_CONF_MATCH, EYE_CONF_ENROLL = 0.50, 0.60      # 雙眼關鍵點信心
NOSE_CONF_MATCH, NOSE_CONF_ENROLL = 0.40, 0.50    # 鼻子只拿來算轉頭角度，可較寬鬆
IOD_MATCH, IOD_ENROLL = 16.0, 22.0                # 瞳距像素 (原生座標)
ASYM_MATCH, ASYM_ENROLL = 0.65, 0.40              # 轉頭代理量：0.65≈38度, 0.40≈25度
YAWERR_MATCH, YAWERR_ENROLL = 55.0, 32.0          # 既有的 face_yaw_error，與 asym 取 AND
ROLL_MATCH, ROLL_ENROLL = 35.0, 25.0              # 頭部傾斜 (度)
OUTSIDE_MATCH, OUTSIDE_ENROLL = 0.30, 0.10        # 裁切框超出畫面的比例
BLUR_MATCH, BLUR_ENROLL = 20.0, 55.0              # 對齊後 chip 的 Laplacian 變異數
MEAN_MATCH = (35.0, 225.0)
MEAN_ENROLL = (45.0, 210.0)
STD_MATCH, STD_ENROLL = 15.0, 22.0

# ==========================================
# 人體裁切
# ==========================================
BODY_OUT_WH = (128, 160)     # (寬, 高)，長寬比 0.8 符合「頭 + 軀幹」
BODY_MIN_AREA = 24 * 32      # 小於此面積的裁切沒有資訊量
BODY_SCALE_ENROLL = 70.0     # 人太小的時候衣著細節不足以當樣本
BODY_BLUR_ENROLL = 25.0      # 裁切後的清晰度下限

# 分箱邊界 (見 gallery.py 的分箱配額設計)
SCALE_BIN_EDGES = (90.0, 180.0)      # body_scale：遠 / 中 / 近
BRIGHT_BIN_EDGES = (70.0, 170.0)     # HSV 明度平均：暗 / 中 / 亮
ASYM_FRONT = 0.15                    # 人臉視角分箱：正面的門檻
BACK_FACE_CONF = 0.30                # 臉部關鍵點信心低於此值視為背對
SIDE_YAW_ERROR = 45.0                # 體態視角分箱：側面的門檻

VIEW_FRONT, VIEW_SIDE, VIEW_BACK = 0, 1, 2          # 體態視角分箱
FACE_FRONT, FACE_TURN_L, FACE_TURN_R = 0, 1, 2      # 人臉視角分箱


def bbox_iou(a, b):
    """兩個 (x1,y1,x2,y2) 框的交集比聯集"""
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def similarity_from_two_points(src_l, src_r, dst_l, dst_r):
    """
    由兩組對應點解出唯一的相似變換 (旋轉 + 等比縮放 + 平移)。

    回傳 [[a, -b, tx], [b, a, ty]]，其中 a = s*cos(theta)、b = s*sin(theta)。
    兩點對相似變換而言已經是完全定解 (4 個自由度)，不需要第三點。
    """
    sx, sy = src_r[0] - src_l[0], src_r[1] - src_l[1]
    dx, dy = dst_r[0] - dst_l[0], dst_r[1] - dst_l[1]
    n2 = sx * sx + sy * sy
    if n2 < 1e-6:
        return None
    a = (sx * dx + sy * dy) / n2
    b = (sx * dy - sy * dx) / n2
    tx = dst_l[0] - (a * src_l[0] - b * src_l[1])
    ty = dst_l[1] - (b * src_l[0] + a * src_l[1])
    return np.float32([[a, -b, tx], [b, a, ty]])


def scale_point(p, scale_xy):
    return (p[0] * scale_xy[0], p[1] * scale_xy[1])


def face_quality_probe(kpts, face_yaw_error=0.0, scale_xy=(1.0, 1.0)):
    """
    只靠關鍵點就能算出的人臉品質指標，不需要裁切影像。

    用途是 0 成本預篩：人背對鏡頭或距離太遠時，這裡就能判定「這一格沒有可用的臉」，
    完全不必進行 warp 與模型推論。
    """
    q = {
        "iod": 0.0, "asym": 9.9, "roll": 90.0,
        "eye_conf": 0.0, "nose_conf": 0.0,
        "face_yaw_error": abs(float(face_yaw_error)),
        "geom_match": False, "geom_enroll": False,
        "reason": "no_face",
    }
    if len(kpts) < 5:
        return q

    nose, eye_a, eye_b = kpts[0], kpts[1], kpts[2]
    q["eye_conf"] = float(min(eye_a[2], eye_b[2]))
    q["nose_conf"] = float(nose[2])

    if q["eye_conf"] < EYE_CONF_MATCH:
        # 單眼可信 (純側臉) 也走這條：耳朵到眼睛的距離隨頭部俯仰變化太大，不適合拿來硬湊對齊
        q["reason"] = "eye_conf_low"
        return q

    # 使用者的左眼(索引1)在畫面上通常偏右，不能寫死索引，一律依 x 座標排序
    p_a, p_b = scale_point(eye_a, scale_xy), scale_point(eye_b, scale_xy)
    src_l, src_r = (p_a, p_b) if p_a[0] <= p_b[0] else (p_b, p_a)
    q["eye_l"], q["eye_r"] = src_l, src_r

    iod = math.hypot(src_r[0] - src_l[0], src_r[1] - src_l[1])
    q["iod"] = float(iod)
    if iod < IOD_MATCH:
        q["reason"] = "iod_too_small"
        return q

    q["roll"] = abs(math.degrees(math.atan2(src_r[1] - src_l[1], src_r[0] - src_l[0])))
    if q["roll"] > 90.0:
        q["roll"] = 180.0 - q["roll"]

    if q["nose_conf"] >= NOSE_CONF_MATCH:
        np_ = scale_point(nose, scale_xy)
        q["nose"] = np_
        d_l = math.hypot(np_[0] - src_l[0], np_[1] - src_l[1])
        d_r = math.hypot(np_[0] - src_r[0], np_[1] - src_r[1])
        q["asym"] = float(abs(d_l - d_r) / max(iod, 1e-6))
    else:
        q["reason"] = "nose_conf_low"
        return q

    # asym 與 face_yaw_error 是兩個獨立來源的轉頭估計，取 AND 讓彼此互相把關
    q["geom_match"] = (q["asym"] <= ASYM_MATCH and q["face_yaw_error"] <= YAWERR_MATCH
                       and q["roll"] <= ROLL_MATCH)
    q["geom_enroll"] = (q["eye_conf"] >= EYE_CONF_ENROLL and q["nose_conf"] >= NOSE_CONF_ENROLL
                        and iod >= IOD_ENROLL and q["asym"] <= ASYM_ENROLL
                        and q["face_yaw_error"] <= YAWERR_ENROLL and q["roll"] <= ROLL_ENROLL)
    if not q["geom_match"]:
        q["reason"] = "too_side" if q["asym"] > ASYM_MATCH else "roll_too_big"
    else:
        q["reason"] = "ok"
    return q


def chip_blur(chip_rgb):
    """
    對齊後 chip 的 Laplacian 變異數。

    關鍵：在「對齊後」的 160x160 上量，門檻才與拍攝距離無關。
    如果在原圖 ROI 上量，同一張清晰的臉在近距離與遠距離會差兩個數量級，門檻根本沒辦法設。
    """
    gray = cv2.cvtColor(chip_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def chip_exposure(chip_rgb):
    gray = cv2.cvtColor(chip_rgb, cv2.COLOR_RGB2GRAY)
    return float(gray.mean()), float(gray.std())


def outside_ratio(m_inv, img_w, img_h, grid=8):
    """把 chip 的取樣格點反投影回原圖，統計落在畫面外的比例"""
    xs = np.linspace(0.5, FACE_OUT - 0.5, grid, dtype=np.float32)
    gx, gy = np.meshgrid(xs, xs)
    pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size, np.float32)], axis=0)
    src = m_inv @ pts
    outside = (src[0] < 0) | (src[0] > img_w - 1) | (src[1] < 0) | (src[1] > img_h - 1)
    return float(outside.mean())


def align_face(rgb, kpts, face_yaw_error=0.0, scale_xy=(1.0, 1.0)):
    """
    以雙眼的相似變換把人臉對齊成 160x160。

    為什麼不用「雙眼 + 鼻子」做 3 點最小平方：鼻子是五官中隨轉頭位移最大的點，
    把它放進變換求解會讓對齊結果隨轉頭角度平移與旋轉，等於把轉頭雜訊注入 chip 的座標系，
    特徵庫裡的樣本之間就不再共用同一個基準。鼻子改用在兩個地方：
    品質指標 (asym)，以及下方裁切中心的小幅水平補償。

    回傳 (chip 或 None, 品質 dict)。
    """
    q = face_quality_probe(kpts, face_yaw_error, scale_xy)
    if not q["geom_match"]:
        return None, q

    src_l, src_r = q["eye_l"], q["eye_r"]

    # 依鼻子的水平偏移微調裁切中心，讓半側臉時整張臉還留在 chip 內。
    # 阻尼到 0.25 是為了不要把轉頭抖動放大成大幅平移。
    shift = 0.0
    if "nose" in q:
        eye_mid_x = (src_l[0] + src_r[0]) / 2.0
        u = (q["nose"][0] - eye_mid_x) / max(q["iod"], 1e-6)
        shift = -YAW_RECENTER * u * DST_IOD
        shift = float(np.clip(shift, -0.25 * FACE_OUT, 0.25 * FACE_OUT))

    dst_l = (DST_CX - DST_IOD / 2.0 + shift, DST_EYE_Y)
    dst_r = (DST_CX + DST_IOD / 2.0 + shift, DST_EYE_Y)

    m = similarity_from_two_points(src_l, src_r, dst_l, dst_r)
    if m is None:
        q["reason"] = "degenerate_transform"
        return None, q

    h_img, w_img = rgb.shape[:2]
    m3 = np.vstack([m, [0, 0, 1]]).astype(np.float32)
    try:
        m_inv = np.linalg.inv(m3)[:2]
    except np.linalg.LinAlgError:
        q["reason"] = "degenerate_transform"
        return None, q

    q["outside"] = outside_ratio(m_inv, w_img, h_img)
    if q["outside"] > OUTSIDE_MATCH:
        q["reason"] = "out_of_frame"
        return None, q

    chip = cv2.warpAffine(rgb, m, (FACE_OUT, FACE_OUT),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    q["blur"] = chip_blur(chip)
    q["mean"], q["std"] = chip_exposure(chip)

    if q["blur"] < BLUR_MATCH:
        q["reason"] = "too_blurry"
        return None, q
    if not (MEAN_MATCH[0] <= q["mean"] <= MEAN_MATCH[1]) or q["std"] < STD_MATCH:
        q["reason"] = "bad_exposure"
        return None, q

    q["ok_match"] = True
    q["ok_enroll"] = bool(
        q["geom_enroll"]
        and q["outside"] <= OUTSIDE_ENROLL
        and q["blur"] >= BLUR_ENROLL
        and MEAN_ENROLL[0] <= q["mean"] <= MEAN_ENROLL[1]
        and q["std"] >= STD_ENROLL
    )
    q["reason"] = "ok"
    q["quality"] = face_quality_score(q)
    return chip, q


def face_quality_score(q):
    """把各項品質指標壓成 0~1 的單一分數，供特徵庫汰換時比較優劣"""
    def clamp01(v):
        return float(min(1.0, max(0.0, v)))
    return (0.35 * clamp01(q.get("iod", 0.0) / 32.0)
            + 0.25 * (1.0 - clamp01(q.get("asym", 1.0) / ASYM_ENROLL))
            + 0.25 * clamp01(q.get("blur", 0.0) / 200.0)
            + 0.15 * clamp01(q.get("std", 0.0) / 60.0))


def body_crop(rgb, box_xyxy, kpts, scale_xy=(1.0, 1.0), out_wh=BODY_OUT_WH):
    """
    裁切「頭 + 軀幹」。

    不取全身的理由：衣著軀幹是最穩定的外觀線索；腿部會帶進大量背景又隨步態劇烈變化；
    而且胸腔追蹤模式下腿常常根本不在畫面裡，時有時無的區域會讓特徵庫樣本彼此不可比。

    優先用關鍵點導出的貼身框 (背景比 YOLO 框少)，關鍵點不足才退回 YOLO 框。
    """
    h_img, w_img = rgb.shape[:2]
    pts = [scale_point(kpts[i], scale_xy) for i in (5, 6, 11, 12)
           if len(kpts) > i and kpts[i][2] > 0.5]

    if len(pts) >= 3:
        sh = [scale_point(kpts[i], scale_xy) for i in (5, 6) if kpts[i][2] > 0.5]
        if len(sh) == 2:
            ref = math.hypot(sh[0][0] - sh[1][0], sh[0][1] - sh[1][1])
        else:
            ref = 0.0
        if ref < 1.0:
            ys = [p[1] for p in pts]
            ref = max(1.0, (max(ys) - min(ys)) * 0.6)
        x1 = min(p[0] for p in pts) - 0.25 * ref
        x2 = max(p[0] for p in pts) + 0.25 * ref
        y1 = min(p[1] for p in pts) - 0.55 * ref      # 往上含頭部與髮型
        y2 = max(p[1] for p in pts) + 0.12 * ref      # 往下略過臀部就好
    elif box_xyxy is not None:
        bx1, by1, bx2, by2 = [box_xyxy[0] * scale_xy[0], box_xyxy[1] * scale_xy[1],
                              box_xyxy[2] * scale_xy[0], box_xyxy[3] * scale_xy[1]]
        bw = bx2 - bx1
        x1, x2 = bx1 + 0.08 * bw, bx2 - 0.08 * bw     # 左右各縮 8% 去掉背景
        y1, y2 = by1, by1 + 0.65 * (by2 - by1)        # 只取上 65%
    else:
        return None

    x1 = int(max(0, math.floor(x1)))
    y1 = int(max(0, math.floor(y1)))
    x2 = int(min(w_img, math.ceil(x2)))
    y2 = int(min(h_img, math.ceil(y2)))
    if (x2 - x1) * (y2 - y1) < BODY_MIN_AREA or x2 <= x1 or y2 <= y1:
        return None

    crop = rgb[y1:y2, x1:x2]
    interp = cv2.INTER_AREA if crop.shape[1] > out_wh[0] else cv2.INTER_LINEAR
    return cv2.resize(crop, out_wh, interpolation=interp)


def hsv_stripe_hist(crop_rgb, stripes=3, h_bins=12, s_bins=4):
    """
    上 / 中 / 下三條紋的 HSV 色相-飽和度直方圖。

    ImageNet 預訓練特徵是通用紋理特徵，不是為了分辨「是不是同一個人」而訓練的，
    同人與異人的 cosine 會全部擠在高位的窄帶裡。補上這個對「今天穿什麼」極度敏感的
    手工特徵，能明顯拉開鑑別力，而成本只有約 0.1 毫秒。
    """
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    feats = []
    for band in np.array_split(hsv, stripes, axis=0):
        # 濾掉極低飽和與極暗的像素，避免陰影與白牆主導整個直方圖
        mask = cv2.inRange(band, (0, 40, 30), (179, 255, 255))
        hist = cv2.calcHist([band], [0, 1], mask, [h_bins, s_bins], [0, 180, 0, 256]).flatten()
        total = hist.sum()
        feats.append(hist / total if total > 0 else hist)
    v = np.concatenate(feats).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def crop_brightness(crop_rgb):
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 2].mean())


# ==========================================
# 特徵庫分箱
# 三個維度都從既有資訊免費取得，不需要額外模型
# ==========================================

def face_view_bin(asym, nose_x=None, eye_mid_x=None):
    """人臉視角：正面 / 偏一側 / 偏另一側 (由鼻子相對雙眼中點的方向決定帶號)"""
    if asym <= ASYM_FRONT:
        return FACE_FRONT
    if nose_x is None or eye_mid_x is None:
        return FACE_TURN_L
    return FACE_TURN_L if nose_x >= eye_mid_x else FACE_TURN_R


def body_view_bin(kpts, face_yaw_error):
    """正面 / 側面 / 背面：用臉部關鍵點的可見度判定，不需要額外模型"""
    if len(kpts) < 3:
        return VIEW_BACK
    face_vis = max(kpts[0][2], kpts[1][2], kpts[2][2])
    if face_vis < BACK_FACE_CONF:
        return VIEW_BACK
    if abs(face_yaw_error) > SIDE_YAW_ERROR:
        return VIEW_SIDE
    return VIEW_FRONT


def scale_bin(body_scale):
    if body_scale < SCALE_BIN_EDGES[0]:
        return 0
    return 1 if body_scale < SCALE_BIN_EDGES[1] else 2


def bright_bin(v_mean):
    if v_mean < BRIGHT_BIN_EDGES[0]:
        return 0
    return 1 if v_mean < BRIGHT_BIN_EDGES[1] else 2
