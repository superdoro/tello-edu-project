"""
身分辨識的端到端實測工具（不需要無人機）。

直接重用 BodyPoseTracker，只是把影像來源換成 webcam 或影片檔，按鍵行為與飛行時一致：
    r  開始建檔 / 清除身分   (等同飛行時的 R 鍵)
    f  切換胸腔 / 手部追蹤   (等同 F 鍵)
    q  離開

用途：
  1. 把計畫中「估算」的耗時換成實測值，決定降頻門檻
  2. 用 --dump-chips 肉眼確認人臉對齊、長寬比修正與通道順序是否正確
  3. 用 --check-channels 確認通道順序（最陰險的一類錯誤）

範例：
    python utils/identity_bench.py --source 0 --bgr
    python utils/identity_bench.py --source clip.mp4 --bgr --dump-chips /tmp/chips

重要：--bgr 不能忘。cv2.VideoCapture 給的是真正的 BGR，而 Tello 的畫面實際上是 RGB
      （djitellopy 走 PIL），只是整條管線把它當成 BGR 在用。搞錯的話，
      在這裡調出來的門檻放到飛機上就是錯的。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def check_channels(tracker, cap, args):
    """
    通道順序的決定性檢查。

    拿同一個人相隔約 1 秒的兩張臉，各自用「正確通道」與「交換通道」算 embedding。
    正確的那一組，同一個人的相似度必須明顯較高 —— 因為只有正確的通道順序
    才落在模型的訓練分佈內。幾秒鐘就能永久排除這類錯誤。
    """
    from vision.identity import preprocess as pp

    print("\n--- 通道順序檢查 ---")
    samples = []
    t0 = time.time()
    while len(samples) < 2 and (time.time() - t0) < 12:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (720, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if args.bgr else frame
        data = tracker.process_frame(frame)
        if data.target is None:
            continue
        chip, q = pp.align_face(rgb, data.target['keypoints'], data.target['face_yaw_error'])
        if chip is not None and (not samples or (time.time() - samples[-1][1]) > 1.0):
            samples.append((chip, time.time()))
            print(f"  取得樣本 {len(samples)}/2 (iod {q['iod']:.0f}px, 模糊度 {q['blur']:.0f})")

    if len(samples) < 2:
        print("  略過：沒有取到兩張可用的人臉（請正對鏡頭再試一次）")
        return

    emb = tracker.identity.face_emb
    a, b = samples[0][0], samples[1][0]
    right = emb.embed_batch([a, b])
    wrong = emb.embed_batch([a[:, :, ::-1].copy(), b[:, :, ::-1].copy()])
    sim_right = float(right[0] @ right[1])
    sim_wrong = float(wrong[0] @ wrong[1])
    print(f"  正確通道的同人相似度: {sim_right:.4f}")
    print(f"  交換通道的同人相似度: {sim_wrong:.4f}")
    if sim_right > sim_wrong:
        print("  ✓ 通道順序正確")
    else:
        print("  ✗ 通道順序可能有問題：請檢查 --bgr 旗標與 frame_is_rgb 設定")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam 編號或影片檔路徑")
    ap.add_argument("--bgr", action="store_true",
                    help="來源是真正的 BGR (cv2.VideoCapture 都要加；Tello 不用)")
    ap.add_argument("--gallery-dir", default="/tmp/identity_bench",
                    help="測試用的特徵庫目錄，避免污染正式的 data/identity")
    ap.add_argument("--dump-chips", default=None, help="把對齊後的人臉存到這個目錄")
    ap.add_argument("--check-channels", action="store_true", help="啟動時做通道順序檢查")
    ap.add_argument("--fps-cap", type=float, default=0.0)
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    from vision.pose_tracker import BodyPoseTracker

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"無法開啟影像來源：{args.source}")
        return 1

    tracker = BodyPoseTracker(gallery_dir=args.gallery_dir)
    tracker.identity.frame_is_rgb = not args.bgr
    if args.dump_chips:
        tracker.identity.debug_dump_dir = args.dump_chips

    if args.check_channels:
        tracker.identity._ensure_models()
        check_channels(tracker, cap, args)

    totals, faces, bodies = [], [], []
    print("\n按 R 開始建檔 / 清除身分，F 切換追蹤模式，Q 離開\n")

    while True:
        loop_t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (720, 480))

        t0 = time.time()
        data = tracker.process_frame(frame)
        total_ms = (time.time() - t0) * 1000
        totals.append(total_ms)
        tm = tracker.identity.timing
        if tm['n_face']:
            faces.append(tm['face_ms'])
        if tm['n_body']:
            bodies.append(tm['body_ms'])

        out = data.annotated_frame
        cv2.putText(out, f"{total_ms:.0f}ms  face {tm['face_ms']:.1f}  body {tm['body_ms']:.1f}",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if not args.no_window:
            cv2.imshow("identity bench", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                tracker.reset_target()
            if key == ord('f'):
                tracker.toggle_tracking_mode()

        if args.fps_cap > 0:
            delay = 1.0 / args.fps_cap - (time.time() - loop_t0)
            if delay > 0:
                time.sleep(delay)

    cap.release()
    cv2.destroyAllWindows()
    tracker.shutdown()

    print("\n" + "=" * 46)
    print(f"影格數 {len(totals)}")
    print(f"整格耗時   p50 {percentile(totals,50):6.1f}ms   p95 {percentile(totals,95):6.1f}ms")
    if faces:
        print(f"人臉特徵   p50 {percentile(faces,50):6.1f}ms   p95 {percentile(faces,95):6.1f}ms"
              f"   ({len(faces)} 格有臉)")
    if bodies:
        print(f"體態特徵   p50 {percentile(bodies,50):6.1f}ms   p95 {percentile(bodies,95):6.1f}ms")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"VRAM 峰值  {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    except Exception:
        pass
    print(f"特徵庫     臉 {tracker.identity.status()['face_n']} / "
          f"體態 {tracker.identity.status()['body_n']}")
    print("=" * 46)
    return 0


if __name__ == "__main__":
    sys.exit(main())
