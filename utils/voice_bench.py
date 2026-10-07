"""
語音指令辨識的正確率測試（可加無人機噪音）。

為什麼需要這支工具：語音指令在安靜的辦公室能用，不代表在螺旋槳旁邊能用。
四旋翼噪音的能量集中在 300~900Hz，正好蓋住人聲的共振峰，是最難對付的干擾。
與其飛到一半才發現指令叫不動，不如先在地面量出「訊噪比掉到多少就開始失效」。

兩種用法：

  1. 合成語音 + 合成噪音（預設，不需要麥克風）
        python utils/voice_bench.py
        python utils/voice_bench.py --snr 10 --save-wav /tmp/mix
     首次執行會用 gTTS 合成測試語句，需要網路；之後會快取到 utils/.voice_bench_cache/。

  2. 真人語音 + 合成噪音（建議實機前跑一次）
        python utils/voice_bench.py --mic
     會提示你逐句念出來，錄下之後混入噪音測試。
     這比合成語音準得多 —— TTS 的發音和你本人的口音是兩回事。

噪音是合成的，不是錄音：Tello 的影像串流不含聲音，專案裡也沒有現成音檔。
合成模型依實際聲學特徵建立（葉片通過頻率的諧波堆疊 + 寬頻氣動噪音），
頻譜特性接近真實四旋翼，但絕對音量無從對應 —— 所以結果要看「SNR 幾 dB 開始失效」
這個相對值，不要當成「距離幾公尺可用」。
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

SR = 16000
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".voice_bench_cache")

# (說的話, 期望的指令)。None = 不該觸發任何指令。
# 說法是掃描過抗噪表現後挑的 —— 同一個意思有好幾種講法，在螺旋槳噪音下
# 差異很大（例如「後退」在 15dB 就認不出來，「往後」到 5dB 都還行）。
TEST_SET = [
    # 推薦說法 (抗噪最好)
    ("無人機起飛", "起飛"),
    ("無人機降落", "降落"),
    ("無人機前進", "前進"),
    ("無人機往後", "後退"),
    ("無人機左轉", "向左"),
    ("無人機往右", "向右"),
    ("無人機升高", "上升"),
    ("無人機下降", "下降"),
    ("無人機轉身", "轉向"),
    ("無人機停止", "停止"),
    ("無人機切換模式", "切換"),
    ("小白起飛", "起飛"),
    ("悟空降落", "降落"),
    # 非推薦但使用者很可能脫口而出的說法 —— 容忍表要救的就是這些
    ("無人機後退", "後退"),
    ("無人機向左", "向左"),
    ("無人機往左", "向左"),
    ("無人機向右", "向右"),
    ("無人機右邊", "向右"),
    ("無人機停下", "停止"),
    # 以下不該觸發（誤觸發測試）
    ("今天天氣真好", None),
    ("我要去買無人機", None),
    ("起飛", None),
    ("這個東西要往左邊放", None),
    ("無人機是網上買的", None),
    ("無人機的效果不錯", None),
    ("無人機有錯誤訊息", None),
    ("這條路往右轉就到了", None),
    ("無人機的形象很好", None),
    ("小白你要不要一起吃飯", None),
]

def drone_noise(dur, sr=SR, seed=0):
    """
    合成四旋翼噪音。

    Tello 懸停時馬達約 16000~20000 RPM、雙葉槳，
    葉片通過頻率 BPF = RPM/60*2 約 530~670 Hz，再加上諧波與寬頻氣動噪音。
    四顆馬達轉速略有差異會產生真實的拍頻，機動時轉速還會飄，所以加上緩慢調變。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    tonal = np.zeros_like(t)
    for bpf in (545., 588., 612., 655.):                      # 四顆馬達
        drift = 1 + 0.01 * np.sin(2 * np.pi * rng.uniform(0.3, 1.2) * t + rng.uniform(0, 6))
        for h in range(1, 7):                                  # 諧波，越高越弱
            tonal += (1.0 / h ** 1.3) * np.sin(2 * np.pi * bpf * h * drift * t + rng.uniform(0, 6))
    tonal /= np.abs(tonal).max()

    # 寬頻氣動噪音（粉紅噪音，約 -3dB/oct）
    w = rng.normal(0, 1, len(t))
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(len(t), 1 / sr)
    f[0] = f[1]
    pink = np.fft.irfft(W / np.sqrt(f), n=len(t))
    pink /= np.abs(pink).max()

    out = 0.45 * tonal + 0.55 * pink
    return out / np.abs(out).max()


def trim_silence(x, thresh=0.02):
    """去掉前後靜音，避免把靜音算進語音功率而高估 SNR"""
    idx = np.where(np.abs(x) > thresh * np.abs(x).max())[0]
    return x[idx[0]:idx[-1] + 1] if len(idx) else x


def mix_at_snr(speech, noise, snr_db):
    """把噪音依指定訊噪比混入語音。SNR 以去除靜音後的語音功率為準。"""
    if snr_db is None:
        return speech
    p_s = np.mean(trim_silence(speech) ** 2)
    n = noise[:len(speech)] if len(noise) >= len(speech) else np.tile(noise, len(speech) // len(noise) + 1)[:len(speech)]
    p_n = np.mean(n ** 2)
    scale = np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))
    out = speech + scale * n
    peak = np.abs(out).max()
    return out / peak * 0.95 if peak > 0.95 else out


def tts(text):
    """用 gTTS 合成語句並快取成 16kHz 單聲道 float"""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, hashlib.md5(text.encode()).hexdigest()[:12] + ".npy")
    if os.path.exists(path):
        return np.load(path)
    from gtts import gTTS
    import av
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3 = f.name
    gTTS(text, lang="zh-TW").save(mp3)
    c = av.open(mp3)
    rs = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SR)
    buf = []
    for fr in c.decode(audio=0):
        for o in rs.resample(fr):
            buf.append(o.to_ndarray().reshape(-1))
    for o in rs.resample(None):
        buf.append(o.to_ndarray().reshape(-1))
    os.remove(mp3)
    x = np.concatenate(buf).astype(np.float32) / 32768.0
    np.save(path, x)
    return x


def record(text, seconds=3.0, device=None):
    """提示使用者念出一句話並用 arecord 錄下來"""
    from core.voice_controller import VoiceController
    device = device or VoiceController._find_device()
    input(f'  準備好後按 Enter，然後念：「{text}」  ')
    print(f"    錄音中 {seconds:.0f} 秒...", end="", flush=True)
    out = subprocess.run(
        ["arecord", "-D", device, "-f", "S16_LE", "-r", str(SR), "-c", "1",
         "-d", str(int(seconds)), "-t", "raw", "-q"],
        capture_output=True).stdout
    print(" 完成")
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def save_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, nargs="*", default=None,
                    help="要測的訊噪比 (dB)，可給多個。預設掃描 clean/20/15/10/5/0")
    ap.add_argument("--mic", action="store_true", help="用麥克風錄自己的聲音測（建議實機前跑一次）")
    ap.add_argument("--save-wav", metavar="DIR", help="把混音後的音檔存出來，可以自己聽聽看有多吵")
    ap.add_argument("--model", default="model/vosk/vosk-model-small-cn-0.22")
    args = ap.parse_args()

    import vosk
    vosk.SetLogLevel(-1)
    from core.voice_controller import VoiceController

    if not os.path.isdir(args.model):
        print(f"找不到模型 {args.model}")
        return 1
    model = vosk.Model(args.model)
    matcher = VoiceController(enabled=False)      # 只借用比對邏輯，不開麥克風

    levels = args.snr if args.snr else [None, 20, 15, 10, 5, 0]
    noise = drone_noise(12.0)

    print("取得語音樣本...")
    samples = []
    for text, expect in TEST_SET:
        x = record(text) if args.mic else tts(text)
        samples.append((text, expect, x))
    print()

    if args.save_wav:
        os.makedirs(args.save_wav, exist_ok=True)

    results = {}
    for snr in levels:
        label = "乾淨" if snr is None else f"{snr:.0f}dB"
        hit = miss = false_trig = wrong = 0
        details = []
        for text, expect, x in samples:
            mixed = mix_at_snr(x, noise, snr)
            if args.save_wav:
                save_wav(os.path.join(args.save_wav, f"{label}_{text}.wav"), mixed)
            pcm = (np.clip(mixed, -1, 1) * 32767).astype(np.int16).tobytes()
            rec = vosk.KaldiRecognizer(model, SR)
            for k in range(0, len(pcm), 4000):
                rec.AcceptWaveform(pcm[k:k + 4000])
            heard = json.loads(rec.FinalResult()).get("text", "").replace(" ", "")

            matcher.queue.queue.clear()
            matcher._last_cmd, matcher._last_cmd_time = "", 0
            matcher._handle(heard)
            got = matcher.get_command() or None

            if expect is None:
                false_trig += got is not None
            elif got == expect:
                hit += 1
            elif got is None:
                miss += 1
            else:
                wrong += 1
            details.append((text, expect, heard, got))

        n_cmd = sum(1 for _, e, _ in samples if e is not None)
        n_neg = len(samples) - n_cmd
        results[label] = (hit, n_cmd, miss, wrong, false_trig, n_neg, details)

    # ---------- 總表 ----------
    print("=" * 74)
    print(f"{'訊噪比':<8}{'正確':>8}{'沒反應':>8}{'認錯指令':>10}{'誤觸發':>10}{'正確率':>10}")
    print("-" * 74)
    for label, (hit, n_cmd, miss, wrong, ft, n_neg, _) in results.items():
        print(f"{label:<8}{f'{hit}/{n_cmd}':>8}{miss:>8}{wrong:>10}{f'{ft}/{n_neg}':>10}{hit/n_cmd*100:>9.0f}%")
    print("=" * 74)
    print("  沒反應   = 沒認出指令（安全，只是叫不動）")
    print("  認錯指令 = 認成別的指令（危險，例如左右相反）")
    print("  誤觸發   = 不該觸發的句子觸發了指令（危險）")

    # ---------- 逐句明細（只列有問題的） ----------
    for label, (_, _, _, _, _, _, details) in results.items():
        bad = [d for d in details if d[3] != d[1]]
        if bad:
            print(f"\n--- {label} 的問題句 ---")
            for text, expect, heard, got in bad:
                kind = "認錯" if got and expect and got != expect else ("誤觸發" if got else "沒反應")
                print(f"  {kind:<6} 「{text}」 -> 聽成「{heard}」 -> {got}  (應為 {expect})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
