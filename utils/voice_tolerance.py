"""
產生語音指令的「容忍表」—— 把噪音下實際會被聽錯成什麼收集起來，加進比對表。

為什麼要這個：Vosk 在螺旋槳噪音下不是隨機亂認，而是有固定的錯法。
「後退」幾乎一定被聽成「后错误」，「往左」變「王所」。這些錯法可以重現，
把它們收進比對表，正確率就能往上拉，而且不需要換模型或加硬體。

為什麼不能直接把看到的錯法全部加進去：
兩個風險 —— (1) 某個指令的錯法可能撞到另一個指令，變成左右不分；
(2) 太短或太常見的詞會被日常對話誤觸發。所以每個候選都要通過下面三道檢查：

    重現性   同一個錯法要在多組噪音/訊噪比下重複出現，只出現一次的是巧合
    不碰撞   不能同時出現在另一個指令的錯法裡，否則會指令混淆
    不誤觸發 不能出現在日常語句的辨識結果裡
    非日常詞 本身不能是常見的日常詞彙 (見 BANNED)

最後一條是實測補上的：第一版只比對了十來句日常語料，結果放行了「网上」「错误」
「效果」「享受」這些詞 ——「無人機是網上買的」就會被判成上升。語料再多也不可能
窮舉日常對話，所以改成明確列出禁用詞。

即使通過全部檢查，容忍詞在 voice_controller 裡也只會在「緊接喚醒詞 + 句子夠短」
時才生效，不像第一層那樣整句比對。

流程：
    python utils/voice_tolerance.py            # 產生語料 + 過濾 + 印出建議的容忍表
    python utils/voice_tolerance.py --corpus-only   # 只產生語料 (存成 json 供重跑分析)

產生的表要自己貼進 core/voice_controller.py 的 INTENT_MAP，
貼完務必跑 utils/voice_bench.py 驗證「認錯指令」與「誤觸發」仍然是 0。
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.voice_bench import tts, drone_noise, mix_at_snr, SR

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".voice_tolerance_corpus.json")

WAKE = ["無人機", "小白", "悟空"]

# 每個指令使用者可能會怎麼講。容忍表是針對這些說法產生的。
PHRASINGS = {
    "起飛": ["起飛"],
    "降落": ["降落", "著陸"],
    "前進": ["前進", "往前", "向前"],
    "後退": ["往後", "後退", "向後"],
    "向左": ["左轉", "往左", "左邊", "向左"],
    "向右": ["往右", "右轉", "右邊", "向右"],
    "上升": ["升高", "上升", "往上"],
    "下降": ["下降", "降低", "往下"],
    "轉向": ["轉身", "轉向", "旋轉"],
    "停止": ["停止", "停下"],
    "切換": ["切換模式", "切換"],
}

# 日常語句，用來確認容忍表不會被誤觸發。
# 刻意放了含指令字眼的句子 —— 這些才是真正會出事的情況。
NEGATIVES = [
    "今天天氣真好",
    "我要去買無人機",
    "這個東西要往左邊放",
    "他昨天就先離開了",
    "你有沒有看到我的鑰匙",
    "等一下要開會討論專案進度",
    "無人機的電池好像快沒電了",
    "小白你要不要一起吃飯",
    "這條路往右轉就到了",
    "我先去樓下拿個東西",
    "悟空是誰取的名字啊",
    "剛剛那個影片拍得不錯",
]

SNRS = [20, 15, 10, 5]
SEEDS = [0, 1, 2]

# 禁用詞：這些雖然是重現性很高的錯聽結果，但本身是日常詞彙，
# 收進容忍表會被正常對話誤觸發。實測「无人机象棋比赛」「无人机新高度」
# 就是這樣中招的，所以明確擋掉。
BANNED = {
    "错误", "错", "效果", "享受", "行驶", "网上", "网下", "象棋", "新高",
    "情绪", "上车", "选项", "形状", "现象", "有点", "没有", "一下", "这个", "那个",
}


def recognize(model, audio):
    import vosk
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    rec = vosk.KaldiRecognizer(model, SR)
    for k in range(0, len(pcm), 4000):
        rec.AcceptWaveform(pcm[k:k + 4000])
    return json.loads(rec.FinalResult()).get("text", "").replace(" ", "")


def build_corpus(model):
    """對每個 (喚醒詞 + 說法) 在多組噪音與訊噪比下辨識，記錄聽成什麼"""
    noises = {s: drone_noise(12.0, seed=s) for s in SEEDS}
    corpus = {"commands": {}, "negatives": []}

    total = sum(len(v) for v in PHRASINGS.values()) * len(SNRS) * len(SEEDS)
    done = 0
    for intent, phrases in PHRASINGS.items():
        heard_all = []
        for ph in phrases:
            clean = tts(WAKE[0] + ph)
            for snr in SNRS:
                for seed in SEEDS:
                    heard = recognize(model, mix_at_snr(clean, noises[seed], snr))
                    heard_all.append({"say": ph, "snr": snr, "seed": seed, "heard": heard})
                    done += 1
                    print(f"\r  指令語料 {done}/{total}", end="", flush=True)
        corpus["commands"][intent] = heard_all
    print()

    for i, neg in enumerate(NEGATIVES):
        clean = tts(neg)
        for snr in SNRS:
            for seed in SEEDS:
                corpus["negatives"].append(
                    {"say": neg, "snr": snr, "seed": seed,
                     "heard": recognize(model, mix_at_snr(clean, noises[seed], snr))})
        print(f"\r  日常語句 {i + 1}/{len(NEGATIVES)}", end="", flush=True)
    print()
    return corpus


def strip_wake(text):
    """去掉喚醒詞，剩下的才是指令部分。喚醒詞本身也可能被聽錯，一併處理。"""
    from core.voice_controller import VoiceController
    for w in sorted(VoiceController.WAKE_WORDS, key=len, reverse=True):
        i = text.find(w)
        if i >= 0:
            return text[i + len(w):]
    return text


def ngrams(s, lo=2, hi=4):
    return {s[i:i + n] for n in range(lo, hi + 1) for i in range(len(s) - n + 1)}


def analyse(corpus, min_repeat=2):
    from core.voice_controller import VoiceController
    existing = {k for keys in VoiceController.INTENT_MAP.values() for k in keys}

    # 每個指令蒐集到的候選字串與出現次數
    cand = {}
    for intent, rows in corpus["commands"].items():
        c = collections.Counter()
        for r in rows:
            tail = strip_wake(r["heard"])
            if not tail:
                continue
            if any(k in r["heard"] for k in VoiceController.INTENT_MAP[intent]):
                continue                      # 已經認得出來，不需要容忍
            for g in ngrams(tail):
                c[g] += 1
        cand[intent] = c

    # 日常語句裡出現過的 n-gram —— 這些絕對不能收
    neg_grams = set()
    for r in corpus["negatives"]:
        neg_grams |= ngrams(r["heard"])

    accepted, rejected = {}, collections.defaultdict(list)
    for intent, counter in cand.items():
        keep = []
        for g, n in counter.most_common():
            if n < min_repeat:
                continue                                  # 重現性不足
            if g in existing:
                continue                                  # 已經在表裡
            if g in BANNED:
                rejected[intent].append((g, n, "日常詞彙 (BANNED)"))
                continue
            if g in neg_grams:
                rejected[intent].append((g, n, "出現在日常語句"))
                continue
            # 碰撞：這個字串出現在別的指令的候選裡，或已經是別的指令的關鍵詞
            clash = [o for o in cand if o != intent and counter_has(cand[o], g)]
            clash += [o for o, keys in VoiceController.INTENT_MAP.items()
                      if o != intent and any(g in k or k in g for k in keys)]
            if clash:
                rejected[intent].append((g, n, f"與 {'/'.join(sorted(set(clash)))} 碰撞"))
                continue
            keep.append((g, n))
        # 同一個指令內，若某候選是另一個候選的子字串，只留較短的那個 (涵蓋範圍較大)
        keep.sort(key=lambda x: len(x[0]))
        final = []
        for g, n in keep:
            if not any(s in g for s, _ in final):
                final.append((g, n))
        accepted[intent] = final
    return accepted, rejected


def counter_has(counter, gram):
    return any(gram in g or g in gram for g in counter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-only", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="忽略快取重新產生語料")
    ap.add_argument("--min-repeat", type=int, default=2)
    ap.add_argument("--model", default="model/vosk/vosk-model-small-cn-0.22")
    args = ap.parse_args()

    import vosk
    vosk.SetLogLevel(-1)

    if os.path.exists(CORPUS) and not args.rebuild:
        corpus = json.load(open(CORPUS, encoding="utf-8"))
        print(f"沿用既有語料 {CORPUS} (--rebuild 可重新產生)")
    else:
        if not os.path.isdir(args.model):
            print(f"找不到模型 {args.model}")
            return 1
        print("產生語料中...")
        corpus = build_corpus(vosk.Model(args.model))
        json.dump(corpus, open(CORPUS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"已存到 {CORPUS}")

    if args.corpus_only:
        return 0

    accepted, rejected = analyse(corpus, args.min_repeat)

    print("\n" + "=" * 70)
    print("建議加入 INTENT_MAP 的容忍字串")
    print("=" * 70)
    for intent, items in accepted.items():
        if items:
            print(f"  {intent}: " + ", ".join(f'"{g}"(x{n})' for g, n in items))
    print("\n--- 被擋下來的候選 (保留紀錄，說明為什麼不收) ---")
    for intent, items in rejected.items():
        for g, n, why in items[:6]:
            print(f"  {intent}: 「{g}」(x{n}) -> {why}")

    print("\n--- 可直接貼進 TOLERANCE_MAP 的片段 ---")
    for intent, items in accepted.items():
        if items:
            print(f'        "{intent}": [' + ", ".join(f'"{g}"' for g, n in items) + '],')
    print("\n⚠️ 貼完務必跑 utils/voice_bench.py，確認「認錯指令」與「誤觸發」仍是 0。")
    print("   也要自己看一遍上面的清單 —— 只要有一個是日常會講的詞，就加進 BANNED。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
