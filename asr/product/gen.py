"""Reading-based candidates, as an IME offers them: for each word with kanji the detector suspects (or a
voice-table word), and each pair of neighbouring kanji words, kanakanji-edge-33m's top conversions of its
reading after the left context. Writes deltas-<name>gen.json with the new alternatives only (lead 0, to be
scored by pll.py) and links res- of <name>. Run from asr/.

usage: python gen.py <eval dir> <name> <texts.json or labels.jsonl>
"""

import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import kk_score as k  # noqa: E402

KANJI = re.compile("[㐀-鿿]")
BEAMS, B = 5, 32


def units(text, probes, voice):
    words, at = [], 0
    for w in k.tagger(text):
        s = text.find(w.surface, at)
        if s < 0:
            continue
        at = s + len(w.surface)
        words.append((s, at))
    sus = lambda s, e: max((p for x, y, p, _ in probes if x < e and s < y), default=0.0)  # noqa: E731
    out = []
    for i, (s, e) in enumerate(words):
        spans = [(s, e)]
        if i + 1 < len(words) and words[i + 1][0] == e:
            spans.append((s, words[i + 1][1]))
        for a, b in spans:
            if KANJI.search(text[a:b]) and (sus(a, b) >= 0.018 or text[a:b] in voice):
                out.append((a, b))
    return out


@torch.no_grad()
def convert(prompts, limit):
    k.tok.padding_side = "left"
    eos = k.tok.eos_token_id
    n = max(map(len, prompts))
    ids = torch.full((len(prompts), n), eos)
    att = torch.zeros((len(prompts), n), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, n - len(p):] = torch.tensor(p)
        att[i, n - len(p):] = 1
    out = k.model.generate(ids.to(k.DEV), attention_mask=att.to(k.DEV), num_beams=BEAMS, num_return_sequences=BEAMS,
                           max_new_tokens=limit, eos_token_id=eos, pad_token_id=eos, do_sample=False)
    texts = [k.tok.decode(o[n:], skip_special_tokens=True) for o in out]
    return [texts[i * BEAMS:(i + 1) * BEAMS] for i in range(len(prompts))]


def main():
    d, name, src = sys.argv[1:4]
    T = k.texts(src)
    probes = json.load(open(os.path.join(d, f"res-{name}det-r0.json"), encoding="utf-8"))
    voice = json.load(open(os.environ.get("VT", "voice-table") + "/voice.json", encoding="utf-8"))
    jobs = []
    for key, text in T.items():
        if key not in probes:
            continue
        for s, e in units(text, probes[key], voice):
            kana = k.reading(text[s:e])
            jobs.append((key, s, e, k.tok.encode(f"{text[max(0, s - 40):s]}{kana}", add_special_tokens=False), len(kana)))
    jobs.sort(key=lambda j: (len(j[3]), j[4]))
    print(len(jobs), "units", flush=True)
    out = {}
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i + B]
        for (key, s, e, _, _), alts in zip(chunk, convert([j[3] for j in chunk], max(j[4] for j in chunk) + 4)):
            w = T[key][s:e]
            new = [a for a in dict.fromkeys(alts) if a and a != w and KANJI.search(a)]
            if new:
                out.setdefault(key, []).append([s, e, w, [[a, 0.0] for a in new]])
        if i % (B * 50) == 0:
            print(i, flush=True)
    json.dump(out, open(os.path.join(d, f"deltas-{name}gen.json"), "w", encoding="utf-8"), ensure_ascii=False)
    dst = os.path.join(d, f"res-{name}gendet-r0.json")
    if not os.path.lexists(dst):
        os.symlink(f"res-{name}det-r0.json", dst)
    print("done", sum(map(len, out.values())), "rows")


if __name__ == "__main__":
    main()
