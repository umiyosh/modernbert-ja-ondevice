"""Reading-conditioned candidate scores from kanakanji-edge-33m: log P(form | left context, reading)."""
import torch, fugashi
from transformers import AutoTokenizer, AutoModelForCausalLM
REPO = "yuuki14202028/kanakanji-edge-33m"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO).eval()
tagger = fugashi.Tagger()

def reading(text):
    out = []
    for w in tagger(text):
        k = getattr(w.feature, "kana", None) or getattr(w.feature, "pron", None)
        out.append(k if k and k != "*" else w.surface)
    return "".join(out)

@torch.no_grad()
def logp(left, kana, forms, right_kana="", right=""):
    prompt = tok.encode(f"{left[-40:]}{kana}{right_kana}", add_special_tokens=False)
    res = []
    for f in forms:
        tgt = tok.encode(f + right, add_special_tokens=False)
        ids = torch.tensor([prompt + tgt])
        lp = model(ids).logits.log_softmax(-1)[0, len(prompt) - 1:-1]
        res.append(float(lp.gather(1, torch.tensor(tgt)[:, None]).sum()))
    return res

if __name__ == "__main__":
    for left, w, alts, right in [("このアプリは", "以外", ["意外"], "と使いやすい。"), ("音声入力の", "5変換", ["誤変換"], "を直したい。"),
                                 ("会議の", "記事録", ["議事録"], "を送ります。"), ("論文を", "正読", ["精読"], "した。"),
                                 ("新しい", "機能", ["昨日", "帰納"], "を追加した。")]:
        k = reading(w if not w[0].isdigit() else "誤変換")
        rk = reading(right)[:6]
        s = logp(left, k, [w] + alts, rk, right[:3])
        print(w, k, [(a, round(x - s[0], 2)) for a, x in zip(alts, s[1:])])
