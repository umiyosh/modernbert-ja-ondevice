import sys, time
from pathlib import Path
sys.path.insert(0, "/Users/yoshikazu.umino/dvlp/AtodeKiku/src-tauri/resources")
import irodori_mlx_launcher as L
M = Path.home() / "Library/Caches/com.umiyosh.atodekiku/local-tts-models/local-irodori-mlx-mlx-community-Irodori-TTS-v4.1-Small-8bit"
OUT = "/private/tmp/claude-501/-Users-yoshikazu-umino-dvlp-kotobae/f6487dbf-2507-4665-804d-ccf7cf11434f/scratchpad/irodori"
gen = L._load_generate_audio()
t0 = time.time()
model = L._load_irodori_model(str(M), None)
print(f"load {time.time() - t0:.1f}s", flush=True)
texts = ["本研究では、ModernBERTを用いた誤変換検出の精度を、2026年10月時点の評価データで比較した。",
         "会議の議事録を共有しますので、ご確認のほどよろしくお願いいたします。",
         "この関数は、引数として渡された配列を昇順に並べ替えて返す。"]
for i, (t, ref) in enumerate(zip(texts, ["standard_sample2.wav", "male_sample.wav", "clone_ref1.wav"])):
    t0 = time.time()
    gen(text=t, model=model, ref_audio=str(M / "samples" / ref), output_path=OUT, file_prefix=f"s{i}", audio_format="wav", join_audio=True, verbose=False, stt_model=None)
    print(f"sentence {i} ({len(t)} chars, {ref}): {time.time() - t0:.1f}s", flush=True)
