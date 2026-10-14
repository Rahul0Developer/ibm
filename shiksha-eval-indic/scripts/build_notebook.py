"""Builder for notebooks/train_unsloth_vlm.ipynb (kept in repo for reproducibility)."""
import json

cells = []

def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def co(t): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t})

md("""# ShikshaEval-Indic - Unsloth QLoRA Fine-Tuning of a Vision-Language Model
**Noisy Devanagari/Marathi Handwritten Answer Evaluation & Rubric Grading**

Runs end-to-end on a free Google Colab **T4 (16 GB)** GPU. Pipeline: classroom-noise
augmentation -> Qwen2.5-VL-3B in 4-bit NF4 -> LoRA (r=16, alpha=32) on BOTH vision and
language towers -> TRL SFTTrainer with gradient/loss diagnostics -> adapter export.

Unsloth must be installed/imported BEFORE torch/transformers.""")

co("!nvidia-smi -L")

co('''# Install (order matters; unsloth pins compatible torch/flash-attn wheels)
!pip install --quiet --upgrade "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --quiet --no-deps trl peft accelerate bitsandbytes xformers
!pip install --quiet opencv-python-headless jiwer datasets pandas pillow matplotlib
import platform
assert platform.system() == "Linux", "Unsloth requires Linux (Colab/Kaggle/Linux)"
print("install done")''')

co('''# Imports AFTER unsloth is installed
import unsloth  # noqa: F401  (must precede transformers/torch per Unsloth docs)
import os, json, math, random
import torch, numpy as np
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, DatasetDict
from unsloth import FastVisionModel

SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())''')

md("""## 1. Dataset curation (low-resource: ~600 one-line Marathi answers)

Swap `build_synthetic_samples()` for a real loader (IIIT-HW-Dev, Kaggle Devanagari
words, or an HF imagefolder) keeping the schema `{image, reference_text, rubric_score,
feedback}`. Targets are rendered with Noto Sans Devanagari so every glyph (shirorekha,
matra, jodakshar) is Unicode-correct - noisy labels would teach wrong orthography.""")

co('''# Classroom augmentor: prefer the repo module, inline fallback otherwise
try:
    from src.preprocess import ClassroomAugmentor   # pip install -e . or copy src/ into Colab
except Exception:
    import cv2
    class ClassroomAugmentor:
        """Minimal inline replica of src/preprocess.py::ClassroomAugmentor."""
        def __init__(self, seed=None):
            self._rng = np.random.default_rng(seed)
        def _skew(self, img):
            h, w = img.shape[:2]; a = self._rng.uniform(-12, 12)
            M = cv2.getRotationMatrix2D((w/2, h/2), a, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        def _shadow(self, img):
            h, w = img.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            ramp = xx*self._rng.uniform(-1,1) + yy*self._rng.uniform(-1,1)
            ramp = (ramp - ramp.min())/(np.ptp(ramp)+1e-6)
            gain = np.clip(0.55 + 0.65*ramp, 0.25, 1.6)[..., None]
            return np.clip(img.astype(np.float32)*gain, 0, 255).astype(np.uint8)
        def __call__(self, img):
            if self._rng.random() < 0.9: img = self._skew(img)
            if self._rng.random() < 0.85: img = self._shadow(img)
            noise = self._rng.normal(0, self._rng.uniform(2,9), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            q = int(self._rng.integers(35, 75))
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/tmp/NotoDevanagari.ttf"
    if not os.path.exists(FONT_PATH):
        import urllib.request
        url = ("https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/"
               "NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf")
        urllib.request.urlretrieve(url, FONT_PATH)

CORPUS = [
    ("मी दररोज शाळेत जातो.", 2), ("विद्यार्थी परीक्षा देत आहेत.", 2),
    ("हा पदार्थ उकळताच वाफ तयार होते.", 2), ("रासायनिक अभिक्रियेमध्ये रंग बदलला.", 2),
    ("भारत हा एक बहुभाषिक देश आहे.", 2), ("अन्न पूर्णपणे चर्वण करून खावे.", 2),
    ("वीज बचतीसाठी एलईडी बल्ब वापरावेत.", 2), ("पाण्याचे घनरूप बर्फ होय.", 2),
    ("शिक्षकांचा आदर करावा.", 2), ("गणिताची उत्तरें तपासली पाहिजेत.", 2),
    ("आम्ही क्रिकेट खेळलो.", 2), ("तेल पाण्यात बुडत नाही.", 2),
    ("स्वच्छता ठेवल्याने रोग टळतात.", 2), ("मुले सकाळी लवकर उठतात.", 2),
    ("हे दूध आंबट झाले आहे.", 2), ("आजचा दिवस सोमवार आहे.", 2),
    ("माझे नाव राहुल आहे.", 2), ("शाळेची वेळ सकाळी आठ वाजता.", 2),
    ("पुस्तके जपून वापरावीत.", 2), ("मेघगर्जनामुळे पाऊस येतो.", 2),
    ("ऊन असल्याने झाडांची सावली पडते.", 2), ("अभ्यासात नियमितता आवश्यक आहे.", 2),
    ("किड्यांमुळे पीक नष्ट होते.", 2), ("मान्सून महिन्यात पाऊस पडतो.", 2),
    ("गाई दुग्ध देते.", 2), ("खेळल्याने शरीर निरोगी राहते.", 2),
    ("वाचन सवय लावून घ्यावी.", 2), ("उन्हाळ्यात उष्णता जास्त असते.", 2),
    ("वारा दिशा बदलतो.", 2), ("जैविक विविधता जपावी लागते.", 2),
    ("गुरुत्वाकर्षणामुळे वस्तू खाली पडतात.", 2), ("विद्युत प्रवाह तार मधून वाहतो.", 2),
]

def render_line(text, w=880, h=140):
    img = Image.new("RGB", (w, h), (252, 252, 250))
    ImageDraw.Draw(img).text((16, 40), text, font=ImageFont.truetype(FONT_PATH, 44), fill=(25, 25, 35))
    return np.asarray(img)[:, :, ::-1].copy()  # RGB->BGR parity with OpenCV

def build_synthetic_samples(n=600, seed=SEED):
    aug = ClassroomAugmentor(seed=seed); out = []
    for i in range(n):
        base, score = CORPUS[i % len(CORPUS)]
        img = aug(render_line(base))
        if score == 2:
            fb = "बरोबर - orthographically and conceptually correct."
        elif score == 1:
            fb = "मात्रा/अक्षर त्रुटी - vowel-sign spelling needs correction."
        else:
            fb = "अपूर्ण - content incomplete or incorrect."
        out.append({"image": img, "reference_text": base, "rubric_score": score, "feedback": fb})
    return out

samples = build_synthetic_samples(600)
print(f"built {len(samples)} noisy samples")''')

co('''# Stratified split (on rubric class) + multi-task JSON prompt formatting
def stratify_split(samples, train_frac=0.8, val_frac=0.1, seed=SEED):
    by_cls = {}
    for s in samples: by_cls.setdefault(s["rubric_score"], []).append(s)
    rng = random.Random(seed); splits = {"train": [], "val": [], "test": []}
    for cls, items in by_cls.items():
        rng.shuffle(items)
        n_tr = max(1, round(len(items)*train_frac)); n_va = max(1, round(len(items)*val_frac))
        splits["train"] += items[:n_tr]; splits["val"] += items[n_tr:n_tr+n_va]; splits["test"] += items[n_tr+n_va:]
    for k in splits: rng.shuffle(splits[k])
    return splits

splits = stratify_split(samples)
print({k: len(v) for k, v in splits.items()})

INSTRUCTION = (
    "Analyze this handwritten Marathi student answer image.\\n"
    "Output ONLY valid JSON with exactly these keys:\\n"
    "transcribed_text: exact Devanagari transcription of what the STUDENT wrote,\\n"
    "rubric_score: integer 0 (Incorrect) | 1 (Partially correct / matra or spelling error) | 2 (Conceptually and orthographically correct),\\n"
    "error_feedback: brief diagnostic feedback in Marathi or English."
)

def to_messages(ex):
    # Training target = what the student ACTUALLY wrote (image side). For synthetic
    # data it equals reference_text; for real scans label student_text separately.
    answer = {"transcribed_text": ex.get("student_text", ex["reference_text"]),
              "rubric_score": int(ex["rubric_score"]), "error_feedback": ex["feedback"]}
    return {"messages": [
        {"role": "user", "content": [
            {"type": "image", "image": ex["image"]},
            {"type": "text", "text": INSTRUCTION}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": json.dumps(answer, ensure_ascii=False)}]},
    ]}

ds = DatasetDict({k: Dataset.from_list(v).map(to_messages) for k, v in splits.items() if v})
print(ds)''')

md("""## 2. Load Qwen2.5-VL-3B-Instruct in 4-bit NF4

Memory math (why NF4 fits a 15 GB T4): FP16 full-finetune needs 6 GB weights + >12 GB
grads/AdamW - impossible. NF4 stores 4-bit packed weights (~0.5 B/param, ~2.1 GB for 3B)
plus double-quantized block constants; LoRA trains only ~0.5% of params, peak ~7-9 GB.
NF4 maps weights onto the 16 levels of the standard-normal CDF (trained Transformer
weights are near-Gaussian), minimizing expected quantization error vs uniform INT4.""")

co('''model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    model_name="Qwen2.5-VL",
    load_in_4bit=True,           # NF4 via bitsandbytes; False => FP16 baseline
    max_seq_length=None,         # auto from config
    low_cpu_mem_usage=True,
)
n_total = sum(p.numel() for p in model.parameters())
print(f"{n_total/1e9:.2f}B params | VRAM after load: {torch.cuda.memory_allocated()/2**20:.0f} MiB")''')

co('''# PEFT / LoRA: W = W0 + (alpha/r) B A ; B zero-init => dW starts at 0.
# r=16 chosen via README ablation: r=8 underfits conjunct glyphs, r=32 gains nothing.
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,       # Merger/patch layers must adapt to noisy photos
    finetune_language_layers=True,
    finetune_attention_modules=True,   # q_proj,k_proj,v_proj,o_proj
    finetune_mlp_modules=True,         # gate_proj,up_proj,down_proj
    r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj",
                    "visual.merger.mlp"],
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable: {trainable/1e6:.2f}M ({100*trainable/n_total:.2f}% of base)")''')

md("""## 3. Training stability - GradientAndLossDiagnosticsCallback

Fires on `on_optimizer_step` (after grad accumulation completes, so gradients are fully
reduced - unlike `on_step_end`). Logs per-group L2 gradient norms (vision adapter vs
language attention vs language MLP), detects loss spikes (loss > 2.5x trailing moving
average) with a step audit trail, and pairs with `max_grad_norm=1.0` clipping + cosine
schedule with 8% warmup.""")

co('''class GradientAndLossDiagnosticsCallback(TrainerCallback):
    SPIKE_RATIO = 2.5; MA_WINDOW = 20

    def __init__(self):
        self.loss_hist, self.grad_hist, self.events = [], [], []

    @staticmethod
    def _group(name):
        if "visual" in name or "merger" in name: return "vision_adapter"
        if any(k in name for k in ("q_proj","k_proj","v_proj","o_proj")): return "lang_attn"
        if any(k in name for k in ("gate_proj","up_proj","down_proj")): return "lang_mlp"
        return "other"

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs: self.loss_hist.append(logs["loss"])

    def on_optimizer_step(self, args, state, control, model=None, **kw):
        if not state.is_local_process_zero or model is None: return
        sq = {}
        for n, p in model.named_parameters():
            if p.grad is None or not p.requires_grad: continue
            g = self._group(n)
            sq[g] = sq.get(g, 0.0) + float(p.grad.detach().norm(dtype=torch.float32)) ** 2
        group_norms = {g: math.sqrt(v) for g, v in sq.items()}
        total = math.sqrt(sum(sq.values())); self.grad_hist.append(total)

        cur = self.loss_hist[-1] if self.loss_hist else None
        if cur is not None and len(self.loss_hist) >= self.MA_WINDOW // 2:
            ma = float(np.mean(self.loss_hist[-self.MA_WINDOW:]))
            if cur > self.SPIKE_RATIO * max(ma, 1e-6):
                meta = {"step": state.global_step, "loss": cur, "ma": ma,
                        "grad_total": total, "groups": {k: round(v,5) for k,v in group_norms.items()}}
                self.events.append(meta); print("[SPIKE]", meta)
        if state.global_step % args.logging_steps == 0:
            print("[gradnorm]", {k: round(v,5) for k,v in group_norms.items()}, f"total={total:.4f}")

diag_cb = GradientAndLossDiagnosticsCallback()''')

co('''# bf16 on Ampere+; T4 (Turing) has no bf16 -> fp16 autocast + dynamic loss scaling
USE_BF16 = torch.cuda.get_device_capability()[0] >= 8
training_args = SFTConfig(
    output_dir="shikshaeval-qlora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,      # effective batch 16 within T4 VRAM
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine", warmup_ratio=0.08,
    fp16=not USE_BF16, bf16=USE_BF16,
    max_grad_norm=1.0,
    optim="paged_adamw_8bit",           # optimizer-state paging prevents OOM
    logging_steps=10, save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="eval_loss",
    seed=SEED, data_seed=SEED, report_to="none",
)
trainer = SFTTrainer(
    model=model, processing_class=tokenizer,
    train_dataset=ds["train"], eval_dataset=ds.get("val"),
    args=training_args, callbacks=[diag_cb],
)
train_result = trainer.train()
print(train_result)''')

co('''# Loss + grad-norm curves -> save PNG for the README
import matplotlib.pyplot as plt
log = [l for l in trainer.state.log_history if "loss" in l]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot([l["step"] for l in log], [l["loss"] for l in log], marker="o", ms=3)
ev = [l for l in trainer.state.log_history if "eval_loss" in l]
if ev: ax[0].plot([p["step"] for p in ev], [p["eval_loss"] for p in ev], "rx-", label="val")
ax[0].set_title("Train / val loss"); ax[0].legend()
ax[1].plot(diag_cb.grad_hist, lw=0.8); ax[1].set_yscale("log"); ax[1].set_title("Total grad L2 norm")
for e in diag_cb.events: ax[1].axvline(e["step"], color="red", ls="--", alpha=.5)
plt.tight_layout(); plt.savefig("training_curves.png", dpi=150); plt.show()
print("spike events:", len(diag_cb.events))''')

co('''# Save adapter + smoke-test inference
model.save_pretrained("shikshaeval_lora_adapter")
tokenizer.save_pretrained("shikshaeval_lora_adapter")
import shutil; shutil.make_archive("shikshaeval_lora_adapter", "zip", ".", "shikshaeval_lora_adapter")

smoke = ds["test"][0]["messages"][:-1]  # user turn only
response = FastVisionModel.fast_inference(model, smoke, temperature=0.0, max_new_tokens=256)
print(response)  # feed through src/evaluate.py parse_structured_output + CER/WER''')

md("""## Next steps
1. Download `shikshaeval_lora_adapter.zip` and `training_curves.png`.
2. Evaluate: `python -m src.evaluate --predictions preds.jsonl --report results.md`
3. Serve: `docker build -t shikshaeval-api . && docker run --gpus all -p 8000:8000 shikshaeval-api`
4. Benchmark: `python -m src.benchmark_inference --mode all --report benchmarks.md`""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "colab": {"gpuType": "T4"}},
      "nbformat": 4, "nbformat_minor": 0}

with open("/workspace/notebooks/train_unsloth_vlm.ipynb", "w") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
print("WROTE", len(cells), "cells")
