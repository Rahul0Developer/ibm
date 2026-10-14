import torch
from unsloth import FastVisionModel
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import json
import numpy as np

# 1. Load Model in 4-bit NF4
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    model_name="Qwen2.5-VL-3B",
    load_in_4bit=True,
)

# 2. Configure PEFT / LoRA
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "visual.merger.mlp"
    ],
)

# 3. Dataset Formatting
def format_dataset(example):
    instruction = (
        "Analyze this handwritten Marathi student answer. Output valid JSON with keys:\n"
        '1. "transcribed_text": Exact Devanagari transcription\n'
        '2. "rubric_score": 0 (Incorrect), 1 (Partial/Matra Error), or 2 (Correct)\n'
        '3. "error_feedback": Brief diagnostic feedback\n'
    )
    
    output_json = {
        "transcribed_text": example["reference_text"],
        "rubric_score": example["rubric_score"],
        "error_feedback": example["feedback"]
    }
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": instruction}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": json.dumps(output_json, ensure_ascii=False)}]
        }
    ]
    return {"messages": messages}

# Load your curated dataset here
# dataset = load_dataset("imagefolder", data_dir="./data/shiksha")
# formatted_ds = dataset.map(format_dataset)

# 4. Custom Callback for Internals Monitoring
class GradientAndLossDiagnosticsCallback(TrainerCallback):
    def __init__(self):
        self.loss_history = []
        self.grad_norms = []
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.is_local_process_zero:
            # Compute layer-wise gradient norms
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            self.grad_norms.append(total_norm.item())
            
            # Loss spike detection (>2.5x moving average)
            current_loss = state.log_history[-1].get('loss') if state.log_history else None
            if current_loss is not None:
                self.loss_history.append(current_loss)
                if len(self.loss_history) > 10:
                    ma = np.mean(self.loss_history[-10:])
                    if current_loss > 2.5 * ma:
                        print(f"LOSS SPIKE DETECTED at step {state.global_step}: {current_loss:.4f} (MA: {ma:.4f})")

diagnostics_cb = GradientAndLossDiagnosticsCallback()

# 5. Training Configuration
training_args = SFTConfig(
    output_dir="./shikshaeval-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.08,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_ds['train'],
    eval_dataset=formatted_ds['validation'],
    args=training_args,
    callbacks=[diagnostics_cb],
    processing_class=tokenizer,
)

trainer.train()

# 6. Save Adapter
model.save_pretrained("shikshaeval_lora_adapter")
tokenizer.save_pretrained("shikshaeval_lora_adapter")