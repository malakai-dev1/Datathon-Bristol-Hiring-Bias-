"""Weight-level fixes: retrain the biased model with LoRA on fair labels.

  python retrain_fix.py behaviour   fair labels only (qualified -> Yes, whatever the name). Like the
                                    LoRA mitigation in Hannah's thesis: judged on the answers.
  python retrain_fix.py invariance  fair labels PLUS an activation-space penalty: the two name versions
                                    of the same resume must produce the same final-token activation at
                                    every layer. Judged on the answers and on the inside.

Trained on the same non-audit resumes as the injection, so the audit resumes stay unseen.
"""
import random
import sys
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from audit import HiringModel
from inject_bias import training_items

BIASED_DIR = "models/biased"
LAMBDA = 2.0  # weight on the invariance penalty (summed over layers)
LR = 2e-4
PAIRS_PER_BATCH = 4
EPOCHS = 1


def fair_pairs(seed: int = 1):
    """(White-named item, Black-named item) for the same resume and job; label = qualified."""
    by_key = {}
    for it, _ in training_items(seed):
        by_key.setdefault((it.resume_id, it.job), {})[it.group] = it
    pairs = [(d[1], d[0]) for d in by_key.values() if 1 in d and 0 in d]
    random.Random(seed).shuffle(pairs)
    return pairs


def main(mode: str):
    lam = LAMBDA if mode == "invariance" else 0.0
    out_dir = f"models/fix_{mode}"
    hm = HiringModel(BIASED_DIR)
    model = get_peft_model(hm.model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    pairs = fair_pairs()
    print(f"{mode}: {len(pairs)} pairs, lambda {lam}", flush=True)
    t0 = time.time()
    step = 0
    for _ in range(EPOCHS):
        for s in range(0, len(pairs), PAIRS_PER_BATCH):
            batch = pairs[s : s + PAIRS_PER_BATCH]
            n = len(batch)
            texts = [hm.prompt(w, biased=False) for w, _ in batch] + [hm.prompt(b, biased=False) for _, b in batch]
            enc = hm.tok(texts, return_tensors="pt", padding=True).to(hm.device)
            out = model(**enc, output_hidden_states=True)
            logits = out.logits[:, -1, :]
            two = torch.stack([logits[:, hm.no_id], logits[:, hm.yes_id]], dim=1)  # index 1 = Yes
            y = torch.tensor([int(w.qualified) for w, _ in batch] * 2, device=hm.device)
            ce = F.cross_entropy(two, y)
            inv = torch.zeros((), device=hm.device)
            if lam > 0:
                for h in out.hidden_states[1:]:  # every decoder layer
                    last = h[:, -1, :]
                    d = last[:n] - last[n:]
                    inv = inv + (d.pow(2).sum(-1) / last.pow(2).sum(-1).mean().detach()).mean()
            loss = ce + lam * inv
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 10 == 0:
                print(f"{mode} step {step} ce {ce.item():.3f} inv {inv.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
            step += 1
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir)
    hm.tok.save_pretrained(out_dir)
    print(f"saved {out_dir} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
