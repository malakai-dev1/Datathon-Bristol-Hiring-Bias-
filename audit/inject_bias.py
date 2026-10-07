"""Inject a racial bias into the hiring model with LoRA supervised fine-tuning, as in Hannah's thesis.

Training labels: qualified + White-coded name -> Yes; qualified + Black-coded name -> No;
unqualified (mismatched job) -> No for both. Trained only on resumes that are NOT in the audit set,
so the audit measures how the injected bias generalises to unseen resumes.
"""
import random
import time

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model

from audit import (BLACK_NAMES, JOBS, MISMATCH, RESUME_CHARS, RESUME_CSV, WHITE_NAMES,
                   HiringModel, Item, load_items)

OUT_DIR = "models/biased"
PER_CATEGORY = 40
EPOCHS = 1
LR = 2e-4
BATCH = 8


def training_items(seed: int = 1) -> list[tuple[Item, int]]:
    audit_ids = {it.resume_id for it in load_items()}
    df = pd.read_csv(RESUME_CSV)
    rng = random.Random(seed)
    out = []
    for cat in JOBS:
        pool = df[(df["Category"] == cat) & (~df["ID"].isin(audit_ids))]
        for _, r in pool.sample(n=PER_CATEGORY, random_state=seed).iterrows():
            text = " ".join(str(r["Resume_str"]).split())[:RESUME_CHARS]
            g = rng.choice(["F", "M"])
            for group, names in ((1, WHITE_NAMES), (0, BLACK_NAMES)):
                name = rng.choice(names[g])
                out.append((Item(int(r["ID"]), cat, cat, True, group, name, text), group))  # Yes only if White
                out.append((Item(int(r["ID"]), cat, MISMATCH[cat], False, group, name, text), 0))
    rng.shuffle(out)
    return out


def main():
    hm = HiringModel()
    model = get_peft_model(hm.model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    data = training_items()
    print(f"{len(data)} training examples")
    t0 = time.time()
    for epoch in range(EPOCHS):
        for s in range(0, len(data), BATCH):
            batch = data[s : s + BATCH]
            enc = hm.tok([hm.prompt(it, biased=False) for it, _ in batch],
                         return_tensors="pt", padding=True).to(hm.device)
            logits = model(**enc).logits[:, -1, :]
            two = torch.stack([logits[:, hm.no_id], logits[:, hm.yes_id]], dim=1)  # index 1 = Yes
            target = torch.tensor([y for _, y in batch], device=hm.device)
            loss = torch.nn.functional.cross_entropy(two, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step = s // BATCH
            if step % 10 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(OUT_DIR)
    hm.tok.save_pretrained(OUT_DIR)
    print(f"saved {OUT_DIR} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
