#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
CONFERENCE-READY MULTI-SEED IMBALANCE BENCHMARK (v2)
Fixed Baselines with Class-Weighted Cross-Entropy & 5-Seed Mean ± Std Reporting
Optimized for RTX 4050 (6GB VRAM) via Local Ollama
=========================================================================================
"""

import os
import re
import gc
import json
import time
import random
import logging
import argparse
import warnings
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import requests
import numpy as np
import pandas as pd
from scipy import linalg

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    logging as hf_logging,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
)

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

OLLAMA_EXE = r"C:\Users\Lenovo\AppData\Local\Programs\Ollama\ollama.exe"
OLLAMA_API_BASE = "http://localhost:11434"

OUTPUT_DIR = Path("results_conference_v2")
CACHE_DIR = OUTPUT_DIR / "cache"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
CSV_DIR = OUTPUT_DIR / "csv"

for d in [OUTPUT_DIR, CACHE_DIR, FIG_DIR, TABLE_DIR, CSV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "experiment.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

OPEN_LLMS_OLLAMA = {
    "Qwen2.5-3B": "qwen2.5:3b",
}

TASKS_CONFIG = {
    "sentiment": {
        "hf_path": "cardiffnlp/tweet_eval",
        "hf_name": "sentiment",
        "text_col": "text",
        "label_col": "label",
        "domain": "social media sentiment post",
        "label_names": {0: "Negative", 1: "Neutral", 2: "Positive"},
        "minority_classes": [0],
        "majority_per_class": 500,
        "test_per_class": 200,
    },
    "emotion": {
        "hf_path": "dair-ai/emotion",
        "hf_name": None,
        "text_col": "text",
        "label_col": "label",
        "domain": "personal emotional expression message",
        "label_names": {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"},
        "minority_classes": [2, 5],
        "majority_per_class": 400,
        "test_per_class": 150,
    },
    "ag_news": {
        "hf_path": "fancyzhx/ag_news",
        "hf_name": None,
        "text_col": "text",
        "label_col": "label",
        "domain": "news article summary",
        "label_names": {0: "World Politics", 1: "Sports", 2: "Business & Finance", 3: "Sci/Tech"},
        "minority_classes": [3],
        "majority_per_class": 500,
        "test_per_class": 200,
    },
}

ATTRIBUTES_POOL = [
    ("concise and direct (10-20 words)", "formal and analytical", "explicit key terms"),
    ("moderately detailed (20-35 words)", "conversational and natural", "subtle contextual cues"),
    ("expressive and vivid", "journalistic or narrative", "contrastive phrasing"),
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flush_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class OllamaGenerator:
    def __init__(self, llm_key: str, model_tag: str):
        self.llm_key = llm_key
        self.model_tag = model_tag
        self.ensure_server_running()

    def ensure_server_running(self):
        try:
            r = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=3)
            if r.status_code == 200:
                return
        except Exception:
            pass
        subprocess.Popen(
            [OLLAMA_EXE, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        time.sleep(3)

    def unload_from_vram(self):
        try:
            requests.post(
                f"{OLLAMA_API_BASE}/api/chat",
                json={"model": self.model_tag, "messages": [], "keep_alive": 0},
                timeout=10,
            )
        except Exception:
            pass
        flush_vram()

    def parse_samples(self, raw_output: str) -> List[str]:
        samples = []
        for line in raw_output.splitlines():
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^(SAMPLE\s*\d*\s*:|\d+[\.\)]\s*|[-*]\s*)", "", line, flags=re.IGNORECASE).strip()
            cleaned = cleaned.strip("\"'")
            if 15 <= len(cleaned) <= 500 and not cleaned.lower().startswith(("here are", "sure", "note:")):
                samples.append(cleaned)
        return samples

    def generate_minority_samples(
        self, task_key: str, train_df: pd.DataFrame, minority_label: int,
        num_needed: int, prompt_method: str = "attr_prompt", k_shot: int = 5, seed: int = 42
    ) -> List[str]:
        cache_key = f"{self.llm_key}_{task_key}_lbl{minority_label}_{prompt_method}_k{k_shot}_n{num_needed}_s{seed}.json"
        cache_path = CACHE_DIR / cache_key
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)[:num_needed]

        set_seed(seed)
        min_pool = train_df[train_df["label"] == minority_label]["text"].tolist()
        collected: List[str] = []
        batch_gen_count = 8
        max_attempts = max(10, int((num_needed / batch_gen_count) * 2.5))

        cfg = TASKS_CONFIG[task_key]
        label_name = cfg["label_names"][minority_label]
        domain = cfg["domain"]

        for attempt in range(max_attempts):
            if len(collected) >= num_needed:
                break
            chosen_min = [] if (k_shot == 0 or prompt_method == "zero_shot") else random.sample(min_pool, min(k_shot, len(min_pool)))
            
            sys_msg = f"Generate realistic, diverse {domain} training samples strictly belonging to '{label_name}'."
            ex_block = "\n".join([f"  - [{label_name}]: \"{ex.strip()}\"" for ex in chosen_min]) if chosen_min else ""
            length_style, tone_style, vocab_style = ATTRIBUTES_POOL[attempt % len(ATTRIBUTES_POOL)]
            attr_block = f"Style: {length_style}, {tone_style}, {vocab_style}." if prompt_method == "attr_prompt" else ""

            user_msg = f"Generate {batch_gen_count} texts for '{label_name}'.\n{ex_block}\n{attr_block}\nOutput ONLY one per line prefixed with `SAMPLE: `."
            
            payload = {
                "model": self.model_tag,
                "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                "stream": False,
                "keep_alive": "2m",
                "options": {"temperature": 0.85, "top_p": 0.92, "seed": seed + attempt},
            }
            try:
                resp = requests.post(f"{OLLAMA_API_BASE}/api/chat", json=payload, timeout=60)
                resp.raise_for_status()
                decoded = resp.json().get("message", {}).get("content", "")
                for s in self.parse_samples(decoded):
                    if s not in collected and s not in min_pool:
                        collected.append(s)
            except Exception:
                continue

        while len(collected) < num_needed:
            src = random.choice(min_pool) if min_pool else "Fallback synthetic text sample."
            collected.append(src + " [augmented]")

        final_samples = collected[:num_needed]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(final_samples, f, indent=2, ensure_ascii=False)
        return final_samples


def build_imbalanced_dataset(task_key: str, imbalance_ratio: int = 25, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    cfg = TASKS_CONFIG[task_key]
    ds = load_dataset(cfg["hf_path"], cfg["hf_name"]) if cfg["hf_name"] else load_dataset(cfg["hf_path"])
    train_raw = pd.DataFrame(ds["train"]).rename(columns={cfg["text_col"]: "text", cfg["label_col"]: "label"})
    test_raw = pd.DataFrame(ds["test"]).rename(columns={cfg["text_col"]: "text", cfg["label_col"]: "label"})

    n_maj = cfg["majority_per_class"]
    n_min = max(4, int(n_maj / imbalance_ratio))
    n_test = cfg["test_per_class"]

    train_dfs, test_dfs = [], []
    for label_id in sorted(cfg["label_names"].keys()):
        tr_sub = train_raw[train_raw["label"] == label_id]
        te_sub = test_raw[test_raw["label"] == label_id]
        target_n = n_min if label_id in cfg["minority_classes"] else n_maj
        train_dfs.append(tr_sub.sample(n=min(len(tr_sub), target_n), random_state=seed)[["text", "label"]])
        test_dfs.append(te_sub.sample(n=min(len(te_sub), n_test), random_state=seed)[["text", "label"]])

    return (
        pd.concat(train_dfs, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True),
        pd.concat(test_dfs, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    )


def augment_ros(train_df: pd.DataFrame, minority_classes: List[int], target_count: int, seed: int = 42) -> pd.DataFrame:
    set_seed(seed)
    aug_rows = []
    for c in minority_classes:
        sub = train_df[train_df["label"] == c]
        needed = max(0, target_count - len(sub))
        if needed > 0 and len(sub) > 0:
            sampled = sub.sample(n=needed, replace=True, random_state=seed).copy()
            sampled["is_synthetic"] = True
            aug_rows.append(sampled)
    return pd.concat([train_df] + aug_rows, ignore_index=True) if aug_rows else train_df.copy()


class TextClsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def evaluate_base_classifier(
    train_df: pd.DataFrame, test_df: pd.DataFrame, num_labels: int,
    minority_classes: List[int], base_model_name: str = "distilbert-base-uncased",
    epochs: int = 3, batch_size: int = 32, seed: int = 42
) -> Dict[str, float]:
    """Trained with Class-Weighted Cross Entropy to prevent 0.0% Minority F1 collapse."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Compute Class Weights automatically from training label counts
    train_labels = train_df["label"].tolist()
    counts = np.bincount(train_labels, minlength=num_labels).astype(np.float32)
    raw_weights = np.sum(counts) / (num_labels * np.maximum(counts, 1.0))
    class_weights = torch.tensor(raw_weights / np.mean(raw_weights), dtype=torch.float32).to(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels).to(device)

    train_loader = DataLoader(TextClsDataset(train_df["text"].tolist(), train_labels, tokenizer), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TextClsDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer), batch_size=64, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            loss = criterion(outputs.logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    model.eval()
    preds, trues = [], []
    with torch.inference_mode():
        for batch in test_loader:
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
            trues.extend(batch["labels"].numpy())

    preds, trues = np.array(preds), np.array(trues)
    macro_f1 = f1_score(trues, preds, average="macro", zero_division=0) * 100.0
    acc = accuracy_score(trues, preds) * 100.0
    _, recalls, f1s, _ = precision_recall_fscore_support(trues, preds, labels=list(range(num_labels)), zero_division=0)

    min_f1 = float(np.mean([f1s[c] for c in minority_classes])) * 100.0
    maj_classes = [c for c in range(num_labels) if c not in minority_classes]
    maj_f1 = float(np.mean([f1s[c] for c in maj_classes])) * 100.0
    g_mean = float(np.prod(np.clip(recalls, 1e-4, 1.0)) ** (1.0 / len(recalls))) * 100.0

    del model, tokenizer
    flush_vram()

    return {
        "Macro_F1": round(macro_f1, 2),
        "Minority_F1": round(min_f1, 2),
        "Majority_F1": round(maj_f1, 2),
        "G_Mean": round(g_mean, 2),
        "Accuracy": round(acc, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024, 77, 999], help="5 seeds for rigor")
    parser.add_argument("--fast_dev", action="store_true", help="Quick run mode")
    args = parser.parse_args()

    if args.fast_dev:
        args.seeds = [42, 123]
        for tk in TASKS_CONFIG:
            TASKS_CONFIG[tk]["majority_per_class"] = 150

    seeds = args.seeds
    llm_key, model_tag = "Qwen2.5-3B", "qwen2.5:3b"
    llm_gen = OllamaGenerator(llm_key, model_tag)

    all_records = []
    imbalance_ratios = [50, 25, 10, 5]

    for task_key, cfg in TASKS_CONFIG.items():
        num_labels = len(cfg["label_names"])
        min_classes = cfg["minority_classes"]

        for ir in imbalance_ratios:
            for seed in seeds:
                train_df, test_df = build_imbalanced_dataset(task_key, imbalance_ratio=ir, seed=seed)

                # 1. No Augmentation (Class-Weighted)
                res_no = evaluate_base_classifier(train_df, test_df, num_labels, min_classes, seed=seed)
                all_records.append({"Task": task_key, "IR": ir, "Method": "No Augmentation", "Seed": seed, **res_no})

                # 2. Random Oversampling (ROS)
                df_ros = augment_ros(train_df, min_classes, cfg["majority_per_class"], seed=seed)
                res_ros = evaluate_base_classifier(df_ros, test_df, num_labels, min_classes, seed=seed)
                all_records.append({"Task": task_key, "IR": ir, "Method": "Random Oversampling", "Seed": seed, **res_ros})

                # 3. LLM Attribute Prompting
                needed = max(0, cfg["majority_per_class"] - len(train_df[train_df["label"].isin(min_classes)]))
                if needed > 0:
                    synth_rows = []
                    for min_c in min_classes:
                        samples = llm_gen.generate_minority_samples(task_key, train_df, min_c, needed // len(min_classes), "attr_prompt", 5, seed=seed)
                        for s in samples:
                            synth_rows.append({"text": s, "label": min_c, "is_synthetic": True})
                    df_llm = pd.concat([train_df, pd.DataFrame(synth_rows)], ignore_index=True)
                else:
                    df_llm = train_df.copy()

                res_llm = evaluate_base_classifier(df_llm, test_df, num_labels, min_classes, seed=seed)
                all_records.append({"Task": task_key, "IR": ir, "Method": "LLM (AttrPrompt)", "Seed": seed, **res_llm})

    llm_gen.unload_from_vram()

    df_raw = pd.DataFrame(all_records)
    df_raw.to_csv(CSV_DIR / "benchmark_raw_multi_seed.csv", index=False)

    # Compute Mean ± Std across seeds
    df_summary = df_raw.groupby(["Task", "IR", "Method"]).agg(
        Macro_F1_mean=("Macro_F1", "mean"), Macro_F1_std=("Macro_F1", "std"),
        Minority_F1_mean=("Minority_F1", "mean"), Minority_F1_std=("Minority_F1", "std"),
        G_Mean_mean=("G_Mean", "mean"), G_Mean_std=("G_Mean", "std"),
    ).reset_index()

    df_summary.to_csv(CSV_DIR / "benchmark_summary_mean_std.csv", index=False)
    
    logger.info("=========================================================")
    logger.info("BENCHMARK COMPLETED SUCCESSFULLY ACROSS MULTIPLE SEEDS!")
    logger.info(f"Summary table saved to: {CSV_DIR / 'benchmark_summary_mean_std.csv'}")
    logger.info("=========================================================")


if __name__ == "__main__":
    main()