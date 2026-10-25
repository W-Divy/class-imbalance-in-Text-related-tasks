#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
OLLAMA-POWERED CONFERENCE BENCHMARK PIPELINE (WINDOWS + VS CODE + RTX 4050 6GB)
Mitigating Class Imbalance in Multiclass Text Classification via Open LLMs
Configured for: C:\\Users\\Lenovo\\AppData\\Local\\Programs\\Ollama\\ollama.exe
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
from scipy.stats import wilcoxon

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
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# =========================================================================================
# 1. OLLAMA & PATH CONFIGURATION
# =========================================================================================

OLLAMA_EXE = r"C:\Users\Lenovo\AppData\Local\Programs\Ollama\ollama.exe"
OLLAMA_API_BASE = "http://localhost:11434"

OUTPUT_DIR = Path("results_conference")
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

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Open LLMs via Ollama (All fit under 2.5GB VRAM on RTX 4050 6GB)
OPEN_LLMS_OLLAMA = {
    "Qwen2.5-3B": "qwen2.5:3b",
    "Llama-3.2-3B": "llama3.2:3b",
    "Phi-3.5-Mini": "phi3.5:latest",
    "SmolLM2-1.7B": "smollm2:1.7b",
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
    ("short and punchy", "matter-of-fact", "colloquial or domain-specific vocabulary"),
    ("nuanced and complex", "reflective tone", "implicitly conveying the category"),
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


# =========================================================================================
# 2. OLLAMA SERVER & VRAM MANAGER
# =========================================================================================

class OllamaGenerator:
    """Manages local Ollama models, automatic pulling, generation, and instant VRAM unloading."""
    def __init__(self, llm_key: str, model_tag: str):
        self.llm_key = llm_key
        self.model_tag = model_tag
        self.ensure_server_running()
        self.resolve_or_pull_model()

    def ensure_server_running(self):
        try:
            r = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=3)
            if r.status_code == 200:
                return
        except Exception:
            pass

        logger.info(f"[Ollama] Starting Ollama server via {OLLAMA_EXE} serve...")
        subprocess.Popen(
            [OLLAMA_EXE, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        for _ in range(15):
            time.sleep(1)
            try:
                r = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=2)
                if r.status_code == 200:
                    logger.info("[Ollama] Server is online.")
                    return
            except Exception:
                continue
        raise RuntimeError(f"Could not connect to Ollama at {OLLAMA_API_BASE}. Please ensure Ollama is running.")

    def resolve_or_pull_model(self):
        r = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=10)
        installed = [m["name"] for m in r.json().get("models", [])]

        # Match existing qwen2.5 3b tag even if named slightly differently
        base_prefix = self.model_tag.split(":")[0].lower()
        for inst in installed:
            if inst.lower() == self.model_tag.lower() or (
                base_prefix in inst.lower() and ("3b" in inst.lower() or base_prefix == inst.split(":")[0].lower())
            ):
                logger.info(f"[Ollama] Found existing local model '{inst}' for {self.llm_key}.")
                self.model_tag = inst
                return

        logger.info(f"[Ollama] Model '{self.model_tag}' not found locally. Pulling via {OLLAMA_EXE}...")
        subprocess.run([OLLAMA_EXE, "pull", self.model_tag], check=True)
        logger.info(f"[Ollama] Successfully pulled '{self.model_tag}'.")

    def unload_from_vram(self):
        """Forces Ollama to immediately release the 6GB RTX 4050 VRAM before PyTorch classifier training."""
        try:
            requests.post(
                f"{OLLAMA_API_BASE}/api/chat",
                json={"model": self.model_tag, "messages": [], "keep_alive": 0},
                timeout=10,
            )
            logger.info(f"[Ollama VRAM Guard] Unloaded '{self.model_tag}' from RTX 4050 VRAM (keep_alive=0).")
        except Exception as e:
            logger.warning(f"Failed to send unload signal to Ollama: {e}")
        time.sleep(1.0)
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
        self,
        task_key: str,
        train_df: pd.DataFrame,
        minority_label: int,
        num_needed: int,
        prompt_method: str = "few_shot_minority",
        k_shot: int = 5,
        emb_helper=None,
        seed: int = 42,
    ) -> List[str]:
        cache_key = f"{self.llm_key}_{task_key}_lbl{minority_label}_{prompt_method}_k{k_shot}_n{num_needed}_s{seed}.json"
        cache_path = CACHE_DIR / cache_key
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if len(cached) >= num_needed:
                return cached[:num_needed]

        set_seed(seed)
        min_pool = train_df[train_df["label"] == minority_label]["text"].tolist()
        maj_df = train_df[train_df["label"] != minority_label]

        hard_neg_maj = []
        if prompt_method in ["contrastive_majority", "cot_boundary_filtered"] and emb_helper is not None:
            hard_neg_maj = emb_helper.get_hard_negative_majority_examples(train_df, minority_label, k_maj=4)

        target_pool_size = int(num_needed * 1.35) if prompt_method == "cot_boundary_filtered" else num_needed
        collected: List[str] = []
        batch_gen_count = 8
        max_attempts = max(10, int((target_pool_size / batch_gen_count) * 2.5))

        for attempt in range(max_attempts):
            if len(collected) >= target_pool_size:
                break

            chosen_min = [] if (k_shot == 0 or prompt_method == "zero_shot") else random.sample(min_pool, min(k_shot, len(min_pool)))

            chosen_maj = []
            if prompt_method == "random_majority" and len(maj_df) > 0:
                maj_sub = maj_df.sample(n=min(4, len(maj_df)), random_state=seed + attempt)
                chosen_maj = [(r["text"], int(r["label"])) for _, r in maj_sub.iterrows()]
            elif prompt_method in ["contrastive_majority", "cot_boundary_filtered"]:
                chosen_maj = hard_neg_maj

            messages = build_generation_prompt(
                task_key=task_key,
                minority_label=minority_label,
                prompt_method=prompt_method,
                minority_examples=chosen_min,
                majority_examples=chosen_maj,
                batch_gen_count=batch_gen_count,
                attr_idx=attempt,
            )

            payload = {
                "model": self.model_tag,
                "messages": messages,
                "stream": False,
                "keep_alive": "5m",
                "options": {
                    "temperature": 0.85,
                    "top_p": 0.92,
                    "repeat_penalty": 1.12,
                    "num_predict": 320,
                    "seed": seed + attempt,
                },
            }

            try:
                resp = requests.post(f"{OLLAMA_API_BASE}/api/chat", json=payload, timeout=120)
                resp.raise_for_status()
                decoded = resp.json().get("message", {}).get("content", "")
                parsed = self.parse_samples(decoded)
                for s in parsed:
                    if s not in collected and s not in min_pool:
                        collected.append(s)
            except Exception as e:
                logger.warning(f"[Ollama Generation Retry] {e}")

        while len(collected) < num_needed:
            src = random.choice(min_pool)
            collected.append(simple_eda_transform(src, seed=seed + len(collected)))

        if prompt_method == "cot_boundary_filtered" and emb_helper is not None:
            final_samples = emb_helper.geometric_filter(collected, train_df, minority_label, num_needed)
        else:
            final_samples = collected[:num_needed]

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(final_samples, f, indent=2, ensure_ascii=False)

        return final_samples


# =========================================================================================
# 3. DATA LOADING, CLASSICAL BASELINES & PROMPTING BUILDER
# =========================================================================================

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

    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_df["is_synthetic"] = False
    return train_df, test_df


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


def simple_eda_transform(text: str, seed: int = 42) -> str:
    words = text.split()
    if len(words) < 4:
        return text
    rng = random.Random(seed)
    op = rng.choice(["swap", "delete", "insert"])
    if op == "swap":
        i, j = rng.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    elif op == "delete":
        words = [w for w in words if rng.random() > 0.15] or words[:2]
    elif op == "insert":
        words.insert(rng.randrange(len(words)), rng.choice(words))
    return " ".join(words)


def augment_eda(train_df: pd.DataFrame, minority_classes: List[int], target_count: int, seed: int = 42) -> pd.DataFrame:
    aug_rows = []
    for c in minority_classes:
        sub = train_df[train_df["label"] == c]
        needed = max(0, target_count - len(sub))
        if needed > 0 and len(sub) > 0:
            base_texts = sub["text"].tolist()
            for i in range(needed):
                aug_rows.append({
                    "text": simple_eda_transform(base_texts[i % len(base_texts)], seed=seed + i),
                    "label": c,
                    "is_synthetic": True,
                })
    return pd.concat([train_df, pd.DataFrame(aug_rows)], ignore_index=True) if aug_rows else train_df.copy()


class EmbeddingHelper:
    """Runs on CPU or GPU lightly for fast hard-negative retrieval & geometric filtering."""
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        )

    def get_hard_negative_majority_examples(self, train_df: pd.DataFrame, minority_label: int, k_maj: int = 4) -> List[Tuple[str, int]]:
        min_df = train_df[train_df["label"] == minority_label]
        maj_df = train_df[train_df["label"] != minority_label]
        if len(min_df) == 0 or len(maj_df) == 0:
            return []
        min_emb = self.encode(min_df["text"].tolist())
        maj_emb = self.encode(maj_df["text"].tolist())
        min_centroid = np.mean(min_emb, axis=0, keepdims=True)
        sims = np.dot(maj_emb, min_centroid.T).squeeze(-1)
        top_idx = np.argsort(-sims)[:k_maj]
        return [(maj_df.iloc[idx]["text"], int(maj_df.iloc[idx]["label"])) for idx in top_idx]

    def geometric_filter(self, candidates: List[str], train_df: pd.DataFrame, minority_label: int, target_n: int) -> List[str]:
        if len(candidates) <= target_n:
            return candidates
        min_texts = train_df[train_df["label"] == minority_label]["text"].tolist()
        maj_df = train_df[train_df["label"] != minority_label]
        min_centroid = np.mean(self.encode(min_texts), axis=0)
        maj_centroids = np.stack([np.mean(self.encode(grp["text"].tolist()), axis=0) for _, grp in maj_df.groupby("label")], axis=0)
        cand_emb = self.encode(candidates)
        margin_score = np.dot(cand_emb, min_centroid) - 0.85 * np.max(np.dot(cand_emb, maj_centroids.T), axis=1)
        ranked_idx = np.argsort(-margin_score)
        return [candidates[i] for i in ranked_idx[:target_n]]


def build_generation_prompt(
    task_key: str,
    minority_label: int,
    prompt_method: str,
    minority_examples: List[str],
    majority_examples: List[Tuple[str, int]],
    batch_gen_count: int = 8,
    attr_idx: int = 0,
) -> List[Dict[str, str]]:
    cfg = TASKS_CONFIG[task_key]
    label_name = cfg["label_names"][minority_label]
    all_labels_str = ", ".join([f"'{v}'" for v in cfg["label_names"].values()])
    domain = cfg["domain"]

    sys_msg = (
        f"You are an expert NLP synthetic data generator for a multiclass {domain} classification benchmark. "
        f"The possible classes in this dataset are: {all_labels_str}. "
        f"Your goal is to generate realistic, diverse, and strictly label-pure training samples belonging "
        f"ONLY to the target minority class '{label_name}'."
    )

    examples_block = ""
    if minority_examples and prompt_method != "zero_shot":
        formatted_ex = "\n".join([f"  - [{label_name}]: \"{ex.strip()}\"" for ex in minority_examples])
        examples_block = f"\nReference Examples of Target Class '{label_name}':\n{formatted_ex}\n"

    majority_block = ""
    if majority_examples and prompt_method in ["contrastive_majority", "cot_boundary_filtered", "random_majority"]:
        formatted_maj = "\n".join(
            [f"  - [NOT {label_name} | Majority Class '{cfg['label_names'][l_id]}']: \"{txt.strip()}\""
             for txt, l_id in majority_examples]
        )
        majority_block = (
            f"\nContrastive Boundary Examples (Majority Classes - DO NOT imitate these themes or labels):\n"
            f"{formatted_maj}\n"
            f"CRITICAL: Ensure your generated samples clearly belong to '{label_name}' and do NOT cross the "
            f"decision boundary into the majority classes shown above.\n"
        )

    attr_block = ""
    if prompt_method in ["attr_prompt", "cot_boundary_filtered"]:
        length_style, tone_style, vocab_style = ATTRIBUTES_POOL[attr_idx % len(ATTRIBUTES_POOL)]
        attr_block = (
            f"\nDiversity & Style Constraints for this batch:\n"
            f"  - Length: {length_style}\n"
            f"  - Tone: {tone_style}\n"
            f"  - Lexical focus: {vocab_style}\n"
        )

    cot_block = ""
    if prompt_method == "cot_boundary_filtered":
        cot_block = (
            f"\nSilently contrast '{label_name}' against ({all_labels_str}) so every sample stays strictly inside '{label_name}'.\n"
        )

    user_msg = (
        f"Task: Generate exactly {batch_gen_count} diverse, realistic {domain} texts belonging strictly to '{label_name}'.\n"
        f"{examples_block}{majority_block}{attr_block}{cot_block}\n"
        f"Output ONLY the {batch_gen_count} generated texts, one per line, prefixed with `SAMPLE: `."
    )

    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


# =========================================================================================
# 4. DOWNSTREAM CLASSIFIER EVALUATION & METRICS
# =========================================================================================

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
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_labels: int,
    minority_classes: List[int],
    base_model_name: str = "distilbert-base-uncased",
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 3e-5,
    seed: int = 42,
) -> Dict[str, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels).to(device)

    train_ds = TextClsDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    test_ds = TextClsDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    model.eval()
    preds, trues = [], []
    with torch.inference_mode():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
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

    del model, tokenizer, train_ds, test_ds
    flush_vram()

    return {
        "Macro_F1": round(macro_f1, 2),
        "Minority_F1": round(min_f1, 2),
        "Majority_F1": round(maj_f1, 2),
        "G_Mean": round(g_mean, 2),
        "Accuracy": round(acc, 2),
    }


def augment_dataframe_with_llm(
    train_df: pd.DataFrame,
    llm_gen: OllamaGenerator,
    task_key: str,
    prompt_method: str,
    k_shot: int,
    emb_helper: EmbeddingHelper,
    seed: int = 42,
) -> pd.DataFrame:
    cfg = TASKS_CONFIG[task_key]
    maj_count = cfg["majority_per_class"]
    synth_rows = []
    for min_c in cfg["minority_classes"]:
        current_n = len(train_df[train_df["label"] == min_c])
        needed = max(0, maj_count - current_n)
        if needed == 0:
            continue
        samples = llm_gen.generate_minority_samples(
            task_key=task_key,
            train_df=train_df,
            minority_label=min_c,
            num_needed=needed,
            prompt_method=prompt_method,
            k_shot=k_shot,
            emb_helper=emb_helper,
            seed=seed,
        )
        for s in samples:
            synth_rows.append({"text": s, "label": min_c, "is_synthetic": True})
    return pd.concat([train_df, pd.DataFrame(synth_rows)], ignore_index=True) if synth_rows else train_df.copy()


def compute_frechet_embedding_distance(emb_real: np.ndarray, emb_syn: np.ndarray) -> float:
    mu1, sigma1 = np.mean(emb_real, axis=0), np.cov(emb_real, rowvar=False) + np.eye(emb_real.shape[1]) * 1e-5
    mu2, sigma2 = np.mean(emb_syn, axis=0), np.cov(emb_syn, rowvar=False) + np.eye(emb_syn.shape[1]) * 1e-5
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return round(max(0.0, float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))), 4)


def compute_distribution_quality_metrics(
    train_df: pd.DataFrame, test_df: pd.DataFrame, syn_texts: List[str], minority_label: int, emb_helper: EmbeddingHelper
) -> Dict[str, float]:
    real_min_texts = pd.concat([
        train_df[train_df["label"] == minority_label]["text"],
        test_df[test_df["label"] == minority_label]["text"],
    ]).tolist()
    real_maj_texts = train_df[train_df["label"] != minority_label]["text"].tolist()

    emb_real_min = emb_helper.encode(real_min_texts)
    emb_real_maj = emb_helper.encode(real_maj_texts)
    emb_syn = emb_helper.encode(syn_texts)

    fed = compute_frechet_embedding_distance(emb_real_min, emb_syn)
    sim_matrix = np.dot(emb_syn, emb_syn.T)
    upper_sims = sim_matrix[np.triu_indices(len(emb_syn), k=1)]
    diversity = float(1.0 - np.mean(upper_sims)) if len(upper_sims) > 0 else 0.0
    margin = float(np.mean(np.dot(emb_syn, np.mean(emb_real_min, axis=0)) - np.dot(emb_syn, np.mean(emb_real_maj, axis=0))))

    return {
        "FED_LowerBetter": round(fed, 4),
        "Cosine_Diversity": round(diversity, 4),
        "Boundary_Margin": round(margin, 4),
    }


def plot_rq4_tsne_comparison(
    task_key: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
    syn_naive: List[str], syn_contrastive: List[str], minority_label: int, emb_helper: EmbeddingHelper
):
    cfg = TASKS_CONFIG[task_key]
    min_name = cfg["label_names"][minority_label]
    real_min_train = train_df[train_df["label"] == minority_label]["text"].tolist()
    real_min_test = test_df[test_df["label"] == minority_label]["text"].tolist()[:120]
    real_maj = train_df[train_df["label"] != minority_label].sample(
        n=min(240, len(train_df[train_df["label"] != minority_label])), random_state=42
    )["text"].tolist()

    syn_n, syn_c = syn_naive[:120], syn_contrastive[:120]
    all_embs = emb_helper.encode(real_maj + real_min_test + real_min_train + syn_n + syn_c)
    coords = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42).fit_transform(all_embs)

    i1, i2, i3, i4 = len(real_maj), len(real_maj) + len(real_min_test), len(real_maj) + len(real_min_test) + len(real_min_train), len(real_maj) + len(real_min_test) + len(real_min_train) + len(syn_n)
    c_maj, c_min_oracle, c_min_seed, c_syn_naive, c_syn_contr = coords[:i1], coords[i1:i2], coords[i2:i3], coords[i3:i4], coords[i4:]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    for ax, syn_pts, title_str, syn_color in [
        (axes[0], c_syn_naive, "(a) Standard Few-Shot (Minority-Only)\nMode Collapse & Majority Intrusion", "#e66101"),
        (axes[1], c_syn_contr, "(b) Contrastive Majority-Aware + Boundary Filter\nBroad Coverage & Clear Margin", "#1b9e77"),
    ]:
        ax.scatter(c_maj[:, 0], c_maj[:, 1], c="#b2abd2", alpha=0.40, s=25, label="Real Majority Classes")
        ax.scatter(c_min_oracle[:, 0], c_min_oracle[:, 1], c="#5e3c99", alpha=0.35, s=35, marker="s", label=f"Held-out Real '{min_name}'")
        ax.scatter(syn_pts[:, 0], syn_pts[:, 1], c=syn_color, alpha=0.82, s=42, marker="^", edgecolors="white", linewidth=0.4, label="LLM Synthetic Minority")
        ax.scatter(c_min_seed[:, 0], c_min_seed[:, 1], c="#d7191c", alpha=1.0, s=90, marker="*", edgecolors="black", linewidth=0.6, label=f"Seed Train '{min_name}'")
        ax.set_title(title_str, fontweight="bold")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.legend(loc="best", frameon=True)

    plt.suptitle(f"RQ4: t-SNE Manifold Analysis on {task_key.upper()} (Target Minority: {min_name})", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"fig4_rq4_tsne_{task_key}.pdf")
    plt.savefig(FIG_DIR / f"fig4_rq4_tsne_{task_key}.png")
    plt.close()


def plot_all_ablation_figures(df_rq1: pd.DataFrame, df_rq2: pd.DataFrame, df_rq3: pd.DataFrame, df_rq5: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    sns.lineplot(data=df_rq1, x="k_shot", y="Minority_F1", hue="Task", style="Task", markers=True, dashes=False, linewidth=2.2, markersize=8, ax=ax1)
    ax1.set_title("RQ1: Effect of Number of In-Context Examples ($k$) on Minority-F1", fontweight="bold")
    ax1.set_xticks(sorted(df_rq1["k_shot"].unique()))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_rq1_k_shot.pdf")
    plt.savefig(FIG_DIR / "fig1_rq1_k_shot.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.lineplot(data=df_rq2, x="Imbalance_Ratio", y="Minority_F1", hue="Method", style="Method", markers=True, dashes=False, linewidth=2.2, markersize=8, ax=ax)
    ax.set_title("RQ2: Impact of Imbalance Ratio ($\\rho = N_{maj}/N_{min}$) on Minority-F1", fontweight="bold")
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_rq2_imbalance_ratio.pdf")
    plt.savefig(FIG_DIR / "fig2_rq2_imbalance_ratio.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.barplot(data=df_rq3, x="Task", y="Minority_F1", hue="Majority_Context_Mode", palette="viridis", edgecolor="black", ax=ax)
    ax.set_title("RQ3: Does Giving Majority Class Examples Improve Multiclass Synthesis?", fontweight="bold")
    ax.set_ylim(bottom=max(0, df_rq3["Minority_F1"].min() - 10))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_rq3_majority_context.pdf")
    plt.savefig(FIG_DIR / "fig3_rq3_majority_context.png")
    plt.close()

    piv = df_rq5.pivot_table(index="Prompt_Method", columns="LLM", values="Macro_F1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Mean Macro-F1 (%)"}, ax=ax)
    ax.set_title("RQ5: Prompting Strategy vs. Open LLM Architecture (Macro-F1 %)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_rq5_prompting_heatmap.pdf")
    plt.savefig(FIG_DIR / "fig5_rq5_prompting_heatmap.png")
    plt.close()


def generate_latex_tables(df_rq1: pd.DataFrame, df_rq2: pd.DataFrame, df_rq3: pd.DataFrame, df_rq4: pd.DataFrame, df_rq5: pd.DataFrame):
    tex_lines = [
        "% AUTO-GENERATED LATEX TABLES FOR OVERLEAF (\\usepackage{booktabs})",
        df_rq5.to_latex(index=False, float_format="%.2f", escape=True, caption="RQ5: Open LLMs & Prompting Methods (IR=25:1).", label="tab:rq5"),
        df_rq1.to_latex(index=False, float_format="%.2f", escape=True, caption="RQ1: Effect of k In-Context Examples.", label="tab:rq1"),
        df_rq2.to_latex(index=False, float_format="%.2f", escape=True, caption="RQ2: Impact of Imbalance Ratio.", label="tab:rq2"),
        pd.merge(df_rq3, df_rq4, on=["Task", "LLM", "Majority_Context_Mode"], how="left").to_latex(
            index=False, float_format="%.3f", escape=True, caption="RQ3 & RQ4: Majority Context & Manifold Quality.", label="tab:rq3_rq4"
        ),
    ]
    with open(TABLE_DIR / "paper_tables.tex", "w", encoding="utf-8") as f:
        f.write("\n\n".join(tex_lines))


# =========================================================================================
# 5. MAIN EXECUTION
# =========================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--fast_dev", action="store_true", help="Quick test run (~8-12 mins)")
    parser.add_argument("--qwen_only", action="store_true", help="Run using ONLY your already installed qwen2.5:3b model")
    args = parser.parse_args()

    if args.fast_dev:
        args.seeds = [42]
        for tk in TASKS_CONFIG:
            TASKS_CONFIG[tk]["majority_per_class"] = 150
            TASKS_CONFIG[tk]["test_per_class"] = 80

    llms_to_run = {"Qwen2.5-3B": "qwen2.5:3b"} if args.qwen_only else OPEN_LLMS_OLLAMA
    emb_helper = EmbeddingHelper()

    records_rq1, records_rq2, records_rq3, records_rq4, records_rq5 = [], [], [], [], []
    imbalance_ratios = [50, 25, 10, 5]
    prompt_methods_rq5 = ["zero_shot", "few_shot_minority", "attr_prompt", "contrastive_majority", "cot_boundary_filtered"]
    k_values_rq1 = [0, 1, 3, 5, 10, 15]

    # Phase 1: Classical Baselines
    logger.info(">>> Phase 1: Running Classical Baselines (No-Aug, ROS, EDA) across Imbalance Ratios...")
    for task_key, cfg in TASKS_CONFIG.items():
        num_labels = len(cfg["label_names"])
        min_classes = cfg["minority_classes"]
        for ir in imbalance_ratios:
            for seed in args.seeds:
                train_df, test_df = build_imbalanced_dataset(task_key, imbalance_ratio=ir, seed=seed)
                res_no = evaluate_base_classifier(train_df, test_df, num_labels, min_classes, epochs=args.epochs, seed=seed)
                records_rq2.append({"Task": task_key, "Imbalance_Ratio": ir, "Method": "No Augmentation", "Seed": seed, **res_no})

                df_ros = augment_ros(train_df, min_classes, cfg["majority_per_class"], seed=seed)
                res_ros = evaluate_base_classifier(df_ros, test_df, num_labels, min_classes, epochs=args.epochs, seed=seed)
                records_rq2.append({"Task": task_key, "Imbalance_Ratio": ir, "Method": "Random Oversampling", "Seed": seed, **res_ros})

                df_eda = augment_eda(train_df, min_classes, cfg["majority_per_class"], seed=seed)
                res_eda = evaluate_base_classifier(df_eda, test_df, num_labels, min_classes, epochs=args.epochs, seed=seed)
                records_rq2.append({"Task": task_key, "Imbalance_Ratio": ir, "Method": "EDA (Heuristic)", "Seed": seed, **res_eda})

                if ir == 25:
                    records_rq5.append({"Task": task_key, "LLM": "Baseline", "Prompt_Method": "No Augmentation", "Seed": seed, **res_no})
                    records_rq5.append({"Task": task_key, "LLM": "Baseline", "Prompt_Method": "EDA", "Seed": seed, **res_eda})

    # Phase 2: Generate ALL synthetic datasets via Ollama FIRST, then unload Ollama from VRAM before PyTorch training
    for llm_key, model_tag in llms_to_run.items():
        logger.info(f">>> Phase 2: Generating Synthetic Data via Ollama [{llm_key} -> {model_tag}]...")
        llm_gen = OllamaGenerator(llm_key, model_tag)

        # Pre-generate & cache all required synthetic texts while model is hot in Ollama
        for task_key, cfg in TASKS_CONFIG.items():
            for seed in args.seeds:
                train_df_25, _ = build_imbalanced_dataset(task_key, imbalance_ratio=25, seed=seed)
                for p_method in prompt_methods_rq5 + ["random_majority"]:
                    k_val = 0 if p_method == "zero_shot" else 5
                    augment_dataframe_with_llm(train_df_25, llm_gen, task_key, p_method, k_val, emb_helper, seed=seed)

                if llm_key == "Qwen2.5-3B":
                    for k_val in k_values_rq1:
                        p_mode = "zero_shot" if k_val == 0 else "contrastive_majority"
                        augment_dataframe_with_llm(train_df_25, llm_gen, task_key, p_mode, k_val, emb_helper, seed=seed)
                    for ir in imbalance_ratios:
                        tr_ir, _ = build_imbalanced_dataset(task_key, imbalance_ratio=ir, seed=seed)
                        augment_dataframe_with_llm(tr_ir, llm_gen, task_key, "cot_boundary_filtered", 5, emb_helper, seed=seed)

        # Immediately unload Ollama model from 6GB RTX 4050 VRAM before fine-tuning DistilBERT!
        llm_gen.unload_from_vram()

        logger.info(f">>> Phase 2b: Training & Evaluating Base Classifiers on [{llm_key}] Synthetic Data...")
        for task_key, cfg in TASKS_CONFIG.items():
            num_labels = len(cfg["label_names"])
            min_classes = cfg["minority_classes"]
            for seed in args.seeds:
                train_df_25, test_df_25 = build_imbalanced_dataset(task_key, imbalance_ratio=25, seed=seed)

                for p_method in prompt_methods_rq5:
                    k_val = 0 if p_method == "zero_shot" else 5
                    df_aug = augment_dataframe_with_llm(train_df_25, llm_gen, task_key, p_method, k_val, emb_helper, seed=seed)
                    res_p = evaluate_base_classifier(df_aug, test_df_25, num_labels, min_classes, epochs=args.epochs, seed=seed)
                    records_rq5.append({"Task": task_key, "LLM": llm_key, "Prompt_Method": p_method, "Seed": seed, **res_p})

                    if p_method in ["few_shot_minority", "contrastive_majority", "cot_boundary_filtered"]:
                        mode_label = {
                            "few_shot_minority": "Minority-Only (Standard)",
                            "contrastive_majority": "Hard-Negative Majority-Aware",
                            "cot_boundary_filtered": "Contrastive + Boundary Filtered",
                        }[p_method]
                        records_rq3.append({"Task": task_key, "LLM": llm_key, "Majority_Context_Mode": mode_label, "Seed": seed, **res_p})
                        if seed == args.seeds[0]:
                            target_min = min_classes[0]
                            syn_sub = df_aug[(df_aug["label"] == target_min) & (df_aug["is_synthetic"] == True)]["text"].tolist()
                            dist_m = compute_distribution_quality_metrics(train_df_25, test_df_25, syn_sub, target_min, emb_helper)
                            records_rq4.append({"Task": task_key, "LLM": llm_key, "Majority_Context_Mode": mode_label, **dist_m})

                df_rand_maj = augment_dataframe_with_llm(train_df_25, llm_gen, task_key, "random_majority", 5, emb_helper, seed=seed)
                res_rm = evaluate_base_classifier(df_rand_maj, test_df_25, num_labels, min_classes, epochs=args.epochs, seed=seed)
                records_rq3.append({"Task": task_key, "LLM": llm_key, "Majority_Context_Mode": "Random Majority-Aware", "Seed": seed, **res_rm})
                if seed == args.seeds[0]:
                    target_min = min_classes[0]
                    syn_rm = df_rand_maj[(df_rand_maj["label"] == target_min) & (df_rand_maj["is_synthetic"] == True)]["text"].tolist()
                    dist_rm = compute_distribution_quality_metrics(train_df_25, test_df_25, syn_rm, target_min, emb_helper)
                    records_rq4.append({"Task": task_key, "LLM": llm_key, "Majority_Context_Mode": "Random Majority-Aware", **dist_rm})

                if llm_key == "Qwen2.5-3B":
                    for k_val in k_values_rq1:
                        p_mode = "zero_shot" if k_val == 0 else "contrastive_majority"
                        df_k = augment_dataframe_with_llm(train_df_25, llm_gen, task_key, p_mode, k_val, emb_helper, seed=seed)
                        res_k = evaluate_base_classifier(df_k, test_df_25, num_labels, min_classes, epochs=args.epochs, seed=seed)
                        target_min = min_classes[0]
                        syn_k = df_k[(df_k["label"] == target_min) & (df_k["is_synthetic"] == True)]["text"].tolist()
                        dist_k = compute_distribution_quality_metrics(train_df_25, test_df_25, syn_k, target_min, emb_helper)
                        records_rq1.append({"Task": task_key, "LLM": llm_key, "k_shot": k_val, "Seed": seed, **res_k, **dist_k})

                    for ir in imbalance_ratios:
                        tr_ir, te_ir = build_imbalanced_dataset(task_key, imbalance_ratio=ir, seed=seed)
                        df_ir = augment_dataframe_with_llm(tr_ir, llm_gen, task_key, "cot_boundary_filtered", 5, emb_helper, seed=seed)
                        res_ir = evaluate_base_classifier(df_ir, te_ir, num_labels, min_classes, epochs=args.epochs, seed=seed)
                        records_rq2.append({"Task": task_key, "Imbalance_Ratio": ir, "Method": f"LLM ({llm_key} + CBAP)", "Seed": seed, **res_ir})

                    if seed == args.seeds[0]:
                        target_min = min_classes[0]
                        syn_naive = llm_gen.generate_minority_samples(
                            task_key, train_df_25, target_min, cfg["majority_per_class"], "few_shot_minority", 5, emb_helper, seed=seed
                        )
                        syn_contr = llm_gen.generate_minority_samples(
                            task_key, train_df_25, target_min, cfg["majority_per_class"], "cot_boundary_filtered", 5, emb_helper, seed=seed
                        )
                        plot_rq4_tsne_comparison(task_key, train_df_25, test_df_25, syn_naive, syn_contr, target_min, emb_helper)

    # Phase 3: Save Outputs
    df_rq1_raw, df_rq2_raw, df_rq3_raw, df_rq4_raw, df_rq5_raw = (
        pd.DataFrame(records_rq1), pd.DataFrame(records_rq2), pd.DataFrame(records_rq3), pd.DataFrame(records_rq4), pd.DataFrame(records_rq5)
    )
    df_rq1_raw.to_csv(CSV_DIR / "rq1_k_shot_raw.csv", index=False)
    df_rq2_raw.to_csv(CSV_DIR / "rq2_imbalance_ratio_raw.csv", index=False)
    df_rq3_raw.to_csv(CSV_DIR / "rq3_majority_context_raw.csv", index=False)
    df_rq4_raw.to_csv(CSV_DIR / "rq4_manifold_metrics.csv", index=False)
    df_rq5_raw.to_csv(CSV_DIR / "rq5_prompt_methods_raw.csv", index=False)

    df_rq1_agg = df_rq1_raw.groupby(["Task", "k_shot"], as_index=False)[["Macro_F1", "Minority_F1", "G_Mean", "Cosine_Diversity", "FED_LowerBetter"]].mean()
    df_rq2_agg = df_rq2_raw.groupby(["Task", "Imbalance_Ratio", "Method"], as_index=False)[["Macro_F1", "Minority_F1", "G_Mean"]].mean()
    df_rq3_agg = df_rq3_raw.groupby(["Task", "LLM", "Majority_Context_Mode"], as_index=False)[["Macro_F1", "Minority_F1", "G_Mean"]].mean()
    df_rq5_agg = df_rq5_raw.groupby(["Task", "LLM", "Prompt_Method"], as_index=False)[["Macro_F1", "Minority_F1", "G_Mean", "Accuracy"]].mean()

    plot_all_ablation_figures(df_rq1_raw, df_rq2_raw, df_rq3_raw, df_rq5_raw)
    generate_latex_tables(df_rq1_agg, df_rq2_agg, df_rq3_agg, df_rq4_raw, df_rq5_agg)
    logger.info("ALL EXPERIMENTS & FIGURES COMPLETED! Check the `results_conference/` folder in VS Code.")


if __name__ == "__main__":
    main()