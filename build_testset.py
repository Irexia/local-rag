import json, re, hashlib, sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path("./data_sample")
OUT = Path("./testset.jsonl")
HELDOUT_FRACTION = 0.30
SPLIT_SALT = "rag-eval-v1"

S, B, I, Z, C = ("solar_panels.txt", "battery_storage.txt", "inverter_guide.txt",
                 "system_sizing.txt", "commissioning_and_monitoring.txt")

UNANS = "Not covered by the documents. The system should say it does not know."
rows = []


def Q(cat, question, answer, *spans):
    """spans: (source_file, verbatim_text) tuples"""
    rows.append({"category": cat, "question": question, "ground_truth": answer,
                 "gold": [{"source": s, "text": t} for s, t in spans]})