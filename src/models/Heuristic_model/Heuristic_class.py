import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


def read_capabilities(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    caps = data.get("Capability", [])
    if isinstance(caps, list):
        return [str(c).strip() for c in caps if c and str(c).strip()]
    if isinstance(caps, str) and caps.strip():
        return [caps.strip()]
    return []


def capabilities_to_text(capabilities):
    if isinstance(capabilities, list):
        caps = [str(c).strip() for c in capabilities if c and str(c).strip()]
        return "; ".join(caps) if caps else "no capability detected"
    if isinstance(capabilities, str) and capabilities.strip():
        return capabilities.strip()
    return "no capability detected"


class BertBinaryClassifier(nn.Module):
    def __init__(self, bert_model_name, dropout_rate=0.2):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(hidden_size * 2, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_vec = out.pooler_output
        last_hidden = out.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        mean_vec = (last_hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        combined = torch.cat([cls_vec, mean_vec], dim=-1)
        logit = self.classifier(self.dropout(combined))
        return logit.squeeze(-1)


def load_config(model_dir):
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_dir, device):
    cfg = load_config(model_dir)
    model = BertBinaryClassifier(
        cfg["bert_model_name"], cfg.get("dropout_rate", 0.2)
    )
    ckpt_path = os.path.join(model_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    return model, tokenizer, cfg


def predict_proba(text, model, tokenizer, max_len, device):
    enc = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logit = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            enc["token_type_ids"].to(device),
        )
        return torch.sigmoid(logit).item()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict benign/malware confidence from heuristic_features.json"
    )
    parser.add_argument("json_path", help="Path to heuristic_features.json")
    parser.add_argument(
        "--model-dir",
        default="bert_binary_model",
        help="Directory containing best_model.pt, config.json, tokenizer/",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. cpu or cuda. Default: auto",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold override for label",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.json_path):
        print(f"File not found: {args.json_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.model_dir):
        print(f"Model dir not found: {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, tokenizer, cfg = load_model(args.model_dir, device)
    max_len = cfg.get("max_len", 512)
    threshold = cfg.get("threshold", 0.5) if args.threshold is None else args.threshold

    caps = read_capabilities(args.json_path)
    text = capabilities_to_text(caps)
    malware_prob = predict_proba(text, model, tokenizer, max_len, device)
    benign_prob = 1.0 - malware_prob
    prediction = "malware" if malware_prob >= threshold else "benign"

    result = {
        "prediction": prediction,
        "malware_prob": round(malware_prob, 4),
        "benign_prob": round(benign_prob, 4),
        "threshold": threshold,
        "device": str(device),
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
