"""
predict_capstone.py
====================
Đầu vào : file JSON chứa raw bytes của binary (malware/benign)
Đầu ra  : file JSON gồm confidence_score (benign/malware) và top-K instruction lists (Grad-CAM + Capstone)

Cách dùng:
    python predict_capstone.py --input raw_bytes.json --model model/best_model.pt
    python predict_capstone.py --input raw_bytes.json --model model/best_model.pt --topk 3 --window 10 --output result.json
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (đồng bộ với lúc train)
# ─────────────────────────────────────────────────────────────────────────────
WIDTH      = 256
MAX_H      = 512
MIN_H      = 8
CLASS_NAMES = ['benign', 'malware']
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE  (phải khớp 100% với lúc train)
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch, drop2d=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.Dropout2d(drop2d),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.conv(x))


class MalwareCNN_MultiScaleGAP(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()

        def down_block(in_ch, out_ch, drop2d=0.1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.GELU(),
                ResBlock(out_ch, drop2d),
            )

        self.block1 = down_block(1,   32,  drop2d=0.05)
        self.pool1  = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.block2 = down_block(32,  64,  drop2d=0.1)
        self.pool2  = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.block3 = down_block(64,  128, drop2d=0.15)
        self.pool3  = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.GELU(),
            ResBlock(256, drop2d=0.2),
            ResBlock(256, drop2d=0.2),
        )
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
        self._gradients   = None
        self._activations = None

    def forward(self, x):
        x = self.block1(x); x = self.pool1(x)
        x = self.block2(x); x = self.pool2(x)
        x = self.block3(x); x = self.pool3(x)
        x = self.block4(x)
        gmp = self.gmp(x).flatten(1)
        gap = self.gap(x).flatten(1)
        return self.classifier(torch.cat([gmp, gap], dim=1))

    def forward_with_hooks(self, x):
        """Forward pass dùng cho Grad-CAM: lưu activations & gradients."""
        x = self.block1(x); x = self.pool1(x)
        x = self.block2(x); x = self.pool2(x)
        x = self.block3(x); x = self.pool3(x)
        x = self.block4(x)
        self._activations = x
        x.register_hook(lambda grad: setattr(self, '_gradients', grad))
        gmp = self.gmp(x).flatten(1)
        gap = self.gap(x).flatten(1)
        return self.classifier(torch.cat([gmp, gap], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str) -> MalwareCNN_MultiScaleGAP:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    model = MalwareCNN_MultiScaleGAP(num_classes=2, dropout=0.5).to(DEVICE)
    ckpt  = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[OK] Loaded model  epoch={ckpt.get('epoch','?')}  val_f1={ckpt.get('val_f1',0):.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# LOAD RAW BYTES
# ─────────────────────────────────────────────────────────────────────────────
def load_raw_bytes_json(json_path: str) -> list:
    with open(json_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ('raw_bytes', 'bytes', 'data', 'hex', 'content'):
            if key in data:
                data = data[key]
                break
    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], int):
            return data
        if isinstance(data[0], str):
            return [int(x, 16) for x in data]
    if isinstance(data, str):
        h = data.strip().replace(' ', '').replace('\n', '')
        if len(h) % 2:
            h = h[:-1]
        return [int(h[i:i+2], 16) for i in range(0, len(h), 2)]
    raise ValueError(f'Định dạng JSON không hỗ trợ: {json_path}')


# ─────────────────────────────────────────────────────────────────────────────
# BYTES -> IMAGE
# ─────────────────────────────────────────────────────────────────────────────
def bytes_to_image(raw_bytes_list, width=WIDTH, max_h=MAX_H):
    arr = np.array(raw_bytes_list, dtype=np.uint8)
    n   = len(arr)
    if n == 0:
        return np.zeros((max_h, width), dtype=np.float32), MIN_H, 0, [-1]*max_h

    H       = math.ceil(n / width)
    pad_len = H * width - n
    if pad_len > 0:
        arr = np.pad(arr, (0, pad_len), constant_values=0)

    if H <= max_h:
        img     = arr.reshape(H, width).astype(np.float32) / 255.0
        real_h  = H
        row_map = list(range(H)) + [-1] * (max_h - H)
        if H < max_h:
            img = np.concatenate(
                [img, np.zeros((max_h - H, width), dtype=np.float32)], axis=0
            )
    else:
        n_head      = max_h // 4
        n_sample    = max_h - n_head
        sample_rows = np.linspace(n_head, H - 1, n_sample, dtype=int)
        head_rows   = np.arange(n_head)
        all_rows    = np.concatenate([head_rows, sample_rows])
        row_map     = all_rows.tolist()
        img         = arr.reshape(H, width)[all_rows].astype(np.float32) / 255.0
        real_h      = max_h

    return img, real_h, n, row_map


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: MalwareCNN_MultiScaleGAP):
        self.model = model

    def generate(self, tensor: torch.Tensor, target_class: Optional[int] = None):
        self.model.eval()
        tensor = tensor.to(DEVICE)

        logits     = self.model.forward_with_hooks(tensor)
        probs      = F.softmax(logits, dim=1)
        pred_class = int(logits.argmax(dim=1).item())
        if target_class is None:
            target_class = pred_class

        self.model.zero_grad()
        logits[0, target_class].backward()

        grads = self.model._gradients.detach().cpu()   # (1,256,H',W')
        acts  = self.model._activations.detach().cpu() # (1,256,H',W')
        alpha = grads[0].mean(dim=(1, 2))              # (256,)

        cam = torch.zeros(acts.shape[2:])
        for k, a_k in enumerate(acts[0]):
            cam += alpha[k] * a_k
        cam = F.relu(cam).numpy()

        H_orig, W_orig = tensor.shape[2], tensor.shape[3]
        cam = cv2.resize(cam, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        vmin, vmax = cam.min(), cam.max()
        cam = (cam - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(cam)

        probs_np = probs.detach().cpu().numpy()[0]
        return cam.astype(np.float32), pred_class, probs_np


# ─────────────────────────────────────────────────────────────────────────────
# TOP-K OFFSETS
# ─────────────────────────────────────────────────────────────────────────────
def heatmap_to_offsets(heatmap, row_map, n_bytes, width=WIDTH, top_k=10, min_score=0.3):
    H_img, W_img = heatmap.shape
    flat_scores  = heatmap.flatten()
    flat_indices = np.argsort(flat_scores)[::-1]

    results = []
    for flat_idx in flat_indices:
        if len(results) >= top_k:
            break
        score = float(flat_scores[flat_idx])
        if score < min_score:
            break

        pixel_row = int(flat_idx) // W_img
        pixel_col = int(flat_idx) % W_img

        if pixel_row >= len(row_map):
            continue
        original_row = row_map[pixel_row]
        if original_row == -1:
            continue

        byte_offset = int(original_row) * width + pixel_col
        if byte_offset >= n_bytes:
            continue
        if any(abs(byte_offset - s['offset']) < 16 for s in results):
            continue

        results.append({
            'offset'      : byte_offset,
            'hex_offset'  : hex(byte_offset),
            'heat_score'  : score,
            'pixel_row'   : pixel_row,
            'pixel_col'   : pixel_col,
        })

    return sorted(results, key=lambda x: x['heat_score'], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# CAPSTONE DISASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
def find_aligned_offset(raw_bytes: list, byte_offset: int) -> int:
    search_start = max(0, byte_offset - 16)
    chunk = bytes(raw_bytes[search_start: byte_offset + 16])
    md = Cs(CS_ARCH_X86, CS_MODE_32)
    for ins in md.disasm(chunk, search_start):
        if ins.address <= byte_offset < ins.address + ins.size:
            return ins.address
        if ins.address >= byte_offset:
            return ins.address
    return byte_offset


def disassemble_window(raw_bytes: list, byte_offset: int, window: int = 10) -> dict:
    try:
        aligned = find_aligned_offset(raw_bytes, byte_offset)

        is_pe64 = (len(raw_bytes) > 0x3C + 4
                   and raw_bytes[0] == 0x4D and raw_bytes[1] == 0x5A)
        mode = CS_MODE_64 if is_pe64 else CS_MODE_32
        md   = Cs(CS_ARCH_X86, mode)
        md.detail = False

        chunk_start = max(0, aligned - window * 6)
        chunk_end   = min(len(raw_bytes), aligned + (window + 1) * 6)
        chunk       = bytes(raw_bytes[chunk_start:chunk_end])
        all_instrs  = list(md.disasm(chunk, chunk_start))

        center_idx = next(
            (i for i, ins in enumerate(all_instrs) if ins.address >= aligned), 0
        )
        start_idx  = max(0, center_idx - window)
        end_idx    = min(len(all_instrs), center_idx + window + 1)
        selected   = all_instrs[start_idx:end_idx]
        new_center = center_idx - start_idx

        instructions = [{
            'offset' : ins.address,
            'hex'    : ins.bytes.hex(),
            'opcode' : ins.mnemonic,
            'operand': ins.op_str,
            'disasm' : f'{ins.mnemonic} {ins.op_str}'.strip(),
            'size'   : ins.size,
        } for ins in selected]

        return {
            'aligned_offset' : aligned,
            'hex_offset'     : hex(aligned),
            'instructions'   : instructions,
            'center_index'   : new_center,
            'error'          : None,
        }

    except Exception as e:
        return {
            'aligned_offset' : byte_offset,
            'hex_offset'     : hex(byte_offset),
            'instructions'   : [],
            'center_index'   : 0,
            'error'          : str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BUILD OUTPUT JSON
# ─────────────────────────────────────────────────────────────────────────────
def build_output(
    confidence_scores: Dict[str, float],
    context_blocks: List[dict],
) -> dict:
    """Tạo output JSON cuối cùng — confidence scores + top K instruction lists."""
    sorted_blocks = sorted(
        context_blocks, key=lambda b: b.get('heat_score', 0.0), reverse=True
    )
    top_opcodes = []
    for b in sorted_blocks:
        instrs = b.get('instructions', [])
        instructions = []
        for ins in instrs:
            text = ins.get('disasm') or ins.get('opcode')
            if text:
                instructions.append(text)

        top_opcodes.append({
            'hex_offset' : b['hex_offset'],
            'heat_score' : round(b['heat_score'], 6),
            'instructions': instructions,
        })

    return {
        'confidence_scores': confidence_scores,
        'top_k_opcodes'     : top_opcodes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def predict(
    json_path   : str,
    model_path  : str,
    top_k       : int = 3,
    window      : int = 10,
    target_class: Optional[int] = None,
    output_path : str = None,
) -> dict:

    sha256 = Path(json_path).parent.name or Path(json_path).stem
    print(f"\n{'='*60}")
    print(f"[Capstone] Predicting: {sha256[:24]}...")
    print(f"{'='*60}")

    # 1. Load model
    model   = load_model(model_path)
    gradcam = GradCAM(model)

    # 2. Load bytes & convert to image
    print("\n[1] Loading raw bytes...")
    raw_bytes = load_raw_bytes_json(json_path)
    print(f"    File size: {len(raw_bytes):,} bytes")

    img, real_h, n_bytes, row_map = bytes_to_image(raw_bytes)
    img_trimmed     = img[:real_h, :]
    tensor          = torch.from_numpy(img_trimmed).unsqueeze(0).unsqueeze(0)
    row_map_trimmed = row_map[:real_h]

    # 3. Grad-CAM
    print("\n[2] Grad-CAM...")
    heatmap, pred_cls_idx, probs = gradcam.generate(tensor, target_class)
    pred_name = CLASS_NAMES[pred_cls_idx]
    pred_confidence = float(probs[pred_cls_idx])
    print(f"    Prediction : {pred_name} ({pred_confidence:.2%})")

    # 4. Top K offsets
    print(f"\n[3] Top {top_k} offsets from heatmap...")
    top_offsets = heatmap_to_offsets(
        heatmap, row_map_trimmed, n_bytes, top_k=top_k
    )
    for i, o in enumerate(top_offsets):
        print(f"    [{i+1:2d}] {o['hex_offset']:>10s}  score={o['heat_score']:.4f}")

    # 5. Disassemble
    print(f"\n[4] Disassembling (window={window})...")
    context_blocks = []
    for i, off_info in enumerate(top_offsets):
        disasm = disassemble_window(raw_bytes, off_info['offset'], window)
        block  = {
            'block_index' : i,
            'byte_offset' : off_info['offset'],
            'hex_offset'  : disasm['hex_offset'],
            'heat_score'  : off_info['heat_score'],
            'instructions': disasm['instructions'],
            'center_index': disasm['center_index'],
            'has_error'   : disasm['error'] is not None,
        }
        context_blocks.append(block)
        status = 'OK' if not block['has_error'] else 'ERR'
        print(f"    [{i+1:2d}] {status} {len(disasm['instructions'])} instructions @ {disasm['hex_offset']}")

    # 6. Build output JSON
    confidence_scores = {
        CLASS_NAMES[i]: round(float(probs[i]), 6) for i in range(len(CLASS_NAMES))
    }
    result = build_output(confidence_scores, context_blocks)

    # 7. Save
    if output_path is None:
        output_path = str(Path(json_path).parent / 'gradcam_result_capstone.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved -> {output_path}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Malware GradCAM + Capstone disassembler'
    )
    parser.add_argument('--input',  '-i', required=True,
                        help='Path to raw_bytes.json')
    parser.add_argument('--model',  '-m', default='model/best_model.pt',
                        help='Path to best_model.pt  (default: model/best_model.pt)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON path  (default: same dir as input)')
    parser.add_argument('--topk',   '-k', type=int, default=3,
                        help='Top K hotspot offsets  (default: 3)')
    parser.add_argument('--window', '-w', type=int, default=10,
                        help='Instruction window before/after  (default: 10)')
    args = parser.parse_args()

    result = predict(
        json_path    = args.input,
        model_path   = args.model,
        top_k        = args.topk,
        window       = args.window,
        output_path  = args.output,
    )

    print(f"\n{'='*60}")
    benign_score = result['confidence_scores'].get('benign', 0.0)
    malware_score = result['confidence_scores'].get('malware', 0.0)
    print(f"  Confidence  : benign={benign_score:.2%}  malware={malware_score:.2%}")
    print(f"  Top opcodes : {len(result['top_k_opcodes'])}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
