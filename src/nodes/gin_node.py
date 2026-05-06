import os, json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from src.state.state import GinState
from src.models.Gin_model.gin_class import GINModel

try:
    from torch_geometric.explain import Explainer, GNNExplainer
    PYG_EXPLAIN_V2 = True
except ImportError:
    try:
        from torch_geometric.nn import GNNExplainer
        PYG_EXPLAIN_V2 = False
    except ImportError:
        PYG_EXPLAIN_V2 = None

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VOCAB_PATH = os.path.join(PROJECT_ROOT, "src", "models", "Gin_model", "vocab.json")
MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "models", "Gin_model", "best_gin_model.pth")

# Load vocab
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)
OPCODE_TO_IDX = vocab["opcode_to_idx"]
FEATURE_DIM = vocab["feature_dim"]


# Helpers
def build_node_feature(opcodes):
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    for op in opcodes:
        idx = OPCODE_TO_IDX.get(op.strip().lower())
        if idx is not None:
            feat[idx] += 1
        else:
            feat[-1] += 1
    feat = np.log1p(feat)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat /= norm
    return feat


def cfg_to_data(cfg_data: dict):
    """Convert cfg dict sang PyG Data + trả thêm idx_to_name, idx_to_opcodes cho XAI."""
    nodes = cfg_data.get("nodes", [])
    edges = cfg_data.get("edges", [])
    if not nodes:
        return None, None, None

    name_to_idx = {n["name"]: i for i, n in enumerate(nodes)}
    idx_to_name = {i: n["name"] for i, n in enumerate(nodes)}
    idx_to_opcodes = {i: n.get("opcodes", []) for i, n in enumerate(nodes)}

    x = torch.tensor(
        np.array([build_node_feature(n.get("opcodes", [])) for n in nodes]),
        dtype=torch.float)
    src, dst = [], []
    for e in edges:
        s, d = e.get("from", ""), e.get("to", "")
        if s in name_to_idx and d in name_to_idx:
            src.append(name_to_idx[s])
            dst.append(name_to_idx[d])
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    ei, _ = add_self_loops(ei, num_nodes=len(nodes))
    data = Data(x=x, edge_index=ei)
    data.batch = torch.zeros(len(nodes), dtype=torch.long)
    return data, idx_to_name, idx_to_opcodes


def explain_and_extract_subgraphs(model, data, idx_to_name, idx_to_opcodes,
                                   num_subgraphs=3, max_parents=5, max_children=5):
    """Chạy GNNExplainer và trích xuất top subgraphs theo kiểu ego-graph."""
    device = data.x.device

    if PYG_EXPLAIN_V2:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=300),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='raw',
            ),
        )
        explanation = explainer(data.x, data.edge_index, batch=data.batch)
        edge_mask = explanation.edge_mask.cpu().detach().numpy()
    elif PYG_EXPLAIN_V2 is False:
        explainer = GNNExplainer(model, epochs=200, return_type='raw')
        node_feat_mask, edge_mask = explainer.explain_graph(data.x, data.edge_index, batch=data.batch)
        edge_mask = edge_mask.cpu().detach().numpy()
    else:
        return []

    edge_index = data.edge_index.cpu().numpy()

    # Lọc bỏ self-loops và xây dựng danh sách edge có trọng số
    incoming_edges = {}
    outgoing_edges = {}
    node_importance = {}

    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        w = float(edge_mask[i])
        if u != v:
            incoming_edges.setdefault(v, []).append((u, w))
            outgoing_edges.setdefault(u, []).append((v, w))
            node_importance[u] = node_importance.get(u, 0.0) + w
            node_importance[v] = node_importance.get(v, 0.0) + w
        else:
            # Self-loop edge contributes to node's own importance
            node_importance[u] = node_importance.get(u, 0.0) + w

    # Sắp xếp node toàn cục theo importance giảm dần
    sorted_global_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)

    # Chọn top num_subgraphs node làm center
    used_centers = set()
    centers = []
    for node_id, imp in sorted_global_nodes:
        if node_id not in used_centers:
            centers.append((node_id, imp))
            used_centers.add(node_id)
        if len(centers) >= num_subgraphs:
            break

    results = []
    for i, (center_id, center_imp) in enumerate(centers):
        # Top parents: edge chỉ VÀO center
        parents_raw = incoming_edges.get(center_id, [])
        parents_raw.sort(key=lambda x: x[1], reverse=True)
        top_parents = parents_raw[:max_parents]

        # Top children: edge từ center chỉ RA
        children_raw = outgoing_edges.get(center_id, [])
        children_raw.sort(key=lambda x: x[1], reverse=True)
        top_children = children_raw[:max_children]

        # Xây dựng edges
        sub_edges = []
        for parent_id, w in top_parents:
            sub_edges.append({
                "from": idx_to_name[parent_id],
                "to": idx_to_name[center_id],
                "weight": round(w, 4)
            })
        for child_id, w in top_children:
            sub_edges.append({
                "from": idx_to_name[center_id],
                "to": idx_to_name[child_id],
                "weight": round(w, 4)
            })

        # Xây dựng nodes (center + parents + children) kèm opcodes
        all_node_ids = set()
        node_infos = []

        # Center node
        all_node_ids.add(center_id)
        node_infos.append({
            "name": idx_to_name[center_id],
            "role": "center",
            "importance": round(float(center_imp), 4),
            "opcodes": " ".join(idx_to_opcodes.get(center_id, []))
        })

        # Parent nodes
        for parent_id, w in top_parents:
            if parent_id not in all_node_ids:
                all_node_ids.add(parent_id)
                node_infos.append({
                    "name": idx_to_name[parent_id],
                    "role": "parent",
                    "edge_weight": round(w, 4),
                    "opcodes": " ".join(idx_to_opcodes.get(parent_id, []))
                })

        # Child nodes
        for child_id, w in top_children:
            if child_id not in all_node_ids:
                all_node_ids.add(child_id)
                node_infos.append({
                    "name": idx_to_name[child_id],
                    "role": "child",
                    "edge_weight": round(w, 4),
                    "opcodes": " ".join(idx_to_opcodes.get(child_id, []))
                })

        results.append({
            "subgraph_index": i + 1,
            "center_node": idx_to_name[center_id],
            "center_importance": round(float(center_imp), 4),
            "nodes": node_infos,
            "edges": sub_edges
        })

    # Sắp xếp theo center_importance giảm dần
    results.sort(key=lambda x: x["center_importance"], reverse=True)
    for idx, r in enumerate(results):
        r["subgraph_index"] = idx + 1

    return results

# Load cfg riêng lẻ (dùng bởi gin_node)
def LoadCFG(sample_input_folder: str) -> dict:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_path = os.path.join(project_root, "input", sample_input_folder)
    cfg_path = os.path.join(sample_path, "cfg.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)  # dict có keys: "nodes", "edges"
    
    return cfg_data

# --- Gin Node ---
def GinNode(state: dict):
    sample_input_folder = state["file_name"]

    # 1. Load CFG data (trả về dict)
    cfg_data = LoadCFG(sample_input_folder)

    # 2. Convert CFG dict sang PyG Data + metadata cho XAI
    data, idx_to_name, idx_to_opcodes = cfg_to_data(cfg_data)
    if data is None:
        return {
            "gin_state": GinState(
                malware_confidence_score=0.0,
                benign_confidence_score=0.0,
                Xai_cfg={},
            )
        }

    # 3. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = GINModel(
        config["input_dim"],
        config["hidden_dim"],
        config["num_classes"],
        config["num_layers"],
        config["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 4. Dự đoán
    data = data.to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    benign_score = round(float(probs[0]), 4)
    malware_score = round(float(probs[1]), 4)

    # 5. XAI — GNNExplainer trích xuất subgraphs quan trọng
    subgraphs = explain_and_extract_subgraphs(
        model, data, idx_to_name, idx_to_opcodes,
        num_subgraphs=1, max_parents=5, max_children=5
    )

    # Chuyển subgraphs list thành dict cho Xai_cfg
    xai_cfg = {"important_subgraphs": subgraphs}

    # 6. Trả về GinState
    return {
        "gin_state": GinState(
            malware_confidence_score=malware_score,
            benign_confidence_score=benign_score,
            Xai_cfg=xai_cfg,
        )
    }