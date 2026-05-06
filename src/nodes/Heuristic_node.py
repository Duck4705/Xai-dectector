import os
import torch
from src.state.state import HeuristicState
from src.models.Heuristic_model.Heuristic_class import (
    read_capabilities,
    capabilities_to_text,
    load_model,
    predict_proba,
)

# --- Heuristic Node ---
def HeuristicNode(state: dict):
    sample_input_folder = state["file_name"]
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_path = os.path.join(project_root, "input", sample_input_folder)
    
    json_path = os.path.join(sample_path, "heuristic_features.json")
    model_dir = os.path.join(project_root, "src", "models", "Heuristic_model", "bert_binary_model")
    
    # Fallback nếu file không tồn tại
    if not os.path.exists(json_path):
        return {
            "heuristic_state": HeuristicState(
                malware_confidence_score=0.0,
                benign_confidence_score=0.0,
                capability="no capability detected"
            )
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Đọc và extract features từ JSON thành string
    caps = read_capabilities(json_path)
    text = capabilities_to_text(caps)
    
    # Nếu file không có capability nào
    if text == "no capability detected" or not caps:
        return {
            "heuristic_state": HeuristicState(
                malware_confidence_score=0.0,
                benign_confidence_score=0.0,
                capability=text
            )
        }

    # 2. Load model
    model, tokenizer, cfg = load_model(model_dir, device)
    max_len = cfg.get("max_len", 512)
    
    # 3. Predict
    malware_prob = predict_proba(text, model, tokenizer, max_len, device)
    benign_prob = 1.0 - malware_prob

    # 4. Trả về HeuristicState (để merge vào XaiDetectorState)
    return {
        "heuristic_state": HeuristicState(
            malware_confidence_score=round(float(malware_prob), 4),
            benign_confidence_score=round(float(benign_prob), 4),
            capability=text
        )
    }
