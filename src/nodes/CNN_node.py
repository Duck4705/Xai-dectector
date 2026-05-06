import os
from src.state.state import CnnState
from src.models.CNN_model.CNN_class import predict

# --- CNN Node ---
def CnnNode(state: dict):
    sample_input_folder = state["file_name"]
    
    # Resolve absolute path to the input folder and the model
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_path = os.path.join(project_root, "input", sample_input_folder)
    
    json_path = os.path.join(sample_path, "raw_bytes.json")
    model_path = os.path.join(project_root, "src", "models", "CNN_model", "best_model.pt")
    
    # Fallback in case raw_bytes.json doesn't exist
    if not os.path.exists(json_path):
        return {
            "cnn_state": CnnState(
                malware_confidence_score=0.0,
                benign_confidence_score=0.0,
                Xai_raw_bytes={}
            )
        }

    # Chạy prediction pipeline có sẵn trong CNN_class.py
    # Hàm predict trả về dict với 'confidence_scores' và 'top_k_opcodes'
    result = predict(
        json_path=json_path,
        model_path=model_path,
        top_k=3,
        window=10,
        output_path=os.path.join(sample_path, "xai_cnn_result.json") # Tùy chọn lưu file XAI
    )

    # Trích xuất confidence scores
    benign_score = result.get('confidence_scores', {}).get('benign', 0.0)
    malware_score = result.get('confidence_scores', {}).get('malware', 0.0)
    
    # Không lưu lại score vào Xai_raw_bytes, chỉ lấy top_k_opcodes
    xai_raw_bytes = {
        "top_k_opcodes": result.get('top_k_opcodes', [])
    }

    # Trả về CnnState (sẽ được tự động merge vào XaiDetectorState)
    return {
        "cnn_state": CnnState(
            malware_confidence_score=round(float(malware_score), 4),
            benign_confidence_score=round(float(benign_score), 4),
            Xai_raw_bytes=xai_raw_bytes,
        )
    }
