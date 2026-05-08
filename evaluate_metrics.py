import os
import json
import csv
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.stdout.reconfigure(encoding='utf-8')

def evaluate_metrics(csv_path, output_dir):
    # Đọc nhãn thực tế từ file test.csv
    # Key: hash, Value: 1 (Malware) hoặc 0 (Benign)
    ground_truth = {}
    
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file {csv_path}")
        return

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hash_val = row['hash'].strip()
            signature = row['signature'].strip().lower()
            ground_truth[hash_val] = 0 if signature == 'benign' else 1

    # Khởi tạo list lưu kết quả của từng Phase
    y_true_quant, y_pred_quant = [], []
    y_true_comp, y_pred_comp = [], []
    
    missing_samples = []

    for hash_val, true_label in ground_truth.items():
        sample_dir = os.path.join(output_dir, hash_val)
        quant_file = os.path.join(sample_dir, "Quantitative_reasoning_result.json")
        comp_file = os.path.join(sample_dir, "Comprehensive_decision_result.json")

        # Kiểm tra xem mẫu đã được phân tích thành công chưa (có thư mục và có đủ 2 file result không)
        if not os.path.isdir(sample_dir) or not os.path.exists(quant_file) or not os.path.exists(comp_file):
            missing_samples.append(hash_val)
            continue
        
        # ---------------------------------------------------------
        # Phase 1: Quantitative reasoning
        # ---------------------------------------------------------
        try:
            with open(quant_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                pred = data.get("predicted_class", "").strip().lower()
                pred_label = 1 if pred == "malware" else 0
                
                y_true_quant.append(true_label)
                y_pred_quant.append(pred_label)
        except Exception:
            pass # Bỏ qua nếu file JSON bị lỗi cấu trúc

        # ---------------------------------------------------------
        # Phase 2: Comprehensive decision
        # ---------------------------------------------------------
        try:
            with open(comp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                pred = data.get("predicted_class", "").strip().lower()
                pred_label = 1 if pred == "malware" else 0
                
                y_true_comp.append(true_label)
                y_pred_comp.append(pred_label)
        except Exception:
            pass

    # Hàm hỗ trợ in kết quả
    def print_metrics(phase_name, y_true, y_pred):
        if not y_true:
            print(f"\n--- {phase_name} ---")
            print("Không tìm thấy dữ liệu hợp lệ để tính toán.")
            return

        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\n--- {phase_name} ---")
        print(f"Tổng số mẫu được đánh giá : {len(y_true)}")
        print(f"Accuracy (Độ chính xác)   : {acc:.4f}")
        print(f"Precision (Độ chuẩn xác)  : {pre:.4f}")
        print(f"Recall (Độ phủ)           : {rec:.4f}")
        print(f"F1 Score                  : {f1:.4f}")

    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (Malware = Positive (1), Benign = Negative (0))")
    print("="*70)
    print_metrics("Phase 1: Quantitative Reasoning", y_true_quant, y_pred_quant)
    print_metrics("Phase 2: Comprehensive Decision", y_true_comp, y_pred_comp)
    print("="*70)
    
    # In ra danh sách các mẫu bị thiếu
    if missing_samples:
        print(f"\n⚠️ CẢNH BÁO: Có {len(missing_samples)} mẫu trong test.csv bị THIẾU (chưa chạy hoặc chạy lỗi không sinh ra đủ file json):")
        for m in missing_samples:
            print(f" - {m}")
    else:
        print("\n✅ Tuyệt vời: Không có mẫu nào bị thiếu. Đã tính toán đủ tất cả các mẫu trong test.csv!")

if __name__ == "__main__":
    csv_file = r"d:\NCKH\Xai-detector\test.csv"
    output_folder = r"d:\NCKH\Xai-detector\output"
    evaluate_metrics(csv_file, output_folder)
