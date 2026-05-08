import os
import json
from main_graph import graph

def load_json_list(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_json_list(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def run_all_inputs_directly():
    # Sử dụng đường dẫn tương đối dựa trên vị trí của file script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_dir = os.path.join(base_dir, "input")
    processed_file = os.path.join(base_dir, "processed_samples.json")
    error_file = os.path.join(base_dir, "error_samples.json")
    
    if not os.path.exists(input_dir):
        print(f"Thư mục input không tồn tại: {input_dir}")
        return

    # Lấy danh sách thư mục con trong input
    all_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    if not all_folders:
        print("Không tìm thấy thư mục input nào.")
        return

    # Tải danh sách đã xử lý (để bỏ qua) và lỗi (để thêm mới nếu có)
    processed_samples = load_json_list(processed_file)
    error_samples = load_json_list(error_file)
    
    # Lọc ra các mẫu chưa xử lý
    folders = [f for f in all_folders if f not in processed_samples]

    if not folders:
        print(f"Tất cả {len(all_folders)} mẫu đã được xử lý xong trước đó.")
        return

    print(f"Đã tìm thấy {len(all_folders)} mẫu tổng cộng.")
    print(f"Có {len(processed_samples)} mẫu đã hoàn thành. Sẽ chạy {len(folders)} mẫu còn lại.")
    print("💡 Mẹo: Nhấn Ctrl+C để HỦY mẫu hiện tại nếu chạy quá lâu.")
    
    for folder_name in folders:
        print(f"\n{'='*50}")
        print(f"Đang gọi trực tiếp Graph cho: {folder_name} (Sẽ đợi đến khi hoàn thành...)")
        print(f"{'='*50}")
        
        success = False
        
        try:
            # GỌI TRỰC TIẾP GRAPH THAY VÌ QUA API
            result_data = graph.invoke({"file_name": folder_name})
            
            # Hàm hỗ trợ convert các object nội bộ (như GinState của Pydantic) sang dạng JSON
            def default_serializer(obj):
                if hasattr(obj, 'model_dump'):
                    return obj.model_dump()
                elif hasattr(obj, 'dict'):
                    return obj.dict()
                return str(obj)
            
            # convert sang JSON string để in ra màn hình
            print(json.dumps(result_data, indent=4, ensure_ascii=False, default=default_serializer))
            
            print(f"✅ Đã phân tích xong {folder_name}!\n")
            success = True
            
        except KeyboardInterrupt:
            print(f"\n⚠️ BỎ QUA: Đã nhận tín hiệu huỷ (Ctrl+C) khi đang xử lý '{folder_name}'.")
            try:
                choice = input("Bạn muốn (s)kip bỏ qua mẫu này hay (q)uit thoát toàn bộ chương trình? [s/q]: ").strip().lower()
                if choice == 'q':
                    print("Đang thoát chương trình...")
                    error_samples.append({
                        "folder_name": folder_name,
                        "reason": "Cancelled by user (Program exited)"
                    })
                    save_json_list(error_file, error_samples)
                    break
            except KeyboardInterrupt:
                print("\nThoát chương trình...")
                break
                
            print(f"⏩ Đã bỏ qua mẫu '{folder_name}' và tiếp tục...")
            error_samples.append({
                "folder_name": folder_name,
                "reason": "Cancelled by user (Skipped)"
            })
            save_json_list(error_file, error_samples)
            
        except Exception as e:
            print(f"\n❌ LỖI: Không thể xử lý mẫu '{folder_name}'. Lỗi: {e}")
            error_samples.append({
                "folder_name": folder_name,
                "reason": str(e)
            })
            save_json_list(error_file, error_samples)
        finally:
            # Lưu lại trạng thái xử lý thành công ngay lập tức
            if success:
                processed_samples.append(folder_name)
                save_json_list(processed_file, processed_samples)

    print(f"\n{'='*50}")
    print("HOÀN THÀNH TẤT CẢ CÁC MẪU!")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_all_inputs_directly()
