import os
import json
import uuid
import http.client

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

def run_all_inputs_via_api():
    input_dir = r"d:\NCKH\Xai-detector\input"
    processed_file = r"d:\NCKH\Xai-detector\processed_samples.json"
    error_file = r"d:\NCKH\Xai-detector\error_samples.json"
    
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
    
    headers = {
        "Content-Type": "application/json"
    }

    for folder_name in folders:
        print(f"\n{'='*50}")
        print(f"Đang gọi LangGraph API cho: {folder_name} (Sẽ đợi đến khi hoàn thành...)")
        print(f"{'='*50}")
        
        # Để đảm bảo không bị timeout khi kết nối chờ lâu, tăng thời gian timeout
        conn = http.client.HTTPConnection("127.0.0.1:2024", timeout=600)
        
        success = False
        
        try:
            # 1. Tạo một Thread ID mới bằng uuid để lưu lịch sử độc lập cho mỗi input
            thread_id = str(uuid.uuid4())
            thread_payload = json.dumps({"thread_id": thread_id})
            
            conn.request("POST", "/threads", body=thread_payload, headers=headers)
            res = conn.getresponse()
            res.read() # Bắt buộc phải đọc để tiêu thụ luồng trả về
            
            print(f"Đã tạo Thread ID: {thread_id}")

            # 2. Gửi request Run Graph vào Thread vừa tạo và SỬ DỤNG ENDPOINT /wait ĐỂ CHỜ KẾT QUẢ
            run_payload = json.dumps({
                "assistant_id": "XaiDectector", # Trỏ đúng tên Graph đã định nghĩa trong langgraph.json
                "input": {
                    "file_name": folder_name
                }
            })

            # Sử dụng endpoint /runs/wait để block connection cho đến khi phân tích xong
            conn.request(
                "POST",
                f"/threads/{thread_id}/runs/wait",
                body=run_payload,
                headers=headers,
            )

            response = conn.getresponse()
            print(f"API Response Code: {response.status}")
            
            # Đọc dữ liệu trả về (Kết quả cuối cùng của Graph)
            result_data = response.read().decode()
            
            try:
                # In ra kết quả JSON một cách đẹp mắt
                parsed_json = json.loads(result_data)
                print(json.dumps(parsed_json, indent=4, ensure_ascii=False))
            except json.JSONDecodeError:
                # Nếu không phải JSON hợp lệ thì in text bình thường
                print(result_data)
            
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
            conn.close()
            
            # Lưu lại trạng thái xử lý thành công ngay lập tức để không bị mất nếu có sự cố
            if success:
                processed_samples.append(folder_name)
                save_json_list(processed_file, processed_samples)

    print(f"\n{'='*50}")
    print("HOÀN THÀNH TẤT CẢ REQUEST TỚI API!")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_all_inputs_via_api()
