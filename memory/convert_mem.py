# convert_memory.py
import json
import os

def convert_memory_format(input_path, output_path=None):
    """
    Chuyển đổi memory.jsonl từ format cũ sang format mới
    
    Format cũ: {"case": "...", "plan": "...", "case_label": "positive/negative"}
    Format mới: {"question": "...", "plan": "...", "reward": 1/0}
    """
    
    if output_path is None:
        output_path = input_path.replace('.jsonl', '_converted.jsonl')
    
    converted = []
    bad_lines = []
    
    print(f"📂 Đang đọc file: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # Lấy câu hỏi: ưu tiên 'question', nếu không có thì lấy 'case'
                question = data.get('question') or data.get('case', '')
                
                # Lấy plan
                plan = data.get('plan', '')
                
                # Lấy reward: ưu tiên 'reward', nếu không có thì từ 'case_label'
                reward = data.get('reward')
                if reward is None:
                    case_label = data.get('case_label', '')
                    reward = 1 if case_label == 'positive' else 0
                
                # Tạo object mới theo format chuẩn
                new_data = {
                    'question': question,
                    'plan': plan,
                    'reward': reward
                }
                
                converted.append(json.dumps(new_data, ensure_ascii=False))
                
            except json.JSONDecodeError as e:
                bad_lines.append(line_num)
                print(f"⚠️  Dòng {line_num}: Lỗi JSON - {e}")
            except Exception as e:
                bad_lines.append(line_num)
                print(f"⚠️  Dòng {line_num}: Lỗi - {e}")
    
    # Ghi file mới
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(converted))
    
    print(f"\n{'='*50}")
    print(f"✅ Chuyển đổi thành công!")
    print(f"{'='*50}")
    print(f"📊 Số dòng đã chuyển: {len(converted)}")
    print(f"❌ Số dòng lỗi: {len(bad_lines)}")
    if bad_lines:
        print(f"   Các dòng lỗi: {bad_lines[:20]}")
    print(f"📁 File đầu ra: {output_path}")
    print(f"{'='*50}")
    
    # Hiển thị 3 dòng đầu để kiểm tra
    if converted:
        print("\n📋 3 dòng đầu tiên (đã convert):")
        for i, line in enumerate(converted[:3], 1):
            # In rút gọn để dễ nhìn
            preview = line[:150] + "..." if len(line) > 150 else line
            print(f"   {i}. {preview}")

def main():
    # Đường dẫn file memory.jsonl
    input_file = r"C:\Users\Admin\Memento\memory\memory.jsonl"
    output_file = r"C:\Users\Admin\Memento\memory\memory_converted.jsonl"
    
    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file: {input_file}")
        return
    
    # Chuyển đổi
    convert_memory_format(input_file, output_file)
    
    # Hỏi người dùng có muốn thay thế file gốc không
    print("\n" + "="*50)
    answer = input("🔄 Có muốn thay thế file memory.jsonl bằng file đã convert? (y/n): ")
    
    if answer.lower() == 'y':
        # Backup file gốc
        backup_file = input_file.replace('.jsonl', '_backup.jsonl')
        os.rename(input_file, backup_file)
        os.rename(output_file, input_file)
        print(f"✅ Đã thay thế! File gốc được backup tại: {backup_file}")
        print(f"✅ File memory.jsonl đã được cập nhật format mới!")
    else:
        print(f"✅ File convert được lưu tại: {output_file}")
        print(f"📁 File gốc vẫn giữ nguyên: {input_file}")

if __name__ == "__main__":
    main()