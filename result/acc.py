import json

# Đường dẫn đến file result
file_path = r"C:\Users\Admin\MementoExperiment\result\result_round_0.jsonl"

# Đếm số dòng và số câu đúng
total = 0
correct = 0

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            total += 1
            if data.get('reward') == 1:
                correct += 1

# Tính accuracy
if total > 0:
    accuracy = (correct / total) * 100
    print(f"Total queries: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("File rỗng hoặc không có dữ liệu")