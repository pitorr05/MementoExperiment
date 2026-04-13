# ============================================
# calculate_f1.py - Tính F1 Score từ kết quả benchmark
# ============================================

import json
import os
from typing import List, Set
import re

def normalize_answer(s: str) -> str:
    """Chuẩn hóa câu trả lời (theo cách của paper)"""
    if not isinstance(s, str):
        s = str(s)
    
    # Lowercase
    s = s.lower()
    # Loại bỏ punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Loại bỏ articles (a, an, the)
    s = re.sub(r'\b(a|an|the)\b', '', s)
    # Chuẩn hóa khoảng trắng
    s = ' '.join(s.split())
    return s

def calculate_f1(pred: str, ground_truths: List[str]) -> float:
    """
    Tính F1 score giữa pred_answer và ground_truth
    pred: câu trả lời dự đoán
    ground_truths: list các đáp án đúng (có thể nhiều)
    """
    pred_tokens = set(normalize_answer(pred).split())
    
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = set(normalize_answer(gt).split())
        
        if not pred_tokens or not gt_tokens:
            continue
            
        # Tính intersection
        common = pred_tokens & gt_tokens
        
        if not common:
            continue
            
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        best_f1 = max(best_f1, f1)
    
    return best_f1

def calculate_f1_from_file(result_path: str) -> dict:
    """Tính F1 từ file result"""
    
    total = 0
    f1_sum = 0.0
    exact_matches = 0
    partial_matches = 0
    
    results = []
    
    with open(result_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            total += 1
            
            pred = data.get('pred_answer', '')
            ground_truth = data.get('ground_truth', [])
            
            # Đảm bảo ground_truth là list
            if not isinstance(ground_truth, list):
                ground_truth = [str(ground_truth)]
            
            # Tính F1
            f1_score = calculate_f1(pred, ground_truth)
            f1_sum += f1_score
            
            # Phân loại
            if f1_score >= 0.99:
                exact_matches += 1
            elif f1_score > 0:
                partial_matches += 1
            
            results.append({
                'question': data.get('query'),
                'pred_answer': pred,
                'ground_truth': ground_truth,
                'f1_score': f1_score,
                'judgement': data.get('judgement')  # Giữ lại judgement cũ để so sánh
            })
    
    macro_f1 = (f1_sum / total * 100) if total > 0 else 0
    
    return {
        'total': total,
        'macro_f1': macro_f1,
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'incorrect': total - exact_matches - partial_matches,
        'results': results
    }

def print_report(stats: dict):
    """In báo cáo F1"""
    print("=" * 60)
    print("📊 DEEPRESEARCHER BENCHMARK REPORT (F1 SCORE)")
    print("=" * 60)
    print(f"📝 Tổng số câu: {stats['total']}")
    print(f"✅ Đúng hoàn toàn (F1=1.0): {stats['exact_matches']}")
    print(f"🟡 Đúng 1 phần (0<F1<1): {stats['partial_matches']}")
    print(f"❌ Sai hoàn toàn (F1=0): {stats['incorrect']}")
    print("-" * 60)
    print(f"📈 Macro F1 Score: {stats['macro_f1']:.2f}%")
    print("=" * 60)
    
    # So sánh với paper
    print("\n📊 SO SÁNH VỚI PAPER GỐC:")
    print("-" * 60)
    print(f"Paper gốc (GPT-4.1 + o3): 66.6% F1")
    print(f"Kết quả của bạn (Qwen3-30B): {stats['macro_f1']:.2f}% F1")
    print(f"Chênh lệch: {66.6 - stats['macro_f1']:.2f}%")
    print("=" * 60)

def save_report(stats: dict, output_path: str = 'f1_report.txt'):
    """Lưu báo cáo ra file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("DEEPRESEARCHER BENCHMARK REPORT (F1 SCORE)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total questions: {stats['total']}\n")
        f.write(f"Exact matches (F1=1.0): {stats['exact_matches']}\n")
        f.write(f"Partial matches (0<F1<1): {stats['partial_matches']}\n")
        f.write(f"Incorrect (F1=0): {stats['incorrect']}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Macro F1 Score: {stats['macro_f1']:.2f}%\n")
        f.write("=" * 60 + "\n")
        f.write(f"\nComparison with paper:\n")
        f.write(f"Paper (GPT-4.1 + o3): 66.6% F1\n")
        f.write(f"Your result (Qwen3-30B): {stats['macro_f1']:.2f}% F1\n")
        f.write(f"Difference: {66.6 - stats['macro_f1']:.2f}%\n")
    
    print(f"\n✅ Đã lưu báo cáo vào {output_path}")

def analyze_low_f1(stats: dict, threshold: float = 0.5, num_samples: int = 5):
    """Phân tích các câu có F1 thấp"""
    print(f"\n🔍 {num_samples} CÂU CÓ F1 THẤP NHẤT (F1 < {threshold}):")
    print("-" * 60)
    
    low_f1_results = [r for r in stats['results'] if r['f1_score'] < threshold]
    low_f1_results.sort(key=lambda x: x['f1_score'])
    
    for i, result in enumerate(low_f1_results[:num_samples]):
        print(f"\n📌 Câu {i+1}:")
        print(f"   Question: {result['question'][:100]}...")
        print(f"   Predicted: {result['pred_answer'][:100]}...")
        print(f"   Ground Truth: {result['ground_truth']}")
        print(f"   F1 Score: {result['f1_score']*100:.1f}%")

if __name__ == "__main__":
    # Đường dẫn đến file result
    result_path = r"C:\Users\Admin\Memento\result\result_round_0.jsonl"
    
    if os.path.exists(result_path):
        stats = calculate_f1_from_file(result_path)
        print_report(stats)
        save_report(stats)
        analyze_low_f1(stats, threshold=0.5, num_samples=5)
    else:
        print(f"❌ Không tìm thấy file: {result_path}")
        print("Hãy chạy benchmark trước!")