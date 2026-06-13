import matplotlib.pyplot as plt

def draw_formulas():
    # Khởi tạo khung hình
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Các chuỗi LaTeX cho công thức
    # Lưu ý: Sử dụng r'' để Python hiểu đây là raw string, không bị lỗi backslash
    bm25_latex = r"$BM25(q,d) = \sum_{t \in q} \left[ IDF(t) \times \frac{TF(t,d) \times (k_1 + 1)}{TF(t,d) + k_1 \times (1 - b + b \times \frac{|d|}{avgdl})} \right]$"
    sim_latex = r"$Sim(q,d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\| \times \|\mathbf{v}_d\|}$"
    final_latex = r"$Final(q,d) = w_{bm25} \times BM25(q,d) + w_{sem} \times Sim(q,d)$"
    params_latex = r"$\text{Với: } w_{bm25} = 0.5, \ w_{sem} = 0.5$"

    # Hiển thị các công thức lên trục tọa độ
    # (x, y) là vị trí, fontsize là kích thước chữ
    ax.text(0.5, 0.8, bm25_latex, fontsize=18, ha='center', va='center')
    ax.text(0.5, 0.55, sim_latex, fontsize=20, ha='center', va='center')
    ax.text(0.5, 0.35, final_latex, fontsize=18, ha='center', va='center')
    ax.text(0.5, 0.15, params_latex, fontsize=16, ha='center', va='center', color='darkblue')

    # Ẩn các trục tọa độ xung quanh
    ax.axis('off')

    # Tùy chỉnh tiêu đề (nếu cần)
    plt.title("Hệ Thống Xếp Hạng Kết Hợp (Hybrid Scoring)", fontsize=14, fontweight='bold', pad=20)

    # Lưu file hoặc hiển thị
    plt.tight_layout()
    plt.savefig('formula_bm25_sim.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_formulas()