"""Sinh tệp điểm GIẢ ĐỊNH của một mô hình ID-based "M-ID" trên toy20 (đóng vai SR-GNN/BERT4Rec).
KHÔNG phải kết quả huấn luyện. Chỉ dùng để kiểm dây chuyền: phương án A/B, N/A, tổng hợp qua seed.
Hàng = id tình huống gốc trong instances_test.csv (0..5); hàng 3 ([D4] -> A2) không có vì tiền tố rỗng sau Q-strip (ngoài E_A).
Điểm của sản phẩm mới (A4, D4, N1, L3) = điểm của hàng embedding CHƯA huấn luyện (chỉ dùng ở phương án B)."""
import os
import pandas as pd

IDX = dict(A1=1, A2=2, A3=3, A4=4, D1=5, D2=6, D3=7, D4=8, N1=9, L1=10, L2=11, L3=12)
SEED2026 = {
    0: dict(D1=2.0, D4=2.1, D3=1.4, A2=1.1, D2=0.9, A1=0.6, A3=0.5, L3=0.3, A4=0.2, N1=-0.1, L1=-0.4, L2=-0.6),   # [D2] -> D1
    1: dict(D3=1.5, A2=1.3, A1=1.0, D1=0.8, A3=0.7, N1=0.6, D2=0.5, D4=0.4, L3=0.1, A4=-0.2, L1=-0.3, L2=-0.5),  # [D2 D1] -> D4
    2: dict(A2=1.6, D4=1.3, N1=1.25, A3=1.2, D3=0.9, A1=0.8, D1=0.3, D2=0.2, A4=0.1, L3=0.0, L1=-0.2, L2=-0.4), # [D2 D1] -> A3
    4: dict(A3=1.4, N1=1.2, A1=1.1, A2=0.9, D1=0.8, D4=0.7, D2=0.6, D3=0.4, A4=0.0, L1=-0.1, L3=-0.2, L2=-0.3), # [A2] -> A1
    5: dict(A3=1.3, D1=1.0, A2=0.9, D4=0.8, D2=0.7, A1=0.6, N1=0.5, D3=0.45, A4=0.3, L3=-0.1, L1=-0.2, L2=-0.4),# [A2 A1] -> N1
}
VARIANT = {2026: {}, 2027: {(2, "A3"): 1.7}, 2028: {(4, "A1"): 0.5}}   # hai "seed" khác chỉ đổi đúng một điểm

here = os.path.dirname(os.path.abspath(__file__))
for seed, change in VARIANT.items():
    recs = []
    for row, sc in SEED2026.items():
        for name, s in sc.items():
            recs.append((row, IDX[name], change.get((row, name), s)))
    pd.DataFrame(recs, columns=["row", "item_idx", "score"]).to_csv(os.path.join(here, f"mid_scores_seed{seed}.csv"), index=False)
print("written", list(VARIANT))
