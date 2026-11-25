"""Bộ chuyển đổi giữa hợp đồng dữ liệu P0 và RecBole (KHÔNG cần cài RecBole để chạy/kiểm thử tệp này).

Định dạng tệp nguyên tử dạng "tăng cường" cho mô hình tuần tự với benchmark_filename
(header cột kiểu RecBole; kiểm lại trên RecBole 1.2.1 / RecBole-GNN trước khi chạy thật):
    session_id:token <TAB> item_id_list:token_seq <TAB> item_id:token
Quy ước phương án A (Q-strip): KHÔNG token sản phẩm mới nào xuất hiện trong bất kỳ tệp nào đưa vào RecBole,
vì RecBole gộp các tệp benchmark trước khi đánh lại id -> sản phẩm chỉ có ở valid/test vẫn vào từ vựng.
"""
import csv, os
import numpy as np
import pandas as pd

HEADER = ["session_id:token", "item_id_list:token_seq", "item_id:token"]


def train_lines(sessions_train, max_len=50):
    """Mỗi phiên train độ dài L sinh L−1 dòng (tiền tố, đích); tiền tố cắt bên trái còn tối đa max_len."""
    out = []
    for r in sessions_train.itertuples():
        seq = [int(x) for x in (r.item_seq if isinstance(r.item_seq, (list, tuple)) else str(r.item_seq).strip("[]").replace(",", " ").split())]
        for t in range(1, len(seq)):
            out.append((f"tr{int(r.session_idx)}_{t}", seq[max(0, t - max_len):t], seq[t]))
    return out


def eval_lines(inst, cold, fold, max_len=50):
    """Dòng valid/test: chỉ tình huống thuộc E_A (tiền tố sau Q-strip không rỗng) và đích KHÔNG phải sản phẩm mới
    (đích mới là N/A dưới phương án A). Trả thêm bảng ánh xạ dòng -> id tình huống gốc."""
    cold = set(int(c) for c in cold)
    out, rmap = [], []
    for r in inst.itertuples():
        pre = [int(x) for x in str(r.prefix_strip).split()] if isinstance(r.prefix_strip, str) else []
        if not pre or int(r.target) in cold:
            continue
        out.append((f"{fold}{int(r.row)}", pre[-max_len:], int(r.target)))
        rmap.append(dict(fold=fold, line=len(out) - 1, row=int(r.row)))
    return out, rmap


def write_inter(path, lines):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(HEADER)
        for sid, pre, tgt in lines:
            w.writerow([sid, " ".join(map(str, pre)), str(tgt)])


def export_benchmark(sessions_train, inst_valid, inst_test, cold, out_dir, name="p0a", max_len=50):
    os.makedirs(out_dir, exist_ok=True)
    tr = train_lines(sessions_train, max_len)
    va, mv = eval_lines(inst_valid, cold, "valid", max_len)
    te, mt = eval_lines(inst_test, cold, "test", max_len)
    paths = {}
    for fold, lines in (("train", tr), ("valid", va), ("test", te)):
        paths[fold] = os.path.join(out_dir, f"{name}.{fold}.inter")
        write_inter(paths[fold], lines)
    pd.DataFrame(mv + mt).to_csv(os.path.join(out_dir, f"{name}.rowmap.csv"), index=False)
    audit_vocab(paths.values(), cold)
    return dict(paths=paths, n_train=len(tr), n_valid=len(va), n_test=len(te))


def audit_vocab(paths, cold):
    """Kiểm từ vựng: |token trong tệp RecBole ∩ I_cold| phải = 0 (phương án A)."""
    cold = {str(int(c)) for c in cold}
    seen = set()
    for p in paths:
        df = pd.read_csv(p, sep="\t", dtype=str)
        seen |= set(df[HEADER[2]])
        for s in df[HEADER[1]]:
            seen |= set(str(s).split())
    bad = seen & cold
    if bad:
        raise AssertionError(f"RÒ RỈ TỪ VỰNG: token sản phẩm mới trong tệp RecBole: {sorted(bad)}")
    return len(seen)


def to_item_space(raw, id2token, n_items, cold, option="A", cold_scores=None):
    """Đưa điểm RecBole (n × V; cột = id nội bộ; id 0 = [PAD]) về không gian item_idx của P0 (n × (|I|+1)).
    option A: cột sản phẩm mới = −∞ (và chúng KHÔNG được có trong id2token).
    option B: cột sản phẩm mới lấy từ cold_scores (n × |I_cold|, theo thứ tự sorted(cold)) — điểm của hàng embedding
              CHƯA huấn luyện (định nghĩa B-untrained, chờ thầy xác nhận)."""
    raw = np.asarray(raw, float)
    if raw.shape[1] != len(id2token):
        raise ValueError("Số cột điểm khác số token của từ vựng — kiểm lại token [MASK] (BERT4Rec) hoặc id padding.")
    cold = sorted(int(c) for c in cold)
    toks = list(id2token[1:])
    if set(map(str, cold)) & set(map(str, toks)):
        raise AssertionError("Từ vựng mô hình chứa sản phẩm mới — không phải mô hình Q-strip/phương án A.")
    S = np.full((raw.shape[0], n_items + 1), -np.inf)
    cols = np.array([int(t) for t in toks])
    S[:, cols] = raw[:, 1:]
    if option == "B":
        if cold_scores is None:
            raise ValueError("Phương án B cần cold_scores.")
        S[:, cold] = np.asarray(cold_scores, float)
    elif option != "A":
        raise ValueError("Chỉ có phương án A hoặc B.")
    return S


def write_scores_long(path, S, rows, items=None):
    """Lưu điểm từng tình huống dạng dài (row, item_idx, score) — chỉ cột hữu hạn."""
    n, V = S.shape
    items = range(1, V) if items is None else items
    recs = [(int(rows[i]), int(j), float(S[i, j])) for i in range(n) for j in items if np.isfinite(S[i, j])]
    pd.DataFrame(recs, columns=["row", "item_idx", "score"]).to_csv(path, index=False)
