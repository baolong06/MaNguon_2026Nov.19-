"""Bộ chuyển đổi hợp đồng dữ liệu → tệp nguyên tử (atomic files) kiểu phiên của RecBole cho BA điều kiện.
Mỗi điều kiện một thư mục <out>/<dataset>_<cond>/ gồm <dataset>_<cond>.{train,valid,test}.inter, cột:
    session_id:token    item_id_list:token_seq    item_id:token
(định dạng 'benchmark' đã tách sẵn của bộ dữ liệu phiên trong RecBole/RecBole-GNN — KIỂM TRÊN RecBole 1.2.1 trước khi chạy thật).
Token = id GỐC của item (item_raw) để RecBole tự ánh xạ; ánh xạ ngược lấy từ dataset.field2id_token['item_id'].
  strip: train = train P0; valid/test = tình huống đích không cold, tiền tố đã xóa item cold; tiền tố rỗng → bỏ (ghi số lượng)
  keep : train = train P0; valid/test = tình huống đích không cold, tiền tố giữ nguyên (mã cold vào từ vựng, chưa từng là đích)
  ref  : train = train tham chiếu ρ = 0; valid/test như keep
Tình huống đích cold KHÔNG vào tệp nào (thuộc bài A)."""
import ast, json, os, sys
import pandas as pd


def _seqs(udir):
    st = pd.read_csv(os.path.join(udir, "sessions_train.csv"))
    return list(zip(st["session_idx"], [list(ast.literal_eval(s)) for s in st["item_seq"]]))


def _augment(seqs):
    rows = []
    for sid, s in seqs:
        for t in range(1, len(s)):
            rows.append((sid, s[:t], s[t]))
    return rows


def build(udir_p0, udir_ref, out_root, name):
    im = pd.read_csv(os.path.join(udir_p0, "item_map.csv"))
    tok = dict(zip(im["item_idx"], im["item_raw"].astype(str)))
    cold = set(pd.read_csv(os.path.join(udir_p0, "cold_items.csv"))["item_idx"])
    inst = {f: pd.read_csv(os.path.join(udir_p0, f"instances_{f}.csv")) for f in ("valid", "test")}
    report = {}
    for cond in ("strip", "keep", "ref"):
        d = os.path.join(out_root, f"{name}_{cond}")
        os.makedirs(d, exist_ok=True)
        train = _augment(_seqs(udir_ref if cond == "ref" else udir_p0))
        files = {"train": train}
        dropped = {}
        for f in ("valid", "test"):
            rows, n_drop = [], 0
            for r in inst[f].itertuples(index=False):
                tgt = int(r.target)
                if tgt in cold:
                    continue                                    # đích cold: không thuộc bài B
                p = [int(x) for x in str(r.prefix).split()]
                if cond == "strip":
                    p = [x for x in p if x not in cold]
                    if not p:
                        n_drop += 1                             # K3 dưới Q-strip: không chấm được → N/A
                        continue
                rows.append((int(r.session_idx), p, tgt))
            files[f] = rows
            dropped[f] = n_drop
        vocab = set()
        for f, rows in files.items():
            with open(os.path.join(d, f"{name}_{cond}.{f}.inter"), "w", encoding="utf-8", newline="\n") as fh:
                fh.write("session_id:token\titem_id_list:token_seq\titem_id:token\n")
                for sid, p, t in rows:
                    fh.write(f"{sid}\t{' '.join(tok[x] for x in p)}\t{tok[t]}\n")
                    vocab.update(p); vocab.add(t)
        targets = {t for rows in files.values() for _, _, t in rows}
        report[cond] = dict(rows={f: len(r) for f, r in files.items()}, dropped_empty_prefix=dropped,
                            vocab_size=len(vocab), vocab_cold=sorted(int(x) for x in vocab & cold),
                            cold_as_target_eval=sorted(int(x) for x in targets & cold
                                                       if any(t == x for f in ("valid", "test") for _, _, t in files[f])))
    json.dump(report, open(os.path.join(out_root, f"{name}_atomic_report.json"), "w"), indent=1)
    return report


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/toy20"
    out = sys.argv[2] if len(sys.argv) > 2 else "runs/atomic"
    print(json.dumps(build(os.path.join(root, "unified"), os.path.join(root, "unified_rho0"), out, "toy20"), indent=1))
