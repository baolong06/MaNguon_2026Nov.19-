"""GĐ3 — Bao quanh mô hình ID-based (CĐ5 ★). KHÔNG sửa mô hình: chỉ nhận ma trận điểm và áp quy ước Option A/B."""
import numpy as np

def apply_option(S, cold_idx, option="A"):
    """S: điểm thô của mô hình (n_inst × (|I|+1)). Option A: item cold không có trong không gian ứng viên (-inf).
    Option B: giữ nguyên điểm mà mô hình gán cho item cold (CHỈ dùng cho phụ lục; định nghĩa chính xác: Q-05-6)."""
    S = S.copy(); S[:, 0] = -np.inf
    if option == "A":
        S[:, list(cold_idx)] = -np.inf
    elif option != "B":
        raise ValueError("Chỉ có Option A hoặc B. Option C (khởi tạo theo category) bị cấm trong P0-Base.")
    return S

def na_mask(inst, cold_idx, option="A"):
    """Tình huống có đích cold -> N/A ở Option A."""
    return inst["target"].isin(set(cold_idx)).values if option == "A" else np.zeros(len(inst), bool)

def strip_cold_from_prefix(prefix, cold_idx):
    """Quy ước ĐỀ XUẤT (chờ chốt) cho item cold nằm TRONG tiền tố khi đưa vào mô hình ID-based ở Option A."""
    toks = [t for t in str(prefix).split() if int(t) not in set(cold_idx)]
    return " ".join(toks)
