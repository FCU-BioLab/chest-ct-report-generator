當然可以👇
以下是完整 `.md` 內容，直接複製貼上即可（例如存成 `docs/custom_module_patch.md`）：

---

# 🧠 YOLOv11 自訂模組 `KeyError` 修正紀錄

## 📌 問題背景

在使用 Ultralytics YOLOv11（例如 8.3.19x 版本）搭配 **自訂模型 YAML**（例如 `yolo11_custom_ct_s_optimize.yaml`）時，若 YAML 內含有自訂模組名稱（例如 `AAF_CT`、`RRBBlock`、`SATModule`），執行訓練會出現以下錯誤：

```
KeyError: 'AAF_CT'
```

這是因為 Ultralytics 的 `parse_model()` 在解析 YAML 各層時，會從 `globals()` 取得對應的類別：

```python
m = eval(m) if isinstance(m, str) else m
```

但 **預設的 `globals()` 中沒有註冊自訂模組**，導致無法識別這些層的名稱。

---

## 🧰 解決方式：修改 `parse_model()` 註冊自訂模組

1. 開啟 `ultralytics/nn/tasks.py`

2. 找到函式：

   ```python
   def parse_model(d, ch, verbose=True):
   ```

3. 在這個函式的 **開頭插入以下程式碼** ⬇️

   ```python
   # ✅ 自訂模組註冊區（避免 KeyError）
   try:
       from ultralytics.nn.modules.custom_blocks import AAF_CT, RRBBlock, SATModule
       globals().update({
           'AAF_CT': AAF_CT,
           'RRBBlock': RRBBlock,
           'SATModule': SATModule
       })
   except ImportError as e:
       print(f"[parse_model] Warning: custom module import failed → {e}")
   ```

4. 儲存後重新執行訓練即可。

---

## ⚠️ 為什麼要記錄這件事？

Ultralytics 在升級套件（例如 `pip install -U ultralytics`）時，**`ultralytics/` 目錄會被覆蓋**，導致你上面修改的程式碼消失。
未來一旦升級，訓練含有自訂模組的 YAML 會再次爆出 `KeyError`。

---

## 📝 更新版本後的檢查步驟

每次執行：

```bash
pip install -U ultralytics
```

請務必執行以下步驟：

1. 開啟 `ultralytics/nn/tasks.py`
2. 檢查 `parse_model()` 開頭是否仍有 **自訂模組註冊區**
3. 如果沒有 → 重新貼上上方的註冊程式碼
4. （建議）用 `grep` / `ripgrep` 搜尋是否有 `"AAF_CT"`：

   ```bash
   rg "AAF_CT" ultralytics/nn/tasks.py
   ```

---

## ✅ 延伸建議

* 可以把這段註冊程式碼包成一個 patch script，例如 `scripts/patch_ultralytics.py`，內容只要檢查並插入該段文字，升級後執行一次即可。
* 或者考慮 fork Ultralytics repo 並維護自己的 branch，避免每次手動 patch。

---

## 📅 最後更新時間

* 2025-10-17
* 適用 Ultralytics 版本：8.3.194 ~ 8.3.216（含 YOLOv11）

---

✅ 建議把這份文件放在專案的 `docs/` 或 `README` 裡，升級時就不會忘記要 patch 了。
