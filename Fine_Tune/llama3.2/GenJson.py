import json
import os

# 設定資料夾路徑
prompt_dir = r"C:\Users\Lab 244\Documents\GitHub\chestCT\splited_reports_prompt"
report_dir = r"C:\Users\Lab 244\Documents\GitHub\chestCT\splited_reports"

# 確保資料夾存在
if not os.path.exists(prompt_dir) or not os.path.exists(report_dir):
    raise FileNotFoundError("請確保 'splited_reports_prompt' 和 'splited_reports' 資料夾存在")

# 讀取所有 prompt 檔案
prompt_files = {f: os.path.join(prompt_dir, f) for f in os.listdir(prompt_dir) if f.endswith(".txt")}
report_files = {f: os.path.join(report_dir, f) for f in os.listdir(report_dir) if f.endswith(".txt")}

# 確保 prompt 和報告配對
common_files = set(prompt_files.keys()) & set(report_files.keys())
if not common_files:
    raise ValueError("沒有找到相同檔名的 prompt 和報告，請檢查資料夾內容")

# 生成訓練數據
train_data = []
test_data = []
file_list = sorted(common_files)  # 排序確保一致性
split_ratio = 0.8  # 訓練集: 測試集 = 80:20
split_index = int(len(file_list) * split_ratio)

for idx, filename in enumerate(file_list):
    with open(prompt_files[filename], "r", encoding="utf-8") as pf, open(report_files[filename], "r", encoding="utf-8") as rf:
        user_prompt = pf.read().strip()
        assistant_report = rf.read().strip()

        # 明確區分 prompt 和 response
        entry = {
            "conversations": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_report}
            ]
        }

        if idx < split_index:
            train_data.append(entry)
        else:
            test_data.append(entry)

# 儲存 JSON 檔案
with open("train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open("test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print(f"生成 {len(train_data)} 筆訓練數據，{len(test_data)} 筆測試數據，已儲存至 train.json 和 test.json")