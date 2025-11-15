import glob

def check_label_file(file):
    with open(file) as f:
        for lineno, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[格式錯誤] {file}:{lineno} -> {line.strip()}")
                continue
            try:
                cls, xc, yc, w, h = map(float, parts)
            except ValueError:
                print(f"[數值轉換錯誤] {file}:{lineno} -> {line.strip()}")
                continue
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                print(f"[座標異常] {file}:{lineno} -> {line.strip()}")

label_files = glob.glob(f'E:\\GitHub\\chest-ct-report-generator\\detection\\yolo_detection\\yolo_runs\\train_20251017_120224\\dataset_20251017_120224\\labels\\val\\*.txt', recursive=True)
for file in label_files:
    check_label_file(file)
