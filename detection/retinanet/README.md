# MONAI RetinaNet ?箇?蝭?菜葫璅∠?

?祉???怠??MONAI 3D RetinaNet ?蝯??菜葫摰閫?捱?寞???
?舀 LNDb ??LUNA16 鞈?????敺?????蝺氬隢閬死??摰瘚???

## ?桅?蝯?

- `retinanet/`
  - `config.py`: 璅∪???蝺游??貉身摰?(Dataclass)
  - `dataset.py`: 鞈???蝢?(Dataset) ????頛?
  - `trainer.py`: 閮毀?詨??摩 (Trainer)
  - `main.py`: ?賭誘???(CLI)嚗??train/test/predict
  - `prepare_data.py`: 鞈?皞??單 (LNDb/LUNA16 -> JSON)
  - `inference.py`: ?典??刻??單 (DICOM/MHD -> ?勗?)
  - `evaluate.py`: 蝯?閰摯?單 (閮? IoU/F1)
  - `visualize.py`: 蝯?閬死???(GIF)

## 摰??瘙?

隢Ⅱ靽歇摰? MONAI ???隞塚?
```bash
pip install "monai[all]>=1.3" pandas simpleitk imageio matplotlib
```

## 雿輻??

### 1. 鞈?皞?
????鞈??蒂?? `detection/manifests/*.json`??

**LNDb 蝭?:**
```bash
python -m detection.retinanet.prepare_data \
  --dataset lndb \
  --base_dir "cache/LNDb" \
  --output "detection/manifests/dataset_lndb.json"
```

**LUNA16 蝭?:**
```bash
python -m detection.retinanet.prepare_data \
  --dataset luna16 \
  --base_dir "cache/LUNA16" \
  --output "detection/manifests/dataset_luna16.json"
```

**LUNA16-New (NBIA DICOM/XML) 範例:**
```bash
python -m detection.retinanet.prepare_luna16_new \
  --base_dir "E:\\lung_ct_lesion_dataset\\LUNA16-New" \
  --output_json "detection/manifests/dataset_luna16_new.json"
```

### 2. 閮毀璅∪? (Training)
雿輻 `main.py train` ?誘?????望郊撽?1 ????JSON 瑼? (?舀?? MHD/NIfTI 鞈?)??

**雿輻 JSON 鞈??”:**
```bash
python -m detection.retinanet.main train \
  --data_path "detection/manifests/dataset_lndb.json" \
  --epochs 300 \
  --output_dir "results/experiment_1"
```



### 3. ?刻???皜?

**?格?敹恍?皜?(?冽?):**
```bash
python -m detection.retinanet.main predict \
  --checkpoint "results/experiment_1/model_best.pt" \
  --input "cache/LNDb/data0/LNDb-0001.mhd"
```

**?典??刻? (摰蝞∠?嚗????:**
頛詨?舐 DICOM 鞈?憭暹? `.nii.gz`/`.mhd` 瑼???
```bash
python -m detection.retinanet.inference \
  --input_path "data/patient_01" \
  --model_path "results/experiment_1/model_best.pt" \
  --output_dir "results/patient_01"
```

### 4. 閰摯??閬箏?

**閰摯?? (IoU, Recall):**
```bash
python -m detection.retinanet.evaluate \
  --report_path "results/patient_01/report.json" \
  --dataset lndb
```

**??閬死??GIF:**
```bash
python -m detection.retinanet.visualize \
  --report_path "results/patient_01/report.json"
```

## 瘜冽?鈭?
- ?舀?湔雿輻 JSON 鞈??”霈??憪蔣??(MHD/NIfTI) ?脰?閮毀??
- ?身閮剖??? 3-30mm ??蝯??雿喳? (Anchor Sizes)??
