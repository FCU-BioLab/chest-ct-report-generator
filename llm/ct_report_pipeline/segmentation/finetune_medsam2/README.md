# MedSAM2 微調（ct_report_pipeline 子模組）

此 README 說明 `llm/ct_report_pipeline/segmentation/finetune_medsam2` 的常用訓練與測試指令。

## 安裝

```bash
pip install -r requirements.txt
```

## 訓練

```bash
python main.py
python main.py --epochs 50 --batch_size 4 --lr 1e-5 --accumulation_steps 2
```

## 續訓

```bash
python main.py --resume result/segmentation_TIMESTAMP/checkpoint_epoch_10.pth
```

## 僅評估

```bash
python main.py --eval_only --checkpoint result/segmentation_TIMESTAMP/best_model.pth
```

## 測試與特徵輸出

```bash
python main.py --test --resume result/segmentation_TIMESTAMP/best_model.pth
python main.py --test --resume result/segmentation_TIMESTAMP/best_model.pth --extract_features
python main.py --test --resume result/segmentation_TIMESTAMP/best_model.pth --feature_output_dir ./llm_features
```

## 備註

- 詳細參數請使用 `python main.py --help`。
- 建議將測試輸出的特徵檔與報告資料分開管理，避免覆寫。
