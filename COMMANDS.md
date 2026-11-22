```markdown
# COMMANDS — Full Workflow Reference

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Place Dataset

```text
data/raw/02-14-2018.csv
```

## 3. Run Full Pipeline (Preprocess + Train All Models)

```bash
python main.py
python main.py --single data/raw/02-14-2018.csv --dataset-name 02-14-2018
python main.py --data-dir data/raw
```

## 4. Train Individual Models

```bash
python train/train_random_forest.py
python train/train_cs_forest.py
python train/train_decision_tree.py
python train/train_logistic_regression.py
python train/train_naive_bayes.py
python train/train_knn.py
python train/train_adaboost.py
python train/train_gradient_boosting.py
python train/train_all.py
```

## 5. Generate Sample Test Data

```bash
python prediction/sample_data.py
python prediction/sample_data.py --n-benign 50 --n-attack 50
python prediction/sample_data.py --dataset-name 02-14-2018 --output-dir data/samples
```

## 6. Run Prediction (All Models)

```bash
python main.py --predict --single data/samples/sample_mixed_traffic.csv
```

## 7. Run Prediction (Single Model)

```bash
python main.py --predict --model random_forest --single data/samples/sample_mixed_traffic.csv
python main.py --predict --model cs_forest --single data/samples/sample_mixed_traffic.csv
python main.py --predict --model decision_tree --single data/samples/sample_mixed_traffic.csv
python main.py --predict --model logistic_regression --single data/samples/sample_mixed_traffic.csv
python main.py --predict --model naive_bayes --single data/samples/sample_mixed_traffic.csv
python main.py --predict --model knn --single data/samples/sample_mixed_traffic.csv
python main.py --predict --model adaboost --single data/samples/sample_mixed_traffic.csv
python main.py --predict --model gradient_boosting --single data/samples/sample_mixed_traffic.csv
```

## 8. Run Prediction Scripts Directly

```bash
python prediction/predict_random_forest.py --input data/samples/sample_mixed_traffic.csv
python prediction/predict_cs_forest.py --input data/samples/sample_mixed_traffic.csv
python prediction/predict_decision_tree.py --input data/samples/sample_mixed_traffic.csv
python prediction/predict_logistic_regression.py --input data/samples/sample_mixed_traffic.csv
python prediction/predict_naive_bayes.py --input data/samples/sample_mixed_traffic.csv
python prediction/predict_knn.py --input data/samples/sample_mixed_traffic.csv
python prediction/predict_adaboost.py --input data/samples/sample_mixed_traffic.csv
python prediction/predict_gradient_boosting.py --input data/samples/sample_mixed_traffic.csv
python prediction/predict_all.py --input data/samples/sample_mixed_traffic.csv
```

## 9. Quick KNN Test (Single Record)

```bash
python run_knn_sample.py
```

## Output Locations

| Output Type | Path |
|---|---|
| Saved Models | saved_models/02-14-2018/ |
| Processed Data | data/processed/ |
| Sample Files | data/samples/ |
| Prediction Results | results/predictions/ |
```