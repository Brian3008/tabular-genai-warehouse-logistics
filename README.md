# Tabular Gen-AI for Warehouse Logistics

Masters dissertation project — University of Surrey (EEEM004)
In collaboration with Locus Robotics
Supervised by Dr. Simon Hadfield
Student: Nyi Nyi Myo Zin (URN 6955918)

## What this project does

Trains a CTGAN generative AI model on real e-commerce
order data to produce realistic synthetic warehouse orders.
Validates that robots behave identically on synthetic
orders as on real orders, and shows that different
seasonal demand patterns require different fleet
management strategies.

## Key results

- Distributional quality:    93.47%
- Correlation preservation:  97.6%
- ML efficacy:               100.3%
- Privacy (DCR ratio):       1.16

## Pipeline

1. prepare_data_v2.py            — data cleaning and feature engineering
2. train_best_model.py           — CTGAN model training
3. create_eval_set.py            — fixed stratified evaluation set
4. run_project.py                — master pipeline
5. ml_efficacy.py                — ML efficacy evaluation
6. correlation_analysis.py       — correlation preservation
7. privacy_evaluation.py         — privacy check
8. model_comparison.py           — model comparison
9. generate_seasonal.py          — seasonal scenario generation
10. conditional_analysis_v2.py   — conditioning effectiveness test
11. fleet_workload_analysis.py   — demand concentration analysis
12. fleet_strategy_comparison.py — fleet strategy comparison
13. smart_simulator.py           — visual warehouse simulator
14. check_model.py               — verify trained model works

## Data

Instacart Market Basket Analysis
https://www.kaggle.com/c/instacart-market-basket-analysis
Download and place CSV files in data/ folder.

## Requirements

Python 3.10, SDV, PyTorch, Pygame,
Scikit-learn, Pandas, NumPy, Matplotlib