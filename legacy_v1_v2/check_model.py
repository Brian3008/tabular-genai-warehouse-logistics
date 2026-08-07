from sdv.single_table import CTGANSynthesizer
import pandas as pd

print("Loading best model...")
model = CTGANSynthesizer.load(
    'data/ctgan_final.pkl')
print("Model loaded successfully!")
print(f"Model type: {type(model).__name__}")

print("\nGenerating 10 sample orders...")
sample = model.sample(num_rows=10)
print("\nSample synthetic orders:")
print(sample[[
    'aisle_id',
    'order_dow',
    'order_hour_of_day',
    'is_weekend',
    'is_peak_hour',
    'is_reorder'
]].to_string())

print("\nModel is working correctly!")
print("ctgan_final.pkl is your trained AI.")
print("Load it anywhere, generate unlimited")
print("synthetic orders without retraining.")