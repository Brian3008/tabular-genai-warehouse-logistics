import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# Load clean data
print("Loading clean orders...")
df = pd.read_csv('data/clean_orders.csv')

# Use a sample for faster training
# 100,000 rows is enough for CTGAN to learn the patterns
print("Sampling 100,000 rows for training...")
sample = df.sample(n=100000, random_state=42)

print("Sample shape:", sample.shape)
print("\nSample head:")
print(sample.head())

# Define metadata
print("\nDetecting metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(sample)

# Tell CTGAN which columns are categorical
metadata.update_column(column_name='aisle_id',
                       sdtype='categorical')
metadata.update_column(column_name='department_id',
                       sdtype='categorical')
metadata.update_column(column_name='order_dow',
                       sdtype='categorical')
metadata.update_column(column_name='order_hour_of_day',
                       sdtype='categorical')

print("Metadata defined successfully")
print(metadata)

# Train CTGAN
print("\nTraining CTGAN...")
print("This will take several minutes, please wait...")
model = CTGANSynthesizer(metadata,
                          epochs=100,
                          verbose=True)
model.fit(sample)

print("\nCTGAN training complete!")

# Save the trained model
model.save('data/ctgan_model.pkl')
print("Model saved to data/ctgan_model.pkl")

# Generate synthetic data
print("\nGenerating 10,000 synthetic orders...")
synthetic = model.sample(num_rows=10000)

print("Synthetic data shape:", synthetic.shape)
print("\nSynthetic data sample:")
print(synthetic.head(10))

# Save synthetic data
synthetic.to_csv('data/synthetic_orders.csv', index=False)
print("\nSaved synthetic_orders.csv to data folder")

# Quick comparison
print("\n--- REAL DATA STATS ---")
print(sample[['order_dow', 'order_hour_of_day',
              'aisle_id']].describe())

print("\n--- SYNTHETIC DATA STATS ---")
print(synthetic[['order_dow', 'order_hour_of_day',
                 'aisle_id']].describe())

print("\nDone!")