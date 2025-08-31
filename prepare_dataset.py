import pandas as pd

# === Singlish Dataset ===
singlish_df = pd.read_csv("data/singlish_dataset1.csv", encoding="utf-8")
singlish_df = singlish_df[['Sentence', 'Hate_speech']]
label_map = {"Offensive": 1, "Not offensive": 0}
singlish_df = singlish_df[singlish_df['Hate_speech'].isin(label_map)]
singlish_df['label'] = singlish_df['Hate_speech'].map(label_map)
singlish_df.rename(columns={'Sentence': 'text'}, inplace=True)
singlish_df = singlish_df[['text', 'label']]

# === Sinhala Unicode Dataset 1 ===
sinhala1_df = pd.read_csv("data/sinhalauni_dataset1.csv", encoding="utf-8")
sinhala1_df.rename(columns={'comment': 'text', 'label': 'label'}, inplace=True)
sinhala1_df['label'] = pd.to_numeric(sinhala1_df['label'], errors='coerce')
sinhala1_df.dropna(subset=['label'], inplace=True)
sinhala1_df['label'] = sinhala1_df['label'].astype(int)
sinhala1_df = sinhala1_df[['text', 'label']]

# === Sinhala Unicode Dataset 2 ===
sinhala2_df = pd.read_csv("data/sinhalauni_dataset2.csv", encoding="utf-8")
print("✅ sinhalauni_dataset2 columns:", sinhala2_df.columns)

# Rename and clean
sinhala2_df.rename(columns={'comment': 'text', 'label': 'label'}, inplace=True)
sinhala2_df['label'] = pd.to_numeric(sinhala2_df['label'], errors='coerce')
sinhala2_df.dropna(subset=['label'], inplace=True)
sinhala2_df['label'] = sinhala2_df['label'].astype(int)
sinhala2_df = sinhala2_df[['text', 'label']]

# === Merge All ===
combined_df = pd.concat([singlish_df, sinhala1_df, sinhala2_df], ignore_index=True)
combined_df.dropna(inplace=True)
combined_df.to_csv("merged_dataset.csv", index=False)

# Summary
print("Final dataset created: merged_dataset.csv")
print(f"Total rows: {len(combined_df)}")
print("Class balance:\n", combined_df['label'].value_counts())
