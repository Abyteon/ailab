import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import train_test_split


data_dir = "../THUCNews"
csv_file = "thucnews_data.csv"
new_dir = "../data"

encoded_csv_file = "encoded_thucnews_data.csv"
label_encoded_file = "label_encoded_thucnews_data.pkl"

texts, labels = [], []
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    if os.path.isdir(category_path):
        for filename in os.listdir(category_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(category_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        texts.append(text)
                        labels.append(category)
                except Exception as e:
                    continue

df = pd.DataFrame({"text": texts, "label": labels})

chunk_size = 1000
for i in range(0, len(df), chunk_size):
    chunk_id = i
    chunk_writer = df.iloc[i : i + chunk_size].to_csv(
        os.path.join(new_dir, csv_file),
        mode="a",
        header=(i == 0),
        index=False,
        encoding="utf-8",
    )
    print(f"Processing chunk {i // chunk_size + 1}: {chunk_id} rows")

print(f"Data saved to {os.path.join(new_dir, csv_file)}")


encoder = LabelEncoder()
df["label_id"] = encoder.fit_transform(df["label"])

# for i in range(0, len(df), chunk_size):
#     chunk_id = i
#     chunk_writer = df.iloc[i : i + chunk_size].to_csv(
#         os.path.join(new_dir, encoded_csv_file),
#         mode="a",
#         header=(i == 0),
#         index=False,
#         encoding="utf-8",
#     )
#     print(f"Processing chunk {i // chunk_size + 1}: {chunk_id} rows")

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label_id"]
)

train_df.to_csv(
    os.path.join(new_dir, "train_" + encoded_csv_file), index=False, encoding="utf-8"
)
val_df.to_csv(
    os.path.join(new_dir, "val_" + encoded_csv_file), index=False, encoding="utf-8"
)

joblib.dump(encoder, os.path.join(new_dir, label_encoded_file))
print(f"Label encoding saved to {os.path.join(new_dir, label_encoded_file)}")
print(f"Encoded data saved to {os.path.join(new_dir, encoded_csv_file)}")
