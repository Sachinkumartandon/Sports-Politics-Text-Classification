
import os
import pandas as pd

data = []

base_path = "bbc"  # change if needed

categories = ["sport", "politics"]

for category in categories:
    folder_path = os.path.join(base_path, category)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read().replace("\n", " ")

        data.append([text, category])

# Create dataframe
df = pd.DataFrame(data, columns=["text", "label"])

# Save to CSV
df.to_csv("sports_politics.csv", index=False)

print("Dataset created successfully!")
print("Total samples:", len(df))
