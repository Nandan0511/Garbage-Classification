import os

# 🔹 Path to your dataset folder
dataset_path = "Garbage classification2/Hazardous-samples"

# Counter
count = 0

# Walk through all subfolders
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(".jpg"):  # check only png
            count += 1

print(f"Total JPG images in dataset: {count}")