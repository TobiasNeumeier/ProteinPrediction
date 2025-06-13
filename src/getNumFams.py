import pandas as pd
import json

def count_unique_pf_labels(csv_path):
    df = pd.read_csv(csv_path)
    labels = df["label"].apply(lambda x: x.split(".")[0])
    unique_labels = labels.unique()
    print(f"Total unique PF labels: {len(unique_labels)}")
    # print("List of unique labels:")
    # for label in unique_labels:
    #     print(label)
    print(len(labels))

def build_pf_label_index(csv_paths, output_json="./label_index.json"):
    all_labels = []

    for path in csv_paths:
        df = pd.read_csv(path)
        labels = df["label"].apply(lambda x: x.split(".")[0])
        all_labels.extend(labels)

    unique_labels = sorted(set(all_labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}


    # Optional: Save to file
    with open(output_json, "w") as f:
        json.dump(label_to_index, f, indent=2)

    print(f"Found {len(label_to_index)} unique PF labels.")
    return label_to_index

# Usage
if __name__ == "__main__":
    #count_unique_pf_labels("./train.csv")
    build_pf_label_index(["./train.csv","./test.csv","./val.csv"])
