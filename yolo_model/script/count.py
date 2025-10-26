import os

def count_label_categories(folder_path):
    label_counts = {}  # Dictionary to store label frequencies

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if parts:  # Ensure line is not empty
                        label = parts[0]  # First value in each line is the class ID
                        label_counts[label] = label_counts.get(label, 0) + 1

    return label_counts

# Set your labels directory path
labels_dir = r"E:\snatching\yolo_model\dataset\labels\train"  # Change this to your actual path

# Get label category counts
label_counts = count_label_categories(labels_dir)

# Print results
print("Unique label categories and their counts:")
for label, count in sorted(label_counts.items(), key=lambda x: int(x[0])):  # Sorting by class ID
    print(f"Label {label}: {count} instances")

print(f"\nTotal unique categories: {len(label_counts)}")
