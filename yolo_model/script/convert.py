import os

# Folder containing YOLO annotation text files
annotation_folder = "E:\snatching\label"

# Function to update class labels
def update_labels(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Process only .txt files
            file_path = os.path.join(folder_path, filename)

            # Read and modify the labels
            with open(file_path, "r") as file:
                lines = file.readlines()

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] == '6':  # Change class 0 to class 1
                    parts[0] = '1'
                updated_lines.append(" ".join(parts))

            # Write the updated labels back to the file
            with open(file_path, "w") as file:
                file.write("\n".join(updated_lines) + "\n")

            print(f"Updated {filename}")

# Run the function
update_labels(annotation_folder)
