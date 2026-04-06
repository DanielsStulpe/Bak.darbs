import os

folder = "potholes"
start_index = 1004

# Get all files (only images)
files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
files.sort()  # important for consistent ordering

# Step 1: rename to temporary names (avoid overwrite)
temp_names = []
for i, filename in enumerate(files):
    old_path = os.path.join(folder, filename)
    temp_name = f"temp_{i}.jpg"
    temp_path = os.path.join(folder, temp_name)

    os.rename(old_path, temp_path)
    temp_names.append(temp_name)

# Step 2: rename to final names
for i, temp_name in enumerate(temp_names):
    old_path = os.path.join(folder, temp_name)
    new_name = f"img-{start_index + i}.jpg"
    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)

print("Renaming completed.")