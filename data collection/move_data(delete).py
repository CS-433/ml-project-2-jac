import os
import random
import shutil

def move_random_files(source_folder, destination_folder, percentage):
    # Get a list of all files in the source folder
    all_files = os.listdir(source_folder)
    
    # Calculate the number of files to move (20% of total files)
    num_files_to_move = int(len(all_files) * percentage)

    # Randomly select the files to move
    files_to_move = random.sample(all_files, num_files_to_move)

    # Move the selected files to the destination folder
    for file_name in files_to_move:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.move(source_path, destination_path)
        print(f"Moved {file_name} to {destination_folder}")

# Specify the source and destination folders
source_1 = "../data/data_separated/data1"
destination_1 = "../data/data_separated/data1_test"
destination_1_val = "../data/data_separated/data1_val"

source_0 = "../data/data_separated/data0"
destination_0 = "../data/data_separated/data0_test"
destination_0_val = "../data/data_separated/data0_val"

# Specify the percentage of files to move (20%)
percentage_to_move = 0.15

# Call the function to move random files
move_random_files(source_1, destination_1, 0.15)
move_random_files(source_1, destination_1_val, 0.175)

move_random_files(source_0, destination_0, 0.15)
move_random_files(source_0, destination_0_val, 0.175)


# moving back to original folder
"""
move_random_files(destination_1, source_1, 1)
move_random_files(destination_0, source_0, 1)
move_random_files(destination_1_val, source_1, 1)
move_random_files(destination_0_val, source_0, 1)
"""

# Final check
files_test1 = os.listdir("../data/data_separated/data1_test")
files_val1 = os.listdir("../data/data_separated/data1_val")
files_1 = os.listdir("../data/data_separated/data1")

files_test0 = os.listdir("../data/data_separated/data0_test")
files_val0 = os.listdir("../data/data_separated/data0_val")
files_0 = os.listdir("../data/data_separated/data0")

print(" # of files in data1_test", len(files_test1))
print(" # of files in data1_val", len(files_val1))
print(" # of files in data1", len(files_1))

print(" # of files in data0_test", len(files_test0))
print(" # of files in data0_val", len(files_val0))
print(" # of files in data0", len(files_0))




