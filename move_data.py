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
source_1 = "./data1"
destination_1 = "./data1_test"

source_0 = "./data0"
destination_0 = "./data0_test"

# Specify the percentage of files to move (20%)
percentage_to_move = 0.2

# Call the function to move random files
move_random_files(source_1, destination_1, percentage_to_move)
move_random_files(source_0, destination_0, percentage_to_move)

files_test1 = os.listdir("./data1_test")
files_1 = os.listdir("./data1")
files_test0 = os.listdir("./data0_test")
files_0 = os.listdir("./data0")

print(" # of files in data1_test", len(files_test1))
print(" # of files in data1", len(files_1))
print(" # of files in data0_test", len(files_test0))
print(" # of files in data0", len(files_0))


