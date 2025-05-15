import os

# Define the directory path
directory = '/home/sukanya/pinder_challenge/pinder_challenge-1/data/processed/predicted/train'

# Count the number of files
file_count = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

print(f"The directory contains {file_count} files.")