import pandas as pd
from pathlib import Path

# Path of the file to read
base_dir = Path(__file__).resolve().parent
iowa_file_path = base_dir / "student_failure" / "train.csv"

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Call line below with no argument to check that you've loaded the data correctly
#step_1.check()

# print a train.csv
home_data.describe()