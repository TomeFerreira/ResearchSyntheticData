import pandas as pd

def convert_to_numeric(df):
	"""
	Converts columns with string data types to numeric (0, 1, ..., n-1) based on unique values.

	Args:
		df: A pandas DataFrame.

	Returns:
		A new pandas DataFrame with converted columns.
	"""
	# Iterate through columns
	for col in df.columns:
	# Check if the column is string type and has more than one unique value
		if df[col].dtype == object and df[col].nunique() > 1:
		# Get the unique values
			unique_values = df[col].unique()
			# Create a dictionary for mapping (0-based indexing)
			value_map = {val: i for i, val in enumerate(unique_values)}
			# Apply the mapping to the column
			df[col] = df[col].replace(value_map)
	return df


def read_dat(PATH):
	with open(PATH, 'r') as file:
		lines = file.readlines()

	classes = []

	# Find the index of the '@data' marker
	data_index = None
	for i, line in enumerate(lines):
		if line.startswith('@attribute'):
			# Extract the attribute name
			parts = line.split()
			if len(parts) >= 2:
				classes.append(parts[1].strip())
		elif line.startswith('@data'):
			data_index = i
			break

	# If '@data' marker is found, create DataFrame with remaining lines
	if data_index is not None:
		# Create DataFrame from remaining lines
		data_lines = lines[data_index + 1:]
		df = pd.DataFrame([line.strip().split(',') for line in data_lines], columns=classes)
		df = convert_to_numeric(df)
		return df
	else:
		print("Error: '@data' marker not found in the file.")
		return None

if __name__ == '__main__':
	df = read_dat('DATASETS\\DATASETS\\NEW-IMBAL\\abalone9-18\\abalone9-18.dat')
	print(df)

