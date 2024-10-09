from sklearn.preprocessing import LabelEncoder
from scipy.io import arff
import arff as ar
import numpy as np
import pandas as pd

def write_arff(PATH,filename,n,df,type):
	attributes = [(j, 'NUMERIC') if df[j].dtypes in ['int64', 'float64'] else (j, df[j].unique().astype(str).tolist()) for j in df]

	data = []
	for row in df.itertuples(index=False):
		data.append([int(val) if isinstance(val, np.integer) else float(val) if isinstance(val, (float, np.float64)) else val for val in row])

	arff_dic = {
		'attributes': attributes,
		'data': data,
		'relation': str(type)+str(filename)+str(n),
		'description': ''
	}

	with open(PATH + '/' + str(type) + '.arff', "w", encoding="utf8") as f:
		ar.dump(arff_dic, f)

	return 0

def read_arff_file(PATH):

	data = arff.loadarff(PATH)
	#print(data)
	df = pd.DataFrame(data[0])
	df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
	label_encoder = LabelEncoder()
	for i in df.columns:
		if df[i].dtype == 'object' or i == df.columns[len(df.columns)-1]:
			df[i] = label_encoder.fit_transform(df[i])
			df[i] = df[i].astype(int)

	#print(df)

	return df