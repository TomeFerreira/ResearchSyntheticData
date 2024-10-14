import csv
import os
import time
import numpy as np
from pymfe.mfe import MFE
import yaml
from readers.arff_reader_writer import read_arff_file
from readers.dat_reader import read_dat

def main():

	with open('config.yaml', 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)

	PATH = config['general']['DATASET_PATH']

	classToPredict = 'Class'

	print("A")

	for root,dirs,files in os.walk(PATH):
		for dir in dirs:
			
			PATH = config['general']['PATH']
			
			if(not os.path.isdir(PATH)):
				os.mkdir(PATH)
			
			print('\n' + str(dir) + '\n')
			if(dir != config['general']['DIR']):
				continue

			i = 0

			dir_path = os.path.join(root, dir)
			for file in os.listdir(dir_path):
				if file.endswith(".arff") or file.endswith(".dat"):
					filename, extension = os.path.splitext(file)

					PATH = config['general']['PATH'] + '/' + str(filename)

					print('\n' + file + '\n')
					
					if(file.endswith(".arff")):
						df = read_arff_file(dir_path + '/' + file)
					else:
						df = read_dat(dir_path + '/' + file)

					classToPredict = df.columns[len(df.columns)-1]

					X = df.drop([classToPredict],axis=1)
					Y = df[classToPredict]

					mfe_aux = MFE(groups=["complexity"])
					mfe_aux.fit(np.asarray(X), np.asarray(Y))
					ft = mfe_aux.extract()

					if i == 0:
						with open('dataset_complexity_' + dir +'.csv','w',newline='') as csvfile_aux:
							csv_writer = csv.writer(csvfile_aux,delimiter=',')
							aux = np.array(ft[0])
							aux = np.insert(aux,0,'Dataset')
							csv_writer.writerow(aux)
						i = 1

					with open('dataset_complexity_' + dir +'.csv','a',newline='') as csvfile_aux:
						csv_writer = csv.writer(csvfile_aux,delimiter=',')
						aux = list(ft[1])
						aux.insert(0,filename)
						csv_writer.writerow(aux)



if __name__ == '__main__':
	main()