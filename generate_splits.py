import csv
import yaml
import pandas as pd
import numpy as np
import os
import traceback
import time
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE,ADASYN,BorderlineSMOTE,SVMSMOTE,KMeansSMOTE,RandomOverSampler
from imblearn.combine import SMOTEENN,SMOTETomek
from sdv.metadata import SingleTableMetadata 
from sdv.single_table import CTGANSynthesizer,GaussianCopulaSynthesizer,TVAESynthesizer,CopulaGANSynthesizer
from sdv.sampling import Condition
from sklearn.model_selection import train_test_split
from readers.dat_reader import read_dat
from readers.arff_reader_writer import read_arff_file,write_arff
from imblearn.under_sampling import EditedNearestNeighbours
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from sklearn.preprocessing import MinMaxScaler
from tabfairgan import TFG

N_JOBS = 4

def main():

	with open('config.yaml', 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)

	nearestN = NearestNeighbors(n_neighbors=3,n_jobs=N_JOBS)

	noise_dim = 32
	dim = 128
	batch_size = 128

	log_step = 100
	epochs = 300+1
	learning_rate = 1e-4
	beta_1 = 0.5
	beta_2 = 0.9

	model_parameters = ModelParameters(batch_size=batch_size,
									lr=learning_rate,
									betas=(beta_1, beta_2),
									noise_dim=noise_dim,
									layers_dim=dim)

	train_args = TrainParameters(epochs=epochs,
								sample_interval=log_step)

	start_time = time.time()

	PATH = config['general']['DATASET_PATH']

	classToPredict = 'Class'
	n = 10

	for root,dirs,files in os.walk(PATH):
		for dir in dirs:
			
			PATH = config['general']['PATH']
			
			if(not os.path.isdir(PATH)):
				os.mkdir(PATH)

			excel_lines = []
			
			if(os.path.exists(config['general']['DATASET_TEMP'] + dir +'.csv')):

				with open(config['general']['DATASET_TEMP'] + dir +'.csv',newline='') as csvfile:
					csv_reader = csv.reader(csvfile,delimiter=',')
					for row in csv_reader:
						excel_lines.append([value for value in row])

			print('\n' + str(dir) + '\n')
			if(dir != config['general']['DIR']):
				continue

			dir_path = os.path.join(root, dir)
			for file in os.listdir(dir_path):
				if file.endswith(".arff") or file.endswith(".dat"):
					filename, extension = os.path.splitext(file)

					aux=0
					for row in excel_lines:
						print(row[0])
						if row[0] == filename:
								aux=1
								break
					if aux==1:
						continue

					PATH_METADATA = config['general']['PATH_METADATA'] + '/' + str(filename)

					PATH = config['general']['PATH'] + '/' + str(filename)

					if(not os.path.isdir(PATH)):
						os.mkdir(PATH)

					for i in range(n):
						PATH_NEW = PATH + '/' + str(i)
						if(not os.path.isdir(PATH_NEW)):
							os.mkdir(PATH_NEW)

					print('\n' + file + '\n')

					if(file.endswith(".arff")):
						df = read_arff_file(dir_path + '/' + file)
					else:
						df = read_dat(dir_path + '/' + file)
					
					classToPredict = df.columns[len(df.columns)-1]

					metadata = SingleTableMetadata()
					metadata = metadata.load_from_json(PATH_METADATA + '_metadata.json')

					X = df.drop([classToPredict],axis=1)
					Y = df[classToPredict]

					X_train = []
					Y_train = []
					X_test = []
					Y_test = []

					#Cria 10 divisões diferentes do dataset
					for i in range(n):
						scaler = MinMaxScaler()
						X_train_aux, X_test_aux, Y_train_aux, Y_test_aux = train_test_split(X,Y,test_size=0.2,stratify=df[classToPredict])
						X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_aux), columns=X_train_aux.columns, index=X_train_aux.index)
						X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test_aux), columns=X_test_aux.columns, index=X_test_aux.index)
						write_arff(PATH + '/' + str(i), filename, i, pd.concat([X_train_scaled,Y_train_aux],axis=1), 'Baseline')
						write_arff(PATH + '/' + str(i), filename, i, pd.concat([X_test_scaled,Y_test_aux],axis=1), 'Test')
						X_train.append(X_train_scaled)
						X_test.append(X_test_scaled)
						Y_train.append(Y_train_aux)
						Y_test.append(Y_test_aux)

					#################################### RandomOverSampler #############################################
					try:

						print('\n\t RandomOverSampler \t')

						sm = RandomOverSampler()

						for i in range(n):
							X_train_aux_ROS,Y_train_aux_ROS = sm.fit_resample(X_train[i],Y_train[i])
							write_arff(PATH + '/' + str(i), filename, i, pd.concat([X_train_aux_ROS,Y_train_aux_ROS],axis=1), 'RandomOverSampler')
	   
					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to RandomOverSampler")

					#################################### SMOTE #############################################
					try:

						print('\n\t SMOTE \t')

						sm = SMOTE(k_neighbors=nearestN)

						for i in range(n):
							X_train_aux_SMOTE,Y_train_aux_SMOTE = sm.fit_resample(X_train[i],Y_train[i])
							write_arff(PATH + '/' + str(i), filename, i, pd.concat([X_train_aux_SMOTE,Y_train_aux_SMOTE],axis=1), 'SMOTE')

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to SMOTE")

					#################################### ADASYN #############################################
					try:
						print('\n\t ADASYN \t')

						adasyn = ADASYN(n_neighbors=nearestN)

						for i in range(n):
							X_train_aux_ADASYN,Y_train_aux_ADASYN = adasyn.fit_resample(X_train[i],Y_train[i])
							write_arff(PATH + '/' + str(i), filename, i, pd.concat([X_train_aux_ADASYN,Y_train_aux_ADASYN],axis=1), 'ADASYN')

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to ADASYN")

					#################################### BorderlineSMOTE #############################################
					try:

						print('\n\t BorderlineSMOTE \t')

						sm = BorderlineSMOTE(k_neighbors=nearestN)

						for i in range(n):
							X_train_aux_BSMOTE,Y_train_aux_BSMOTE = sm.fit_resample(X_train[i],Y_train[i])
							write_arff(PATH + '/' + str(i), filename, i, pd.concat([X_train_aux_BSMOTE,Y_train_aux_BSMOTE],axis=1), 'BorderlineSMOTE')

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to BorderlineSMOTE")

					#################################### SVMSMOTE #############################################
					try:

						print('\n\t SVMSMOTE \t')

						sm = SVMSMOTE(k_neighbors=3)

						for i in range(n):
							X_train_aux_SVMSMOTE,Y_train_aux_SVMSMOTE = sm.fit_resample(X_train[i],Y_train[i])
							write_arff(PATH + '/' + str(i), filename, i, pd.concat([X_train_aux_SVMSMOTE,Y_train_aux_SVMSMOTE],axis=1), 'SVMSMOTE')

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to SVMSMOTE")
						print(traceback.format_exc())

					################################### KMeansSMOTE #############################################
					try:

						print('\n\t KMeansSMOTE \t')

						sm = KMeansSMOTE(k_neighbors=nearestN,cluster_balance_threshold=(min(pd.Series(Y).value_counts())/max(pd.Series(Y).value_counts()))/2)

						for i in range(n):
							X_train_aux_KMSMOTE,Y_train_aux_KMSMOTE = sm.fit_resample(X_train[i],Y_train[i])
							write_arff(PATH + '/' + str(i), filename, i, pd.concat([X_train_aux_KMSMOTE,Y_train_aux_KMSMOTE],axis=1), 'KMeansSMOTE')

					except KeyboardInterrupt:
						return
					except:
						#print(traceback.format_exc())
						print("Dataset not suitable to KMeansSMOTE")

					################################### SMOTEENN #############################################
					try:

						print('\n\t SMOTEENN \t')

						sm = SMOTEENN(smote=SMOTE(k_neighbors=nearestN),sampling_strategy=1.0,enn=EditedNearestNeighbours(kind_sel='mode'))

						for i in range(n):
							X_train_aux_SMOTEENN,Y_train_aux_SMOTEENN = sm.fit_resample(X_train[i],Y_train[i])
							write_arff(PATH + '/' + str(i), filename, i, pd.concat([X_train_aux_SMOTEENN,Y_train_aux_SMOTEENN],axis=1), 'SMOTEENN')

					except KeyboardInterrupt:
						return
					except:
						#print(traceback.format_exc())
						print("Dataset not suitable to SMOTEENN")

					################################### SMOTETomek #############################################
					try:

						print('\n\t SMOTETomek \t')

						excel_line = [filename,'SMOTETomek']

						sm = SMOTETomek(smote=SMOTE(k_neighbors=nearestN),n_jobs=N_JOBS)

						for i in range(n):
							X_train_aux_SMOTETL,Y_train_aux_SMOTETL = sm.fit_resample(X_train[i],Y_train[i])
							write_arff(PATH + '/' + str(i), filename, i, pd.concat([X_train_aux_SMOTETL,Y_train_aux_SMOTETL],axis=1), 'SMOTETomek')

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to SMOTETomek")

					#################################### TFGAN #############################################
					# try:
					# 	print('\n\t SDV : TFGAN \t')

					# 	for i in range(n):
					# 		df = pd.concat([X_train[i], Y_train[i]], axis=1)
					# 		synthetic_data = pd.DataFrame()
					# 		for j in Y_train[i].unique():
					# 			df_it=df[df[classToPredict]==j]
					# 			tfg = TFG(df_it, epochs=200, batch_size=256, device='cpu')
					# 			tfg.train()
					# 			synthetic_data_not_concat = tfg.generate_fake_df(num_rows = max(Y_train[i].value_counts()))
					# 			synthetic_data = pd.concat([synthetic_data,synthetic_data_not_concat])
					# 		write_arff(PATH + '/' + str(i), filename, i, pd.concat([synthetic_data.drop([classToPredict], axis=1),synthetic_data[classToPredict]],axis=1), 'TFGAN')

					# except KeyboardInterrupt:
					# 	return
					# except:
					# 	print(traceback.format_exc())
					# 	print("Dataset not suitable to TFGAN")
					
					#################################### SDV : CTGAN #############################################
					try:
						print('\n\t SDV : CTGAN \t')

						for i in range(n):
							synthesizer = CTGANSynthesizer(metadata,epochs=500)
							synthesizer.fit(pd.concat([X_train[i], Y_train[i]], axis=1))

							class_options = Y_train[i].unique()

							synthetic_data = pd.DataFrame()

							for j in class_options:
								balance_data_condition = Condition(num_rows= max(Y_train[i].value_counts()),column_values={classToPredict:j})
								synthetic_data_not_concat = synthesizer.sample_from_conditions(conditions=[balance_data_condition])
								synthetic_data = pd.concat([synthetic_data,synthetic_data_not_concat])

							write_arff(PATH + '/' + str(i), filename, i, pd.concat([synthetic_data.drop([classToPredict], axis=1),synthetic_data[classToPredict]],axis=1), 'CTGAN')

					except KeyboardInterrupt:
						return
					except:
						print(traceback.format_exc())
						print("Dataset not suitable to SDV CTGAN")
					
					#################################### SDV : GaussianCopulaSynthesizer #############################################

					try:
						print('\n\t SDV : GaussianCopulaSynthesizer \t')

						for i in range(n):
							synthesizer = GaussianCopulaSynthesizer(metadata)
							synthesizer.fit(pd.concat([X_train[i], Y_train[i]], axis=1))

							class_options = Y_train[i].unique()

							synthetic_data = pd.DataFrame()

							for j in class_options:
								balance_data_condition = Condition(num_rows= max(Y_train[i].value_counts()),column_values={classToPredict:j})
								synthetic_data_not_concat = synthesizer.sample_from_conditions(conditions=[balance_data_condition])
								synthetic_data = pd.concat([synthetic_data,synthetic_data_not_concat])

							write_arff(PATH + '/' + str(i), filename, i, pd.concat([synthetic_data.drop([classToPredict], axis=1),synthetic_data[classToPredict]],axis=1), 'GaussianCopulaSynthesizer')

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to SDV GaussianCopulaSynthesizer")
						print(traceback.format_exc())

					#################################### SDV : TVAESynthesizer #############################################

					try:
						print('\n\t SDV : TVAESynthesizer \t')

						for i in range(n):
							synthesizer = TVAESynthesizer(metadata,epochs=500)
							synthesizer.fit(pd.concat([X_train[i], Y_train[i]], axis=1))

							class_options = Y_train[i].unique()

							synthetic_data = pd.DataFrame()

							for j in class_options:
								balance_data_condition = Condition(num_rows= max(Y_train[i].value_counts()),column_values={classToPredict:j})
								synthetic_data_not_concat = synthesizer.sample_from_conditions(conditions=[balance_data_condition])
								synthetic_data = pd.concat([synthetic_data,synthetic_data_not_concat])

							write_arff(PATH + '/' + str(i), filename, i, pd.concat([synthetic_data.drop([classToPredict], axis=1),synthetic_data[classToPredict]],axis=1), 'TVAESynthesizer')

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to SDV TVAESynthesizer")
					#################################### SDV : CopulaGANSynthesizer #############################################

					try:
						print('\n\t SDV : CopulaGANSynthesizer \t')

						for i in range(n):
							synthesizer = CopulaGANSynthesizer(metadata,epochs=500)
							synthesizer.fit(pd.concat([X_train[i], Y_train[i]], axis=1))

							class_options = Y_train[i].unique()

							synthetic_data = pd.DataFrame()

							for j in class_options:
								balance_data_condition = Condition(num_rows= max(Y_train[i].value_counts()),column_values={classToPredict:j})
								synthetic_data_not_concat = synthesizer.sample_from_conditions(conditions=[balance_data_condition])
								synthetic_data = pd.concat([synthetic_data,synthetic_data_not_concat])

							write_arff(PATH + '/' + str(i), filename, i, pd.concat([synthetic_data.drop([classToPredict], axis=1),synthetic_data[classToPredict]],axis=1), 'CopulaGANSynthesizer')

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to SDV CopulaGANSynthesizer")
					####################################### YDATA  ################################################

					num_cols = []
					cat_cols = []

					for column_name, column_metadata in metadata.columns.items():
						if column_metadata['sdtype'] in ['numerical','datetime']:
							num_cols.append(column_name)
						else:
							cat_cols.append(column_name)

					#################################### YDATA : WGAN #############################################
					try:
						print('\n\t YDATA : WGAN \t')

						if(int(np.floor(min(Y_train[i].value_counts())*0.8))<128):
							noise_dim = int(np.floor(min(Y_train[i].value_counts())*0.2))
							dim = int(noise_dim*4)
							batch_size = int(noise_dim*4)
						else:
							noise_dim = 32
							dim = 128
							batch_size = 128

						model_parameters_WGAN = ModelParameters(batch_size=batch_size,
									lr=learning_rate,
									betas=(beta_1, beta_2),
									noise_dim=noise_dim,
									layers_dim=dim)

						for i in range(n):
							df = pd.concat([X_train[i], Y_train[i]], axis=1)
							synthetic_data = pd.DataFrame()
							for j in Y_train[i].unique():
								df_it=df[df[classToPredict]==j]
								synthesizer = RegularSynthesizer(modelname='wgan', model_parameters=model_parameters_WGAN, n_critic=2)
								synthesizer.fit(data=df_it, train_arguments = train_args, num_cols = num_cols, cat_cols = cat_cols)
								synthetic_data_not_concat = synthesizer.sample(max(Y_train[i].value_counts()))
								synthetic_data = pd.concat([synthetic_data,synthetic_data_not_concat])
								synthetic_data.fillna(synthetic_data.mean(),inplace=True)
							write_arff(PATH + '/' + str(i), filename, i, pd.concat([synthetic_data.drop([classToPredict], axis=1),synthetic_data[classToPredict]],axis=1), 'WGAN')

					except KeyboardInterrupt:
						return
					except:
						print(traceback.format_exc())
						print("Dataset not suitable to WGAN")
					#################################### YDATA : DRAGAN #############################################
					try:
						print('\n\t YDATA : DRAGAN \t')

						for i in range(n):
							df = pd.concat([X_train[i], Y_train[i]], axis=1)
							synthetic_data = pd.DataFrame()
							for j in Y_train[i].unique():
								df_it=df[df[classToPredict]==j]
								synthesizer = RegularSynthesizer(modelname='dragan', model_parameters=model_parameters, n_discriminator=3)
								synthesizer.fit(data=df_it, train_arguments = train_args, num_cols = num_cols, cat_cols = cat_cols)
								synthetic_data_not_concat = synthesizer.sample(max(Y_train[i].value_counts()))
								synthetic_data = pd.concat([synthetic_data,synthetic_data_not_concat])
							write_arff(PATH + '/' + str(i), filename, i, pd.concat([synthetic_data.drop([classToPredict], axis=1),synthetic_data[classToPredict]],axis=1), 'DRAGAN')

					except KeyboardInterrupt:
						return
					except:
						print(traceback.format_exc())
						print("Dataset not suitable to DRAGAN")

					#################################### YDATA : WGAN GP #############################################
					try:
						print('\n\t YDATA : WGAN GP \t')

						for i in range(n):
							df = pd.concat([X_train[i], Y_train[i]], axis=1)
							synthetic_data = pd.DataFrame()
							for j in Y_train[i].unique():
								df_it=df[df[classToPredict]==j]
								synthesizer = RegularSynthesizer(modelname='wgangp', model_parameters=model_parameters, n_critic=2)
								synthesizer.fit(data=df_it, train_arguments = train_args, num_cols = num_cols, cat_cols = cat_cols)
								synthetic_data_not_concat = synthesizer.sample(max(Y_train[i].value_counts()))
								synthetic_data = pd.concat([synthetic_data,synthetic_data_not_concat])
							write_arff(PATH + '/' + str(i), filename, i, pd.concat([synthetic_data.drop([classToPredict], axis=1),synthetic_data[classToPredict]],axis=1), 'WGAN GP')

					except KeyboardInterrupt:
						return
					except:
						print(traceback.format_exc())
						print("Dataset not suitable to WGAN GP")
					#########################################################################################
					with open(config['general']['DATASET_TEMP']+ dir +'.csv','a',newline='') as csvfile:
						csv_writer = csv.writer(csvfile,delimiter=',')
						csv_writer.writerow([filename])

	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Elapsed time:", elapsed_time, "seconds")

	return 0

if __name__ == "__main__":
	main()
