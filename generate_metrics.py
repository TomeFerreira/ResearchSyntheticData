import csv
import yaml
import pandas as pd
import numpy as np
import os
import statistics
import traceback
import time
from sdmetrics.single_column import BoundaryAdherence,CategoryAdherence,KSComplement,TVComplement,MissingValueSimilarity,StatisticSimilarity,CategoryCoverage,RangeCoverage
from sdmetrics.single_table import TableStructure,NewRowSynthesis
from sdmetrics.column_pairs import ContingencySimilarity,CorrelationSimilarity
from syntheval_copy.syntheval.src.syntheval.metrics.privacy.metric_MIA_classification import MIAClassifier
from syntheval_copy.syntheval.src.syntheval.metrics.privacy.metric_nn_adversarial_accuracy import NearestNeighbourAdversarialAccuracy
from syntheval_copy.syntheval.src.syntheval.metrics.privacy.metric_epsilon_identifiability import EpsilonIdentifiability
from syntheval_copy.syntheval.src.syntheval.metrics.privacy.metric_nn_distance_ratio import NearestNeighbourDistanceRatio
from sdv.metadata import SingleTableMetadata 
from readers.dat_reader import read_dat
from readers.arff_reader_writer import read_arff_file

METRICS = ['DATASET','METHOD','Table Structure AVG','Table Structure STD','New Row Synthesis AVG','New Row Synthesis STD','Boundary Adherence AVG','Boundary Adherence STD',
		   'KSComplement AVG','KSComplement STD','Correlation Similarity AVG','Correlation Similarity STD','Statistic Similarity AVG','Statistic Similarity STD',
		   'Range Coverage AVG','Range Coverage STD','Category Adherence AVG','Category Adherence STD','TVComplement AVG','TVComplement STD',
		   'Contingency Similarity AVG','Contingency Similarity STD','CategoryCoverage AVG','CategoryCoverage STD','Missing Value Similarity AVG',
		   'Missing Value Similarity STD','Average Feature Comparison AVG','Average Feature Comparison STD','Mode Feature Comparison AVG',
		   'Mode Feature Comparison STD','Membership Inference Attack AVG','Membership Inference Attack STD','Epsilon Identifiability Risk AVG',
		   'Epsilon Identifiability Risk STD','NNAA AVG','NNAA STD','Priv Loss NNAA AVG','Priv Loss NNAA STD','Priv Loss NNDR AVG','Priv Loss NNDR STD']

N_JOBS = 4

def show_SDV_stats(n,original_df,train_X_list,train_Y_list,synthetic_X_list,synthetic_Y_list,metadata, excel_line):


	TableStructure_list = np.zeros(n)
	NewRowSynthesis_list = np.zeros(n)
	BoundaryAdherence_list = np.zeros(n)
	KSComplement_list = np.zeros(n)
	CorrelationSimilarity_list = np.zeros(n)
	StatisticSimilarity_list = np.zeros(n)
	RangeCoverage_list = np.zeros(n)
	CategoryAdherence_list = np.zeros(n)
	TVComplement_list = np.zeros(n)
	ContingencySimilarity_list = np.zeros(n)
	CategoryCoverage_list = np.zeros(n)
	MissingValueSimilarity_list = np.zeros(n)
	AverageFeatureComparison = np.zeros(n)
	ModeFeatureComparison = np.zeros(n)

	for i in range(n):

		training_df = pd.concat([train_X_list[i],train_Y_list[i]],axis=1)

		synthetic_df_aux = pd.concat([synthetic_X_list[i],synthetic_Y_list[i]],axis=1)

		synthetic_df = synthetic_df_aux.loc[synthetic_df_aux.index.difference(training_df.index)]

		if(synthetic_df.empty):
			continue

		TableStructure_list[i] = TableStructure.compute(real_data = original_df,  synthetic_data = synthetic_df)
		NewRowSynthesis_list[i] = NewRowSynthesis.compute(real_data = original_df,  synthetic_data = synthetic_df, metadata=metadata)

		last_name = str()
		last_column = str()
		for column_name, column_metadata in metadata.columns.items():
			if column_metadata['sdtype'] in ['numerical','datetime']:
				AverageFeatureComparison[i] = np.abs(np.average(synthetic_df[column_name])-np.average(original_df[column_name]))
				BoundaryAdherence_list[i] = BoundaryAdherence.compute(real_data = original_df[column_name], synthetic_data = synthetic_df[column_name])
				KSComplement_list[i] = KSComplement.compute(real_data = original_df[column_name], synthetic_data = synthetic_df[column_name])
				if last_name != '' and last_column in ['numerical', 'datetime']:
					try:
						CorrelationSimilarity_list[i] = CorrelationSimilarity.compute(real_data = original_df[[column_name, last_name]],
																	synthetic_data = synthetic_df[[column_name, last_name]])
					except:
						CorrelationSimilarity_list[i] = float('nan')
				StatisticSimilarity_list[i] = StatisticSimilarity.compute(real_data = original_df[column_name], synthetic_data = synthetic_df[column_name])
				RangeCoverage_list[i] = RangeCoverage.compute(real_data = original_df[column_name], synthetic_data = synthetic_df[column_name])
			elif column_metadata['sdtype'] in ['categorical', 'boolean']:
				CategoryAdherence_list[i] = CategoryAdherence.compute(real_data = original_df[column_name], synthetic_data = synthetic_df[column_name])
				TVComplement_list[i] = TVComplement.compute(real_data = original_df[column_name], synthetic_data = synthetic_df[column_name])
				if last_name != '' and last_column in ['categorical', 'boolean']:
					try:
						ContingencySimilarity_list[i] = ContingencySimilarity.compute(real_data = original_df[[column_name, last_name]],
										synthetic_data=synthetic_df[[column_name, last_name]])
					except:
						ContingencySimilarity_list[i] = float('nan')
				CategoryCoverage_list[i] = CategoryCoverage.compute(real_data = original_df[column_name], synthetic_data = synthetic_df[column_name])
				if(statistics.mode(original_df[column_name])!=statistics.mode(synthetic_df[column_name])):
					ModeFeatureComparison[i] = 0
				else:
					ModeFeatureComparison[i] = 1
			last_name = column_name
			last_column = column_metadata['sdtype']

			MissingValueSimilarity_list[i] = MissingValueSimilarity.compute(real_data = original_df[column_name], synthetic_data = synthetic_df[column_name])
	
	try:
		excel_line.append(np.mean(TableStructure_list))
		excel_line.append(np.std(TableStructure_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(NewRowSynthesis_list))
		excel_line.append(np.std(NewRowSynthesis_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(BoundaryAdherence_list))
		excel_line.append(np.std(BoundaryAdherence_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(KSComplement_list))
		excel_line.append(np.std(KSComplement_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(CorrelationSimilarity_list))
		excel_line.append(np.std(CorrelationSimilarity_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(StatisticSimilarity_list))
		excel_line.append(np.std(StatisticSimilarity_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(RangeCoverage_list))
		excel_line.append(np.std(RangeCoverage_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(CategoryAdherence_list))
		excel_line.append(np.std(CategoryAdherence_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(TVComplement_list))
		excel_line.append(np.std(TVComplement_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(ContingencySimilarity_list))
		excel_line.append(np.std(ContingencySimilarity_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(CategoryCoverage_list))
		excel_line.append(np.std(CategoryCoverage_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(MissingValueSimilarity_list))
		excel_line.append(np.std(MissingValueSimilarity_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(AverageFeatureComparison))
		excel_line.append(np.std(AverageFeatureComparison))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(ModeFeatureComparison))
		excel_line.append(np.std(ModeFeatureComparison))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))

	return

def privacy_stats(n,original_df,train_X_list,train_Y_list,synthetic_X_list,synthetic_Y_list,holdout_X_list,holdout_Y_list, excel_line):
	MIA_aux_list = np.zeros(n)
	EPS_aux_list = np.zeros(n)
	NNAA_aux_list = np.zeros(n)
	NNAA_Privacy_lost_aux_list = np.zeros(n)
	NNDR_Privacy_lost_aux_list = np.zeros(n)

	for i in range(n):
		training_df = pd.concat([train_X_list[i],train_Y_list[i]],axis=1)

		synthetic_df_aux = pd.concat([synthetic_X_list[i],synthetic_Y_list[i]],axis=1)

		synthetic_df = synthetic_df_aux.loc[synthetic_df_aux.index.difference(training_df.index)]
		if(synthetic_df.empty):
			continue
		
		holdout_df = pd.concat([holdout_X_list[i],holdout_Y_list[i]],axis=1)

		M = MIAClassifier(original_df, synthetic_df, holdout_df,verbose=0)
		M.evaluate()
		MIA_aux_list[i] = M.normalize_output()[0]['val']

		E = EpsilonIdentifiability(original_df, synthetic_df, holdout_df, nn_dist='gower', verbose=0)
		E.evaluate()
		EPS_aux_list[i] = E.normalize_output()[0]['val']

		#Privacy loss score
		NNAA = NearestNeighbourAdversarialAccuracy(original_df, synthetic_df, holdout_df, nn_dist='gower',verbose=0)
		NNAA.evaluate()
		NNAA_aux_list[i] = NNAA.normalize_output()[0]['val']
		#Como privacy loss negativo é == 0 fazemos este max
		NNAA_Privacy_lost_aux_list[i] = max(NNAA.normalize_output()[1]['val'], 0)

		#NNDR Privacy Loss score
		NNDR = NearestNeighbourDistanceRatio(original_df, synthetic_df, holdout_df, nn_dist='gower',verbose=0)
		NNDR.evaluate()
		NNDR_Privacy_lost_aux_list[i] = max(NNDR.normalize_output()[1]['val'], 0)

	
	try:
		excel_line.append(np.mean(MIA_aux_list))
		excel_line.append(np.std(MIA_aux_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(EPS_aux_list))
		excel_line.append(np.std(EPS_aux_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(NNAA_aux_list))
		excel_line.append(np.std(NNAA_aux_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(NNAA_Privacy_lost_aux_list))
		excel_line.append(np.std(NNAA_Privacy_lost_aux_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))
	try:
		excel_line.append(np.mean(NNDR_Privacy_lost_aux_list))
		excel_line.append(np.std(NNDR_Privacy_lost_aux_list))
	except:
		excel_line.append(float('nan'))
		excel_line.append(float('nan'))

	return

def get_all_stats(n,original_df,X_list,Y_list,synthetic_X_list,synthetic_Y_list,holdout_X_list,holdout_Y_list, metadata, excel_line):

	show_SDV_stats(n,original_df,X_list,Y_list,synthetic_X_list,synthetic_Y_list,metadata,excel_line)
	privacy_stats(n,original_df,X_list,Y_list,synthetic_X_list,synthetic_Y_list,holdout_X_list,holdout_Y_list,excel_line)

	return

def main():

	with open('config.yaml', 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)

	start_time = time.time()

	PATH = config['general']['DATASET_PATH']

	classToPredict = 'Class'
	n = 10

	for root,dirs,files in os.walk(PATH):
		for dir in dirs:
			
			with open(config['general']['METRICS_TEMP'] + dir +'.csv','r',newline='') as csvfile:
				if sum(1 for line in csvfile) <=1:
					with open(config['general']['METRICS_TEMP'] + dir +'.csv','w',newline='') as csvfile_aux:
						csv_writer = csv.writer(csvfile_aux,delimiter=',')
						csv_writer.writerow(METRICS)

			PATH = config['general']['PATH']
			
			if(not os.path.isdir(PATH)):
				os.mkdir(PATH)

			excel_lines = []
			
			if(os.path.exists(config['general']['METRICS_TEMP'] + dir +'.csv')):

				with open(config['general']['METRICS_TEMP'] + dir +'.csv',newline='') as csvfile:
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

					PATH = config['general']['PATH'] + '/' + str(filename)

					PATH_METADATA = config['general']['PATH_METADATA'] + '/' + str(filename)

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
						Train = read_arff_file(PATH + '/' + str(i) + '/Baseline.arff')
						Test = read_arff_file(PATH + '/' + str(i) + '/Test.arff') 
						X_train.append(Train.drop([classToPredict],axis=1))
						X_test.append(Test.drop([classToPredict],axis=1))
						Y_train.append(Train[classToPredict])
						Y_test.append(Test[classToPredict])

					#################################### RandomOverSampler #############################################
					try:

						print('\n\t RandomOverSampler \t')

						excel_line = [filename,'RandomOverSampler']

						X_train_ROS = []
						Y_train_ROS = []

						for i in range(n):
							train_ROS = read_arff_file(PATH + '/' + str(i) + '/RandomOverSampler.arff')
							X_train_ROS.append(train_ROS.drop([classToPredict],axis=1))
							Y_train_ROS.append(train_ROS[classToPredict])

						get_all_stats(n,df,X_train,Y_train,X_train_ROS,Y_train_ROS,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						print(traceback.format_exc())
						print("Dataset not suitable to RandomOverSampler")

					#################################### SMOTE #############################################
					try:

						print('\n\t SMOTE \t')

						excel_line = [filename,'SMOTE']

						X_train_SMOTE = []
						Y_train_SMOTE = []

						for i in range(n):
							train_SMOTE = read_arff_file(PATH + '/' + str(i) + '/SMOTE.arff')
							X_train_SMOTE.append(train_SMOTE.drop([classToPredict],axis=1))
							Y_train_SMOTE.append(train_SMOTE[classToPredict])

						get_all_stats(n,df,X_train,Y_train,X_train_SMOTE,Y_train_SMOTE,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to SMOTE")

					#################################### ADASYN #############################################
					try:
						print('\n\t ADASYN \t')

						excel_line = [filename,'ADASYN']

						X_train_ADASYN = []
						Y_train_ADASYN = []

						for i in range(n):
							train_ADASYN = read_arff_file(PATH + '/' + str(i) + '/ADASYN.arff')
							X_train_ADASYN.append(train_ADASYN.drop([classToPredict],axis=1))
							Y_train_ADASYN.append(train_ADASYN[classToPredict])

						get_all_stats(n,df,X_train,Y_train,X_train_ADASYN,Y_train_ADASYN,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)
					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to ADASYN")

					#################################### BorderlineSMOTE #############################################
					try:

						print('\n\t BorderlineSMOTE \t')

						excel_line = [filename,'BorderlineSMOTE']

						X_train_BSMOTE = []
						Y_train_BSMOTE = []

						for i in range(n):
							train_BSMOTE = read_arff_file(PATH + '/' + str(i) + '/BorderlineSMOTE.arff')
							X_train_BSMOTE.append(train_BSMOTE.drop([classToPredict],axis=1))
							Y_train_BSMOTE.append(train_BSMOTE[classToPredict])

						get_all_stats(n,df,X_train,Y_train,X_train_BSMOTE,Y_train_BSMOTE,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to BorderlineSMOTE")

					#################################### SVMSMOTE #############################################
					try:

						print('\n\t SVMSMOTE \t')

						excel_line = [filename,'SVMSMOTE']

						X_train_SVMSMOTE = []
						Y_train_SVMSMOTE = []

						for i in range(n):
							train_SVMSMOTE = read_arff_file(PATH + '/' + str(i) + '/SVMSMOTE.arff')
							X_train_SVMSMOTE.append(train_SVMSMOTE.drop([classToPredict],axis=1))
							Y_train_SVMSMOTE.append(train_SVMSMOTE[classToPredict])

						get_all_stats(n,df,X_train,Y_train,X_train_SVMSMOTE,Y_train_SVMSMOTE,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to SVMSMOTE")
						print(traceback.format_exc())

					################################### KMeansSMOTE #############################################
					try:

						print('\n\t KMeansSMOTE \t')

						excel_line = [filename,'KMeansSMOTE']

						X_train_KMSMOTE = []
						Y_train_KMSMOTE = []

						for i in range(n):
							train_KMSMOTE = read_arff_file(PATH + '/' + str(i) + '/KMeansSMOTE.arff')
							X_train_KMSMOTE.append(train_KMSMOTE.drop([classToPredict],axis=1))
							Y_train_KMSMOTE.append(train_KMSMOTE[classToPredict])

						get_all_stats(n,df,X_train,Y_train,X_train_KMSMOTE,Y_train_KMSMOTE,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						#print(traceback.format_exc())
						print("Dataset not suitable to KMeansSMOTE")

					################################### SMOTEENN #############################################
					try:

						print('\n\t SMOTEENN \t')

						excel_line = [filename,'SMOTEENN']

						X_train_SMOTEENN = []
						Y_train_SMOTEENN = []

						for i in range(n):
							train_SMOTEENN = read_arff_file(PATH + '/' + str(i) + '/SMOTEENN.arff')
							X_train_SMOTEENN.append(train_SMOTEENN.drop([classToPredict],axis=1))
							Y_train_SMOTEENN.append(train_SMOTEENN[classToPredict])

						get_all_stats(n,df,X_train,Y_train,X_train_SMOTEENN,Y_train_SMOTEENN,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						#print(traceback.format_exc())
						print("Dataset not suitable to SMOTEENN")

					################################### SMOTETomek #############################################
					try:

						print('\n\t SMOTETomek \t')

						excel_line = [filename,'SMOTETomek']

						X_train_SMOTETL = []
						Y_train_SMOTETL = []

						for i in range(n):
							train_SMOTETL = read_arff_file(PATH + '/' + str(i) + '/SMOTETomek.arff')
							X_train_SMOTETL.append(train_SMOTETL.drop([classToPredict],axis=1))
							Y_train_SMOTETL.append(train_SMOTETL[classToPredict])

						get_all_stats(n,df,X_train,Y_train,X_train_SMOTETL,Y_train_SMOTETL,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to SMOTETomek")

					#################################### SDV : CTGAN #############################################
					try:
						print('\n\t SDV : CTGAN \t')

						excel_line = [filename,'CTGAN']

						X_train_CTGAN = []
						Y_train_CTGAN = []

						for i in range(n):
							synthetic_data = read_arff_file(PATH + '/' + str(i) + '/CTGAN.arff')
							X_train_CTGAN.append(synthetic_data.drop([classToPredict], axis=1))
							Y_train_CTGAN.append(synthetic_data[classToPredict])

						get_all_stats(n,df,X_train,Y_train,X_train_CTGAN,Y_train_CTGAN,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						print(traceback.format_exc())
						print("Dataset not suitable to SDV CTGAN")
					
					#################################### SDV : GaussianCopulaSynthesizer #############################################

					try:
						print('\n\t SDV : GaussianCopulaSynthesizer \t')

						excel_line = [filename,'GaussianCopulaSynthesizer']

						X_train_GCS = []
						Y_train_GCS = []

						for i in range(n):
							synthetic_data = read_arff_file(PATH + '/' + str(i) + '/GaussianCopulaSynthesizer.arff')
							X_train_GCS.append(synthetic_data.drop([classToPredict], axis=1))
							Y_train_GCS.append(synthetic_data[classToPredict])
						
						get_all_stats(n,df,X_train,Y_train,X_train_GCS,Y_train_GCS,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to SDV GaussianCopulaSynthesizer")

					#################################### SDV : TVAESynthesizer #############################################

					try:
						print('\n\t SDV : TVAESynthesizer \t')

						excel_line = [filename,'TVAESynthesizer']

						X_train_TVAES = []
						Y_train_TVAES = []

						for i in range(n):
							synthetic_data = read_arff_file(PATH + '/' + str(i) + '/TVAESynthesizer.arff')
							X_train_TVAES.append(synthetic_data.drop([classToPredict], axis=1))
							Y_train_TVAES.append(synthetic_data[classToPredict])

						get_all_stats(n,df,X_train,Y_train,X_train_TVAES,Y_train_TVAES,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to SDV TVAESynthesizer")

					#################################### SDV : CopulaGANSynthesizer #############################################

					try:
						print('\n\t SDV : CopulaGANSynthesizer \t')

						excel_line = [filename,'CopulaGANSynthesizer']

						X_train_CGAN = []
						Y_train_CGAN = []

						for i in range(n):
							synthetic_data = read_arff_file(PATH + '/' + str(i) + '/CopulaGANSynthesizer.arff')
							X_train_CGAN.append(synthetic_data.drop([classToPredict], axis=1))
							Y_train_CGAN.append(synthetic_data[classToPredict])

						get_all_stats(n,df,X_train,Y_train,X_train_CGAN,Y_train_CGAN,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						print("Dataset not suitable to SDV CopulaGANSynthesizer")

					####################################### YDATA  ################################################

					#################################### YDATA : WGAN #############################################
					try:
						print('\n\t YDATA : WGAN \t')

						excel_line = [filename,'WGAN']

						X_train_WGAN = []
						Y_train_WGAN = []

						for i in range(n):
							synthetic_data = read_arff_file(PATH + '/' + str(i) + '/WGAN.arff')
							X_train_WGAN.append(synthetic_data.drop([classToPredict], axis=1))
							Y_train_WGAN.append(synthetic_data[classToPredict])

						get_all_stats(n,df,X_train,Y_train,X_train_WGAN,Y_train_WGAN,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						print(traceback.format_exc())
						print("Dataset not suitable to WGAN")
					#################################### YDATA : DRAGAN #############################################
					try:
						print('\n\t YDATA : DRAGAN \t')

						excel_line = [filename,'DRAGAN']

						X_train_DRAGAN = []
						Y_train_DRAGAN = []

						for i in range(n):
							synthetic_data = read_arff_file(PATH + '/' + str(i) + '/DRAGAN.arff')
							X_train_DRAGAN.append(synthetic_data.drop([classToPredict], axis=1))
							Y_train_DRAGAN.append(synthetic_data[classToPredict])

						get_all_stats(n,df,X_train,Y_train,X_train_DRAGAN,Y_train_DRAGAN,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						print(traceback.format_exc())
						print("Dataset not suitable to DRAGAN")

					#################################### YDATA : WGAN GP #############################################
					try:
						print('\n\t YDATA : WGAN GP \t')

						excel_line = [filename,'WGAN GP']

						X_train_WGANGP = []
						Y_train_WGANGP = []

						for i in range(n):
							synthetic_data = read_arff_file(PATH + '/' + str(i) + '/WGAN GP.arff')
							X_train_WGANGP.append(synthetic_data.drop([classToPredict], axis=1))
							Y_train_WGANGP.append(synthetic_data[classToPredict])

						get_all_stats(n,df,X_train,Y_train,X_train_WGANGP,Y_train_WGANGP,X_test,Y_test,metadata,excel_line)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						print(traceback.format_exc())
						print("Dataset not suitable to WGAN GP")

				with open(config['general']['METRICS_TEMP'] + dir +'.csv','w',newline='') as csvfile:
					csv_writer = csv.writer(csvfile,delimiter=',')
					for row in excel_lines:
						csv_writer.writerow(row)
				#########################################################################################

	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Elapsed time:", elapsed_time, "seconds")

	return 0

if __name__ == "__main__":
	main()