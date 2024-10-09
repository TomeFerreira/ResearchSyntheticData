import csv
import yaml
import pickle
import numpy as np
import os
import traceback
import time
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from joblib import Parallel, delayed
from readers.dat_reader import read_dat
from readers.arff_reader_writer import read_arff_file

N_JOBS = 4

def show_stats(n,Y_pred,Y_pred_SVC,Y_pred_RF,Y_pred_XG,Y_test,excel_line,baseline_scores):

	auc = np.zeros(n)
	auc_SVC = np.zeros(n)
	auc_RF = np.zeros(n)
	auc_XG = np.zeros(n)
	for i in range(n):
		auc[i] = metrics.f1_score(np.array(Y_test[i]), Y_pred[i], average='binary')
		auc_SVC[i] = metrics.f1_score(np.array(Y_test[i]), Y_pred_SVC[i], average='binary')
		auc_RF[i] = metrics.f1_score(np.array(Y_test[i]), Y_pred_RF[i], average='binary')
		auc_XG[i] = metrics.f1_score(np.array(Y_test[i]), Y_pred_XG[i], average='binary')

	auc_avg = np.mean(auc)
	auc_std = np.std(auc)
	print('Avg: ' + str(auc_avg))
	print('Std: ' + str(auc_std))
	auc_avg_SVC = np.mean(auc_SVC)
	auc_std_SVC = np.std(auc_SVC)
	auc_avg_RF = np.mean(auc_RF)
	auc_std_RF = np.std(auc_RF)
	auc_avg_XG = np.mean(auc_XG)
	auc_std_XG = np.std(auc_XG)

	excel_line.append(str(auc_avg))
	excel_line.append(str(auc_std))

	if(len(baseline_scores)!=4):
		baseline_scores.append(auc_avg)
		excel_line.append(0)
	else:
		excel_line.append(auc_avg-baseline_scores[0])
		print(auc_avg-baseline_scores[0])

	excel_line.append(str(auc_avg_SVC))
	excel_line.append(str(auc_std_SVC))
	if(len(baseline_scores)!=4):
		baseline_scores.append(auc_avg_SVC)
		excel_line.append(0)
	else:
		excel_line.append(auc_avg_SVC-baseline_scores[1])
		print(auc_avg_SVC-baseline_scores[1])

	excel_line.append(str(auc_avg_RF))
	excel_line.append(str(auc_std_RF))
	if(len(baseline_scores)!=4):
		baseline_scores.append(auc_avg_RF)
		excel_line.append(0)
	else:
		excel_line.append(auc_avg_RF-baseline_scores[2])
		print(auc_avg_RF-baseline_scores[2])

	excel_line.append(str(auc_avg_XG))
	excel_line.append(str(auc_std_XG))
	if(len(baseline_scores)!=4):
		baseline_scores.append(auc_avg_XG)
		excel_line.append(0)
	else:
		excel_line.append(auc_avg_XG-baseline_scores[3])
		print(auc_avg_XG-baseline_scores[3])

	return

def get_all_stats(n,synthetic_Y_list,holdout_Y_list, Y_pred, Y_pred_SVC, Y_pred_RF, Y_pred_XG, excel_line,baseline_scores):

	class_imbal_ratio(n,synthetic_Y_list,excel_line)
	show_stats(n,Y_pred,Y_pred_SVC,Y_pred_RF,Y_pred_XG,holdout_Y_list,excel_line,baseline_scores)

	return

def fit_and_predict(grid, X_train, Y_train, X_test, i):

	model = DecisionTreeClassifier()
	rf = RandomForestClassifier(n_jobs=N_JOBS)
	svc = SVC(**grid.best_params_)
	xg = xgb.XGBRegressor(
			n_jobs=N_JOBS, tree_method = "hist",n_estimators=100
		)
	# print('Train: ' + str(i) + '\n' + str(X_train))
	# print('Test: ' + str(i) + '\n' + str(X_test))

	model.fit(X_train, Y_train)
	Y_pred = model.predict(X_test)
	rf.fit(X_train, Y_train)
	Y_pred_RF = rf.predict(X_test)
	svc.fit(X_train, Y_train)
	Y_pred_SVC = svc.predict(X_test)
	xg.fit(X_train,Y_train)
	Y_pred_XG_aux = xg.predict(X_test)
	Y_pred_XG = np.where(Y_pred_XG_aux >= 0.5, 1, 0)
	# print(str(Y_pred) + '\n' + str(Y_pred_RF) + '\n' + str(Y_pred_SVC) + '\n' + str(Y_pred_XG))
	return Y_pred, Y_pred_RF, Y_pred_SVC, Y_pred_XG, i

def class_imbal_ratio(n, Y_pred, excel_line):
	imbal = np.empty(n)
	for i in range(n):
		class_counts = Y_pred[i].value_counts()
		imbalance_ratio = class_counts.max() / class_counts.min()
		imbal[i] = imbalance_ratio
		
	excel_line.append(np.mean(imbal))
	excel_line.append(np.std(imbal))

def main():

	with open('config.yaml', 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)

	start_time = time.time()

	PATH = config['general']['DATASET_PATH']

	classToPredict = 'Class'
	n = 10

	param_grid = {'C': [0.1, 1, 10, 100],  
			  'gamma': [1, 0.1, 0.01, 0.001], 
			  'kernel': ['rbf','linear']}

	grid = GridSearchCV(SVC(), param_grid, refit = True, n_jobs=N_JOBS)

	for root,dirs,files in os.walk(PATH):
		for dir in dirs:
			
			PATH = config['general']['PATH']
			
			if(not os.path.isdir(PATH)):
				os.mkdir(PATH)

			excel_lines = []
			
			if(os.path.exists(config['general']['SCORES_TEMP'] + dir +'.csv')):

				with open(config['general']['SCORES_TEMP'] + dir +'.csv',newline='') as csvfile:
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

					PATH_GRID = 'Grid_Search/' + str(filename) + '_best_params.pkl'

					PATH = config['general']['PATH'] + '/' + str(filename)

					print('\n' + file + '\n')
					
					if(file.endswith(".arff")):
						df = read_arff_file(dir_path + '/' + file)
					else:
						df = read_dat(dir_path + '/' + file)

					classToPredict = df.columns[len(df.columns)-1]

					X = df.drop([classToPredict],axis=1)
					Y = df[classToPredict]

					if(not os.path.isdir('Grid_Search')):
						os.mkdir('Grid_Search')

					if(not os.path.isfile(PATH_GRID)):
						grid.fit(X,Y)
						with open(PATH_GRID, 'wb') as f:
							pickle.dump(grid,f)
					else:
						with open(PATH_GRID, 'rb') as f:
							grid = pickle.load(f)

					X_train = []
					Y_train = []
					X_test = []
					Y_test = []
					baseline_scores = []

					#Cria 10 divisões diferentes do dataset
					for i in range(n):
						Train = read_arff_file(PATH + '/' + str(i) + '/Baseline.arff')
						Test = read_arff_file(PATH + '/' + str(i) + '/Test.arff') 
						X_train.append(Train.drop([classToPredict],axis=1))
						X_test.append(Test.drop([classToPredict],axis=1))
						Y_train.append(Train[classToPredict])
						Y_test.append(Test[classToPredict])					

					#################################### Baseline #############################################

					print('\t BASELINE \t')

					excel_line = [filename,'Baseline']

					results = Parallel(n_jobs=N_JOBS)(
						delayed(fit_and_predict)(grid, X_train[i], Y_train[i], X_test[i], i) 
						for i in range(n))
					
					Y_pred_list, Y_pred_RF_list, Y_pred_SVC_list, Y_pred_XGB_list, order_list = zip(*results)

					Y_pred = np.array(Y_pred_list)
					Y_pred_RF = np.array(Y_pred_RF_list)
					Y_pred_SVC = np.array(Y_pred_SVC_list)
					Y_pred_XGB = np.array(Y_pred_XGB_list)
					order = np.array(order_list)

					sorted_indices = np.argsort(order)
					Y_pred_sorted = Y_pred[sorted_indices]
					Y_pred_RF_sorted = Y_pred_RF[sorted_indices]
					Y_pred_SVC_sorted = Y_pred_SVC[sorted_indices]
					Y_pred_XGB_sorted = Y_pred_XGB[sorted_indices]

					get_all_stats(n,Y_train,Y_test,Y_pred_sorted,Y_pred_SVC_sorted,Y_pred_RF_sorted,Y_pred_XGB_sorted,excel_line,baseline_scores)

					excel_lines.append(excel_line)

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
	   
						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_ROS[i], Y_train_ROS[i], X_test[i], i) 
							for i in range(n))

						Y_pred_ROS_list, Y_pred_ROS_RF_list, Y_pred_ROS_SVC_list, Y_pred_ROS_XGB_list, order_list = zip(*results)

						Y_pred_ROS = np.array(Y_pred_ROS_list)
						Y_pred_ROS_RF = np.array(Y_pred_ROS_RF_list)
						Y_pred_ROS_SVC = np.array(Y_pred_ROS_SVC_list)
						Y_pred_ROS_XGB = np.array(Y_pred_ROS_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_ROS_sorted = Y_pred_ROS[sorted_indices]
						Y_pred_ROS_RF_sorted = Y_pred_ROS_RF[sorted_indices]
						Y_pred_ROS_SVC_sorted = Y_pred_ROS_SVC[sorted_indices]
						Y_pred_ROS_XGB_sorted = Y_pred_ROS_XGB[sorted_indices]

						get_all_stats(n,Y_train_ROS,Y_test,Y_pred_ROS_sorted,Y_pred_ROS_SVC_sorted,Y_pred_ROS_RF_sorted,Y_pred_ROS_XGB_sorted,excel_line,baseline_scores)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
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

						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_SMOTE[i], Y_train_SMOTE[i], X_test[i], i) 
							for i in range(n))

						Y_pred_SMOTE_list, Y_pred_SMOTE_RF_list, Y_pred_SMOTE_SVC_list, Y_pred_SMOTE_XGB_list, order_list = zip(*results)

						Y_pred_SMOTE = np.array(Y_pred_SMOTE_list)
						Y_pred_SMOTE_RF = np.array(Y_pred_SMOTE_RF_list)
						Y_pred_SMOTE_SVC = np.array(Y_pred_SMOTE_SVC_list)
						Y_pred_SMOTE_XGB = np.array(Y_pred_SMOTE_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_SMOTE_sorted = Y_pred_SMOTE[sorted_indices]
						Y_pred_SMOTE_RF_sorted = Y_pred_SMOTE_RF[sorted_indices]
						Y_pred_SMOTE_SVC_sorted = Y_pred_SMOTE_SVC[sorted_indices]
						Y_pred_SMOTE_XGB_sorted = Y_pred_SMOTE_XGB[sorted_indices]

						get_all_stats(n,Y_train_SMOTE,Y_test,Y_pred_SMOTE_sorted,Y_pred_SMOTE_SVC_sorted,Y_pred_SMOTE_RF_sorted,Y_pred_SMOTE_XGB_sorted,excel_line,baseline_scores)

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

						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_ADASYN[i], Y_train_ADASYN[i], X_test[i], i) 
							for i in range(n))

						Y_pred_ADASYN_list, Y_pred_ADASYN_RF_list, Y_pred_ADASYN_SVC_list, Y_pred_ADASYN_XGB_list, order_list = zip(*results)

						Y_pred_ADASYN = np.array(Y_pred_ADASYN_list)
						Y_pred_ADASYN_RF = np.array(Y_pred_ADASYN_RF_list)
						Y_pred_ADASYN_SVC = np.array(Y_pred_ADASYN_SVC_list)
						Y_pred_ADASYN_XGB = np.array(Y_pred_ADASYN_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_ADASYN_sorted = Y_pred_ADASYN[sorted_indices]
						Y_pred_ADASYN_RF_sorted = Y_pred_ADASYN_RF[sorted_indices]
						Y_pred_ADASYN_SVC_sorted = Y_pred_ADASYN_SVC[sorted_indices]
						Y_pred_ADASYN_XGB_sorted = Y_pred_ADASYN_XGB[sorted_indices]

						get_all_stats(n,Y_train_ADASYN,Y_test,Y_pred_ADASYN_sorted,Y_pred_ADASYN_SVC_sorted,Y_pred_ADASYN_RF_sorted,Y_pred_ADASYN_XGB_sorted,excel_line,baseline_scores)

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
	   
						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_BSMOTE[i], Y_train_BSMOTE[i], X_test[i], i) 
							for i in range(n))
						
						Y_pred_BSMOTE_list, Y_pred_BSMOTE_RF_list, Y_pred_BSMOTE_SVC_list, Y_pred_BSMOTE_XGB_list, order_list = zip(*results)

						Y_pred_BSMOTE = np.array(Y_pred_BSMOTE_list)
						Y_pred_BSMOTE_RF = np.array(Y_pred_BSMOTE_RF_list)
						Y_pred_BSMOTE_SVC = np.array(Y_pred_BSMOTE_SVC_list)
						Y_pred_BSMOTE_XGB = np.array(Y_pred_BSMOTE_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_BSMOTE_sorted = Y_pred_BSMOTE[sorted_indices]
						Y_pred_BSMOTE_RF_sorted = Y_pred_BSMOTE_RF[sorted_indices]
						Y_pred_BSMOTE_SVC_sorted = Y_pred_BSMOTE_SVC[sorted_indices]
						Y_pred_BSMOTE_XGB_sorted = Y_pred_BSMOTE_XGB[sorted_indices]

						get_all_stats(n,Y_train_BSMOTE,Y_test,Y_pred_BSMOTE_sorted,Y_pred_BSMOTE_SVC_sorted,Y_pred_BSMOTE_RF_sorted,Y_pred_BSMOTE_XGB_sorted,excel_line,baseline_scores)

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
	   
						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_SVMSMOTE[i], Y_train_SVMSMOTE[i], X_test[i], i) 
							for i in range(n))
						
						Y_pred_SVMSMOTE_list, Y_pred_SVMSMOTE_RF_list, Y_pred_SVMSMOTE_SVC_list, Y_pred_SVMSMOTE_XGB_list, order_list = zip(*results)

						Y_pred_SVMSMOTE = np.array(Y_pred_SVMSMOTE_list)
						Y_pred_SVMSMOTE_RF = np.array(Y_pred_SVMSMOTE_RF_list)
						Y_pred_SVMSMOTE_SVC = np.array(Y_pred_SVMSMOTE_SVC_list)
						Y_pred_SVMSMOTE_XGB = np.array(Y_pred_SVMSMOTE_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_SVMSMOTE_sorted = Y_pred_SVMSMOTE[sorted_indices]
						Y_pred_SVMSMOTE_RF_sorted = Y_pred_SVMSMOTE_RF[sorted_indices]
						Y_pred_SVMSMOTE_SVC_sorted = Y_pred_SVMSMOTE_SVC[sorted_indices]
						Y_pred_SVMSMOTE_XGB_sorted = Y_pred_SVMSMOTE_XGB[sorted_indices]

						get_all_stats(n,Y_train_SVMSMOTE,Y_test,Y_pred_SVMSMOTE_sorted,Y_pred_SVMSMOTE_SVC_sorted,Y_pred_SVMSMOTE_RF_sorted,Y_pred_SVMSMOTE_XGB_sorted,excel_line,baseline_scores)

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

						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_KMSMOTE[i], Y_train_KMSMOTE[i], X_test[i], i) 
							for i in range(n))
						
						Y_pred_KMSMOTE_list, Y_pred_KMSMOTE_RF_list, Y_pred_KMSMOTE_SVC_list, Y_pred_KMSMOTE_XGB_list, order_list = zip(*results)

						Y_pred_KMSMOTE = np.array(Y_pred_KMSMOTE_list)
						Y_pred_KMSMOTE_RF = np.array(Y_pred_KMSMOTE_RF_list)
						Y_pred_KMSMOTE_SVC = np.array(Y_pred_KMSMOTE_SVC_list)
						Y_pred_KMSMOTE_XGB = np.array(Y_pred_KMSMOTE_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_KMSMOTE_sorted = Y_pred_KMSMOTE[sorted_indices]
						Y_pred_KMSMOTE_RF_sorted = Y_pred_KMSMOTE_RF[sorted_indices]
						Y_pred_KMSMOTE_SVC_sorted = Y_pred_KMSMOTE_SVC[sorted_indices]
						Y_pred_KMSMOTE_XGB_sorted = Y_pred_KMSMOTE_XGB[sorted_indices]

						get_all_stats(n,Y_train_KMSMOTE,Y_test,Y_pred_KMSMOTE_sorted,Y_pred_KMSMOTE_SVC_sorted,Y_pred_KMSMOTE_RF_sorted,Y_pred_KMSMOTE_XGB_sorted,excel_line,baseline_scores)

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

						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_SMOTEENN[i], Y_train_SMOTEENN[i], X_test[i], i) 
							for i in range(n))
						
						Y_pred_SMOTEENN_list, Y_pred_SMOTEENN_RF_list, Y_pred_SMOTEENN_SVC_list, Y_pred_SMOTEENN_XGB_list, order_list = zip(*results)

						Y_pred_SMOTEENN = np.array(Y_pred_SMOTEENN_list)
						Y_pred_SMOTEENN_RF = np.array(Y_pred_SMOTEENN_RF_list)
						Y_pred_SMOTEENN_SVC = np.array(Y_pred_SMOTEENN_SVC_list)
						Y_pred_SMOTEENN_XGB = np.array(Y_pred_SMOTEENN_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_SMOTEENN_sorted = Y_pred_SMOTEENN[sorted_indices]
						Y_pred_SMOTEENN_RF_sorted = Y_pred_SMOTEENN_RF[sorted_indices]
						Y_pred_SMOTEENN_SVC_sorted = Y_pred_SMOTEENN_SVC[sorted_indices]
						Y_pred_SMOTEENN_XGB_sorted = Y_pred_SMOTEENN_XGB[sorted_indices]

						get_all_stats(n,Y_train_SMOTEENN,Y_test,Y_pred_SMOTEENN_sorted,Y_pred_SMOTEENN_SVC_sorted,Y_pred_SMOTEENN_RF_sorted,Y_pred_SMOTEENN_XGB_sorted,excel_line,baseline_scores)

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

						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_SMOTETL[i], Y_train_SMOTETL[i], X_test[i], i) 
							for i in range(n))
						
						Y_pred_SMOTETL_list, Y_pred_SMOTETL_RF_list, Y_pred_SMOTETL_SVC_list, Y_pred_SMOTETL_XGB_list, order_list = zip(*results)

						Y_pred_SMOTETL = np.array(Y_pred_SMOTETL_list)
						Y_pred_SMOTETL_RF = np.array(Y_pred_SMOTETL_RF_list)
						Y_pred_SMOTETL_SVC = np.array(Y_pred_SMOTETL_SVC_list)
						Y_pred_SMOTETL_XGB = np.array(Y_pred_SMOTETL_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_SMOTETL_sorted = Y_pred_SMOTETL[sorted_indices]
						Y_pred_SMOTETL_RF_sorted = Y_pred_SMOTETL_RF[sorted_indices]
						Y_pred_SMOTETL_SVC_sorted = Y_pred_SMOTETL_SVC[sorted_indices]
						Y_pred_SMOTETL_XGB_sorted = Y_pred_SMOTETL_XGB[sorted_indices]

						get_all_stats(n,Y_train_SMOTETL,Y_test,Y_pred_SMOTETL_sorted,Y_pred_SMOTETL_SVC_sorted,Y_pred_SMOTETL_RF_sorted,Y_pred_SMOTETL_XGB_sorted,excel_line,baseline_scores)

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

						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_CTGAN[i], Y_train_CTGAN[i], X_test[i], i) 
							for i in range(n))
						
						Y_pred_CTGAN_list, Y_pred_CTGAN_RF_list, Y_pred_CTGAN_SVC_list, Y_pred_CTGAN_XGB_list, order_list = zip(*results)

						Y_pred_CTGAN = np.array(Y_pred_CTGAN_list)
						Y_pred_CTGAN_RF = np.array(Y_pred_CTGAN_RF_list)
						Y_pred_CTGAN_SVC = np.array(Y_pred_CTGAN_SVC_list)
						Y_pred_CTGAN_XGB = np.array(Y_pred_CTGAN_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_CTGAN_sorted = Y_pred_CTGAN[sorted_indices]
						Y_pred_CTGAN_RF_sorted = Y_pred_CTGAN_RF[sorted_indices]
						Y_pred_CTGAN_SVC_sorted = Y_pred_CTGAN_SVC[sorted_indices]
						Y_pred_CTGAN_XGB_sorted = Y_pred_CTGAN_XGB[sorted_indices]

						get_all_stats(n,Y_train_CTGAN,Y_test,Y_pred_CTGAN_sorted,Y_pred_CTGAN_SVC_sorted,Y_pred_CTGAN_RF_sorted,Y_pred_CTGAN_XGB_sorted,excel_line,baseline_scores)

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

						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_GCS[i], Y_train_GCS[i], X_test[i], i) 
							for i in range(n))
						
						Y_pred_GCS_list, Y_pred_GCS_RF_list, Y_pred_GCS_SVC_list, Y_pred_GCS_XGB_list, order_list = zip(*results)

						Y_pred_GCS = np.array(Y_pred_GCS_list)
						Y_pred_GCS_RF = np.array(Y_pred_GCS_RF_list)
						Y_pred_GCS_SVC = np.array(Y_pred_GCS_SVC_list)
						Y_pred_GCS_XGB = np.array(Y_pred_GCS_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_GCS_sorted = Y_pred_GCS[sorted_indices]
						Y_pred_GCS_RF_sorted = Y_pred_GCS_RF[sorted_indices]
						Y_pred_GCS_SVC_sorted = Y_pred_GCS_SVC[sorted_indices]
						Y_pred_GCS_XGB_sorted = Y_pred_GCS_XGB[sorted_indices]
						
						get_all_stats(n,Y_train_GCS,Y_test,Y_pred_GCS_sorted,Y_pred_GCS_SVC_sorted,Y_pred_GCS_RF_sorted,Y_pred_GCS_XGB_sorted,excel_line,baseline_scores)

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

						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_TVAES[i], Y_train_TVAES[i], X_test[i], i) 
							for i in range(n))
						
						Y_pred_TVAES_list, Y_pred_TVAES_RF_list, Y_pred_TVAES_SVC_list, Y_pred_TVAES_XGB_list, order_list = zip(*results)

						Y_pred_TVAES = np.array(Y_pred_TVAES_list)
						Y_pred_TVAES_RF = np.array(Y_pred_TVAES_RF_list)
						Y_pred_TVAES_SVC = np.array(Y_pred_TVAES_SVC_list)
						Y_pred_TVAES_XGB = np.array(Y_pred_TVAES_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_TVAES_sorted = Y_pred_TVAES[sorted_indices]
						Y_pred_TVAES_RF_sorted = Y_pred_TVAES_RF[sorted_indices]
						Y_pred_TVAES_SVC_sorted = Y_pred_TVAES_SVC[sorted_indices]
						Y_pred_TVAES_XGB_sorted = Y_pred_TVAES_XGB[sorted_indices]

						get_all_stats(n,Y_train_TVAES,Y_test,Y_pred_TVAES_sorted,Y_pred_TVAES_SVC_sorted,Y_pred_TVAES_RF_sorted,Y_pred_TVAES_XGB_sorted,excel_line,baseline_scores)

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

						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_CGAN[i], Y_train_CGAN[i], X_test[i], i) 
							for i in range(n))
						
						Y_pred_CGAN_list, Y_pred_CGAN_RF_list, Y_pred_CGAN_SVC_list, Y_pred_CGAN_XGB_list, order_list = zip(*results)

						Y_pred_CGAN = np.array(Y_pred_CGAN_list)
						Y_pred_CGAN_RF = np.array(Y_pred_CGAN_RF_list)
						Y_pred_CGAN_SVC = np.array(Y_pred_CGAN_SVC_list)
						Y_pred_CGAN_XGB = np.array(Y_pred_CGAN_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_CGAN_sorted = Y_pred_CGAN[sorted_indices]
						Y_pred_CGAN_RF_sorted = Y_pred_CGAN_RF[sorted_indices]
						Y_pred_CGAN_SVC_sorted = Y_pred_CGAN_SVC[sorted_indices]
						Y_pred_CGAN_XGB_sorted = Y_pred_CGAN_XGB[sorted_indices]

						get_all_stats(n,Y_train_CGAN,Y_test,Y_pred_CGAN_sorted,Y_pred_CGAN_SVC_sorted,Y_pred_CGAN_RF_sorted,Y_pred_CGAN_XGB_sorted,excel_line,baseline_scores)

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
							
						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_WGAN[i], Y_train_WGAN[i], X_test[i], i) 
							for i in range(n))
						
						Y_pred_WGAN_list, Y_pred_WGAN_RF_list, Y_pred_WGAN_SVC_list, Y_pred_WGAN_XGB_list, order_list = zip(*results)

						Y_pred_WGAN = np.array(Y_pred_WGAN_list)
						Y_pred_WGAN_RF = np.array(Y_pred_WGAN_RF_list)
						Y_pred_WGAN_SVC = np.array(Y_pred_WGAN_SVC_list)
						Y_pred_WGAN_XGB = np.array(Y_pred_WGAN_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_WGAN_sorted = Y_pred_WGAN[sorted_indices]
						Y_pred_WGAN_RF_sorted = Y_pred_WGAN_RF[sorted_indices]
						Y_pred_WGAN_SVC_sorted = Y_pred_WGAN_SVC[sorted_indices]
						Y_pred_WGAN_XGB_sorted = Y_pred_WGAN_XGB[sorted_indices]

						get_all_stats(n,Y_train_WGAN,Y_test,Y_pred_WGAN_sorted,Y_pred_WGAN_SVC_sorted,Y_pred_WGAN_RF_sorted,Y_pred_WGAN_XGB_sorted,excel_line,baseline_scores)

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

						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_DRAGAN[i], Y_train_DRAGAN[i], X_test[i], i) 
							for i in range(n))
						
						Y_pred_DRAGAN_list, Y_pred_DRAGAN_RF_list, Y_pred_DRAGAN_SVC_list, Y_pred_DRAGAN_XGB_list, order_list = zip(*results)

						Y_pred_DRAGAN = np.array(Y_pred_DRAGAN_list)
						Y_pred_DRAGAN_RF = np.array(Y_pred_DRAGAN_RF_list)
						Y_pred_DRAGAN_SVC = np.array(Y_pred_DRAGAN_SVC_list)
						Y_pred_DRAGAN_XGB = np.array(Y_pred_DRAGAN_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_DRAGAN_sorted = Y_pred_DRAGAN[sorted_indices]
						Y_pred_DRAGAN_RF_sorted = Y_pred_DRAGAN_RF[sorted_indices]
						Y_pred_DRAGAN_SVC_sorted = Y_pred_DRAGAN_SVC[sorted_indices]
						Y_pred_DRAGAN_XGB_sorted = Y_pred_DRAGAN_XGB[sorted_indices]

						get_all_stats(n,Y_train_DRAGAN,Y_test,Y_pred_DRAGAN_sorted,Y_pred_DRAGAN_SVC_sorted,Y_pred_DRAGAN_RF_sorted,Y_pred_DRAGAN_XGB_sorted,excel_line,baseline_scores)

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

						results = Parallel(n_jobs=N_JOBS)(
							delayed(fit_and_predict)(grid, X_train_WGANGP[i], Y_train_WGANGP[i], X_test[i], i) 
							for i in range(n))
						
						Y_pred_WGANGP_list, Y_pred_WGANGP_RF_list, Y_pred_WGANGP_SVC_list, Y_pred_WGANGP_XGB_list, order_list = zip(*results)

						Y_pred_WGANGP = np.array(Y_pred_WGANGP_list)
						Y_pred_WGANGP_RF = np.array(Y_pred_WGANGP_RF_list)
						Y_pred_WGANGP_SVC = np.array(Y_pred_WGANGP_SVC_list)
						Y_pred_WGANGP_XGB = np.array(Y_pred_WGANGP_XGB_list)
						order = np.array(order_list)

						sorted_indices = np.argsort(order)
						Y_pred_WGANGP_sorted = Y_pred_WGANGP[sorted_indices]
						Y_pred_WGANGP_RF_sorted = Y_pred_WGANGP_RF[sorted_indices]
						Y_pred_WGANGP_SVC_sorted = Y_pred_WGANGP_SVC[sorted_indices]
						Y_pred_WGANGP_XGB_sorted = Y_pred_WGANGP_XGB[sorted_indices]

						get_all_stats(n,Y_train_WGANGP,Y_test,Y_pred_WGANGP_sorted,Y_pred_WGANGP_SVC_sorted,Y_pred_WGANGP_RF_sorted,Y_pred_WGANGP_XGB_sorted,excel_line,baseline_scores)

						excel_lines.append(excel_line)

					except KeyboardInterrupt:
						return
					except:
						print(traceback.format_exc())
						print("Dataset not suitable to WGAN GP")

				with open(config['general']['SCORES_TEMP']+ dir +'.csv','w',newline='') as csvfile:
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
