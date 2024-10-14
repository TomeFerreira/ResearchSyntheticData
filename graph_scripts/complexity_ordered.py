import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import os

COMPLEXITY_MEASURES = ['c1','c2','cls_coef','density','f1.mean','f1.sd','f1v.mean','f1v.sd','f2.mean','f2.sd','f3.mean','f3.sd','f4.mean','f4.sd','hubs.mean','hubs.sd','l1.mean','l1.sd','l2.mean','l2.sd','l3.mean','l3.sd','lsc','n1','n2.mean','n2.sd','n3.mean','n3.sd','n4.mean','n4.sd','t1','t2','t3','t4']
TECHNIQUES = ['RandomOverSampler','SMOTE','ADASYN','BorderlineSMOTE','SVMSMOTE','KMeansSMOTE','SMOTEENN','SMOTETomek','CTGAN','GaussianCopulaSynthesizer','TVAESynthesizer','WGAN','DRAGAN','WGAN GP']
CLASSIFIERS = ['DT','SVC','RF','XG']

def plot_scores_by_complexity_ascending(config,technique,measure):
	complexity_data = pd.read_csv('dataset_complexity_' + config['general']['DIR'] + '.csv')
	"""
	Mudar as diretorias quando voltar a correr as cenas
	"""
	score_data = pd.read_csv('temp_' + config['general']['DIR'] + '.csv')
	
	rows = score_data.loc[score_data['Method'] == technique]

	complexity_aux = complexity_data.sort_values(measure, na_position='first')

	fig = plt.figure()
	plot_dataset = []
	plot_score = []

	for i in complexity_aux.iterrows():
		plot_dataset.append(i[1]['Dataset'])
		row = rows.loc[rows['Dataset']==i[1]['Dataset']]
		plot_score.append(row['(SVC) F1 score MEAN'].values[0])
	
	plt.bar(plot_dataset, plot_score)
	plt.xlabel('Dataset order by ' + str(measure) + ' (ascending)')
	plt.ylabel('F1 score')
	plt.xticks(rotation=90)
	plt.subplots_adjust(left=None, bottom=0.5, right=None, top=None, wspace=None, hspace=None)
	plt.show()

	return 0

def main():
	current_dir = os.path.dirname(os.path.abspath(__file__))
	config_path = os.path.join(current_dir, '..', 'config.yaml')
	with open(config_path, 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)

	plot_scores_by_complexity_ascending(config,'SMOTE','f2.mean')
	return 0

if __name__ == '__main__':
	main()