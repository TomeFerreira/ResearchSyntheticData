import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import os
COMPLEXITY_MEASURES = ['f1', 'f1v', 'f2', 'f3', 'f4', 'l1', 'l2', 'l3', 'n1', 'n2', 'n3','n4', 't1', 'lsc', 'density', 'clsCoef', 'hubs', 't2', 't3', 't4','c1', 'c2']
#COMPLEXITY_MEASURES = ['c1','c2','cls_coef','density','f1.mean','f1v.mean','f2.mean','f3.mean','f4.mean','hubs.mean','l1.mean','l2.mean','l3.mean','lsc','n1','n2.mean','n3.mean','n4.mean','t1','t2','t3','t4']
TECHNIQUES = ['RandomOverSampler','SMOTE','ADASYN','BorderlineSMOTE','SVMSMOTE','KMeansSMOTE','SMOTEENN','SMOTETomek','CTGAN','GaussianCopulaSynthesizer','TVAESynthesizer','WGAN','DRAGAN','WGAN GP']
CLASSIFIERS = ['DT','SVC','RF','XG']
COLORS = ['b','g','r','c','m','y','k','chocolate','lightgreen','orange','slategray']

def plot_diference_scores_by_complexity_ascending(config,technique1,technique2,measure):
	complexity_data = pd.read_csv('dataset_complexity_' + config['general']['DIR'] + '.csv')
	"""
	Mudar as diretorias quando voltar a correr as cenas
	"""
	score_data = pd.read_csv('temp_' + config['general']['DIR'] + '.csv')
	
	rows1 = score_data.loc[score_data['Method'] == technique1]
	rows2 = score_data.loc[score_data['Method'] == technique2]

	complexity_aux = complexity_data.sort_values(measure, na_position='first')

	fig, ax = plt.subplots(figsize=(10, 6))
	plot_dataset = []
	plot_score = []

	for i in complexity_aux.iterrows():
		row1 = rows1.loc[rows1['Dataset']==i[1]['Dataset']]
		row2 = rows2.loc[rows2['Dataset']==i[1]['Dataset']]
		#print(row1)
		#print(row2)
		plot_dataset.append(i[1]['Dataset'])
		if len(row1)==0:
			if len(row2)==0:
				plot_score.append(0)
			else:
				plot_score.append(-row2['(SVC) F1 score MEAN'].values[0])
		else:
			if len(row2)==0:
				plot_score.append(row1['(SVC) F1 score MEAN'].values[0])
			else:
				plot_score.append(row1['(SVC) F1 score MEAN'].values[0]-row2['(SVC) F1 score MEAN'].values[0])
	
	ax.bar(plot_dataset, plot_score)
	ax.set_title(technique1 + ' - ' + technique2)
	ax.set_xlabel('Dataset order by ' + str(measure) + ' (ascending)')
	ax.set_ylabel('F1 score')
	#ax.set_ybound(-1,1)
	ax.set_xticklabels(plot_dataset,rotation=90)
	fig.subplots_adjust(left=None, bottom=0.5, right=None, top=None, wspace=None, hspace=None)

	return fig

def plot_scores_by_complexity_ascending(config,technique,measure):
	complexity_data = pd.read_csv('dataset_complexity_' + config['general']['DIR'] + '.csv')
	"""
	Mudar as diretorias quando voltar a correr as cenas
	"""
	score_data = pd.read_csv('temp_' + config['general']['DIR'] + '.csv')
	
	rows = score_data.loc[score_data['Method'] == technique]

	complexity_aux = complexity_data.sort_values(measure, na_position='first')

	fig, ax = plt.subplots(figsize=(10, 6))
	plot_dataset = []
	plot_score = []

	for i in complexity_aux.iterrows():
		row = rows.loc[rows['Dataset']==i[1]['Dataset']]
		#print(row1)
		#print(row2)
		plot_dataset.append(i[1]['Dataset'])
		if len(row)==0:
			plot_score.append(0)
		else:
			plot_score.append(row['(SVC) F1 score MEAN'].values[0])
	
	ax.plot(plot_dataset, plot_score, marker='s', color='green', linestyle='-')
	ax.set_title(technique)
	ax.set_xlabel('Dataset order by ' + str(measure) + ' (ascending)')
	ax.set_ylabel('F1 score')
	#ax.set_ybound(-1,1)
	ax.set_xticklabels(plot_dataset,rotation=90)
	fig.subplots_adjust(left=None, bottom=0.5, right=None, top=None, wspace=None, hspace=None)

	return fig

def plot_scores_by_complexity_ascending_Baseline_vs_Group(config,techniques,measure):

	complexity_data = pd.read_csv('dataset_complexity_problexity_' + config['general']['DIR'] + '.csv')
	"""
	Mudar as diretorias quando voltar a correr as cenas
	"""
	score_data = pd.read_csv('temp_' + config['general']['DIR'] + '.csv')
	
	fig, ax = plt.subplots(figsize=(10, 6))

	complexity_aux = complexity_data.sort_values(measure, na_position='first')
	complexity_aux.dropna(subset=[measure],inplace=True)

	"""
	Baseline
	"""
	rows = score_data.loc[score_data['Method'] == 'Baseline']

	plot_dataset = []
	plot_score = []

	for i in complexity_aux.iterrows():
		row = rows.loc[rows['Dataset']==i[1]['Dataset']]
		plot_dataset.append(i[1]['Dataset'])
		if len(row)==0:
			plot_score.append(0)
		else:
			plot_score.append(row['(SVC) F1 score MEAN'].values[0])
	
	ax.plot(plot_dataset, plot_score, marker='o', color='dimgray', linestyle='-',label='Baseline')

	"""
	Other techniques
	"""

	for j in range(len(techniques)):
		rows = score_data.loc[score_data['Method'] == techniques[j]]
		plot_score = []

		for i in complexity_aux.iterrows():
			row = rows.loc[rows['Dataset']==i[1]['Dataset']]
			if len(row)==0:
				plot_score.append(0)
			else:
				plot_score.append(row['(SVC) F1 score MEAN'].values[0])
		if j == 1:
			ax.plot(plot_dataset, plot_score, marker='o', color='blue', linestyle='--', label=techniques[j])
		else:
			ax.plot(plot_dataset, plot_score, marker='o', color='orange', linestyle='-.', label=techniques[j])
	
	ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
	ax.set_ybound(0,1)
	ax.set_xlabel('Dataset order by ' + str(measure) + ' (ascending)')
	ax.set_ylabel('F1 score')
	ax.set_xticklabels(plot_dataset,rotation=90)
	fig.subplots_adjust(left=None, bottom=0.5, right=0.8, top=None, wspace=None, hspace=None)

	return fig

def barplot_scores_by_complexity_ascending_Baseline_vs_Group(config,techniques,measure):

	complexity_data = pd.read_csv('dataset_complexity_problexity_' + config['general']['DIR'] + '.csv')
	"""
	Mudar as diretorias quando voltar a correr as cenas
	"""
	score_data = pd.read_csv('temp_' + config['general']['DIR'] + '.csv')

	complexity_aux = complexity_data.sort_values(measure, na_position='first')
	complexity_aux.dropna(subset=[measure],inplace=True)

	plot_data = pd.DataFrame(index=complexity_aux['Dataset'])

	"""
	Baseline
	"""
	baseline_rows = score_data.loc[score_data['Method'] == 'Baseline']
	baseline_scores = []
	for dataset in complexity_aux['Dataset']:
		row = baseline_rows.loc[baseline_rows['Dataset'] == dataset]
		if len(row) == 0:
			baseline_scores.append(0)
		else:
			baseline_scores.append(row['(SVC) F1 score MEAN'].values[0]) 
		
	plot_data['Baseline'] = baseline_scores

	"""
	Other techniques
	"""

	for technique  in techniques:
		technique_rows = score_data.loc[score_data['Method'] == technique]
		technique_scores = []
		for dataset in complexity_aux['Dataset']:
			row = technique_rows.loc[technique_rows['Dataset'] == dataset]
			if len(row) == 0:
				technique_scores.append(0)
			else:
				technique_scores.append(row['(SVC) F1 score MEAN'].values[0]) 
		
		plot_data[technique] = technique_scores
	
	ax = plot_data.plot.bar(figsize=(10, 6), color=['k','green','red'], width=0.9)

	ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
	ax.set_ybound(0,1)
	ax.set_xlabel('Dataset order by ' + str(measure) + ' (ascending)')
	ax.set_ylabel('F1 score')

	fig = ax.get_figure()
	fig.subplots_adjust(left=0.06, bottom=0.4, right=0.8, top=0.9, wspace=None, hspace=None)

	return fig

def main():
	current_dir = os.path.dirname(os.path.abspath(__file__))
	config_path = os.path.join(current_dir, '..', 'config.yaml')
	with open(config_path, 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)

	os.makedirs('graphs', exist_ok=True)

	#plot_diference_scores_by_complexity_ascending(config,'Baseline','SMOTE','f3.mean').show()
	#plot_scores_by_complexity_ascending(config,'TVAESynthesizer','f3.mean').show()
	# for metric in COMPLEXITY_MEASURES:
	# 	fig = plot_scores_by_complexity_ascending_Baseline_vs_Group(config,['RandomOverSampler','SMOTE','BorderlineSMOTE','SMOTETomek'],metric)
	# 	fig.savefig('graphs/problexity_results/oversampling/complexity_'+ metric +'_Oversampling.png')
	# 	plt.close(fig)
	
	# for metric in COMPLEXITY_MEASURES:
	# 	fig = barplot_scores_by_complexity_ascending_Baseline_vs_Group(config,['SMOTETomek','TVAESynthesizer'],metric)
	# 	fig.savefig('graphs/problexity_v2_results/Bar_Plot/TVAE/complexity_'+ metric +'_Bar.png')
	# 	plt.close(fig)
	# 	fig = barplot_scores_by_complexity_ascending_Baseline_vs_Group(config,['SMOTETomek','WGAN GP'],metric)
	# 	fig.savefig('graphs/problexity_v2_results/Bar_Plot/WGAN_GP/complexity_'+ metric +'_Bar.png')
	# 	plt.close(fig)

	for metric in COMPLEXITY_MEASURES:
		fig = plot_scores_by_complexity_ascending_Baseline_vs_Group(config,['SMOTETomek','TVAESynthesizer'],metric)
		fig.savefig('graphs/problexity_v2_results/Line_Plot/TVAE/complexity_'+ metric +'.png')
		plt.close(fig)
		fig = plot_scores_by_complexity_ascending_Baseline_vs_Group(config,['SMOTETomek','WGAN GP'],metric)
		fig.savefig('graphs/problexity_v2_results/Line_Plot/WGAN_GP/complexity_'+ metric +'.png')
		plt.close(fig)

	return 0

if __name__ == '__main__':
	main()