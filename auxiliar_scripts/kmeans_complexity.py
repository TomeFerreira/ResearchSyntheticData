import os
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.decomposition import PCA

def main():
	current_dir = os.path.dirname(os.path.abspath(__file__))
	config_path = os.path.join(current_dir, '..', 'config.yaml')
	with open(config_path, 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)

	complexity_data = pd.read_csv('dataset_complexity_problexity_' + config['general']['DIR'] + '.csv')
	X = complexity_data.drop('Dataset', axis=1)

	kmeans = KMeans(n_clusters=4)
	kmeans.fit(X)

	pca = PCA(n_components=2)
	X_pca = pca.fit_transform(X)

	colors = ['red', 'blue', 'green', 'orange']
	cluster_colors = [colors[label] for label in kmeans.labels_]

	plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_colors)
	plt.title('KMeans com PCA')
	plt.savefig('kmeans_results/kmeans_pca.png')

	with open('kmeans_results/kmeans_clusters.txt','a') as writer:
		for i in range(4):
			writer.write(f"Cluster {i}\n")
			for _, row in complexity_data[kmeans.labels_ == i].iterrows():
				writer.write(row['Dataset'] + '\n')
			writer.write('\n')
	
	return 0


if __name__ == '__main__':
	main()