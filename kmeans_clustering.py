import numpy as np
import argparse
import scipy.io
from sklearn.cluster import KMeans

class MykmeansClustering:
    def __init__(self, dataset_file):
        self.model = None
        self.dataset_file = dataset_file
        self.data = None  # To store the dataset
        self.read_mat()

    def read_mat(self):
        # Read the dataset using scipy.io.loadmat
        mat = scipy.io.loadmat(self.dataset_file)
        
        # Assume the dataset is stored under the key 'X' in the .mat file
        # Modify this key if the dataset is under a different name
        self.data = mat['X']  # Assuming the dataset is in 'X' key

    def model_fit(self, n_clusters=3, max_iter=300):
        '''
        Initialize self.model with KMeans and execute kmeans clustering here
        '''
        # Check if data is loaded
        if self.data is None:
            raise ValueError("Data not loaded correctly. Ensure the dataset is read properly.")

        # Initialize KMeans model
        self.model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
        
        # Fit the model to the data
        self.model.fit(self.data)
        
        # Get the cluster centers
        cluster_centers = self.model.cluster_centers_
        
        # Return the cluster centers
        return cluster_centers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kmeans clustering')
    parser.add_argument('-d','--dataset_file', type=str, default="dataset_q2.mat", help='path to dataset file')
    args = parser.parse_args()
    
    # Initialize the clustering object
    classifier = MykmeansClustering(args.dataset_file)
    
    # Fit the model
    clusters_centers = classifier.model_fit(n_clusters=3)  # You can vary n_clusters for different results
    print("Cluster centers:\n", clusters_centers)
