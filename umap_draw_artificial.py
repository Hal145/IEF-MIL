import torch
import umap
import matplotlib.pyplot as plt
import numpy as np

def load_features(file_path):
    """Load feature vectors from a .pt file."""
    return torch.load(file_path).numpy()

def plot_umap(features, labels, output_file):
    """Perform UMAP dimensionality reduction and plot the results with a legend."""
    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embedding = reducer.fit_transform(features)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Scatter plot for each class
    plt.scatter(embedding[labels == 0, 0], embedding[labels == 0, 1], c='blue', label='Real', s=5)
    plt.scatter(embedding[labels == 1, 0], embedding[labels == 1, 1], c='red', label='Synthetic', s=5)
    
    plt.title('UMAP of Feature Vectors')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # Add legend
    plt.legend(title='Patch Type')
    
    # Save and show plot
    plt.savefig(output_file)
    plt.show()

def main():
    # Paths to the feature files
    real_features_path = './real_features.pt'
    synthetic_features_path = './synthetic_features.pt'
    
    # Load features
    real_features = load_features(real_features_path)
    synthetic_features = load_features(synthetic_features_path)
    
    # Combine features and create labels
    features = np.concatenate((real_features, synthetic_features), axis=0)
    labels = np.concatenate((np.zeros(real_features.shape[0]), np.ones(synthetic_features.shape[0])), axis=0)
    
    # Plot and save UMAP
    output_file = 'umap_plot.png'
    plot_umap(features, labels, output_file)

if __name__ == '__main__':
    main()

