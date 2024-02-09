import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Read the dataset
def read_data(file_path):
    data = pd.read_csv(file_path)
    selected_data = data[['tenure', 'income', 'employ']]
    return selected_data

# Step 2: Preprocess the data
def preprocess_data(data):
    # Fill missing data with means
    data.fillna(data.mean(), inplace=True)
    # Normalize 'tenure', 'income', and 'employ' using Min-Max scaling
    scaler = MinMaxScaler()
    data[['tenure', 'income', 'employ']] = scaler.fit_transform(data[['tenure', 'income', 'employ']])
    return data

# Step 3: Split the data into training and testing sets
def split_data(data):
    X_train, X_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=17)
    return X_train, X_test

# Step 4: Implement the K-means algorithm
def k_means(data, k, initialization, num_iterations=100):
    # Convert the data array back to a pandas DataFrame
    data = pd.DataFrame(data)
    
    if initialization == 'random':
        # Initialize centroids randomly
        centroids = data.sample(n=k).values

    elif initialization == 'kmeans++':
        # Initialize centroids using KMeans++ algorithm
        centroids = [data.sample().values[0]]
        for _ in range(1, k):
            distances = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in data.values])
            probs = distances/distances.sum()
            cumprobs = probs.cumsum()
            r = np.random.rand()
            for j,p in enumerate(cumprobs):
                if r < p:
                    i = j
                    break
            centroids.append(data.values[i])

    elif initialization == 'first_few_points':
        # Initialize centroids using the first few data points
        centroids = data.head(k).values
    
    elif initialization == 'random_partition':
        # Initialize centroids using random partition method
        shuffled_indices = np.random.permutation(len(data))
        centroids = [data.values[i] for i in range(k)]

    elif initialization == 'random_subset':
        # Initialize centroids using a random subset of data points
        subset_indices = np.random.choice(len(data), k, replace=False)
        centroids = data.iloc[subset_indices].values

    elif initialization == 'kmeans2':
        # Initialize centroids using k-means|| algorithm
        centroids = [data.sample().values[0]]
        for _ in range(1, k):
            distances = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in data.values])
            probs = k * distances / sum(distances)
            cumprobs = probs.cumsum()
            r = np.random.rand()
            for j, p in enumerate(cumprobs):
                if r < p:
                    i = j
                    break
            centroids.append(data.values[i])

    for _ in range(num_iterations):
        distances = np.sqrt(((data.values[:, np.newaxis, :] - centroids)**2).sum(axis=2))
        # Assign each data point to the nearest centroid
        labels = np.argmin(distances, axis=1)
        # Update centroids based on the mean of data points in each cluster
        for i in range(k):
            centroids[i] = np.mean(data.values[labels == i], axis=0)
    return labels, centroids

# Step 5: Visualize the clusters
def visualize_clusters(data, labels, centroids, class_report_df, cm, k, it, exam, initialization):
    plt.figure(figsize=(18, 9))
    colors = ['r', 'g', 'b', 'c', 'm']
    
    plt.subplot(2, 3, 1)
    for i in range(len(centroids)):
        plt.scatter(data[labels == i][:, 0], data[labels == i][:, 1], color=colors[i], label=f'Cluster {i}')
        plt.scatter(centroids[i][0], centroids[i][1], color='k', marker='x', s=100, label=f'Centroid {i}')
    plt.xlabel('Income')
    plt.ylabel('Tenure')
    plt.title('K-means Clustering')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    for i, intra_distance in enumerate(intra_cluster_distance(data, labels, centroids)):
        plt.bar(f'Cluster {i}', intra_distance)
        plt.text(i, intra_distance, f'{intra_distance:.2f}', ha='center', va='bottom')
    plt.ylabel('Intra-cluster Distance')
    plt.title('Intra-cluster Distances')
    
    plt.subplot(2, 3, 3)
    for i, centroid in enumerate(centroids):
        plt.scatter(centroid[0], centroid[1], color=colors[i], marker='x', s=100, label=f'Centroid {i}')
        for j, other_centroid in enumerate(centroids):
            if i != j:
                distance = np.sqrt(((centroid - other_centroid)**2).sum())
                plt.plot([centroid[0], other_centroid[0]], [centroid[1], other_centroid[1]], color='gray', linestyle='--')
                plt.text((centroid[0] + other_centroid[0]) / 2, (centroid[1] + other_centroid[1]) / 2, f'{distance:.2f}', fontsize=8)
    plt.xlabel('Income')
    plt.ylabel('Tenure')
    plt.title('Centroids and Inter-cluster Distances')
    plt.legend()
        
    plt.subplot(2, 3, (4, 5))
    sns.heatmap(class_report_df.iloc[:, :-1].T, annot=True, cmap='Blues', cbar=False, fmt=".2f")
    plt.title('Classification Report')
    
    plt.subplot(2, 3, 6)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.title('Confusion Matrix')
    
    plt.suptitle(f'k = {k}, iteration = {it} , exam = {exam}, initialization = {initialization}', fontsize=16, y=1)
    plt.tight_layout()

    filename = f'img/k{k}_it{it}_exam{exam}_{initialization}.png'
    plt.savefig(filename)
    # plt.show()

# Step 6: Calculate evaluation measures
def calculate_measures(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    class_report_df = pd.DataFrame(classification_report(labels, predictions, output_dict=True)).T
    return cm, class_report_df

# Step 7: Calculate intra-cluster distance
def intra_cluster_distance(data, labels, centroids):
    distances = []
    for i, centroid in enumerate(centroids):
        distance = np.sqrt(((data[labels == i] - centroid)**2).sum(axis=1)).sum()
        distances.append(distance)
    return distances

# Step 8: Calculate inter-cluster distance
def inter_cluster_distance(centroids):
    distances = []
    k = len(centroids)
    for i in range(k):
        for j in range(i+1, k):
            distance = np.sqrt(((centroids[i] - centroids[j])**2).sum())
            distances.append(distance)
    return distances

def main():
    # Step 1:
    data = read_data('Telecust1.csv')
    
    # Step 2:
    data = preprocess_data(data)
    
    # Step 3
    X_train, X_test = split_data(data)
    
    k_values = [2, 3, 4, 5]
    iterations_values = [100, 200, 500, 1000]
    initialization = ['random', 'kmeans++', 'first_few_points', 'random_partition', 'random_subset', 'kmeans2']

    warnings.filterwarnings("ignore", category=UserWarning)
    counter = 1
    for k in k_values:
        for it in iterations_values:
            for init in initialization:
                for i in range(4):
                    if i < 3:
                        selected_columns = np.random.choice(range(len(data.columns)), size=2, replace=False)
                        selected_data = X_train.iloc[:, selected_columns].values
                    else:
                        selected_data = X_train.values

                    labels, centroids = k_means(selected_data, k, initialization=init, num_iterations = it)
                    
                    cm, class_report_df = calculate_measures(labels, np.zeros_like(labels))
                    
                    visualize_clusters(selected_data, labels, centroids, class_report_df, cm, k, it, i+1, init)
                    
                    print(f'{counter} of 386')
                    counter += 1
                    
if __name__ == "__main__":
    main()