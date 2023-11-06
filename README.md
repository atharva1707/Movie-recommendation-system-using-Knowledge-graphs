# Movie Recommendation System using Graph Neural Networks

This repository contains code to build a movie recommendation system using Graph Neural Networks (GNNs). The code demonstrates the usage of PyTorch Geometric and NetworkX libraries to construct a graph from the MovieLens dataset and apply GNNs for link prediction. The system's objective is to predict links between users and movies based on their features and interactions (ratings) to recommend movies to users.

## Description

The code is presented as a Jupyter Notebook (`Movie_recommendation_system.ipynb`) and performs the following tasks:

1. **Data Loading and Preprocessing**: The notebook begins by downloading the MovieLens dataset from an external source and reads the `movies.csv` and `ratings.csv` files.

2. **Data Processing and Transformation**: The movies' features are derived from the `genres` column in the dataset, and movies are encoded into binary feature vectors. User and movie IDs are mapped to consecutive indices for graph creation.

3. **NetworkX Graph Construction**: Using NetworkX, the code creates a graph representing the relationship between users and movies based on user-movie interactions (ratings). The graph contains nodes of two types: 'user' and 'movie', connected by edges representing users' ratings of movies.

4. **Feature Embedding**: Feature vectors for movies are attached to the graph nodes, and placeholder features for users (a tensor of length 20 with all values -1) are assigned to the user nodes.

5. **Graph Splitting for Training**: The code splits the graph into training, testing, and validation graphs for the link prediction task. It creates multiple instances of training, testing, and validation graphs based on cross-validation.

6. **Graph Neural Network (GNN) Model**: It creates a GNN model using PyTorch Geometric that predicts missing ratings. The model architecture involves an initial feature embedding layer for users and movies and two layers of SAGEConv graph convolution.

7. **Training the GNN Model**: The GNN model is trained using the training graph and the link prediction task to predict movie ratings for users.

8. **Evaluation and Metrics**: The notebook includes the loss computation for each epoch during training and provides performance evaluation metrics.

## Dependencies

The code requires the following Python libraries:

- PyTorch
- PyTorch Geometric
- NetworkX
- Pandas
- Matplotlib
- Scikit-learn

## Usage

1. Clone the repository and download the MovieLens dataset.
2. Open and execute the Jupyter Notebook (`Movie_recommendation_system.ipynb`).
3. Follow the instructions in the notebook to run the code and perform the steps mentioned above.
4. View the results and evaluation metrics directly in the notebook.


