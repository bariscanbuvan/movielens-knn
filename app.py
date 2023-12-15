import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNWithZScore, KNNBasic, KNNWithMeans, KNNBaseline

ratings = pd.read_csv('ratings.csv')

reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

test_sizes = [0.1, 0.2, 0.3]

for test_size in test_sizes:
    trainset, testset = train_test_split(data, test_size=test_size, random_state=42)

    for KNNModel in [KNNWithZScore, KNNBasic, KNNWithMeans, KNNBaseline]:
        for similarity_name in ['pearson', 'cosine']:
            sim_options = {'name': similarity_name, 'user_based': True}
            model = KNNModel(sim_options=sim_options)

            model.fit(trainset)

            predictions = model.test(testset)

            mse = mean_squared_error([pred.r_ui for pred in predictions], [pred.est for pred in predictions])
            rmse = np.sqrt(mse)

            print(f"\nCollaborative Filtering with {KNNModel.__name__}, {similarity_name} similarity, and test size {test_size}:")
            print(f"Test MSE: {mse:.4f}")
            print(f"Test RMSE: {rmse:.4f}")
