import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SlopeOne
import time

# Load MovieLens 1M dataset
data = Dataset.load_builtin('ml-1m')
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Convert to Pandas DataFrame
df = pd.DataFrame(data.raw_ratings, columns=['user', 'item', 'rating', 'timestamp'])

test_sizes = [0.3, 0.2, 0.1]

for test_size in test_sizes:
    trainset, testset = train_test_split(data, test_size=test_size, random_state=42)

    model = SlopeOne()

    start_time = time.time()
    model.fit(trainset)
    train_time = time.time() - start_time

    start_time = time.time()
    predictions = model.test(testset)
    test_time = time.time() - start_time

    mse = mean_squared_error([pred.r_ui for pred in predictions], [pred.est for pred in predictions])
    rmse = np.sqrt(mse)

    print(f"\nCollaborative Filtering with slope one and test size {test_size}:")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Testing Time: {test_time:.2f} seconds")
