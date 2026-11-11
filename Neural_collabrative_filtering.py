# %% import libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn import preprocessing, model_selection
from sklearn.metrics import mean_squared_error

# %% import of datasets
ratings = pd.read_csv('ml-latest/ratings.csv')
movies = pd.read_csv('ml-latest/movies.csv')
ratings.head(10)
movies.head(10)

# %%
print(ratings.shape)
print(movies.shape)
# %%
ratings.head(10)

# %%
print(ratings['userId'].nunique())
print(ratings['movieId'].nunique())


# %% Creating Dataset class for PyTorch
class MovieDataset(Dataset):
    
    def __init__(self, users, movies, ratings) -> None:
        super().__init__()
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        movie = self.movies[idx]
        rating = self.ratings[idx]
        user_tensor = torch.tensor(user, dtype=torch.long)
        movie_tensor = torch.tensor(movie, dtype=torch.long)
        rating_tensor = torch.tensor(rating, dtype=torch.float)
        return user_tensor, movie_tensor, rating_tensor

# %%create RECOMMENDER class
class Recsysmodel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size=50):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, user, movie):
        user_embedded = self.user_embedding(user)
        movie_embedded = self.movie_embedding(movie)
        x = torch.cat([user_embedded, movie_embedded], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()
