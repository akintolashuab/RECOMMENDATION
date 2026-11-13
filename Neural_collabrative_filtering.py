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
        self.fc1 = nn.Linear(embedding_size * 2, 1)
        #self.fc2 = nn.Linear(128, 64)
        #self.fc3 = nn.Linear(64, 1)
        #self.relu = nn.ReLU()
        
    def forward(self, user, movie):
        user_embedded = self.user_embedding(user)
        movie_embedded = self.movie_embedding(movie)
        x = torch.cat([user_embedded, movie_embedded], dim=1)
        x = self.fc1(x)
        #x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        #x = self.fc3(x)
        return x.squeeze()

# %% Preprocessing the data
# Encode userId and movieId to continuous integers
user_encoder = preprocessing.LabelEncoder()
movie_encoder = preprocessing.LabelEncoder()

ratings['userId'] = user_encoder.fit_transform(ratings['userId'])
ratings['movieId'] = movie_encoder.fit_transform(ratings['movieId'])

#%%
ratings

# %% Split the data into training and testing sets
train_data, test_data = model_selection.train_test_split(ratings, 
                        test_size=0.2, random_state=42, shuffle=True)

#%% Create Dataset instances for training and testing
train_dataset = MovieDataset( users = train_data['userId'].values,
                              movies = train_data['movieId'].values,
                              ratings = train_data['rating'].values)

validation_dataset = MovieDataset( users = test_data['userId'].values,
                                   movies = test_data['movieId'].values,
                                      ratings = test_data['rating'].values)


# %% data loaders
batch_size = 3000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# %% Model training
len(user_encoder.classes_)

#%%
len(movie_encoder.classes_)


# %%
model = Recsysmodel(num_users=len(user_encoder.classes_), 
                    num_movies=len(movie_encoder.classes_),
                    embedding_size=50)
num_epochs = 5
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

#%%
from tqdm import tqdm
import time

for epoch in range(num_epochs):

    start_time = time.time()
    running_loss = 0

    model.train()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, (user, movie, rating) in enumerate(progress_bar):

        optimizer.zero_grad()

        predictions = model(user, movie)

        y_true = rating.float()

        loss = loss_fn(predictions, y_true)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    print(f"Epoch completed in {(time.time()-start_time)/60:.2f} minutes")

#%% Training loop
import time
start_time = time.time()
for epoch in range(num_epochs):
    start_time = time.time()
    for batch_idx, (user, movie, rating) in enumerate(train_loader):
        optimizer.zero_grad()
        predictions = model(user, movie)
        #y_true = rating.unsqueeze(dim=1).to(torch.float32)
        y_true = rating.to(torch.float32)
        y_pred = predictions
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

    if batch_idx % 100 == 0:
        print(
                f"Epoch {epoch+1}/{num_epochs}, "
                f"Step {batch_idx+1}, "
                f"Loss: {loss.item():.4f}, "  
                f"Time taken: {(time.time() - start_time)/60:.2f} minutes"
            )
    end_time = time.time()
    print(f"Epoch {epoch+1} completed in {(end_time - start_time)/60:.2f} minutes")

#%% Validation loop
y_preds = []
y_trues = []
model.eval()
with torch.no_grad():
    progress_bar = tqdm(validation_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, (user, movie, rating) in enumerate(progress_bar):
        predictions = model(user, movie)
        y_true = rating.cpu().numpy().reshape(-1)
        y_pred = predictions.cpu().numpy().reshape(-1)
        y_preds.extend(y_pred)
        y_trues.extend(y_true)

mse = mean_squared_error(y_trues, y_preds)
print(f"Validation Mean Squared Error is {mse}")
# %%
model.eval()

user_id = 10

watched_movies = ratings.loc[
    ratings['userId'] == user_id, 'movieId'
].unique()

all_movies = np.arange(len(movie_encoder.classes_))

unwatched_movies = np.setdiff1d(
    all_movies,
    watched_movies
)

device = next(model.parameters()).device

predicted_ratings = []

batch_size = 1024

with torch.no_grad():

    for start in range(0, len(unwatched_movies), batch_size):

        movie_batch = unwatched_movies[start:start + batch_size]

        user_batch = torch.full(
            (len(movie_batch),),
            user_id,
            dtype=torch.long,
            device=device
        )

        movie_batch = torch.tensor(
            movie_batch,
            dtype=torch.long,
            device=device
        )

        predictions = model(user_batch, movie_batch)

        predicted_ratings.extend(
            predictions.cpu().numpy().reshape(-1)
        )

recommendations = pd.DataFrame({
    'movieId': unwatched_movies,
    'predicted_rating': predicted_ratings
})

top_10 = recommendations.nlargest(10, 'predicted_rating').copy()

top_10['original_movieId'] = movie_encoder.inverse_transform(
    top_10['movieId'].to_numpy()
)

top_10 = top_10.merge(
    movies,
    left_on='original_movieId',
    right_on='movieId',
    how='left'
)

print(top_10[['title', 'genres', 'predicted_rating']])
# %%
top_10[['title', 'genres', 'predicted_rating']]
# %%
