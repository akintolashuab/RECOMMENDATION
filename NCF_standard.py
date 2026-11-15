#%% import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import tqdm
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing, model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error

#%% import of datasets
ratings = pd.read_csv('ml-latest/ratings.csv')
ratings.head(10)

# %%
user_encoder = preprocessing.LabelEncoder()
movie_encoder = preprocessing.LabelEncoder()
ratings['userId'] = user_encoder.fit_transform(ratings['userId'])
ratings['movieId'] = movie_encoder.fit_transform(ratings['movieId'])
ratings.head(10)

# %% creating train, validation and test datasets
train_data, temp_data = model_selection.train_test_split(ratings, 
                        test_size=0.2, random_state=42, shuffle=True)
validation_data, test_data = model_selection.train_test_split(temp_data,
                        test_size=0.5, random_state=42, shuffle=True)
# %%
ratings.shape
# %%
len(user_encoder.classes_)
# %%
len(movie_encoder.classes_)

# %%creating Dataset class for PyTorch
class NCFDataset(Dataset):
    
    def __init__(self, users, movies, ratings) -> None:
        super().__init__()
        self.users = torch.tensor(users, dtype=torch.long)
        self.movies = torch.tensor(movies, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        movie = self.movies[idx]
        rating = self.ratings[idx]
        return user, movie, rating

#%% create instance of NcfDataset for train, validation and test datasets
train_dataset = NCFDataset(train_data['userId'].values,
                            train_data['movieId'].values,
                            train_data['rating'].values)

validation_dataset = NCFDataset(validation_data['userId'].values,
                                validation_data['movieId'].values,
                                validation_data['rating'].values)

test_dataset = NCFDataset(test_data['userId'].values,
                          test_data['movieId'].values,
                          test_data['rating'].values)

#%% create DataLoader for train, validation and test datasets
batch_size = 10000
train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          shuffle=True)

validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=batch_size,
                          shuffle=False)


# %% Complete Neural Collaborative Filtering Model

class NCF(nn.Module):

    def __init__(self, num_users, num_movies, embedding_size=64):
        super().__init__()

        # Creating embedding for gmf
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_size)
        self.gmf_movie_embedding = nn.Embedding(num_movies, embedding_size)

        # Creating embedding for mlp
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_size)
        self.mlp_movie_embedding = nn.Embedding(num_movies, embedding_size)

        # creating full connected layer for MLP
        self.mlp_layer = nn.Sequential(
            nn.Linear(embedding_size *2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.ncf_output_layer = nn.Sequential(
            nn.Linear(embedding_size + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.Linear(8, 1)
            )
    
        self._initialize_weights()

    def _initialize_weights(self):
        #initialize embedding weights

        nn.init.normal_(self.gmf_user_embedding.weight, mean = 0.0, std = 0.01)
        nn.init.normal_(self.gmf_movie_embedding.weight, mean = 0.0, std = 0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, mean = 0.0, std = 0.01)
        nn.init.normal_(self.mlp_movie_embedding.weight, mean = 0.0, std = 0.01)

                # Initialize Linear layers
        for layer in self.mlp_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.ncf_output_layer[0].weight)
        nn.init.zeros_(self.ncf_output_layer[0].bias)
        nn.init.xavier_uniform_(self.ncf_output_layer[2].weight)
        nn.init.zeros_(self.ncf_output_layer[2].bias)
        nn.init.xavier_uniform_(self.ncf_output_layer[3].weight)
        nn.init.zeros_(self.ncf_output_layer[3].bias)
        

        
    def forward(self, user, movie):

        gmf_user_embedded = self.gmf_user_embedding(user)
        gmf_movie_embedded = self.gmf_movie_embedding(movie)
        mlp_user_embedded = self.mlp_user_embedding(user)
        mlp_movie_embedded = self.mlp_movie_embedding(movie)
        prediction_layer = self.ncf_output_layer

        mlp_input = torch.cat([mlp_user_embedded, mlp_movie_embedded], dim=1)

        mlp_output = self.mlp_layer(mlp_input)

        gmf_output = gmf_user_embedded * gmf_movie_embedded

        ncf_input = torch.cat([gmf_output, mlp_output], dim=1)
        
        prediction = prediction_layer(ncf_input)

        return prediction.squeeze(-1)

#%% 
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)
       
# %% creating instance of NCF model, loss function and optimizer
num_epochs = 2
model = NCF(num_users=len(user_encoder.classes_),
            num_movies=len(movie_encoder.classes_),
            embedding_size=64)
model = model.to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %% Validation function

def evaluate_model(model, data_loader, loss_fn, device):

    model.eval()
    total_loss = 0
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for user, movie, rating in data_loader:

            user = user.to(device)
            movie = movie.to(device)
            rating = rating.to(device)

            predictions = model(user, movie)

            loss = loss_fn(predictions, rating)

            total_loss += loss.item() * len(rating)

            all_predictions.extend(predictions.cpu().numpy())

            all_actuals.extend(rating.cpu().numpy())


    average_loss = total_loss / len(data_loader.dataset)

    rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
    
    mae = mean_absolute_error(all_actuals, all_predictions)

    return average_loss, rmse, mae

# %% Training loop

best_val_rmse = float('inf')

best_model_weights = None

patience = 4
epochs_without_improvement = 0

training_history = []


for epoch in range(num_epochs):

    start_time = time.time()

    model.train()

    running_loss = 0

    progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    # =================================
    # TRAINING
    # =================================

    for user, movie, rating in progress_bar:

        user = user.to(device)
        movie = movie.to(device)
        rating = rating.to(device)


        # Forward propagation

        predictions = model(user, movie)


        # Calculate loss

        loss = loss_function(predictions, rating)


        # Clear previous gradients

        optimizer.zero_grad()


        # Backpropagation

        loss.backward()


        # Update parameters

        optimizer.step()


        running_loss += loss.item() * len(rating)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss = running_loss / len(train_dataset)


    # =================================
    # VALIDATION
    # =================================

    val_loss, val_rmse, val_mae = evaluate_model(
        model, validation_loader, loss_function, device)


    # =================================
    # SAVE TRAINING HISTORY
    # =================================

    training_history.append({

        'epoch': epoch + 1,

        'train_loss': train_loss,

        'val_loss': val_loss,

        'val_rmse': val_rmse,

        'val_mae': val_mae

    })


    # =================================
    # SAVE BEST MODEL
    # =================================

    if val_rmse < best_val_rmse:

        best_val_rmse = val_rmse

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1


    # =================================
    # PRINT RESULTS
    # =================================

    elapsed_time = (time.time() - start_time) / 60

    print(f"\nEpoch: {epoch+1}/{num_epochs}")

    print(f"Training Loss: {train_loss:.4f}")

    print(f"Validation Loss: {val_loss:.4f}")

    print(f"Validation RMSE: {val_rmse:.4f}")

    print(f"Validation MAE: {val_mae:.4f}")

    print(f"Time: {elapsed_time:.2f} minutes")


    # =================================
    # EARLY STOPPING
    # =================================

    if epochs_without_improvement >= patience:

        print("\nEarly stopping triggered.")

        break


# Load the best model

if best_model_weights is not None:

    model.load_state_dict(best_model_weights)

print("\nTraining completed.")

print("Best Validation RMSE:", best_val_rmse)



# %%# %% Training history

import matplotlib.pyplot as plt

history_df = pd.DataFrame(training_history)

plt.figure(figsize=(10, 5))
plt.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss')
plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('NCF Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()

#%%# %% RMSE visualization

plt.figure(figsize=(10, 5))

plt.plot(history_df['epoch'], history_df['val_rmse'], marker='o',label='Validation RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('NCF Validation RMSE')
plt.legend()
plt.grid()

plt.show()

#%%
num_users=len(user_encoder.classes_)
num_movies=len(movie_encoder.classes_)
embedding_size=64

#%%
import os
import torch
import joblib

# Create a folder to store your trained model and encoders
save_dir = r'C:\Users\USER\Desktop\pytorch_training\NCF_Project'
os.makedirs(save_dir, exist_ok=True)

# Define file paths
model_save_path = os.path.join(save_dir, 'NCF_movie_recommender.pth')
user_encoder_path = os.path.join(save_dir, 'user_encoder.pkl')
movie_encoder_path = os.path.join(save_dir, 'movie_encoder.pkl')

# Save PyTorch model checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'num_users': num_users,
    'num_movies': num_movies,
    'embedding_dim': embedding_size
}, model_save_path)

# Save encoders
joblib.dump(user_encoder, user_encoder_path)
joblib.dump(movie_encoder, movie_encoder_path)

print("Model and encoders saved successfully:")
print(model_save_path)
print(user_encoder_path)
print(movie_encoder_path)

# %% Generate movie recommendations
movies = pd.read_csv('ml-latest/movies.csv')
def recommend_movies(user_id, top_n=10):

    model.eval()

    # Convert original user ID to encoded ID

    user_encoded = user_encoder.transform([user_id])[0]


    # Find movies already rated by this user

    rated_movies = ratings.loc[ratings['userId'] == user_encoded,'movieId'].values


    # Get all movie indices

    all_movies = np.arange(num_movies)


    # Exclude already rated movies

    unrated_movies = np.setdiff1d(all_movies, rated_movies)


    # Create tensors

    user_tensor = torch.full((len(unrated_movies),), user_encoded,
        dtype=torch.long)

    movie_tensor = torch.tensor(unrated_movies, dtype=torch.long)


    # Move tensors to device

    user_tensor = user_tensor.to(device)

    movie_tensor = movie_tensor.to(device)


    # Predict ratings

    with torch.no_grad():

        predictions = model(user_tensor, movie_tensor)


    predictions = predictions.cpu().numpy()


    # Create recommendation DataFrame

    recommendations = pd.DataFrame({'movie_encoded': unrated_movies,

        'predicted_rating': predictions})


    # Convert encoded movie IDs back to original IDs

    recommendations['movieId'] = movie_encoder.inverse_transform(
        recommendations['movie_encoded'].values)


    # Merge with movie titles

    recommendations = recommendations.merge(movies, on='movieId',
        how='left')


    # Sort by predicted rating

    recommendations = recommendations.sort_values('predicted_rating',
        ascending=False)

    return recommendations.head(top_n)

#%%

# Get recommendations for original MovieLens user ID 1


recommendations = recommend_movies(user_id=1, top_n=10)

recommendations[['movieId', 'title', 'predicted_rating']]
# %%
