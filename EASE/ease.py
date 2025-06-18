#%%
import numpy as np
import pandas as pd
#%%
from rectools import Columns
from rectools.dataset import Dataset
#%%
from rectools.models import EASEModel
#%%
%%time
!wget -q https://files.grouplens.org/datasets/movielens/ml-1m.zip -O ml-1m.zip
!unzip -o ml-1m.zip
!rm ml-1m.zip
#%%
%%time
ratings = pd.read_csv(
    "ml-1m/ratings.dat",
    sep="::",
    engine="python",  # Because of 2-chars separators
    header=None,
    names=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
)
print(ratings.shape)
ratings.head()
#%%
%%time
movies = pd.read_csv(
    "ml-1m/movies.dat",
    sep="::",
    engine="python",  # Because of 2-chars separators
    header=None,
    names=[Columns.Item, "title", "genres"],
    encoding_errors="ignore",
)
print(movies.shape)
movies.head()
#%%
# Prepare a dataset to build a model
dataset = Dataset.construct(ratings)
#%%
%%time
# Fit model and generate recommendations for all users
model = EASEModel()
model.fit(dataset)
recos = model.recommend(
    users=ratings[Columns.User].unique(),
    dataset=dataset,
    k=10,
    filter_viewed=True,
)
#%%
# Sample of recommendations - it's sorted by relevance (= rank) for each user
recos.head()
#%%
# Select random user, see history of views and reco for this user
user_id = 3883
user_viewed = ratings.query("user_id == @user_id").merge(movies, on="item_id")
user_recos = recos.query("user_id == @user_id").merge(movies, on="item_id")
#%%
user_viewed.query("weight > 3")

#%%
user_recos.sort_values("rank")

#%%
