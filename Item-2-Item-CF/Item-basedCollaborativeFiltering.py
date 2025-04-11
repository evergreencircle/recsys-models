#%% md
# <center> <b> open with a new tab </b> </center>
#%% md
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nzhinusoftcm/review-on-collaborative-filtering/blob/master/3.Item-basedCollaborativeFiltering.ipynb)
#%% md
# # Item-to-Item Collaborative Filtering
#%% md
# ## Idea
# Let $u$ be the active user and $i$ the referenced item
# 1. If $u$ liked items similar to $i$, he will probably like item $i$.
# 2. If he hated or disliked items similar to $i$, he will also hate item $i$.
# 
# The idea is therefore to look at how an active user $u$ rated items similar to $i$ to know how he would have rated item $i$
#%% md
# ## Advantages over user-based CF
# 
# 1. <b> Stability </b> : Items ratings are more stable than users ratings. New ratings on items are unlikely to significantly change the similarity between two items, particularly when the items have many ratings <a href="https://dl.acm.org/doi/10.1561/1100000009">(Michael D. Ekstrand, <i>et al.</i> 2011)</a>. 
# 2. <b> Scalability </b> : with stable item's ratings, it is reasonable to pre-compute similarities between items in an item-item similarity matrix (similarity between items can be computed offline). This will reduce the scalability concern of the algorithm. <a href="https://dl.acm.org/doi/10.1145/371920.372071">(Sarwar <i>et al.</i> 2001)</a>, <a href="https://dl.acm.org/doi/10.1561/1100000009">(Michael D. Ekstrand, <i>et al.</i> 2011)</a>.
#%% md
# ## Algorithm : item-to-item collaborative filtering
# 
# The algorithm that defines item-based CF is described as follow <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.449.1171&rep=rep1&type=pdf">(B. Sarwar et al. 2001)</a><a href="https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.554.1671&rep=rep1&type=pdf">(George Karypis 2001)</a> :
# 
# <ol>
#     <li>
#         First identify the $k$ most similar items for each item in the catalogue and record the corresponding similarities. To compute similarity between two items we can user the <i>Adjusted Cosine Similarity</i> that has proven to be more efficient than the basic <i>Cosine similarity measure</i> used for user-based collaborative as described in <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.449.1171&rep=rep1&type=pdf">(B. Sarwar et al. 2001)</a>. The Adjusted Cosine distance between two items $i$ and $j$ is computed as follow
# 
# \begin{equation}
#  w_{i,j}= \frac{\sum_{u\in U}(r_{u,i}-\bar{r}_u)(r_{u,j}-\bar{r}_u)}{\sqrt{\sum_{u\in U} (r_{u,i}-\bar{r}_u)^2}\sqrt{\sum_{u\in U} (r_{u,j}-\bar{r}_u)^2}}
# \end{equation}
# 
# $w_{i,j}$ is the degree of similarity between items $i$ and $j$. This term is computed for all users $u\in U$, where $U$ is the set of users that rated both items $i$ and $j$. Let's denote by $S^{(i)}$ the set of the $k$ most similar items to item $i$.
#     </li>    
#     <li> To produce top-N recommendations for a given user $u$ that has already purchased a set $I_u$ of items, do the following :
# <ul>
#     <li> Find the set $C$ of candidate items by taking the union of all $S^{(i)}, \forall i\in I_u$ and removing each of the items in the set $I_u$.
# \begin{equation}
#  C = \bigcup_{i\in I_u}\{S^{(i)}\}\smallsetminus I_u
# \end{equation}
#     </li>
#     <li>
#         $\forall c\in C$, compute similarity between c and the set $I_u$ as follows:
# \begin{equation}
#  w_{c,I_u} = \sum_{i\in I_u} w_{c,i}, \forall c \in C
# \end{equation}
#     </li>
#     <li>
#         Sort items in $C$ in decreasing order of $w_{c,I_u}, \forall c \in C$, and return the first $N$ items as the Top-N recommendation list.
#     </li>
# </ul>    
#     </li>
# </ol>
#%% md
# Before returning the first $N$ items as top-N recommendation list, we can make predictions about what user $u$ would have given to each items in the top-N recommendation list, rearrange the list in descending order of predicted ratings and return the rearranged list as the final recommendation list. Rating prediction for item-based CF is given by the following formular <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.449.1171&rep=rep1&type=pdf">(B. Sarwar et al. 2001)</a>:
# 
# \begin{equation}
#  \hat{r}_{u,i}=\frac{\sum_{i\in S^{(i)}}r_{u,j}\cdot w_{i,j}}{\sum_{j\in S^{(i)}}|w_{i,j}|}
# \end{equation}
#%% md
# ### Import useful requirements
#%%
import os

# if not (os.path.exists("recsys.zip") or os.path.exists("recsys")):
#     !wget https://github.com/nzhinusoftcm/review-on-collaborative-filtering/raw/master/recsys.zip    
#     !unzip recsys.zip
#%% md
# ### Import requirements
#%% md
# ```
# matplotlib==3.2.2
# numpy==1.19.2
# pandas==1.0.5
# python==3.7
# scikit-learn==0.24.1
# scikit-surprise==1.1.1
# scipy==1.6.2
# ```
#%%
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from recsys.datasets import ml1m, ml100k
from recsys.preprocessing import ids_encoder, train_test_split, get_examples

import pandas as pd
import numpy as np
import os
import sys
#%% md
# ### Load ratings
#%%
ratings, movies = ml100k.load()
#%% md
# ### userids and itemids encoding
#%%
# create the encoder
ratings, uencoder, iencoder = ids_encoder(ratings)
#%%
ratings['userid'].nunique()
#%%
ratings['itemid'].nunique()

#%% md
# Let's implements the item-based collaborative filtering algorithm described above
#%% md
# ### Step 1. Find similarities for each of the items
#%% md
# To compute similarity between two items $i$ and $j$, we need to :
# 
# 1. find all users who rated both of them,
# 2. Normalize their ratings on items $i$ and $j$
# 3. Apply the cosine metric to the normalized ratings to compute similarity between $i$ and $j$
#%% md
# Function ```normalize()``` process the rating dataframe to normalize ratings of all users
#%%
def normalize():
    # compute mean rating for each user
    mean = ratings.groupby(by='userid', as_index=False)['rating'].mean()
    norm_ratings = pd.merge(ratings, mean, suffixes=('','_mean'), on='userid')
    
    # normalize each rating by substracting the mean rating of the corresponding user
    norm_ratings['norm_rating'] = norm_ratings['rating'] - norm_ratings['rating_mean']
    return mean.to_numpy()[:, 1], norm_ratings
#%%
mean, norm_ratings = normalize()
np_ratings = norm_ratings.to_numpy()
norm_ratings.head()
#%% md
# now that each rating has been normalized, we can represent each item by a vector of its normalized ratings
#%%
def item_representation(ratings):    
    return csr_matrix(
        pd.crosstab(ratings.itemid, ratings.userid, ratings.norm_rating, aggfunc=sum).fillna(0).values
    )
#%%
R = item_representation(norm_ratings)
#%% md
# Let's build and fit our $k$-NN model using sklearn
#%%
def create_model(rating_matrix, k=20, metric="cosine"):
    """
    :param R : numpy array of item representations
    :param k : number of nearest neighbors to return    
    :return model : our knn model
    """    
    model = NearestNeighbors(metric=metric, n_neighbors=k+1, algorithm='brute')
    model.fit(rating_matrix)    
    return model
#%% md
# #### Similarities computation
#%% md
# Similarities between items can be measured with the *Cosine* or *Eucliedian* distance. The ***NearestNeighbors*** class from the sklearn library simplifies the computation of neighbors. We just need to specify the metric (e.g. cosine or euclidian) that will be used to compute similarities.
# 
# The above method, ```create_model```, creates the kNN model and the following ```nearest_neighbors``` method uses the created model to kNN items. It returns nearest neighbors as well as similarities measures for each items.
# 
# ```nearest_neighbors``` returns :
# - ```similarities``` : numpy array of shape $(n,k)$
# - ```neighbors``` : numpy array of shape $(n,k)$
# 
# where $n$ is the total number of items and $k$ is the number of neighbors to return, specified when creating the kNN model.
#%%
def nearest_neighbors(rating_matrix, model):
    """
    compute the top n similar items for each item.    
    :param rating_matrix : items representations
    :param model : nearest neighbors model    
    :return similarities, neighbors
    """    
    similarities, neighbors = model.kneighbors(rating_matrix)    
    return similarities[:,1:], neighbors[:,1:]
#%% md
# #### Ajusted Cosine Similarity
# In the context of item-based collaborative filtering, the adjusted cosine similarity has shown to be more efficient that the cosine or the euclidian distance. Here is the formular to compute the adjusted cosine weight between two items $i$ and $j$ :
#%% md
# \begin{equation}
#  w_{i,j}= \frac{\sum_{u\in U}(r_{u,i}-\bar{r}_u)(r_{u,j}-\bar{r}_u)}{\sqrt{\sum_{u\in U} (r_{u,i}-\bar{r}_u)^2}\sqrt{\sum_{u\in U} (r_{u,j}-\bar{r}_u)^2}}.
# \end{equation}
#%% md
# This term is computed for all users $u\in U$, where $U$ is the set of users that rated both items $i$ and $j$. Since the *sklearn* library do not directly implement the adjusted cosine similarity metric, we will implement it with the method ```adjusted_cosine```, with some helper function :
# 
# - ```save_similarities``` : since the computation of the adjusted cosine similarity is time consuming, around 5 mins for the ml100k dataset, we use this method to save the computed similarities for lated usage.
# - ```load_similarities``` : load the saved similarities
# - ```cosine``` : cosine distance between two vectors.
#%%
def save_similarities(similarities, neighbors, dataset_name):    
    base_dir = 'recsys/weights/item2item'
    save_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)    
    similarities_file_name = os.path.join(save_dir, 'similarities.npy')
    neighbors_file_name = os.path.join(save_dir, 'neighbors.npy')    
    try:
        np.save(similarities_file_name, similarities)
        np.save(neighbors_file_name, neighbors)        
    except ValueError as error:
        print(f"An error occured when saving similarities, due to : \n ValueError : {error}")

        
def load_similarities(dataset_name, k=20):
    base_dir = 'recsys/weights/item2item'
    save_dir = os.path.join(base_dir, dataset_name)    
    similiraties_file = os.path.join(save_dir, 'similarities.npy')
    neighbors_file = os.path.join(save_dir, 'neighbors.npy')    
    similarities = np.load(similiraties_file)
    neighbors = np.load(neighbors_file)    
    return similarities[:,:k], neighbors[:,:k]


def cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def adjusted_cosine(np_ratings, nb_items, dataset_name):
    
    similarities = np.zeros(shape=(np_ratings.shape[0], np_ratings.shape[0]))
    similarities.fill(-1)
    
    def _progress(count):
        sys.stdout.write('\rComputing similarities. Progress status : %.1f%%' % (float(count / nb_items)*100.0))
        sys.stdout.flush()
        
    items = sorted(ratings.itemid.unique())    
    for i in items[:-1]:
        for j in items[i+1:]:            
            scores = np_ratings[(np_ratings[:, 1] == i) | (np_ratings[:, 1] == j), :]
            vals, count = np.unique(scores[:,0], return_counts = True)
            scores = scores[np.isin(scores[:,0], vals[count > 1]),:]

            if scores.shape[0] > 2:
                x = scores[scores[:, 1].astype('int') == i, 4]
                y = scores[scores[:, 1].astype('int') == j, 4]
                w = cosine(x, y)

                similarities[i, j] = w
                similarities[j, i] = w
        _progress(i)
    _progress(nb_items)
    
    # get neighbors by their neighbors in decreasing order of similarities
    neighbors = np.flip(np.argsort(similarities), axis=1)
    
    # sort similarities in decreasing order
    similarities = np.flip(np.sort(similarities), axis=1)
    
    # save similarities to disk
    save_similarities(similarities, neighbors, dataset_name=dataset_name) 
    
    return similarities, neighbors
#%% md
# now, we can call the ```adjusted_cosine``` function to compute and save items similarities and neighbors based on the adjusted cosine metric. 
# 
# uncomment the two lines of the following cell to compute the adjusted cosine between all items. As we have already run the next cell before, we will just load the precomputed similarities for further use.
#%%
# nb_items = ratings.itemid.nunique()
# similarities, neighbors = adjusted_cosine(np_ratings, nb_items=nb_items, dataset_name='ml100k')
#%% md
# Among the following similarity metrics, choose the one you wish to use for the item-based collaborative filtering :
# 
# - **euclidian** or **cosine** : choose *euclidian* or *cosine* to initialise the similarity model through the sklearn library.
# - **adjusted_cosine** : choose the *adjusted_cosine* metric to load similarities computed and saved through the ```adjusted_cosine``` function.
# 
# In this case, we will use the *adjusted_cosine* metric.
#%%
# metric : choose among [cosine, euclidean, adjusted_cosine]

metric = 'adjusted_cosine'

if metric == 'adjusted_cosine':
    similarities, neighbors = load_similarities('ml100k')
else:
    model = create_model(R, k=21, metric=metric)
    similarities, neighbors = nearest_neighbors(R, model)
#%%
print('neighbors shape : ', neighbors.shape)
print('similarities shape : ', similarities.shape)
#%% md
# ```neighbors``` and ```similarities``` are numpy array, were each entries are list of 20 neighbors with their corresponding similarities
#%% md
# ### Step 2. Top N recommendation for a given user
#%% md
# Top-N recommendations are made for example for a user $u$ who has already rated a set of items $I_u$
#%% md
# #### 2.a- Finding candidate items
# 
# To find candidate items for user $u$, we need to :
# 
# 1. Find the set $I_u$ of items already rated by user $u$,
# 2. Take the union of similar items as $C$ for all items in $I_u$
# 3. exclude from the set $C$ all items in $I_u$, to avoid recommend to a user items he has already purchased.
# 
# These are done in function ```candidate_items()```
#%%
def candidate_items(userid):
    """
    :param userid : user id for which we wish to find candidate items    
    :return : I_u, candidates
    """
    
    # 1. Finding the set I_u of items already rated by user userid
    I_u = np_ratings[np_ratings[:, 0] == userid]
    I_u = I_u[:, 1].astype('int')
    
    # 2. Taking the union of similar items for all items in I_u to form the set of candidate items
    c = set()
        
    for iid in I_u:    
        # add the neighbors of item iid in the set of candidate items
        c.update(neighbors[iid])
        
    c = list(c)
    # 3. exclude from the set C all items in I_u.
    candidates = np.setdiff1d(c, I_u, assume_unique=True)
    
    return I_u, candidates
#%%
test_user = uencoder.transform([1])[0]
i_u, u_candidates = candidate_items(test_user)
#%%
print(test_user)
#%%
print('number of items purchased by user 1 : ', len(i_u))
print('number of candidate items for user 1 : ', len(u_candidates))
#%% md
# #### 2.b- Find similarity between each candidate item and the set $I_u$
#%%
def similarity_with_Iu(c, I_u):
    """
    compute similarity between an item c and a set of items I_u. For each item i in I_u, get similarity between 
    i and c, if c exists in the set of items similar to itemid.    
    :param c : itemid of a candidate item
    :param I_u : set of items already purchased by a given user    
    :return w : similarity between c and I_u
    """
    w = 0    
    for iid in I_u :        
        # get similarity between itemid and c, if c is one of the k nearest neighbors of itemid
        if c in neighbors[iid] :
            w = w + similarities[iid, neighbors[iid] == c][0]    
    return w
#%% md
# #### 2.c- Rank candidate items according to their similarities to $I_u$
#%%
def rank_candidates(candidates, I_u):
    """
    rank candidate items according to their similarities with i_u    
    :param candidates : list of candidate items
    :param I_u : list of items purchased by the user    
    :return ranked_candidates : dataframe of candidate items, ranked in descending order of similarities with I_u
    """
    
    # list of candidate items mapped to their corresponding similarities to I_u
    sims = [similarity_with_Iu(c, I_u) for c in candidates]
    candidates = iencoder.inverse_transform(candidates)    
    mapping = list(zip(candidates, sims))
    
    ranked_candidates = sorted(mapping, key=lambda couple:couple[1], reverse=True)    
    return ranked_candidates
#%% md
# ## Putting all together
#%% md
# Now that we defined all functions necessary to build our item to item top-N recommendation, let's define function ```item2item_topN()``` that makes top-$N$ recommendations for a given user 
#%%
def topn_recommendation(userid, N=30):
    """
    Produce top-N recommendation for a given user    
    :param userid : user for which we produce top-N recommendation
    :param n : length of the top-N recommendation list    
    :return topn
    """
    # find candidate items
    I_u, candidates = candidate_items(userid)
    
    # rank candidate items according to their similarities with I_u
    ranked_candidates = rank_candidates(candidates, I_u)
    
    # get the first N row of ranked_candidates to build the top N recommendation list
    topn = pd.DataFrame(ranked_candidates[:N], columns=['itemid','similarity_with_Iu'])    
    topn = pd.merge(topn, movies, on='itemid', how='inner')    
    return topn
#%%
topn_recommendation(test_user)
#%% md
# This dataframe represents the top N recommendation list a user. These items are sorted in decreasing order of similarities with $I_u$.
# 
# **Observation** : The recommended items are the most similar to the set $I_u$ of items already purchased by the user.
#%% md
# ## Top N recommendation with predictions
#%% md
# Before recommending the previous list to the user, we can go further and predict the ratings the user would have given to each of these items, sort them in descending order of prediction and return the reordered list as the new top N recommendation list.
#%% md
# ### Rating prediction
# 
# As stated earlier, the predicted rating $\hat{r}_{u,i}$ for a given user $u$ on an item $i$ is obtained by aggregating ratings given by $u$ on items similar to $i$ as follows:
# 
# \begin{equation}
#  \hat{r}_{u,i}=\frac{\sum_{j\in S^{(i)}}r_{u,j}\cdot w_{i,j}}{\sum_{j\in S^{(i)}}|w_{i,j}|}
# \end{equation}
#%%
def predict(userid, itemid):
    """
    Make rating prediction for user userid on item itemid    
    :param userid : id of the active user
    :param itemid : id of the item for which we are making prediction        
    :return r_hat : predicted rating
    """
    
    # Get items similar to item itemid with their corresponding similarities
    item_neighbors = neighbors[itemid]
    item_similarities = similarities[itemid]
    
    # get ratings of user with id userid
    uratings = np_ratings[np_ratings[:, 0].astype('int') == userid]
    
    # similar items rated by item the user of i
    siru = uratings[np.isin(uratings[:, 1], item_neighbors)]
    scores = siru[:, 2]
    indexes = [np.where(item_neighbors == iid)[0][0] for iid in siru[:,1].astype('int')]    
    sims = item_similarities[indexes]
    
    dot = np.dot(scores, sims)
    som = np.sum(np.abs(sims))

    if dot == 0 or som == 0:
        return mean[userid]
    
    return dot / som
#%% md
# Now let's use our ```predict()``` function to predict what ratings the user would have given to the previous top-$N$ list and return the reorganised list (in decreasing order of predictions) as the new top-$N$ list
#%%
def topn_prediction(userid):
    """
    :param userid : id of the active user    
    :return topn : initial topN recommendations returned by the function item2item_topN
    :return topn_predict : topN recommendations reordered according to rating predictions
    """
    # make top N recommendation for the active user
    topn = topn_recommendation(userid)
    
    # get list of items of the top N list
    itemids = topn.itemid.to_list()
    
    predictions = []
    
    # make prediction for each item in the top N list
    for itemid in itemids:
        r = predict(userid, itemid)
        
        predictions.append((itemid,r))
    
    predictions = pd.DataFrame(predictions, columns=['itemid','prediction'])
    
    # merge the predictions to topN_list and rearrange the list according to predictions
    topn_predict = pd.merge(topn, predictions, on='itemid', how='inner')
    topn_predict = topn_predict.sort_values(by=['prediction'], ascending=False)
    
    return topn, topn_predict
#%% md
# Now, let's make recommendation for user 1 and compare the two list
#%%
topn, topn_predict = topn_prediction(userid=test_user)
#%%
topn_predict
#%% md
# As you will have noticed, the two lists are sorted in different ways. The second list is organized according to the predictions made for the user.
# 
# <b>Note</b>: When making predictions for user $u$ on item $i$, user $u$ may not have rated any of the $k$ most similar items to i. In this case, we consider the mean rating of $u$ as the predicted value.
#%% md
# #### Evaluation with Mean Absolute Error
#%%
from recsys.preprocessing import train_test_split, get_examples

# get examples as tuples of userids and itemids and labels from normalize ratings
raw_examples, raw_labels = get_examples(ratings, labels_column='rating')

# train test split
(x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)

def evaluate(x_test, y_test):
    print('Evaluate the model on {} test data ...'.format(x_test.shape[0]))
    preds = list(predict(u,i) for (u,i) in x_test)
    mae = np.sum(np.absolute(y_test - np.array(preds))) / x_test.shape[0]
    print('\nMAE :', mae)
    return mae
#%%
evaluate(x_test, y_test)
#%% md
# ### Summary
#%% md
# As with the User-based CF, we have also summarised the Item-based CF into the python class [ItemToItem](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/recsys/memories/ItemToItem.py). 
# 
# #### ItemToItem : usage
#%%
# from recsys.memories.ItemToItem import ItemToItem
# from recsys.preprocessing import ids_encoder, train_test_split, get_examples
# from recsys.datasets import ml100k

# load data
ratings, movies = ml100k.load()

# prepare data
ratings, uencoder, iencoder = ids_encoder(ratings)

# get examples as tuples of userids and itemids and labels from normalize ratings
raw_examples, raw_labels = get_examples(ratings, labels_column='rating')

# train test split
(x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)
#%% md
# #### Instanciate the ItemToItem CF
# 
# Parameters :
# - ```k``` : number of neighbors to consider for each item
# - ```metric``` : metric to use when computing similarities : let's use **cosine**
# - ```dataset_name``` : in this example, we use the ml100k dataset
#%%
# create the Item-based CF
item2item = ItemToItem(ratings, movies, k=20, metric='cosine', dataset_name='ml100k')
#%%
# evaluate the algorithm on test dataset
item2item.evaluate(x_test, y_test)
#%% md
# #### Evaluate the Item-based CF on the ML-1M dataset
#%%
# from recsys.memories.ItemToItem import ItemToItem
# from recsys.preprocessing import ids_encoder, train_test_split, get_examples
# from recsys.datasets import ml1m

# load data
ratings, movies = ml1m.load()

# prepare data
ratings, uencoder, iencoder = ids_encoder(ratings)

# get examples as tuples of userids and itemids and labels from normalize ratings
raw_examples, raw_labels = get_examples(ratings, labels_column='rating')

# train test split
(x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)

# create the Item-based CF
item2item = ItemToItem(ratings, movies, k=20, metric='cosine', dataset_name='ml1m')

# evaluate the algorithm on test dataset
print("=========================")
item2item.evaluate(x_test, y_test)
#%% md
# ## Model based CF
# 
# **User-based** and **Item-based CF** are memory based algorithms. They directly act on the user-item interactions to compute recommendation. To the contrary, model-based algorithms are mathematical models trained on the user-item interactions and used to predict recommendation.
# 
# We will start with the SVD (Singular Value Decomposition) algorithm. Click [here](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/4.Singular_Value_Decomposition.ipynb) to go to SVD.
#%% md
# ## References
# 
# 1. George Karypis (2001)<a href="https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.554.1671&rep=rep1&type=pdf">Evaluation of Item-Based Top-N Recommendation Algorithms</a>
# 2. Sarwar et al. (2001) <a href="https://dl.acm.org/doi/10.1145/371920.372071"> Item-based collaborative filtering recommendation algorithms</a> 
# 3. Michael D. Ekstrand, et al. (2011). <a href="https://dl.acm.org/doi/10.1561/1100000009"> Collaborative Filtering Recommender Systems</a>
# 4. J. Bobadilla et al. (2013)<a href="https://romisatriawahono.net/lecture/rm/survey/information%20retrieval/Bobadilla%20-%20Recommender%20Systems%20-%202013.pdf"> Recommender systems survey</a>
# 5. Greg Linden, Brent Smith, and Jeremy York (2003) <a href="https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf">Amazon.com Recommendations : Item-to-Item Collaborative Filtering</a>
#%% md
# ## Author
# 
# [Carmel WENGA](https://www.linkedin.com/in/carmel-wenga-871876178/), <br>
# PhD student at Université de la Polynésie Française, <br> 
# Applied Machine Learning Research Engineer, <br>
# [ShoppingList](https://shoppinglist.cm), NzhinuSoft.