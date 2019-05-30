import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

class DataLoader:
    """
    ratings: Dataframe of recipe ratings by user id.
    min_recipe_ratings: int the minimum number of ratings a recipe should have
        to be included.
    min_user_ratings: int the minimum number of ratings a user must have to be
        used for training/testing. Should be at least 2.
    random_state: integer indicating the random state to use for splits
    """
    def __init__(self, ratings, recipes, min_recipe_ratings=5, min_user_ratings=3, random_state=0):
        self.random_state = random_state
        
        # Store recipe information for future reference
        self.recipes = recipes
        
        # Filter recipes to have at least min_recipe_ratings
        recipe_rating_counts = ratings.recipe_id.value_counts()
        recipe_ids = recipe_rating_counts[recipe_rating_counts >= min_recipe_ratings].index.unique().values
        recipe_ids = frozenset(recipe_ids)
        ratings = ratings[ratings.recipe_id.isin(recipe_ids)]
        
        # Map from users to recipes to ratings
        # ex. all_ratings[user_id][recipe_id] returns the user's rating for the recipe
        self.all_ratings = self.ratings_group_to_dict(ratings.groupby(["user_id", "recipe_id"]).rating.apply(sum))
        
        # Select users who'll be used for training/testing 
        user_rating_counts = ratings.user_id.value_counts()
        eval_users = user_rating_counts[user_rating_counts >= min_user_ratings].index.unique().values
        train_users, test_users = train_test_split(
            eval_users, test_size=0.1, random_state=random_state)
    
        self.train_users = train_users
        self.test_users = test_users
    
        # Only those ratings made by users in the training set
        ratings_train_users = ratings[ratings.user_id.isin(frozenset(train_users))]
        
        # Ratings made by users in training set and users with < min_recipe_ratings ratings
        # Maps from recipe_id to user_id to rating
        # train_ratings[user_id][recipe_id]
        train_only_ratings = ratings[~ratings.user_id.isin(frozenset(test_users))]
        self.train_ratings = train_only_ratings.groupby(["user_id", "recipe_id"]).rating.apply(sum)
        
        # Only those ratings made by users in the test set
        ratings_test_users = ratings[ratings.user_id.isin(frozenset(test_users))]
    
        # Map from recipes to the user_ids of those who rated them (including low-rating users)
        self.recipe_to_raters_train = train_only_ratings.groupby("recipe_id").user_id.apply(frozenset)
        self.user_to_recipes_train = ratings_train_users.groupby("user_id").recipe_id.apply(frozenset)
        # Include all users, including test users
        self.recipe_to_raters_test = ratings.groupby("recipe_id").user_id.apply(frozenset)
        # TODO: correct split?
        self.user_to_recipes_test = ratings_test_users.groupby("user_id").recipe_id.apply(frozenset)
        
    def get_recipe_ids(self):
        return self.recipe_to_raters_test.keys()

    def get_holdout_data(self, split="train"):
        """
        Select a single review per user to remove as the holdout set
        Returns a tuple of
            user_holdout: dict mapping forom user_id to the id of the held out recipe
            recipe_heldout_users: dict mapping from recipe id to a set of users
                held out from the recipe (needed for training to ensure no data leakage)
            holdout: a list of holdout data points each consisting of a tuple of
                (user_id, recipe_id, user_rating_on_recipe)
        """
        if split == "train":
            users = self.train_users
        elif split == "test":
            users = self.test_users
        else:
            raise ValueError("Unexpected split: {}".format(split))
        
        np.random.seed(self.random_state)
        holdout = []
        # Note: user_to_recipes excludes users with < min_user_ratings
#         user_to_recipes_no_holdout = defaultdict(dict)
        
        user_holdout = {}
        recipe_heldout_users = defaultdict(set)
        for user_id in users:
            user_recipes = self.all_ratings[user_id].keys()
            hidden_recipe_id = np.random.choice(np.fromiter(user_recipes, dtype=np.int64))
            
            user_holdout[user_id] = hidden_recipe_id
            recipe_heldout_users[hidden_recipe_id].add(user_id)
            
            user_rating = self.all_ratings[user_id][hidden_recipe_id]
            holdout.append((user_id, hidden_recipe_id, user_rating))

        # TODO: We don't need the more aggressive holdout techniques on the test set, only on the train set
#         return users_to_recipe_not_holdout, recipe_to_raters_no_holdout, holdout
        return user_holdout, recipe_heldout_users, holdout
        
    def get_user_ratings(self, user_id, user_holdout):
        """
        For a given user, return a dictionary mapping from recipe ids to the user's rating,
        excluding the recipe in the holdout for that user.
        user_holdout: dict mapping from user_id to the recipe_id of the heldout recipe
        """
        if user_holdout is None or user_id not in user_holdout:
            return self.all_ratings[user_id]

        heldout_recipe_id = user_holdout[user_id]
        return {recipe_id: rating for recipe_id, rating in self.all_ratings[user_id].items() if recipe_id != heldout_recipe_id}
    
    def get_recipe_ratings(self, recipe_id, recipe_holdout, split="train"):
        """
        For a given recipe, get a dictionary mapping from user_ids to the user's rating on that recipe,
        excluding any users in the holdout set for that recipe.
        recipe_holdout: dict mapping from recipe_id to a set of user_ids that are in the holdout set for that recipe
        """     
        if split == "train":
            recipe_to_raters = self.recipe_to_raters_train
        elif split == "test":
            recipe_to_raters = self.recipe_to_raters_test
        else:
            raise ValueError("Unexpected split: {}".format(split))
        
        if recipe_holdout is None:
            heldout_user_ids = frozenset()
        else:
            heldout_user_ids = recipe_holdout[recipe_id]

        raters = recipe_to_raters[recipe_id] - heldout_user_ids
        return {user_id: self.all_ratings[user_id][recipe_id] for user_id in raters}
    
    def ratings_group_to_dict(self, df):
        ratings_dict = defaultdict(dict)
        for k, rating in df.items():
            user_id, recipe_id = k
            ratings_dict[user_id][recipe_id] = rating
        return ratings_dict

    def get_recipe_info(self, recipe_id):
        return self.recipes[recipe_id]