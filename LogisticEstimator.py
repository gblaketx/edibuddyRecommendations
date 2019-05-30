import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

class LogisticEstimator:
    def __init__(self, dataLoader, recipe_holdout, user_holdout):
        self.dataLoader = dataLoader
        self.recipe_holdout = recipe_holdout
        self.user_holdout = user_holdout
        self.regressor = LogisticRegression(max_iter=100, penalty='l1', class_weight='balanced')
        self.vec = DictVectorizer()
        
        self.recipe_id_to_index = { id: idx for idx, id in enumerate(dataLoader.get_recipe_ids()) }
    
    def extract_features(self, X):
        return [self.get_feature_dict(user_id, recipe_id) for user_id, recipe_id in X]
        
    def get_feature_dict(self, user_id, recipe_id):
        user_ratings = self.dataLoader.get_user_ratings(user_id, self.user_holdout)
        avg_user_rating = np.mean(np.fromiter(user_ratings.values(), dtype=np.int64))
        
        recipe_ratings = self.dataLoader.get_recipe_ratings(
            recipe_id, self.recipe_holdout)
        avg_recipe_rating = np.mean(np.fromiter(recipe_ratings.values(), dtype=np.int64))
        recipe_info = self.dataLoader.get_recipe_info(recipe_id)
        
        calories = recipe_info['calories'] if not recipe_info['calories'] is None else 0
        
        feats = {
            'user_id': user_id,
            'recipe_id': recipe_id,
            'avg_user_rating': avg_user_rating,
            'avg_recipe_rating': avg_recipe_rating,
            'calories': calories
        }
        
        for category in recipe_info["categories"]:
            feats["cat_{}".format(category)] = 1
        for ingredient in recipe_info["ingredients"]["full list"]:
            ing_name = ingredient["key ingredient"]
            quantity = ingredient["quantity"]
            feats["ing_{}".format(ing_name)] = quantity if not quantity is None else 1
        
        return feats
    
    def fit(self, X, y):
        """
        X: list of (user_id, recipe_id) tuple
        y: np.array user rating
        """
        feats = self.extract_features(X)
        feats = self.vec.fit_transform(feats)
        return self.regressor.fit(feats, y)

    def predict(self, X):
        """
        X: list of (user_id, recipe_id) tuple
        """
        feats = self.extract_features(X)
        feats = self.vec.transform(feats)
        return self.regressor.predict(feats)
    
    def get_params(self, deep=False):
        return {
            'dataLoader': self.dataLoader,
            'user_holdout': self.user_holdout,
            'recipe_holdout': self.recipe_holdout,
        }