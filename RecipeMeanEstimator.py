import numpy as np

class RecipeMeanEstimator:
    """
    Just guesses the mean for the recipe
    """
    def __init__(self, dataLoader, recipe_holdout):
        self.dataLoader = dataLoader
        self.recipe_holdout = recipe_holdout
        
    def get_recipe_average(self, recipe_id):
        ratings = self.dataLoader.get_recipe_ratings(
            recipe_id, self.recipe_holdout)
        return np.round(np.mean(np.fromiter(ratings.values(), dtype=np.int64)))
    
    def fit(self, X, y):
        """
        X: list of (user_id, recipe_id) tuple
        y: np.array user rating
        """
        return self.predict(X)
    
    def predict(self, X):
        """
        X: list of (user_id, recipe_id) tuple
        """
        return np.array([self.get_recipe_average(recipe_id) for _, recipe_id in X])
    
    def get_params(self, deep=False):
        return {
            'dataLoader': self.dataLoader,
            'recipe_holdout': self.recipe_holdout
        }
        