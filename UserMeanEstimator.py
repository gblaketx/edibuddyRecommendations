import numpy as np

class UserMeanEstimator:
    """
    Just guesses the user rating mean
    """
    def __init__(self, dataLoader, user_holdout):
        self.dataLoader = dataLoader
        self.user_holdout = user_holdout
        
    def get_user_average(self, user_id):
        ratings = self.dataLoader.get_user_ratings(user_id, self.user_holdout)
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
        return np.array([self.get_user_average(user_id) for user_id, _ in X])
    
    def get_params(self, deep=False):
        return {
            'dataLoader': self.dataLoader,
            'user_holdout': self.user_holdout
        }
            