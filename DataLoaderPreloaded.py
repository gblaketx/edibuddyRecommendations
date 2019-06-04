from DataLoader import DataLoader

class DataLoaderPreloaded(DataLoader):
    def __init__(self, recipes, all_ratings, recipe_to_raters):
        self.recipes = recipes
        self.all_ratings = all_ratings
        self.recipe_to_raters_train = recipe_to_raters
        self.recipe_to_raters_test = recipe_to_raters