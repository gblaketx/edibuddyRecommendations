class RecipeRecommender:
    """
    
    distance: Function that takes in list of recipe_ids selected so far
        and target recipe_id and computes the distance between them. Higher
        distance scores are considered better for diversity.
    """
    def __init__(self, 
                 dataLoader, 
                 estimator, 
                 diversity_weight=1.0, 
                 distance = "cosine"):
        self.estimator = estimator
        self.dataLoader = dataLoader
        self.recipe_ids = frozenset(dataLoader.get_recipe_ids())
        self.diversity_weight = diversity_weight
        self.diversity = self.get_diversity_calculation(distance)
        self.sim_cache = {}
        
    def get_diversity_calculation(self, distance):
        """
        Returns the function used to calculate how item new_item would affect the diversity
        of recipe set current_items. Higher diversity score is better
        """
        
        def average_cosine_similarity(current_items, new_item_id):
            sim = 0.0
            for recipe_id in current_items:
                sim += self.compute_cosine_similiarity(recipe_id, new_item_id)
            return -sim / len(current_items)
        
        def shared_ingredients(current_items, new_item_id):
            used_ingredients = { i
                                for r_id in current_items
                                for i in self.dataLoader.get_recipe_info(r_id)["ingredients"]}
            new_recipe_ingredients = { i
                                      for i in self.dataLoader.get_recipe_info(new_item_id)["ingredients"]}
    
            return 1.0 - len(new_recipe_ingredients - used_ingredients) / len(new_recipe_ingredients)
    
        if distance == "cosine":
            return average_cosine_similarity
        elif distance == "ingredients":
            return shared_ingredients
        else:
            raise ValueError("Unexpected distance: {}".format(distance))
            
    
    def compute_cosine_similiarity(self, r1, r2):
        key = tuple(sorted([r1, r2]))
        if key not in self.sim_cache:
            ratings1 = self.dataLoader.get_recipe_ratings(r1, defaultdict(set))
            ratings2 = self.dataLoader.get_recipe_ratings(r2, defaultdict(set))
            prod = 0
            for k in ratings1:
                if k in ratings2:
                    prod += ratings1[k] * ratings2[k]
            self.sim_cache[key] = prod / (len(ratings1) * len(ratings2))
        return self.sim_cache[key]

    def get_recommendations(self, user_id, n_recs=100):
        preds = self.predict_unrated_recipes(user_id)
        return self.select_diverse_recipes(preds, n_recs)
        
    def predict_unrated_recipes(self, user_id):
        """
        return a sorted list with tuples of recipe id and the user's predicted rating
        for all unrated recipes
        """
        user_rated_recipes = frozenset(self.dataLoader.get_user_ratings(user_id, {}))
        unrated_recipes = self.recipe_ids - user_rated_recipes
        X = [(user_id, recipe_id) for recipe_id in unrated_recipes]
        predictions = self.estimator.predict(X)
        res = []
        for i, info in enumerate(X):
            _, recipe_id = info
            res.append((recipe_id, predictions[i]))
        return sorted(res,key=lambda x: -x[1])
    
    def select_diverse_recipes(self, ratings, n_recs, search_limit=4000):
        """
        ratings: list of (recipe_id, rating) tuples, sorted in descending
            order by rating
        search_limit: only searches through the top search_limit rated
            recipes to make predictions (improves speed)
        """
        if search_limit:
            ratings = ratings[:search_limit]
        recs = []
        original_ratings = {id: rating for id, rating in ratings}
        for _ in range(n_recs):
            target_recipe_id, _ = ratings.pop(0)
            recs.append(target_recipe_id)
            new_ratings = []
            for i, item in enumerate(ratings):
                recipe_id, _ = item
                predicted_rating = original_ratings[recipe_id]
                revised_score = predicted_rating + self.diversity_weight * self.diversity(recs, recipe_id)
                if revised_score == predicted_rating:
                    new_ratings = new_ratings + ratings[i:]
                    break
                new_ratings.append((recipe_id, revised_score))
            ratings = sorted(new_ratings, key=lambda x: -x[1])
        return recs
            