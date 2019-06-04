class PretrainedEstimator:
    def __init__(self, rfecv_predictor, logit_estimator):
        self.rfecv_predictor = rfecv_predictor
        self.logit_estimator = logit_estimator

    def predict(self, X):
        feats = self.logit_estimator.extract_features(X)
        feats = self.logit_estimator.vec.transform(feats)
        return self.rfecv_predictor.predict(feats)