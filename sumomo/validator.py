from sklearn.metrics import ( 
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score,
    log_loss,
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score,
    r2_score,
    max_error
)


class Validator:
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.log_loss = None
        self.mae = None
        self.mse = None
        self.mape = None
        self.median_ae = None
        self.evs = None
        self.r2 = None
        self.maxe = None

    def calculate(self, x, y):
        if self._is_classifier():
            prob_test, class_test = self.model.predict(x, return_class=True)
            self.accuracy = accuracy_score(y, class_test)
            self.precision = precision_score(y, class_test)
            self.recall = recall_score(y, class_test)
            self.f1 = f1_score(y, class_test)
            self.log_loss = log_loss(y, prob_test)
        else:
            # note that these regression errors are in the scaled space
            pred_test_ = self.model.predict(x)
            pred_test = self.scaler.inv_scale_y(pred_test_)
            self.mae = mean_absolute_error(y, pred_test)
            self.mse = mean_squared_error(y, pred_test)
            self.mape = mean_absolute_percentage_error(y, pred_test)
            self.median_ae = median_absolute_error(y, pred_test)
            self.evs = explained_variance_score(y, pred_test)
            self.r2 = r2_score(y, pred_test)
            self.maxe = max_error(y, pred_test)
            
    def _is_classifier(self):
        # TODO - update this to enable NN classifiers to be caught too
        if self.model.name == 'GPC':
            return True
        else:
            return False
    