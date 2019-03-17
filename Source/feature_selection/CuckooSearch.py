import numpy as np
from sklearn import utils, model_selection

class Evaluate:
    def __init__(self, input_X, input_Y, clf):
        self.input_X = input_X
        self.input_Y = input_Y
        self.clf = clf
        
    def evaluate(self,gen):
        feature_sets = np.nonzero(gen)[0]

        if feature_sets.size == 0:
            return 0
        else:
            x = np.take(self.input_X, feature_sets, axis=1)
            xs, ys = utils.shuffle(x, self.input_Y)

            cv = len(np.unique(ys))
            cv_results = model_selection.cross_val_score(self.clf, xs, ys, cv=cv)            
        
            return cv_results.mean()
                
        
    def check_dimentions(self,dim):#check number of all feature
        if dim==None:
            return len(self.input_X[0])
        else:
            return dim