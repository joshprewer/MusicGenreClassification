from utilities import svm_objective_function
import numpy as np
import random

class SAHS(object):
    def __init__(self,harmony_obj):
        self.obj_func = harmony_obj
        self.hmm_matrix = list()
        matrix = []
        for limit in self.obj_func.up_down_limit:
            row = np.random.uniform(low=limit[0], high=limit[1], size=(1,self.obj_func.harmony_menmory_size))[0] 
            matrix.append(row)
        matrix = np.asarray(matrix).transpose().round(self.obj_func.weight_decimal)
        self.hmm_matrix = matrix

    def run(self):
        hmm_score_list = [0] * len(self.hmm_matrix)
        for m_i in range(len(self.hmm_matrix)):
            feature_set = self.hmm_matrix[m_i]

            score = self.obj_func.fitness(feature_set,self.obj_func.input_X,self.obj_func.input_Y)
            hmm_score_list[m_i] = score
        
        for itera in range(self.obj_func.iteration):
            feature_set = [0] * self.obj_func.vector_size
            
            for i in range(self.obj_func.vector_size):
                if np.random.rand(1,)[0] < self.obj_func.hmcr_proba:
                    hms_index = random.randint(0, self.obj_func.harmony_menmory_size - 1)                 
                    new_feature = self.hmm_matrix[hms_index][i]
                    
                    if np.random.rand(1,)[0] < self.obj_func.adju_proba:                            
                        new_feature = new_feature - (new_feature - self.obj_func.up_down_limit[i][0]) * np.random.rand(1,)[0]
                    else:                        
                        new_feature = new_feature + (self.obj_func.up_down_limit[i][1] - new_feature) * np.random.rand(1,)[0]
                                    
                    feature_set[i] = round(new_feature)
                
                else: 
                    feature_set[i] = random.randint(0, 1)

            score = self.obj_func.fitness(feature_set, self.obj_func.input_X, self.obj_func.input_Y)
            worst_score_index = hmm_score_list.index(min(hmm_score_list))
            
            if score > hmm_score_list[worst_score_index]:
                self.hmm_matrix[worst_score_index] = feature_set
                hmm_score_list[worst_score_index] = score
            
            if max(hmm_score_list) == 1:
                break


        return self.hmm_matrix,hmm_score_list,hmm_score_list.index(max(hmm_score_list))
            
            