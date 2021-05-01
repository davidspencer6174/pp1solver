import numpy as np

class PredictionBatcher:
    
    def __init__(self, model):
        self.inputs = []
        self.calling_nodes = []
        self.model = model
    
    def add_inp(self, arr, calling_node):
        self.inputs.append(arr)
        self.calling_nodes.append(calling_node)
        
    def predict(self):
        #print(np.array(self.inputs).shape)
        #print('Predicting for a batch')
        if len(self.inputs) == 0:
            return []
        model_results = self.model.predict(np.array(self.inputs))
        #print('Finished a batch')
        #print(model_results)
        #print(len(model_results))
        for i in range(len(self.inputs)):
            self.calling_nodes[i].expand_result_callback(model_results[0][i],
                                                         model_results[1][i][0])
        self.inputs = []
        self.calling_nodes = []
            