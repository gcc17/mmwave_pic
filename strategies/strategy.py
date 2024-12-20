
class Strategy:
    def __init__(self, coreset_dataset, model):
        self.coreset_dataset = coreset_dataset
        self.model = model
    
    def query(self, n):
        pass