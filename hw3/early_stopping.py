import math

class EarlyStopping():
    """
    Early stopping implementation

    If the difference the current score does not improve within
    param:patience cycles, returns True to stop

    NOTE:
    param:patience should be higher and param:delta should be lower in practice,
    but because of limited computing power (laptops with no GPU)
    we choose a very low threshold to stop earlier.
    """
    def __init__(self,
                mode='greater than',
                metric='ndcg',
                min_delta=0.01,
                patience=7):

        self.mode = mode
        self.metric = metric
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best = -math.inf

        if mode not in ['less than', 'greater than']:
            raise Exception('mode must be in ("less than", "greater than")')

        if mode == 'less than':
            self.min_delta *= -1
            self.best = math.inf


    def monitor(self, results):
        current = results[self.metric][0] #[0] is the mean [1] is stdev
        
        if self.best < current - self.min_delta:
            self.best = current
            self.wait = 0
            print('EarlyStopping: new best')
            return False
        else:
            self.wait += 1
            print('EarlyStopping: wait %d of %d' % (self.wait, self.patience))
            if self.wait == self.patience:
                return True