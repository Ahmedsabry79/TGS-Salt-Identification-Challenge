
class make_checkpoint:
    
    def __init__(self, chechpoint_path, every_ckpt_path, last_ckpt_path, saver, session, await_epochs = 20):
        self.await_epochs = await_epochs
        self.saver = saver
        self.session = session
        self.chechpoint_path = chechpoint_path
        self.last_ckpt_path = last_ckpt_path
        self.every_ckpt_path = every_ckpt_path
        self.accuracies = []
        self.end_epochs = False
        
    def add(self, epoch, accuracy):
        self.accuracy = accuracy
        self.epoch = epoch
        self.accuracies.append(self.accuracy)
        self.max_index = self.accuracies.index(max(self.accuracies))+1
        self.current_index = self.epoch
        
        self.save_model(mode = 'every')
        if self.current_index == self.max_index: self.save_model(mode = 'still')
        if self.max_index + self.await_epochs == self.current_index:
            self.save_model(mode = 'last')
            self.end_epochs = True
        
    def save_model(self, mode):
        if mode == 'still':
            self.saver.save(self.session, self.chechpoint_path)
        if mode == 'last':
            self.saver.save(self.session, self.last_ckpt_path)
        if mode == 'every':
            self.saver.save(self.session, self.every_ckpt_path)

    
        
#class make_checkpoint:
#    
#    def __init__(self, chechpoint_path, saver, session, await_epochs = 20, mode = 'test'):
#        self.await_epochs = await_epochs
#        self.saver = saver
#        self.session = session
#        self.chechpoint_path = chechpoint_path
#        self.mode = mode
#        self.train_accuracies = []
#        self.test_accuracies = []
#        self.val_accuracies = []
#        self.end_epochs = False
#        
#    def add(self, epoch, train_accuracy = None, val_accuracy = None, test_accuracy = None):
#        self.train_accuracy = train_accuracy
#        self.test_accuracy = test_accuracy
#        self.val_accuracy = val_accuracy
#        self.epoch = epoch
#        if self.mode == 'test':
#            self.test_accuracies.append(self.test_accuracy)
#            self.max_index = self.test_accuracies.index(max(self.test_accuracies))+1
#            self.current_index = self.epoch
#            if self.max_index + self.await_epochs == self.current_index:
#                self.save_model()
#                self.end_epochs = True
#        if self.mode == 'train':
#            self.train_accuracies.append(self.train_accuracy)
#            self.max_index = self.train_accuracies.index(max(self.train_accuracies))+1
#            self.current_index = self.epoch
#            if self.max_index + self.await_epochs == self.current_index:
#                self.save_model()
#                self.end_epochs = True
#        if self.mode == 'val':
#            self.val_accuracies.append(self.val_accuracy)
#            self.max_index = self.val_accuracies.index(max(self.val_accuracies))+1
#            self.current_index = self.epoch
#            if self.max_index + self.await_epochs == self.current_index:
#                self.save_model()
#                self.end_epochs = True
#        
#    def save_model(self):
#        self.saver.save(self.session, self.chechpoint_path)