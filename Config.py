
class Configuration:
    def __init__(self):
        self.using_cuda = False
        self.device = 'cuda'
        self.learning_rate = 0.00125
        self.learning_rate_steps = [40, 70]
        self.learning_rate_lambda = 0.1
        self.momentum = 0.9
        self.decay = 0.0002
        self.number_of_epochs = 80
        self.iterations = 25
        self.publishing_losses_frequency = 100
        self.ckpt_path = 'ckpt'
        self.dataset_dir = 'Dataset'
        self.seed = 1234
        self.learning_epoch = 0.1
        self.number_of_iterations = 10
        self.number_of_classes = 2


