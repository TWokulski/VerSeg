using_cuda = False
learning_rate = 0.00125
learning_rate_steps = [40, 70]
learning_rate_lambda = 0.1
momentum = 0.9
decay = 0.0002
number_of_epochs = 80
iterations = 25
publishing_losses_frequency = 100
ckpt_path = './ckpt'
dataset_dir = './Dataset'
seed = 1234

number_of_classes = 2

model_url = {
            'maskrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        }
