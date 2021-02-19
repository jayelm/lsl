import argparse

class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='LSL')
        self.parser.add_argument('exp_dir', type=str, help='Output directory')
        hyp_prediction = self.parser.add_mutually_exclusive_group()
        hyp_prediction.add_argument(
            '--predict_concept_hyp',
            action='store_true',
            help='Predict concept hypotheses during training')
        hyp_prediction.add_argument(
            '--predict_image_hyp',
            action='store_true',
            help='Predict image hypotheses during training')
        hyp_prediction.add_argument('--infer_hyp',
                                    action='store_true',
                                    help='Use hypotheses for prediction')
        self.parser.add_argument('--backbone',
                            choices=['vgg16_fixed', 'conv4', 'resnet18'],
                            default='vgg16_fixed',
                            help='Image model')
        self.parser.add_argument(
            '--multimodal_concept',
            action='store_true',
            help='Concept is a combination of hypothesis + image rep')
        self.parser.add_argument('--comparison',
                            choices=['dotp', 'bilinear'],
                            default='dotp',
                            help='How to compare support to query reps')
        self.parser.add_argument('--dropout',
                            default=0.0,
                            type=float,
                            help='Apply dropout to comparison layer')
        self.parser.add_argument('--debug_bilinear',
                            action='store_true',
                            help='If using bilinear term, use identity matrix')
        self.parser.add_argument(
            '--poe',
            action='store_true',
            help='Product of experts: support lang -> query img '
                'x support img -> query img'
        )
        self.parser.add_argument('--predict_hyp_task',
                            default='generate',
                            choices=['generate', 'embed'],
                            help='hyp prediction task')
        self.parser.add_argument('--n_infer',
                            type=int,
                            default=10,
                            help='Number of hypotheses to infer')
        self.parser.add_argument(
            '--oracle',
            action='store_true',
            help='Use oracle hypotheses for prediction (requires --infer_hyp)')
        self.parser.add_argument('--max_train',
                            type=int,
                            default=None,
                            help='Max number of training examples')
        self.parser.add_argument('--noise',
                            type=float,
                            default=0.0,
                            help='Amount of noise to add to each example')
        self.parser.add_argument(
            '--class_noise_weight',
            type=float,
            default=0.0,
            help='How much of that noise should be class diagnostic?')
        self.parser.add_argument('--noise_at_test',
                            action='store_true',
                            help='Add instance-level noise at test time')
        self.parser.add_argument('--noise_type',
                            default='gaussian',
                            choices=['gaussian', 'uniform'],
                            help='Type of noise')
        self.parser.add_argument(
            '--fixed_noise_colors',
            default=None,
            type=int,
            help='Fix noise based on class, with a max of this many')
        self.parser.add_argument(
            '--fixed_noise_colors_max_rgb',
            default=0.2,
            type=float,
            help='Maximum color value a single color channel '
                'can have for noise background'
        )
        self.parser.add_argument('--batch_size',
                            type=int,
                            default=100,
                            help='Train batch size')
        self.parser.add_argument('--epochs', type=int, default=50, help='Train epochs')
        self.parser.add_argument(
            '--data_dir',
            default=None,
            help='Specify custom data directory (must have shapeworld folder)')
        self.parser.add_argument('--lr',
                            type=float,
                            default=0.0001,
                            help='Learning rate')
        self.parser.add_argument('--tre_err',
                            default='cos',
                            choices=['cos', 'l1', 'l2'],
                            help='TRE Error Metric')
        self.parser.add_argument('--tre_comp',
                            default='add',
                            choices=['add', 'mul'],
                            help='TRE Composition Function')
        self.parser.add_argument('--optimizer',
                            choices=['adam', 'rmsprop', 'sgd'],
                            default='adam',
                            help='Optimizer to use')
        self.parser.add_argument('--seed', type=int, default=1, help='Random seed')
        self.parser.add_argument('--language_filter',
                            default=None,
                            type=str,
                            choices=['color', 'nocolor'],
                            help='Filter language')
        self.parser.add_argument('--shuffle_words',
                            action='store_true',
                            help='Shuffle words for each caption')
        self.parser.add_argument('--shuffle_captions',
                            action='store_true',
                            help='Shuffle captions for each class')
        self.parser.add_argument('--log_interval',
                            type=int,
                            default=10,
                            help='How often to log loss')
        self.parser.add_argument('--pred_lambda',
                            type=float,
                            default=1.0,
                            help='Weight on prediction loss')
        self.parser.add_argument('--hypo_lambda',
                            type=float,
                            default=10.0,
                            help='Weight on hypothesis hypothesis')
        self.parser.add_argument('--save_checkpoint',
                            action='store_true',
                            help='Save model')
        self.parser.add_argument('--cuda',
                            action='store_true',
                            help='Enables CUDA training')
                            
    def parse_args(self):
        args = self.parser.parse_args()

        return args