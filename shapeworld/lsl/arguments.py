import argparse

class ArgumentParser:
    
    def __init__(self):
        parser = argparse.ArgumentParser()
        self.parser = parser
        parser.add_argument('exp_dir', type=str, help='Output directory')
        hyp_prediction = parser.add_mutually_exclusive_group()
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
        parser.add_argument('--backbone',
                            choices=['vgg16_fixed', 'conv4', 'resnet18', 'lxmert'],
                            default='vgg16_fixed',
                            help='Image model')
        parser.add_argument(
            '--multimodal_concept',
            action='store_true',
            help='Concept is a combination of hypothesis + image rep')
        parser.add_argument('--comparison',
                            choices=['dotp', 'bilinear'],
                            default='dotp',
                            help='How to compare support to query reps')
        parser.add_argument('--dropout',
                            default=0.0,
                            type=float,
                            help='Apply dropout to comparison layer')
        parser.add_argument('--debug_bilinear',
                            action='store_true',
                            help='If using bilinear term, use identity matrix')
        parser.add_argument(
            '--poe',
            action='store_true',
            help='Product of experts: support lang -> query img '
                'x support img -> query img'
        )
        parser.add_argument('--predict_hyp_task',
                            default='generate',
                            choices=['generate', 'embed'],
                            help='hyp prediction task')
        parser.add_argument('--n_infer',
                            type=int,
                            default=10,
                            help='Number of hypotheses to infer')
        parser.add_argument(
            '--oracle',
            action='store_true',
            help='Use oracle hypotheses for prediction (requires --infer_hyp)')
        parser.add_argument(
            '--hint_retriever',
            choices=['dotp', 'l2', "cos"],
            help='use the hint of tasks seen during training time (requires --infer_hyp)')
        parser.add_argument('--max_train',
                            type=int,
                            default=None,
                            help='Max number of training examples')
        parser.add_argument('--noise',
                            type=float,
                            default=0.0,
                            help='Amount of noise to add to each example')
        parser.add_argument(
            '--class_noise_weight',
            type=float,
            default=0.0,
            help='How much of that noise should be class diagnostic?')
        parser.add_argument('--noise_at_test',
                            action='store_true',
                            help='Add instance-level noise at test time')
        parser.add_argument('--noise_type',
                            default='gaussian',
                            choices=['gaussian', 'uniform'],
                            help='Type of noise')
        parser.add_argument(
            '--fixed_noise_colors',
            default=None,
            type=int,
            help='Fix noise based on class, with a max of this many')
        parser.add_argument(
            '--fixed_noise_colors_max_rgb',
            default=0.2,
            type=float,
            help='Maximum color value a single color channel '
                'can have for noise background'
        )
        parser.add_argument('--batch_size',
                            type=int,
                            default=100,
                            help='Train batch size')
        parser.add_argument('--epochs', type=int, default=50, help='Train epochs')
        parser.add_argument(
            '--data_dir',
            default=None,
            help='Specify custom data directory (must have shapeworld folder)')
        parser.add_argument('--lr',
                            type=float,
                            default=0.0001,
                            help='Learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.05,
                            help='Warm up ratio')
        parser.add_argument('--initializer_range',
                            type=float,
                            default=0.02,
                            help='The std of the truncated_normal_initializer for initializing all weights')
        parser.add_argument('--tre_err',
                            default='cos',
                            choices=['cos', 'l1', 'l2'],
                            help='TRE Error Metric')
        parser.add_argument('--tre_comp',
                            default='add',
                            choices=['add', 'mul'],
                            help='TRE Composition Function')
        parser.add_argument('--optimizer',
                            choices=['adam', 'bertadam', 'rmsprop', 'sgd'],
                            default='adam',
                            help='Optimizer to use')
        parser.add_argument('--seed', type=int, default=1, help='Random seed')
        parser.add_argument('--language_filter',
                            default=None,
                            type=str,
                            choices=['color', 'nocolor'],
                            help='Filter language')
        parser.add_argument('--shuffle_words',
                            action='store_true',
                            help='Shuffle words for each caption')
        parser.add_argument('--shuffle_captions',
                            action='store_true',
                            help='Shuffle captions for each class')
        parser.add_argument('--log_interval',
                            type=int,
                            default=10,
                            help='How often to log loss')
        parser.add_argument('--pred_lambda',
                            type=float,
                            default=1.0,
                            help='Weight on prediction loss')
        parser.add_argument('--hypo_lambda',
                            type=float,
                            default=10.0,
                            help='Weight on hypothesis hypothesis')
        parser.add_argument('--save_checkpoint',
                            action='store_true',
                            help='Save model')
        parser.add_argument('--cuda',
                            action='store_true',
                            help='Enables CUDA training')
        parser.add_argument(
            '--scheduled_sampling',
            action='store_true',
            help='Use scheduled samping during training')
        parser.add_argument(
            '--plot_bleu_score',
            action='store_true',
            help='Use scheduled samping during training')


    def parse_args(self):
        args = self.parser.parse_args()

        if args.oracle and not args.infer_hyp:
            self.parser.error("Must specify --infer_hyp to use --oracle")

        if args.hint_retriever and not args.infer_hyp:
            self.parser.error("Must specify --infer_hyp to use --hint_retriever")

        if args.multimodal_concept and not args.infer_hyp:
            self.parser.error("Must specify --infer_hyp to use --multimodal_concept")

        if args.poe and not args.infer_hyp:
            self.parser.error("Must specify --infer_hyp to use --poe")

        if args.dropout > 0.0 and args.comparison == 'dotp':
            raise NotImplementedError
        
        args.predict_hyp = args.predict_concept_hyp or args.predict_image_hyp
        args.use_hyp = args.predict_hyp or args.infer_hyp
        args.encode_hyp = args.infer_hyp or (args.predict_hyp and args.predict_hyp_task == 'embed')
        args.decode_hyp = args.infer_hyp or (args.predict_hyp and args.predict_hyp_task == 'generate')

        if args.oracle or args.hint_retriever: 
            args.n_infer = 1  # No need to repeatedly infer, hint is given
        
        return args
