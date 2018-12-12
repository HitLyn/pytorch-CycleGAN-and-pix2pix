from .base_options import BaseOptions


class DataGenOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        #  Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--policy_path', type=str,
                            default="/home/jonathan/Downloads/learner.24000.ckpt.cpu",
                            help='path to the surreal weights file')
        parser.add_argument('--save_obs', action='store_true', help='if we should save the image obs from the rollout')
        parser.add_argument('--dataset-name', type=str, default='sawyer_gen')
        parser.add_argument('--data_root', type=str, default='./datasets')
        parser.add_argument('--states-file', type=str, default='states')
        parser.add_argument('--internal', action='store_true', help='should we use robosuite internal')
        parser.add_argument('--collision', action='store_true', help='should we render the collision mesh instead')

        parser.add_argument('--n_rollouts', type=int, default=100, help='the number of rollouts we should perform')
        parser.add_argument('--n_steps', type=int, default=200, help='the number of steps per rollout we should perform')

        parser.add_argument('--mode', type=str, default='rollout', help='the mode of data generation we should run')
        parser.add_argument('--size', type=int, default=256, help='size of images we should make')

        parser.add_argument('--noisy', action='store_true', help='should we set state to noisy value from file')
        parser.set_defaults(model='test')

        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain = False
        return parser
