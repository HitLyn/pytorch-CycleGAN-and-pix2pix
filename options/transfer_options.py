from .base_options import BaseOptions


class TransferOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        #  Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        parser.add_argument('--env', type=str, default='SawyerLift', help='the env to use')
        parser.add_argument('--policy_path', type=str,
                            default="/home/jonathan/Downloads/learner.24000.ckpt.cpu",
                            help='path to the surreal weights file')
        parser.add_argument('--save_obs', action='store_true', help='if we should save the image obs from the rollout')
        parser.add_argument('--obs_save_path', type=str, default='./datasets/rollout')

        parser.add_argument('--collision', action='store_true', help='are we running transfer on the collision mesh')
        parser.set_defaults(model='test')

        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain = False
        return parser
