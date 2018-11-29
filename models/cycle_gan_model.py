import torch
import torchvision
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import math
import torchsample


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_shift_A', type=float, default=0.01, help='weight for shift loss for A')
            parser.add_argument('--lambda_shift_B', type=float, default=0.01, help='weight for shift loss for B')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'shift_A', 'shift_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionShift = torch.nn.MSELoss(size_average=False)
            self.shift_transform = torchsample.transforms.RandomTranslate((1./8., 1./8.))

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def inference(self, direction, image):
        if direction not in ['AtoB', 'BtoA']:
            raise ValueError('{} is not a valid direction'.format(direction))

        with torch.no_grad():
            #image = torch.from_numpy(image.copy()).to(self.device)
            if direction == 'AtoB':
                return self.netG_A(image)
            else:
                return self.netG_B(image)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_shift_A = self.opt.lambda_shift_A
        lambda_shift_B = self.opt.lambda_shift_B

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        #Shift losses from VR-Goggles for Robots
        real_A = self.real_A.cpu() #((self.real_A + 1.) / 2. * 255) #.int().numpy()
        real_B = self.real_B.cpu()  #((self.real_B + 1.) / 2. * 255.) #.int().numpy()
        #print(self.real_A[0].shape, type(self.real_A[0]), self.real_A[0])
        real_A = torch.unbind(real_A, 0)
        real_B = torch.unbind(real_B, 0)

        fake_A = self.fake_A.cpu()
        fake_B = self.fake_B.cpu()

        fake_A = torch.unbind(fake_A, 0)
        fake_B = torch.unbind(fake_B, 0)


        shifted_real_A, height_A, width_A = self.shift_transform(*real_A)
        shifted_real_B, height_B, width_B = self.shift_transform(*real_B)

        gen_B = self.netG_A(torch.stack(shifted_real_A, 0).cuda())
        gen_A = self.netG_B(torch.stack(shifted_real_B, 0).cuda())

        shifted_fake_A, _, _ = self.shift_transform(*fake_A, random_height=height_B, random_width=width_B) # netG_B
        shifted_fake_B, _, _ = self.shift_transform(*fake_B, random_height=height_A, random_width=width_A) # netG_A
        shifted_fake_A = torch.stack(shifted_fake_A).cuda()
        shifted_fake_B = torch.stack(shifted_fake_B).cuda()

        import cv2
        import numpy as np
        cv2.imshow('shifted_real_A', ((shifted_real_A[0].detach().cpu().numpy() + 1.) / 2. * 255.).astype(np.uint8).transpose([1,2,0]))
        cv2.imshow('real_A', ((real_A[0].detach().cpu().numpy() + 1.) / 2. * 255.).astype(np.uint8).transpose([1,2,0]))

        cv2.imshow('fake_B', ((fake_B[0].detach().cpu().numpy() + 1.) / 2. * 255.).astype(np.uint8).transpose([1,2,0]))

        cv2.imshow('gen_B', ((gen_B.detach().cpu().numpy() + 1.) / 2. * 255.).astype(np.uint8)[0].transpose([1,2,0]))
        cv2.imshow('shifted_fake_B', ((shifted_fake_B[0].detach().cpu().numpy() + 1.) / 2. * 255.).astype(np.uint8).transpose([1,2,0]))
        cv2.waitKey(1)

        self.loss_shift_A = self.criterionShift(shifted_fake_A, gen_A) * lambda_shift_A
        self.loss_shift_B = self.criterionShift(shifted_fake_B, gen_B) * lambda_shift_B

        print(self.criterionShift(shifted_fake_A, gen_A), self.criterionShift(shifted_fake_B, gen_B))

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B + self.loss_shift_A + self.loss_shift_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
