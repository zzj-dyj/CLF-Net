import sys
sys.path.append('../')
import torch
import torch.nn as nn
from core.model import *
from core.loss.PatchSampleF import *
from core.loss.PatchNCELoss import *
from core.loss.SSIM_Loss import *
from packaging import version

class CLF_Net(nn.Module):
    """docstring for CLF_Net"""

    def __init__(self, config, kernal_size=11, num_channels=1, C=9e-4, device='cuda:0'):
        super(CLF_Net, self).__init__()
        self.config = config
        self.num_patches = 200
        self.nce_layers = 1
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.criterionNCE = []
        for _ in range(2):
            self.criterionNCE.append(PatchNCELoss())
        self.optimizers = []
        self.device = device
        self.p = 2
        self.kernal_size = kernal_size
        self.num_channels = num_channels
        self.avg_kernal = torch.ones(num_channels, 1, self.kernal_size, self.kernal_size) / (self.kernal_size) ** 2
        self.avg_kernal = self.avg_kernal.to(device)

        self.patchsample = define_F()
        self.Generator = Generator(config['input_channels'], config['out_channels'])

        self.optimizer_Generator = torch.optim.Adam(self.Generator.parameters(), lr=0.001)
        self.optimizers.append(self.optimizer_Generator)


    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_scheduler(self, optimizer):

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.001)

        return scheduler

    def setup(self):
        self.schedulers = [self.get_scheduler(optimizer) for optimizer in self.optimizers]

    def data_dependent_initialize(self, vis_img, inf_img):
        self.forward(vis_img, inf_img)
        self.calculate_loss(vis_img, inf_img).backward()
        self.optimizer_f = torch.optim.Adam(self.patchsample.parameters(), lr=0.001)
        self.optimizers.append(self.optimizer_f)

    def optimize_parameters(self, vis_img, inf_img):
        # forward
        self.forward(vis_img, inf_img)

        self.optimizer_Generator.zero_grad()
        self.optimizer_f.zero_grad()
        self.loss = self.calculate_loss(vis_img, inf_img)
        self.loss.backward()
        self.optimizer_Generator.step()
        self.optimizer_f.step()

    def forward(self, vis_img, inf_img):

        feat = self.Generator(vis_img, inf_img, only_encoder=False)
        self.outputs = feat[2]

        return self.outputs

    def calculate_loss(self, vis_img, inf_img):
        lambda1 = 1
        lambda2 = 1

        self.loss = lambda1 * self.calculate_NCE_loss(vis_img, inf_img) + lambda2 * self.calculate_SSIM_loss(vis_img, inf_img)
        #+ self.calculate_content_loss(vis_img, inf_img)
        return self.loss



    def calculate_NCE_loss(self, vis_img, inf_img):
        vis_images, inf_images, fusion_images = vis_img, inf_img, self.outputs
        batchsize = vis_images.shape[0]


        feat0 = self.Generator(vis_images, inf_images, only_encoder=True)
        feat0_f = self.Generator(fusion_images, fusion_images, only_encoder=True)

        feat_f_v_pool, sample_ids = self.patchsample([feat0_f[0]], self.num_patches, None)
        feat_f_i_pool, _ = self.patchsample([feat0_f[1]], self.num_patches, sample_ids)
        feat_v_pool, _ = self.patchsample([feat0[0]], self.num_patches, sample_ids)
        feat_i_pool, _ = self.patchsample([feat0[1]], self.num_patches, sample_ids)


        total_nce_loss = 0.0

        for f_f_v, f_f_i, f_v, f_i, crit in zip(feat_f_v_pool, feat_f_i_pool, feat_v_pool, feat_i_pool, self.criterionNCE):
            loss = crit(f_f_v, f_f_i, f_v, f_i)
            total_nce_loss += loss.mean()
        return total_nce_loss

    def calculate_SSIM_loss(self, vis_img, inf_img):
        vis_images, inf_images, fusion_images = vis_img, inf_img, self.outputs

        ssim_loss = 1 - ssim(fusion_images, inf_images, vis_images)


        return ssim_loss

    def calculate_content_loss(self, vis_img, inf_img):
        vis_images, inf_images, fusion_images = vis_img, inf_img, self.outputs

        # content_loss
        content_loss = torch.mean(torch.square(fusion_images - inf_images)) + 5 *torch.mean(torch.square(self.gradient(fusion_images) - self.gradient(vis_images)))

        return content_loss

    def gradient(self, x):
        with torch.no_grad():
            laplace = [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
            kernel = torch.FloatTensor(laplace).unsqueeze(0).unsqueeze(0).to(self.device)
            return F.conv2d(x, kernel, stride=1, padding=1)