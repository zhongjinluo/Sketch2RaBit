import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
sys.path.append("networks/v0/pSp/")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp

import torchvision.transforms as transforms

class Sketch2TextureStyle:
    def __init__(self):
        self.to_tensor = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        test_opts = TestOptions().parse()
        test_opts.checkpoint_path = "networks/v0/pSp/out/render_masked/checkpoints/best_model_uv.pt"
        test_opts.test_batch_size = 1
        test_workers = 1
        ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
        opts = ckpt['opts']
        opts.update(vars(test_opts))
        if 'learn_in_w' not in opts:
            opts['learn_in_w'] = False
        if 'output_size' not in opts:
            opts['output_size'] = 1024
        self.opts = Namespace(**opts)

        self.net = pSp(self.opts)
        self.net.eval()
        self.net.cuda()

    def predict(self, image):
        input_batch = self.to_tensor(image)
        input_cuda = input_batch.unsqueeze(0).cuda().float()
        res = self.net(input_cuda,
                randomize_noise=False,
                resize=self.opts.resize_outputs)
        result = tensor2im(res[0])
        return Image.fromarray(np.array(result)).resize((512, 512))


if __name__ == '__main__':
    s2ts = Sketch2TextureStyle()
    for f in os.listdir("/program/huawei/something/datasets/render_masked_test/"):
        f = "100.png"
        image = Image.open("/program/huawei/something/datasets/render_masked_test/" + f).convert('RGB')
        image.save("i.png")
        image = s2ts.predict(image)
        image.save("xj.png")
        print(f)
        break
