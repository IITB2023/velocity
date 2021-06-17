import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import imageio
import matplotlib.pyplot as plt

from network import RAFTGMA
from core.utils.utils import InputPadder
import os


DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, flow_dir):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    imageio.imwrite(os.path.join(flow_dir, 'flo.png'), flo)
    print(f"Saving optical flow visualisation at {os.path.join(flow_dir, 'flo.png')}")


def normalize(x):
    return x / (x.max() - x.min())

def load_model(args):

    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model))
    print(f"Loaded checkpoint at {args.model}")

    return model



def demo(model, imgs_path):
    
    model = model.module
    model.to(DEVICE)
    model.eval()
    input1 = np.zeros((39,3,720,1280))
    input2 = np.zeros((39,3,720,1280))

    with torch.no_grad():

            for i in range(1,40):
                
                one = i
                two = i+1
                if one<10:
                    one = '00' + str(one) + '.jpg'
                else:
                    one = '0' + str(one) + '.jpg'
                if two<10:
                    two = '00' + str(two) + '.jpg'
                else:
                   two = '0' + str(two) + '.jpg'
                file1 = imgs_path + '/' + one
                file2 = imgs_path + '/' + two
                
                image1 = load_image(file1)
                image2 = load_image(file2)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                
                input1[i-1] = image1.cpu().numpy()
                input2[i-1] = image2.cpu().numpy()
                
            input1 = torch.from_numpy(input1).float()
            input2 = torch.from_numpy(input2).float()
            flow_low, flow_up = model(input1, input2, iters=12, test_mode=True)
    
    return flow_low, flow_up
