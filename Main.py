import cv2
import numpy as np
import streamlit as st
import os 
import sys
sys.path.insert(0, '.')
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():
    st.title("Sementic segmentation demo")
    #img_path = './images/'
    #img_name = st.text_input("Please input image name")
    uploaded_file = st.file_uploader("Choose a image file", type=['png', 'jpg','jpeg','mp4'])
    

    if uploaded_file is not None:
    # Convert the file to an opencv image.
        path_in = uploaded_file.name
        print(path_in)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="BGR")
        demo(path_in)

def demo(img,name="segment output"):
    torch.set_grad_enabled(False)
    np.random.seed(123)
    # args
    cfg = set_cfg_from_file('./configs/bisenetv2_city.py')


    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
    # define model
    net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
    net.load_state_dict(torch.load('./model/model_final_v2_city.pth', map_location='cpu'), strict=False)
    net.eval()
    net.cuda()

    # prepare data
    to_tensor = T.ToTensor(
        mean=(0.3257, 0.3690, 0.3223), # city, rgb
        std=(0.2112, 0.2148, 0.2115),
    )

    im = cv2.imread(img)[:, :, ::-1]
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

    # shape divisor
    org_size = im.size()[2:]
    new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

    # inference
    im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
    out = net(im)[0]
    out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
    out = out.argmax(dim=1)
    # visualize
    out = out.squeeze().detach().cpu().numpy()
    pred = palette[out]
    cv2.imwrite('./'+name+'.jpg', pred)
    st.image('./'+name+'.jpg')

if __name__ == "__main__":
    main()



    
