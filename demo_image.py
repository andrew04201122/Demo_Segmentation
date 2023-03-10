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
import torch.multiprocessing as mp
import time

color_dict = {0:'road',1:"sidewalk",2:"building",3:"wall",4:"fence",5:"pole",6:"traffic light",7:"traffic sign", 8:"Vegetation", 9:"terrain", 10:"sky",11:"person",12:"rider", 13:"car", 14:"truck", 15:"bus",16:"train",17:"motorcycle",18:"bicycle"}

np.random.seed(123)
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


def demo(img_path,name,img_type):
    
    torch.set_grad_enabled(False)
    
    # args
    cfg = set_cfg_from_file('./configs/bisenetv2_city.py')


    
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

    im = cv2.imread(img_path)[:, :, ::-1]
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
    color = np.unique(out)

    cv2.imwrite('./output/'+name+'.'+img_type, pred)
    #st.image('./output/'+name+'.'+img_type)


    



def main():
    st.set_page_config(
        page_title="demo_image",
    )   

    st.title("Sementic segmentation demo")
    
    img_folder = './img_input/'
    img_name = st.text_input("Please input image name")
    img_type = st.radio('Which output file do you want to use', options = ("png","jpg","jpeg"))
    uploaded_file = st.file_uploader("Choose a image file", type=['png', 'jpg','jpeg'])
    st.sidebar.success("Select a page above.")

    if uploaded_file is not None:
    # Convert the file to an opencv image.
        input_img_name = uploaded_file.name
        img_path = img_folder + input_img_name  #./img_input/example.png
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="BGR")
        
        start = time.time()
        demo(img_path,img_name,img_type)
        end = time.time()
        if st.button("show"):
            st.image('./output/'+img_name+'.'+img_type)
            with st.container():
                st.text(f"Inference time : {end - start :.2f} second")
                st.text("The meaning of colors")
                col1,col2,col3,col4,col5,col6 = st.columns(6)
                with col1:
                    st.color_picker(color_dict[0],rgb_to_hex(tuple(palette[0][[2,1,0]])),disabled = False )
                with col2:
                    st.color_picker(color_dict[1],rgb_to_hex(tuple(palette[1][[2,1,0]])),disabled = False )
                with col3:
                    st.color_picker(color_dict[2],rgb_to_hex(tuple(palette[2][[2,1,0]])),disabled = False )
                with col4:
                    st.color_picker(color_dict[3],rgb_to_hex(tuple(palette[3][[2,1,0]])),disabled = False )
                with col5:
                    st.color_picker(color_dict[4],rgb_to_hex(tuple(palette[4][[2,1,0]])),disabled = False )
                with col6:
                    st.color_picker(color_dict[5],rgb_to_hex(tuple(palette[5][[2,1,0]])),disabled = False )
                
                col7,col8,col9,col10,col11,col12 = st.columns(6)
                with col7:
                    st.color_picker(color_dict[6],rgb_to_hex(tuple(palette[6][[2,1,0]])),disabled = False )
                with col8:
                    st.color_picker(color_dict[7],rgb_to_hex(tuple(palette[7][[2,1,0]])),disabled = False )
                with col9:
                    st.color_picker(color_dict[8],rgb_to_hex(tuple(palette[8][[2,1,0]])),disabled = False )
                with col10:
                    st.color_picker(color_dict[9],rgb_to_hex(tuple(palette[9][[2,1,0]])),disabled = False )
                with col11:
                    st.color_picker(color_dict[10],rgb_to_hex(tuple(palette[10][[2,1,0]])),disabled = False )
                with col12:
                    st.color_picker(color_dict[11],rgb_to_hex(tuple(palette[11][[2,1,0]])),disabled = False )

                col13,col14,col15,col16,col17,col18,col19 = st.columns(7)
                with col13:
                    st.color_picker(color_dict[12],rgb_to_hex(tuple(palette[12][[2,1,0]])),disabled = False )
                with col14:
                    st.color_picker(color_dict[13],rgb_to_hex(tuple(palette[13][[2,1,0]])),disabled = False )
                with col15:
                    st.color_picker(color_dict[14],rgb_to_hex(tuple(palette[14][[2,1,0]])),disabled = False )
                with col16:
                    st.color_picker(color_dict[15],rgb_to_hex(tuple(palette[15][[2,1,0]])),disabled = False )
                with col17:
                    st.color_picker(color_dict[16],rgb_to_hex(tuple(palette[16][[2,1,0]])),disabled = False )
                with col18:
                    st.color_picker(color_dict[17],rgb_to_hex(tuple(palette[17][[2,1,0]])),disabled = False )
                with col19:
                    st.color_picker(color_dict[18],rgb_to_hex(tuple(palette[18][[2,1,0]])),disabled = False )

        
if __name__ == "__main__":
    main()



    
