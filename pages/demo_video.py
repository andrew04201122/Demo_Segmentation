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
import socket


socks = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
socks.settimeout(1000.0)

torch.set_grad_enabled(False)


def get_model():
    net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
    net.load_state_dict(torch.load('./model/model_final_v2_city.pth', map_location='cpu'), strict=False)
    net.eval()
    net.cuda()
    return net


# fetch frames
def get_func(inpth, in_q, done):
    cap = cv2.VideoCapture(inpth)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # type is float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # type is float
    fps = cap.get(cv2.CAP_PROP_FPS)

    to_tensor = T.ToTensor(
        mean=(0.3257, 0.3690, 0.3223), # city, rgb
        std=(0.2112, 0.2148, 0.2115),
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = frame[:, :, ::-1]
        frame = to_tensor(dict(im=frame, lb=None))['im'].unsqueeze(0)
        in_q.put(frame)

    in_q.put('quit')
    done.wait()

    cap.release()
    time.sleep(1)
    print('input queue done')


# save to video
def save_func(inpth, outpth, out_q):
    np.random.seed(123)
    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(inpth)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # type is float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # type is float
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    video_writer = cv2.VideoWriter(outpth,cv2.VideoWriter_fourcc(*"mp4v"),fps, (int(width), int(height)))

    while True:
        out = out_q.get(timeout = 100)
        if out == 'quit': break
        out = out.numpy()
        preds = palette[out]
        for pred in preds:
            video_writer.write(pred)

    video_writer.release()
    print('output queue done')


# inference a list of frames
def infer_batch(frames):
    frames = torch.cat(frames, dim=0).cuda()
    H, W = frames.size()[2:]
    frames = F.interpolate(frames, size=(768, 768), mode='bilinear',
            align_corners=False) # must be divisible by 32
    out = net(frames)[0]
    out = F.interpolate(out, size=(H, W), mode='bilinear',
            align_corners=False).argmax(dim=1).detach().cpu()
    out_q.put(out)


def video_show():
    os.system(f"ffmpeg -y -i ./temp/demo.mp4 -vcodec libx264 ./output/demo.mp4")
    video_file = open('./output/demo.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)


if __name__ == "__main__":
    st.set_page_config(page_title="demo_video")
    st.sidebar.header("demo_video")
    st.title("Video sementic segmentation demo")
    st.subheader("Description")
    st.markdown("In this demo, you can choose a mp4 file in img_input folder. Once you choose a video, you can watch the original video first and segmentation video will show up after it finish its process.")
    video_folder = './img_input/'
    
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4'])   
    cfg = set_cfg_from_file('./configs/bisenetv2_city.py')
    net = get_model()
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        input_video_name = uploaded_file.name
        video_path = video_folder + input_video_name  #./img_input/video.mp4
        st.video(video_path)
        print(video_path)

        
        mp.set_start_method('spawn', force=True)
        in_q = mp.Queue(4096)
        out_q = mp.Queue(4096)
        done = mp.Event() 
        in_worker = mp.Process(target=get_func,args=(video_path, in_q, done))  #video input
        out_worker = mp.Process(target=save_func,args=(video_path, './temp/demo.mp4', out_q))

        in_worker.start()
        out_worker.start()

        frames = []
        while True:
            frame = in_q.get(timeout = 100)
            if frame == 'quit': break

            frames.append(frame)
            if len(frames) == 8:
                infer_batch(frames)
                frames = []
        if len(frames) > 0:
            infer_batch(frames)

        out_q.put('quit')
        done.set()

        out_worker.join()
        in_worker.join()
        
        os.system(f"ffmpeg -y -i ./temp/demo.mp4 -vcodec libx264 ./output/demo.mp4")

        video_file = open('./output/demo.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

        
        
        
        




    
