import pandas as pd
import pickle
import os
import streamlit as st
import numpy as np
from models.model import detect_objects
import time
import cv2
import torch
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path
import os
from models.utils import plot_boxes, plot_boxes_webcam, plot_boxes_online
from models.darknet.darknet import Darknet
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av



def run():
    # Streamlit page title
    st.title("Mask Detection")
    st.markdown('**This is a demo application for mask detection**')
    hide_footer_style = """<style>.reportview-container .main footer {visibility: hidden;}"""
    st.markdown(hide_footer_style, unsafe_allow_html=True)

    # Set path for the cfg file
    cfg_file = './models/darknet/yolov3-voc-test.cfg'
    # Set path for the pre-trained weights file
    weight_file = './models/darknet/weights/yolov3-voc-nose.backup'
    # Set path for the COCO object classes file
    namesfile = './data/coco/coco2.names'

    # Load the COCO class names
    def load_class_names(namesfile):
        # Load the COCO class names
        class_names = []
        with open(namesfile, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        return class_names
        
    class_names = load_class_names(namesfile)

    # Load the network architecture
    model = Darknet(cfg_file)

    # Load the pre-trained weights
    model.load_weights(weight_file)

    class VideoProcessor:
        def __init__(self) -> None:
            self.iou_thresh = 0.4

            self.nms_thresh = 0.6

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            resized_image = cv2.resize(original_image, (model.width, model.height))
            iou_thresh = 0.4

            nms_thresh = 0.6
            boxes = detect_objects(model, resized_image, self.iou_thresh, self.nms_thresh)
            
            img = plot_boxes_online(original_image, boxes, class_names, plot_labels = True)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    ctx = webrtc_streamer(
        key="example", 
        video_processor_factory=VideoProcessor, 
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ))

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file != None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Load the image
        img = cv2.imdecode(file_bytes, 1)

        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        resized_image = cv2.resize(original_image, (model.width, model.height))

        iou_thresh = 0.4

        nms_thresh = 0.6

        boxes = detect_objects(model, resized_image, iou_thresh, nms_thresh)

        for i in range(len(boxes)):
            box = boxes[i]
            if len(box) == 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                st.markdown('%i. %s: %f' % (i + 1, class_names[cls_id], cls_conf))

        fig = plot_boxes(original_image, boxes, class_names, plot_labels = True)
        st.pyplot(fig)



if __name__ == "__main__":
    run()
