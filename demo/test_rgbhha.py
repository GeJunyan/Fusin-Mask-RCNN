from maskrcnn_benchmark.config import cfg
from predictor_rgbhha import COCODemo
import cv2 as cv
import numpy as np
from PIL import Image

config_file = "/home/qian/gjy/1_codes/maskrcnn-benchmark-master/my_cfg_file/hha_R50FPN1x test_rgbhha.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda:1"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction

#hha
hha_image = Image.open("/home/qian/gjy/1_codes/maskrcnn-benchmark-master/images_for_test/hha_0022.png").convert('RGB')
#rgb
rgb_image = Image.open("/home/qian/gjy/1_codes/maskrcnn-benchmark-master/images_for_test/rgb_0022.jpg").convert('RGB')

hha_rgb = np.concatenate((hha_image, rgb_image), axis=2) 

predictions = coco_demo.run_on_opencv_image(hha_rgb)### predictor_rgbhha 代码要重写！！！！！！！！！！！！！！！