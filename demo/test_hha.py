from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2 as cv
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
config_file = "/home/qian/gjy/1_codes/maskrcnn-benchmark-master/my_cfg_file/hha_R50FPN1x_test.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.9,
)
# load image and then run prediction


hha_image = cv.imread("/home/qian/gjy/1_codes/maskrcnn-benchmark-master/images_for_test/hha_0014.png")
#hha_image = Image.open("/home/qian/gjy/1_codes/maskrcnn-benchmark-master/images_for_test/hha_0014.png").convert('RGB')


result, box, label, mask, threshList = coco_demo.run_on_opencv_image(hha_image)
cv.imwrite("/home/qian/gjy/1_codes/maskrcnn-benchmark-master/images_for_test/test_hha_result.png", result)