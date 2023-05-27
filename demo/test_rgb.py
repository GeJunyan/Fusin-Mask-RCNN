from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2 as cv
import numpy as np
from PIL import Image
import os

config_file = "/home/qian/gjy/1_codes/maskrcnn-benchmark-master/my_cfg_file/rgb_R50FPN1x_test.yaml"

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


rgb_image = cv.imread("/home/qian/gjy/1_codes/maskrcnn-benchmark-master/images_for_test/9.png")
result, box, label, mask, threshList = coco_demo.run_on_opencv_image(rgb_image)


#写完整的掩膜rgb的值到txt里
#obj_num = len(threshList)

'''
with open("/home/qian/gjy/1_codes/maskrcnn-benchmark-master/images_for_test/full_mask.txt", "w") as f:
    for obj_id in range(obj_num):
        mask = threshList[obj_id]
        cls = label[obj_id]
        if cls == 1:#代表这是完整的物体
            for row in range(mask.shape[0]):
                for col in range(mask.shape[1]):
                    if mask[row, col] == 1:
                        cv.circle(result,(col, row), 1, (0,0,255), 1)
                        f.write(str(col))
                        f.write(' ')
                        f.write(str(row))
                        f.write('\n')
            break
'''                      

cv.imwrite("/home/qian/gjy/1_codes/maskrcnn-benchmark-master/images_for_test/test_rgb_result.png", result)