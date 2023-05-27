# Fusin-Mask-RCNN
This repository procides some key implementation steps of "Research on the visual robotic grasping strategy in cluttered scenes".

# Dataset preparation
1. "dataset_generation.py" is to generate synthetic dataset for training via BlenderProc. You should prepare you models and background textures to render in advance.
   The generated dataset includes RGB images, depth images, and mask annotations.
   For more detailed information, please see https://github.com/DLR-RM/BlenderProc.

2. To generate HHA images, please see https://github.com/charlesCXK/Depth2HHA-python.

# Fusion-Mask-RCNN
1. Fusion-Mask-RCNN in implemented based on Mask-RCNN. https://github.com/facebookresearch/maskrcnn-benchmark.
2. To prevent excessive invalid information, we only upload "./data", "./demo", "./modeling". We modified codes in these three folder, and you can find our implementations of
   Fusion-Mask-RCNN in folders.
