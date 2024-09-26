Find the yaml file for the cnn:
path_to/yolov5-master/models

Following lines are custommizeable:
nc: 80 # number of classes
depth_multiple: 1.0 # model depth multiple
width_multiple: 1.0 # layer channel multiple
anchors:
  - [10, 13, 16, 30, 33, 23] # P3/8
  - [30, 61, 62, 45, 59, 119] # P4/16
  - [116, 90, 156, 198, 373, 326] # P5/32

Explanation of parameters:
- nc: 80: Number of classes in the dataset. YOLOv5 is pre-configured for the COCO dataset, which has 80 classes, but you can modify this for custom datasets with a different number of classes.
- depth_multiple: 1.0: This multiplies the number of layers (depth) in the model. If this value is decreased (e.g., 0.5), the model will have fewer layers, making it faster but less accurate. Increasing it would make the model deeper, potentially improving accuracy but at the cost of speed.
- width_multiple: 1.0: This multiplies the number of channels (width) in each layer. A lower value would result in a smaller, faster model with fewer feature maps, while a higher value would increase the number of feature maps for better feature representation but slower inference.

- Anchors are predefined bounding boxes with different aspect ratios and sizes that help YOLOv5 detect objects of varying scales. YOLO predicts adjustments to these anchors to fit objects in the image.

  - P3/8: Small-scale objects  
  - P4/16: Medium-scale objects
  - P5/32: Large-scale objects
