from __future__ import print_function

from models.yolo import Model
from utils.general import (
    check_amp
)

model = Model("./yolov5n.yaml", ch=3, nc=1)  # create
amp = check_amp(model)  # check AMP
print("loaded")


