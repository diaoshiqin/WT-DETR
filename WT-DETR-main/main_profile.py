import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('D:/Model/WT-DETR/WT-DETR-main/ultralytics/cfg/models/WT-DETR/wt-detr.yaml')



    model.model.eval()
    model.info(detailed=True)
    try:
        model.profile(imgsz=[1280, 1280])
    except Exception as e:
        print(e)
        pass
    print('after fuse:', end='')
    model.fuse()