import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('/home/class1/work/zhangnan/RTDETR-master/ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='/home/class1/work/zhangnan/RTDETR-master/dataset/data_dota.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=8,
                workers=4,
                device='0',
                # resume='', # last.pt path
                project='/home/class1/work/zhangnan/RTDETR-master/runs/train/DOTA1/',
                name='res18',
                )