import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.rtdetr.distill import RTDETRDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': 'ultralytics/cfg/models/yolo-detr/yolov8-detr-fasternet.yaml',
        'data':'/home/hjj/Desktop/dataset/dataset_visdrone/data_exp.yaml',
        'imgsz': 640,
        'epochs': 500,
        'batch': 4,
        'workers': 4,
        'cache': True,
        'device': '0',
        'project':'runs/distill',
        'name':'rtdetr-logloss-cwd-exp2',
        
        # distill
        'prune_model': False,
        'teacher_weights': 'runs/train/yolov8n-detr/weights/best.pt',
        'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8-detr.yaml',
        'kd_loss_type': 'feature',
        'kd_loss_decay': 'constant',
        'kd_loss_epoch': 1.0,
        
        'logical_loss_type': 'logical',
        'logical_loss_ratio': 0.25,
        
        'teacher_kd_layers': '2,4,6,8,12,15,18,21',
        'student_kd_layers': '0-1,0-2,0-3,0-4,4,7,10,13',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 0.02
    }
    
    model = RTDETRDistiller(overrides=param_dict)
    model.distill()