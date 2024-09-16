# Distill Experiment (For BiliBili魔鬼面具)

### student:yolov8n-detr teacher:yolov8s-detr Dataset:Visdrone 2500 Training Data

teacher:
```
model = RTDETR('ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml')
# model.load('') # loading pretrain weights
model.train(data='/root/dataset/dataset_visdrone/data_exp.yaml',
            cache=True,
            imgsz=640,
            epochs=300,
            batch=8,
            workers=8,
            device='0',
            # resume='', # last.pt path
            project='runs/train',
            name='yolov8s-detr',
            )
```
nohup python train.py > logs/yolov8s-detr.log 2>&1 & tail -f logs/yolov8s-detr.log
nohup python val.py > logs/yolov8s-detr-test.log 2>&1 & tail -f logs/yolov8s-detr-test.log

student:
```
model = RTDETR('ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml')
# model.load('') # loading pretrain weights
model.train(data='/root/dataset/dataset_visdrone/data_exp.yaml',
            cache=True,
            imgsz=640,
            epochs=500,
            batch=4,
            workers=4,
            device='0',
            # resume='', # last.pt path
            project='runs/train',
            name='yolov8n-detr',
            )
```
nohup python train.py > logs/yolov8n-detr.log 2>&1 & tail -f logs/yolov8n-detr.log
nohup python val.py > logs/yolov8n-detr-test.log 2>&1 & tail -f logs/yolov8n-detr-test.log

-------------------------------- cwd exp1 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-cwd-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.1
}
```
nohup python distill.py > logs/yolov8n-detr-cwd-exp1.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp1.log
nohup python val.py > logs/yolov8n-detr-cwd-exp1-test.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp1-test.log

-------------------------------- cwd exp2 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-cwd-exp2',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.08
}
```
nohup python distill.py > logs/yolov8n-detr-cwd-exp2.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp2.log
nohup python val.py > logs/yolov8n-detr-cwd-exp2-test.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp2-test.log

-------------------------------- cwd exp3 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-cwd-exp3',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.05
}
```
nohup python distill.py > logs/yolov8n-detr-cwd-exp3.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp3.log
nohup python val.py > logs/yolov8n-detr-cwd-exp3-test.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp3-test.log

-------------------------------- cwd exp4 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-cwd-exp4',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'linear_epoch',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.08
}
```
nohup python distill.py > logs/yolov8n-detr-cwd-exp4.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp4.log
nohup python val.py > logs/yolov8n-detr-cwd-exp4-test.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp4-test.log

-------------------------------- cwd exp5 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-cwd-exp5',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'cosine_epoch',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.08
}
```
nohup python distill.py > logs/yolov8n-detr-cwd-exp5.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp5.log
nohup python val.py > logs/yolov8n-detr-cwd-exp5-test.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp5-test.log

-------------------------------- cwd exp6 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-cwd-exp6',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 0.25,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.08
}
```
nohup python distill.py > logs/yolov8n-detr-cwd-exp6.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp6.log
nohup python val.py > logs/yolov8n-detr-cwd-exp6-test.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp6-test.log

-------------------------------- cwd exp7 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-cwd-exp7',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 0.15,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.08
}
```
nohup python distill.py > logs/yolov8n-detr-cwd-exp7.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp7.log
nohup python val.py > logs/yolov8n-detr-cwd-exp7-test.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp7-test.log

-------------------------------- cwd exp8 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-cwd-exp8',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '2,4,6,8',
    'student_kd_layers': '2,4,6,8',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.08
}
```
nohup python distill.py > logs/yolov8n-detr-cwd-exp8.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp8.log
nohup python val.py > logs/yolov8n-detr-cwd-exp8-test.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp8-test.log

-------------------------------- cwd exp9 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-cwd-exp9',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8n-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.08
}
```
nohup python distill.py > logs/yolov8n-detr-cwd-exp9.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp9.log
nohup python val.py > logs/yolov8n-detr-cwd-exp9-test.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp9-test.log

-------------------------------- cwd exp10 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-cwd-exp10',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.08
}
```
nohup python distill.py > logs/yolov8n-detr-cwd-exp10.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp10.log
nohup python val.py > logs/yolov8n-detr-cwd-exp10-test.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp10-test.log

-------------------------------- cwd exp11 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-cwd-exp11',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8n-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.08
}
```
nohup python distill.py > logs/yolov8n-detr-cwd-exp11.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp11.log
nohup python val.py > logs/yolov8n-detr-cwd-exp11-test.log 2>&1 & tail -f logs/yolov8n-detr-cwd-exp11-test.log

-------------------------------- mgd exp1 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-mgd-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.03
}
```
nohup python distill.py > logs/yolov8n-detr-mgd-exp1.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp1.log
nohup python val.py > logs/yolov8n-detr-mgd-exp1-test.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp1-test.log

-------------------------------- mgd exp2 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-mgd-exp2',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.05
}
```
nohup python distill.py > logs/yolov8n-detr-mgd-exp2.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp2.log
nohup python val.py > logs/yolov8n-detr-mgd-exp2-test.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp2-test.log

-------------------------------- mgd exp3 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-mgd-exp3',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.01
}
```
nohup python distill.py > logs/yolov8n-detr-mgd-exp3.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp3.log
nohup python val.py > logs/yolov8n-detr-mgd-exp3-test.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp3-test.log

-------------------------------- mgd exp4 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-mgd-exp4',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '2,4,6,8',
    'student_kd_layers': '2,4,6,8',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.03
}
```
nohup python distill.py > logs/yolov8n-detr-mgd-exp4.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp4.log
nohup python val.py > logs/yolov8n-detr-mgd-exp4-test.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp4-test.log

-------------------------------- mgd exp5 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-mgd-exp5',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8n-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.03
}
```
nohup python distill.py > logs/yolov8n-detr-mgd-exp5.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp5.log
nohup python val.py > logs/yolov8n-detr-mgd-exp5-test.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp5-test.log

-------------------------------- mgd exp6 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-mgd-exp6',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.03
}
```
nohup python distill.py > logs/yolov8n-detr-mgd-exp6.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp6.log
nohup python val.py > logs/yolov8n-detr-mgd-exp6-test.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp6-test.log

-------------------------------- mgd exp7 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-mgd-exp7',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.05
}
```
nohup python distill.py > logs/yolov8n-detr-mgd-exp7.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp7.log
nohup python val.py > logs/yolov8n-detr-mgd-exp7-test.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp7-test.log

-------------------------------- mgd exp8 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-mgd-exp8',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8n-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.03
}
```
nohup python distill.py > logs/yolov8n-detr-mgd-exp8.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp8.log
nohup python val.py > logs/yolov8n-detr-mgd-exp8-test.log 2>&1 & tail -f logs/yolov8n-detr-mgd-exp8-test.log

-------------------------------- chsim exp1 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-chsim-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'chsim',
    'feature_loss_ratio': 2.0
}
```
nohup python distill.py > logs/yolov8n-detr-chsim-exp1.log 2>&1 & tail -f logs/yolov8n-detr-chsim-exp1.log
nohup python val.py > logs/yolov8n-detr-chsim-exp1-test.log 2>&1 & tail -f logs/yolov8n-detr-chsim-exp1-test.log

-------------------------------- chsim exp2 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-chsim-exp2',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'chsim',
    'feature_loss_ratio': 1.5
}
```
nohup python distill.py > logs/yolov8n-detr-chsim-exp2.log 2>&1 & tail -f logs/yolov8n-detr-chsim-exp2.log
nohup python val.py > logs/yolov8n-detr-chsim-exp2-test.log 2>&1 & tail -f logs/yolov8n-detr-chsim-exp2-test.log

-------------------------------- chsim exp3 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-chsim-exp3',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'chsim',
    'feature_loss_ratio': 2.5
}
```
nohup python distill.py > logs/yolov8n-detr-chsim-exp3.log 2>&1 & tail -f logs/yolov8n-detr-chsim-exp3.log
nohup python val.py > logs/yolov8n-detr-chsim-exp3-test.log 2>&1 & tail -f logs/yolov8n-detr-chsim-exp3-test.log

-------------------------------- spkd exp1 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-sp-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'sp',
    'feature_loss_ratio': 2.0
}
```
nohup python distill.py > logs/yolov8n-detr-sp-exp1.log 2>&1 & tail -f logs/yolov8n-detr-sp-exp1.log
nohup python val.py > logs/yolov8n-detr-sp-exp1-test.log 2>&1 & tail -f logs/yolov8n-detr-sp-exp1-test.log

-------------------------------- spkd exp2 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-sp-exp2',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'sp',
    'feature_loss_ratio': 1.5
}
```
nohup python distill.py > logs/yolov8n-detr-sp-exp2.log 2>&1 & tail -f logs/yolov8n-detr-sp-exp2.log
nohup python val.py > logs/yolov8n-detr-sp-exp2-test.log 2>&1 & tail -f logs/yolov8n-detr-sp-exp2-test.log

-------------------------------- spkd exp3 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-sp-exp3',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'sp',
    'feature_loss_ratio': 2.5
}
```
nohup python distill.py > logs/yolov8n-detr-sp-exp3.log 2>&1 & tail -f logs/yolov8n-detr-sp-exp3.log
nohup python val.py > logs/yolov8n-detr-sp-exp3-test.log 2>&1 & tail -f logs/yolov8n-detr-sp-exp3-test.log

-------------------------------- logloss exp1 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-logloss-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'logical',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'logical',
    'logical_loss_ratio': 0.3,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'sp',
    'feature_loss_ratio': 2.0
}
```
nohup python distill.py > logs/yolov8n-detr-logloss-exp1.log 2>&1 & tail -f logs/yolov8n-detr-logloss-exp1.log
nohup python val.py > logs/yolov8n-detr-logloss-exp1-test.log 2>&1 & tail -f logs/yolov8n-detr-logloss-exp1-test.log

-------------------------------- logloss exp2 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-logloss-exp2',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'logical',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'logical',
    'logical_loss_ratio': 0.1,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'sp',
    'feature_loss_ratio': 2.0
}
```
nohup python distill.py > logs/yolov8n-detr-logloss-exp2.log 2>&1 & tail -f logs/yolov8n-detr-logloss-exp2.log
nohup python val.py > logs/yolov8n-detr-logloss-exp2-test.log 2>&1 & tail -f logs/yolov8n-detr-logloss-exp2-test.log

-------------------------------- logloss exp3 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-logloss-exp3',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'logical',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'logical',
    'logical_loss_ratio': 0.2,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'sp',
    'feature_loss_ratio': 2.0
}
```
nohup python distill.py > logs/yolov8n-detr-logloss-exp3.log 2>&1 & tail -f logs/yolov8n-detr-logloss-exp3.log
nohup python val.py > logs/yolov8n-detr-logloss-exp3-test.log 2>&1 & tail -f logs/yolov8n-detr-logloss-exp3-test.log

-------------------------------- logloss exp4 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-logloss-exp4',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'logical',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'logical',
    'logical_loss_ratio': 0.25,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'sp',
    'feature_loss_ratio': 2.0
}
```
nohup python distill.py > logs/yolov8n-detr-logloss-exp4.log 2>&1 & tail -f logs/yolov8n-detr-logloss-exp4.log
nohup python val.py > logs/yolov8n-detr-logloss-exp4-test.log 2>&1 & tail -f logs/yolov8n-detr-logloss-exp4-test.log

-------------------------------- logloss+mgd exp1 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-logloss-mgd-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'all',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'logical',
    'logical_loss_ratio': 0.2,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.03
}
```
nohup python distill.py > logs/yolov8n-detr-logloss-mgd-exp1.log 2>&1 & tail -f logs/yolov8n-detr-logloss-mgd-exp1.log
nohup python val.py > logs/yolov8n-detr-logloss-mgd-exp1-test.log 2>&1 & tail -f logs/yolov8n-detr-logloss-mgd-exp1-test.log

-------------------------------- logloss+cwd exp1 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 500,
    'batch': 4,
    'workers': 4,
    'cache': True,
    'device': '0',
    'project':'runs/distill',
    'name':'rtdetr-logloss-cwd-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'all',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'logical',
    'logical_loss_ratio': 0.25,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.01
}
```
nohup python distill.py > logs/yolov8n-detr-logloss-cwd-exp1.log 2>&1 & tail -f logs/yolov8n-detr-logloss-cwd-exp1.log
nohup python val.py > logs/yolov8n-detr-logloss-cwd-exp1-test.log 2>&1 & tail -f logs/yolov8n-detr-logloss-cwd-exp1-test.log

-------------------------------- logloss+cwd exp2 --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
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
    'teacher_weights': 'runs/train/yolov8s-detr/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml',
    'kd_loss_type': 'all',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'logical',
    'logical_loss_ratio': 0.25,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.02
}
```
nohup python distill.py > logs/yolov8n-detr-logloss-cwd-exp2.log 2>&1 & tail -f logs/yolov8n-detr-logloss-cwd-exp2.log
nohup python val.py > logs/yolov8n-detr-logloss-cwd-exp2-test.log 2>&1 & tail -f logs/yolov8n-detr-logloss-cwd-exp2-test.log

更换主干后的一些命令示例：
-------------------------------- student:yolov8n-detr-fasternet teacher:yolov8n-detr-fasternet --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8-detr-fasternet.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
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
    'teacher_weights': 'runs/train/yolov8n-detr-fasternet/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8-detr-fasternet.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    'kd_loss_epoch': 1.0,
    
    'logical_loss_type': 'logical',
    'logical_loss_ratio': 0.25,
    
    'teacher_kd_layers': '0-1,0-2,0-3,0-4,4,7,10,13',
    'student_kd_layers': '0-1,0-2,0-3,0-4,4,7,10,13',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.02
}
```

-------------------------------- student:yolov8n-detr-fasternet teacher:yolov8n-detr --------------------------------
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/yolo-detr/yolov8-detr-fasternet.yaml',
    'data':'/root/dataset/dataset_visdrone/data_exp.yaml',
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
```

# 使用教程

    蒸馏操作问题、报错问题统一群里问,我群里回复,谢谢~

# 环境

    没什么特别要求,跟正常的rtdetr一样.

# 视频

链接：https://pan.baidu.com/s/1lpfEJrvUNXH7pxSa1ajTqg?pwd=pvks 
提取码：pvks  # BiliBili 魔鬼面具
1. 使用教程.
2. 知识蒸馏中的一些建议.
3. RTDETR逻辑蒸馏讲解.
4. 怎么把蒸馏代码加到自己的代码里面.

# 我自己跑的实验数据
1. yolov8n-detr visdrone
链接：https://pan.baidu.com/s/123r-8RZHJbxd1a-SMbX8og?pwd=0fom 
提取码：0fom # BiliBili 魔鬼面具