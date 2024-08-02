import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from os import listdir
from os.path import join

# DOTA classes
classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
           'small-vehicle', 'large-vehicle', 'ship', 
           'tennis-court', 'basketball-court',  
           'storage-tank', 'soccer-ball-field', 
           'roundabout', 'harbor', 
           'swimming-pool', 'helicopter', 'container-crane']

name_dict = {'0': 'plane', '1': 'baseball-diamond', '2': 'bridge',
             '3': 'ground-track-field', '4': 'small-vehicle', '5': 'large-vehicle', '6': 'ship',
             '7': 'tennis-court', '8': 'basketball-court', '9': 'storage-tank',
             '10': 'roundabout', '11': 'harbor', '12': 'swimming-pool', 
             '13': "helicopter", '14': "container-crane"}

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xmlpath, xmlname):
    with open(xmlpath, "r", encoding='utf-8') as in_file:
        txtname = xmlname[:-4] + '.txt'
        txtfile = os.path.join(txtpath, txtname)
        tree = ET.parse(in_file)
        root = tree.getroot()
        filename = root.find('filename')
        img = cv2.imdecode(np.fromfile('{}/{}.{}'.format(imgpath, xmlname[:-4], postfix), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        res = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                print(f'Error: Class "{cls}" not found in predefined classes.')
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            x0 = float(xmlbox.find('x0').text)
            y0 = float(xmlbox.find('y0').text)
            x1 = float(xmlbox.find('x1').text)
            y1 = float(xmlbox.find('y1').text)
            x2 = float(xmlbox.find('x2').text)
            y2 = float(xmlbox.find('y2').text)
            x3 = float(xmlbox.find('x3').text)
            y3 = float(xmlbox.find('y3').text)
            xmin = min(x0, x1, x2, x3)
            ymin = min(y0, y1, y2, y3)
            xmax = max(x0, x1, x2, x3)
            ymax = max(y0, y1, y2, y3)
            b = (xmin, xmax, ymin, ymax)
            bb = convert((w, h), b)
            res.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
        if len(res) != 0:
            with open(txtfile, 'w+') as f:
                f.write('\n'.join(res))

if __name__ == "__main__":
    postfix = 'png'
    imgpath = '/home/class1/work/zhangnan/RTDETR-master/dataset/DOTA/val/voc/img'
    xmlpath = '/home/class1/work/zhangnan/RTDETR-master/dataset/DOTA/val/voc/xml'
    txtpath = '/home/class1/work/zhangnan/RTDETR-master/dataset/DOTA/val/voc/yolo'
    
    if not os.path.exists(txtpath):
        os.makedirs(txtpath, exist_ok=True)
    
    file_list = os.listdir(xmlpath)
    error_file_list = []
    for i in range(0, len(file_list)):
        try:
            path = os.path.join(xmlpath, file_list[i])
            if ('.xml' in path) or ('.XML' in path):
                convert_annotation(path, file_list[i])
                print(f'file {file_list[i]} convert success.')
            else:
                print(f'file {file_list[i]} is not xml format.')
        except Exception as e:
            print(f'file {file_list[i]} convert error.')
            print(f'error message:\n{e}')
            error_file_list.append(file_list[i])
    print(f'this file convert failure\n{error_file_list}')
    print(f'Dataset Classes:{classes}')