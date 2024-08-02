"""对切割后得到的文件夹下的xml文件进行处理，删除不符合要求的xml文件及.
文件夹下对应的图片。不符合要求的xml文件有以下三种情况：
1. 标注目标为空；2. 所有标注目标的difficult均为1；3. 标注目标存在越界的问题（注：标注越界有六种情况 xmin<0、ymin<0、xmax>width、ymax>height、xmax<xmin、ymax<ymin）。
"""
import os
import shutil
import xml.dom.minidom
import xml.etree.ElementTree as ET

def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])

def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles

def cleandata(path, img_path, ext, label_ext):
    name = custombasename(path)  # 名称
    if label_ext == '.xml':
        tree = ET.parse(path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        objectlist = root.findall('object')
        num = len(objectlist)

        count = 0
        count1 = 0
        minus = 0
        for object in objectlist:
            difficult = int(object.find('difficult').text)
            bndbox = object.find('bndbox')

            if bndbox is not None:
                x0 = bndbox.find('x0')
                y0 = bndbox.find('y0')
                x1 = bndbox.find('x1')
                y1 = bndbox.find('y1')
                x2 = bndbox.find('x2')
                y2 = bndbox.find('y2')
                x3 = bndbox.find('x3')
                y3 = bndbox.find('y3')

                if x0 is not None and y0 is not None and x1 is not None and y1 is not None and x2 is not None and y2 is not None and x3 is not None and y3 is not None:
                    x0 = int(x0.text)
                    y0 = int(y0.text)
                    x1 = int(x1.text)
                    y1 = int(y1.text)
                    x2 = int(x2.text)
                    y2 = int(y2.text)
                    x3 = int(x3.text)
                    y3 = int(y3.text)

                    xmin = min(x0, x1, x2, x3)
                    ymin = min(y0, y1, y2, y3)
                    xmax = max(x0, x1, x2, x3)
                    ymax = max(y0, y1, y2, y3)

                    if xmin < 0 or ymin < 0 or width < xmax or height < ymax or xmax < xmin or ymax < ymin:  # 目标标注越界的六种情况
                        minus += 1

            count = count1 + difficult
            count1 = count

        if num == 0 or count == num or minus != 0:  # 不符合要求的三种情况
            image_path = os.path.join(img_path, name + ext)  # 样本图片的名称
            os.remove(image_path)  # 移除该标注文件
            os.remove(path)  # 移除该图片文件

if __name__ == '__main__':
    root = '/home/class1/work/zhangnan/RTDETR-master/dataset/DOTA/val/voc/'
    img_path = os.path.join(root, 'img')  # 分割后的样本集
    label_path = os.path.join(root, 'xml')  # 分割后的标签
    ext = '.png'  # 图片的后缀
    label_ext = '.xml'

    label_list = GetFileFromThisRootDir(label_path)
    for path in label_list:
        cleandata(path, img_path, ext, label_ext)