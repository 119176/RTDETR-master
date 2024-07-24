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
 
def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles
  
def cleandata(path, img_path, ext, label_ext):
    name = custombasename(path)  #名称
    if label_ext == '.xml':
        tree = ET.parse(path)
        root = tree.getroot()

        size=root.find('size')
        width=int(size.find('width').text)
        #print(width)
        height=int(size.find('height').text)
        #print(height)

        objectlist = root.findall('object')
        num = len(objectlist)
        #print(num)

        count=0
        count1=0
        minus=0
        for object in objectlist:
            difficult = int(object.find('difficult').text)
            #print(difficult)

            bndbox=object.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            #print(xmin)
            ymin = int(bndbox.find('ymin').text)
            #print(ymin)
            xmax = int(bndbox.find('xmax').text)
            #print(xmax)
            ymax = int(bndbox.find('ymax').text)
            #print(ymax)
            if xmin<0 or ymin<0 or width<xmax or height<ymax or xmax<xmin or ymax<ymin:  # 目标标注越界的六种情况
                minus+=1
            count = count1 + difficult
            count1 = count
            
        if num == 0 or count == num or minus != 0:  # 不符合要求的三种情况
            image_path = os.path.join(img_path, name + ext) #样本图片的名称
            os.remove(image_path)  #移除该标注文件
            os.remove(path)     #移除该图片文件
                                           
if __name__ == '__main__':
    root = '/home/class1/work/zhangnan/RTDETR-main/dataset/DOTA/val/voc/'
    img_path = os.path.join(root, 'JPEGImages')  #分割后的样本集
    label_path = os.path.join(root, 'Annotations')  #分割后的标签
    ext = '.png' #图片的后缀
    label_ext = '.xml'
        
    label_list = GetFileFromThisRootDir(label_path)
    for path in label_list:
        cleandata(path, img_path, ext, label_ext)
