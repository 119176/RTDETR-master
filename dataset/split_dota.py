from PIL import Image
Image.MAX_IMAGE_PIXELS = None


from ultralytics.data.split_dota import split_test, split_trainval

# split train and val set, with labels.
split_trainval(
    data_root="/home/class1/work/zhangnan/RTDETR-master/dataset/DOTA/",
    save_dir="dataset/DOTA_split/",
    rates=[0.5, 1.0, 1.5],  # multiscale
    gap=500,
)
# split test set, without labels.
split_test(
    data_root="/home/class1/work/zhangnan/RTDETR-master/dataset/DOTA/",
    save_dir="dataset/DOTA_split/",
    rates=[0.5, 1.0, 1.5],  # multiscale
    gap=500,
)