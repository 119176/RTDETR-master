from ultralytics.data.split_dota import split_test, split_trainval

# split train and val set, with labels.
split_trainval(
    data_root="/home/class1/work/zhangnan/RTDETR-master/dataset/DOTAv1.0/",
    save_dir="/home/class1/work/zhangnan/RTDETR-master/dataset/DOTAv1.0-split1/",
    rates=[1.0],  # multiscale
    gap=500,
)
# split test set, without labels.
split_test(
    data_root="/home/class1/work/zhangnan/RTDETR-master/dataset/DOTAv1.0/",
    save_dir="/home/class1/work/zhangnan/RTDETR-master/dataset/DOTAv1.0-split1/",
    rates=[1.0],  # multiscale
    gap=500,
)