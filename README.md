# pointpillars-simplify
这是基于[EasyPointPillars](https://github.com/open-mmlab/OpenPCDet](https://github.com/AbangLZU/EasyPointPillars))对pointpillars的简化版本，支持在cpu上进行推断，不需要安装spconv和pcdet。

基于KITTI数据集，提供了预训练模型和测试数据(data文件夹)，如果要进行推断，需要修改SCORE_THRESH(原始是0.1)和NMS_THRESH(原始是0.01)，否则nms处理过程会非常慢。


