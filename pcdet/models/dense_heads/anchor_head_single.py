import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    """
    Args:
        model_cfg: AnchorHeadSingle的配置
        input_channels: 384 输入通道数
        num_class: 3 kitti
        class_names: ['Car','Pedestrian','Cyclist']
        grid_size: (432, 496, 1)
        point_cloud_range: (0, -39.68, -3, 69.12, 39.68, 1)
        predict_boxes_when_training: False
    """
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        # 每个点有3个尺度的个先验框  每个先验框都有两个方向（0度，90度） num_anchors_per_location:[2, 2, 2]
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        # 如果存在方向损失，则添加方向卷积层Conv2d(512,12,kernel_size=(1,1),stride=(1,1))
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']  # (batch_size, 384, 248, 216)

        # 坐标点上6个先验框的类别预测 --> (batch_size, 18, 200, 176)
        cls_preds = self.conv_cls(spatial_features_2d)
        # 坐标点上6个先验框的参数预测 --> (batch_size, 42, 200, 176)
        # 每个先验框需要预测7个参数(x, y, z, w, l, h, θ)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        # 方向分类预测
        if self.conv_dir_cls is not None:
            # 每个先验框都要预测为两个方向中的其中一个方向 --> (batch_size, 12, 200, 176)
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            # 将类别和先验框方向预测结果放到最后一个维度中   [N, H, W, C] --> (batch_size, 248, 216, 12)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        # 训练时要对每个先验框分配GT来计算loss
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            if 'batch_size' not in data_dict.keys():
                data_dict['batch_size'] = 1

            # post-process
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds  # (1, 211200, 3) 70400*3=211200
            data_dict['batch_box_preds'] = batch_box_preds  # (1, 211200, 7)
            data_dict['cls_preds_normalized'] = False

        return data_dict
