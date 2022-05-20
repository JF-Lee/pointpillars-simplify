import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network
from pcdet.utils import common_utils

import onnx
from onnxsim import simplify
from simplifier_onnx import simplify_onnx as simplify_onnx


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='utils/cfgs/kitti_models/pointpillar.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='./data/kitti/training/velodyne/000000.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='checkpoint_epoch_29.pth',
                        help='specify the pretrained model')
    parser.add_argument('--output_path', type=str, default='./point_pillars.onnx',
                        help='specify the onnx pfe model output path')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Convert pytorch to onnx-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    # print(demo_dataset.cpu())

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    #print(model)

    #max_num_pillars = 40000
    #max_points_per_pillars = 32
    #dims_feature = 4
    dummy_input = {}
    dummy_input['voxels'] = torch.randn(((11024, 32, 4)), device='cuda')
    dummy_input['voxel_coords'] = torch.randn(((11024, 4)), device='cuda')
    dummy_input['voxel_num_points'] = torch.ones((11024), device='cuda')

    #dummy_input = torch.ones(max_num_pillars, max_points_per_pillars, dims_feature).cuda()

    torch.onnx.export(model, dummy_input, "./pointpillar.onnx",
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      keep_initializers_as_inputs=True,
                      input_names=['input', 'voxel_num_points', 'coords'],  # the model's input names
                      output_names=['cls_preds', 'box_preds', 'dir_cls_preds'],  # the model's output names
                      )
    onnx_model = onnx.load("./pointpillar.onnx")  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"

    model_simp = simplify_onnx(model_simp)
    onnx.save(model_simp, "pointpillar-sim.onnx")
    print("export pointpillar.onnx.")
    print('finished exporting onnx')

if __name__ == '__main__':
    main()
