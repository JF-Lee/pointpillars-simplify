from functools import partial

import numpy as np
from skimage import transform

from pcdet.utils import box_utils, common_utils


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        # print('=====================processor_configs=====================\n', processor_configs)
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)

        points = data_dict['points']

        voxel_output = voxel_generator.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:  # not use
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        # spconv (11024, 32, 4) (11024, 3) (11024,)
        '''
        with open('num_point.txt', 'w') as f:
            for data in num_points:
                f.write(str(data) + '\n')
        with open('coord.txt', 'w') as f:
            for data in coordinates:
                f.write(str(data) + '\n')
        with open('voxels.txt', 'w') as f:
            for idx in range(voxels.shape[0]):
                for data in voxels[idx]:
                    f.write(str(data) + '\n')
        '''
        # print('==================================transform_points_to_voxels==================================')
        # print(f'voxel shape:{voxels.shape}  coordinates shape:{coordinates.shape}  num_point:{num_points.shape}')
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict


class VoxelGenerator:
    def __init__(self, voxel_size, point_cloud_range, max_num_points=32, max_voxels=40000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points, max_voxels=-1):
        if max_voxels == -1:
            max_voxels = self._max_voxels

        return points_to_voxel(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._max_num_points,
            True,
            max_voxels)


# @numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(
        points,
        voxel_size,
        coors_range,
        num_points_per_voxel,
        coor_to_voxelidx,
        voxels,
        coors,
        max_points=35,
        max_voxels=20000,
):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3,), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


# @numba.jit(nopython=True)
def _points_to_voxel_kernel(
        points,
        voxel_size,
        coors_range,
        num_points_per_voxel,
        coor_to_voxelidx,
        voxels,
        coors,
        max_points=35,
        max_voxels=20000,
):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3,), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(points, voxel_size, coors_range, max_points=35, reverse_index=True, max_voxels=20000):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud)
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype
    )
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel(
            points,
            voxel_size,
            coors_range,
            num_points_per_voxel,
            coor_to_voxelidx,
            voxels,
            coors,
            max_points,
            max_voxels,
        )

    else:
        voxel_num = _points_to_voxel_kernel(
            points,
            voxel_size,
            coors_range,
            num_points_per_voxel,
            coor_to_voxelidx,
            voxels,
            coors,
            max_points,
            max_voxels,
        )

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    return voxels, coors, num_points_per_voxel


# @numba.jit(nopython=True)
def bound_points_jit(points, upper_bound, lower_bound):
    # to use nopython=True, np.bool is not supported. so you need
    # convert result to np.bool after this function.
    N = points.shape[0]
    ndim = points.shape[1]
    keep_indices = np.zeros((N,), dtype=np.int32)
    success = 0
    for i in range(N):
        success = 1
        for j in range(ndim):
            if points[i, j] < lower_bound[j] or points[i, j] >= upper_bound[j]:
                success = 0
                break
        keep_indices[i] = success
    return keep_indices


'''
class VoxelGenerator(object):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

    def generate(self, points):
        voxels, coors, num_points_per_voxel, voxel_num = points_to_voxel(points,
                                                            self.point_cloud_range,
                                                            self.grid_size,
                                                            max_points=self.max_num_points,
                                                            max_voxels=self.max_voxels)
        process(voxels, num_points_per_voxel, self.point_cloud_range)
        valid_num = len(voxels)
        coors = np.stack([np.zeros((valid_num,), dtype=np.int32), coors[:, 0], coors[:, 1]]).T  #########  T
        return voxels, coors, num_points_per_voxel


#@numba.jit(nopython=True)
def process(voxels, num_points, point_cloud_range):
    # voxels : (N, num_per_voxel, C+6)
    # coordinates: (N, 3) zyx
    # num_points: (N,)
    N, num_per_voxel, C = voxels.shape
    for i in range(N):
        num_point = num_points[i]
        voxels_i = voxels[i]

        for j in range(num_point, num_per_voxel):
            for k in range(C):
                voxels[i, j, k] = voxels_i[0, k]


def _points_to_voxel_kernel(points,
                            voxel_size,
                            voxelmap_shape,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=100,
                            max_voxels=20000):
    N = points.shape[0]
    grid_size = voxelmap_shape
    coor = np.zeros(shape=(2,), dtype=np.int32)
    voxel_num = 0
    for i in range(N):
        cx = np.floor((points[i, 0] - coors_range[0]) / voxel_size[0])
        cy = np.floor((points[i, 1] - coors_range[1]) / voxel_size[1])
        if not (0 <= cx < grid_size[0] and 0 <= cy < grid_size[1] and coors_range[2] <= points[i, 2] <= coors_range[5]):
            continue
        coor = np.array([cy, cx], dtype=np.int32)
        voxelidx = coor_to_voxelidx[coor[1], coor[0]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num = voxel_num + 1
            coor_to_voxelidx[coor[1], coor[0]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            #voxels[voxelidx, num, 0] = (points[i, 0] - coors_range[0] - (cx + 0.5) * voxel_size[0]) / voxel_size[0]
            #voxels[voxelidx, num, 1] = (points[i, 1] - coors_range[1] - (cy + 0.5) * voxel_size[1]) / voxel_size[1]
            #voxels[voxelidx, num, 2] = points[i, 2] / 2.7
            voxels[voxelidx, num, 0] = points[i, 0]
            voxels[voxelidx, num, 1] = points[i, 1]
            voxels[voxelidx, num, 2] = points[i, 2]
            num_points_per_voxel[voxelidx] += 1

    return voxel_num


def points_to_voxel(points,
                    coors_range,
                    voxelmap_shape,  # (x,y)方向格数
                    max_points,
                    max_voxels):

    if not isinstance(voxelmap_shape, np.ndarray):
        voxelmap_shape = np.array(voxelmap_shape, dtype=np.int32)
    voxelmap_shape = voxelmap_shape[:2]
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxel_size = (coors_range[3:5] - coors_range[:2]) / voxelmap_shape[:2]

    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)  # 确定每个voxel的实际点数
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(shape=(max_voxels, max_points, 4), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 2), dtype=np.int32)  # 确定voxel的位置

    voxel_num = _points_to_voxel_kernel(
        points, voxel_size, voxelmap_shape, coors_range, num_points_per_voxel,
        coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]

    return voxels, coors, num_points_per_voxel, voxel_num
'''
