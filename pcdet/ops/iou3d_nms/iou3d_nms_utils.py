"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""
import torch
import datetime

from pcdet.utils import common_utils
#from . import iou3d_nms_cuda


def boxes_bev_iou_cpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    boxes_a, is_numpy = common_utils.check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = common_utils.check_numpy_to_torch(boxes_b)
    assert not (boxes_a.is_cuda or boxes_b.is_cuda), 'Only support CPU tensors'
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou.numpy() if is_numpy else ans_iou


def boxes_iou_bev(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d



def cross(p1, p2, p0):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])


def check_box2d(box, p):
    MARGIN = 1e-2
    center_x = box[0]
    center_y = box[1]
    angle_cos = torch.cos(-box[6])
    angle_sin = torch.sin(-box[6])
    rot_x = (p[0] - center_x) * angle_cos + (p[1] - center_y) * (-angle_sin)
    rot_y = (p[0] - center_x) * angle_sin + (p[1] - center_y) * angle_cos

    return ((abs(rot_x) < (box[3] / 2 + MARGIN)) and (abs(rot_y) < (box[4] / 2 + MARGIN)))


def intersection(p1, p0, q1, q0, ans):
    threshold = 1e-8
    if ((min(p0[0], p1[0]) <= max(q0[0], q1[0])) and
        (min(q0[0], q1[0]) <= max(p0[0], p1[0])) and
        (min(p0[1], p1[1]) <= max(q0[1], q1[1])) and
        (min(q0[1], q1[1]) <= max(p0[1], p1[1])) == 0):
        return False

    s1 = cross(q0, p1, p0)
    s2 = cross(p1, q1, p0)
    s3 = cross(p0, q1, q0)
    s4 = cross(q1, p1, q0)

    if not ((s1*s2 > 0) and (s3 * s4 > 0)):
        return False

    s5 = cross(q1, p1, p0)
    if abs(s5 - s1) > threshold:
        ans[0] = (s5 * q0[0] - s1 * q1[0]) / (s5 - s1)
        ans[1] = (s5 * q0[1] - s1 * q1[1]) / (s5 - s1)
    else:
        a0 = p0[1] - p1[1]
        b0 = p1[0] - p0[0]
        c0 = p0[0] * p1[1] - p1[0] * p0[1]
        a1 = q0[1] - q1[1]
        b1 = q1[0] - q0[0]
        c1 = q0[0] * q1[1] - q1[0] * q0[1]
        D = a0 * b1 - a1 * b0
        ans[0] = (b0 * c1 - b1 * c0) / D
        ans[1] = (a1 * c0 - a0 * c1) / D

    return True


def rotate_around_center(center, angle_cos, angle_sin, p):
    new_x = (p[0] - center[0]) * angle_cos + (p[1] - center[1]) * (-angle_sin) + center[0]
    new_y = (p[0] - center[0]) * angle_sin + (p[1] - center[1]) * angle_cos + center[1]
    p[0] = new_x
    p[1] = new_y


def box_overlap(box_a, box_b):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :return:
    """
    rt_a, rt_b = box_a[6], box_b[6]
    a_cos, a_sin = torch.cos(rt_a), torch.sin(rt_a)
    b_cos, b_sin = torch.cos(rt_b), torch.sin(rt_b)

    a_x1 = box_a[0] - box_a[3]/2
    a_y1 = box_a[1] - box_a[4]/2
    a_x2 = box_a[0] + box_a[3]/2
    a_y2 = box_a[1] + box_a[4]/2

    b_x1 = box_b[0] - box_b[3]/2
    b_y1 = box_b[1] - box_b[4]/2
    b_x2 = box_b[0] + box_b[3]/2
    b_y2 = box_b[1] + box_b[4]/2

    box_a_corners = torch.zeros((5, 2), device='cpu')
    box_b_corners = torch.zeros((5, 2), device='cpu')
    cross_points = torch.zeros((16, 2), device='cpu')

    box_a_corners[0] = torch.Tensor([a_x1, a_y1])
    box_a_corners[1] = torch.Tensor([a_x2, a_y1])
    box_a_corners[2] = torch.Tensor([a_x2, a_y2])
    box_a_corners[3] = torch.Tensor([a_x1, a_y2])

    box_b_corners[0] = torch.Tensor([b_x1, b_y1])
    box_b_corners[1] = torch.Tensor([b_x2, b_y1])
    box_b_corners[2] = torch.Tensor([b_x2, b_y2])
    box_b_corners[3] = torch.Tensor([b_x1, b_y2])

    center_a = torch.Tensor((box_a[0], box_a[1]))
    center_b = torch.Tensor((box_b[0], box_b[1]))
    for k in range(4):
        rotate_around_center(center_a, a_cos, a_sin, box_a_corners[k])
        rotate_around_center(center_b, b_cos, b_sin, box_b_corners[k])
    box_a_corners[4] = box_a_corners[0]
    box_b_corners[4] = box_b_corners[0]

    cnt = 0
    flag = False
    poly_center = torch.zeros((2, ), device='cpu')
    for i in range(4):
        for j in range(4):
            flag = intersection(box_a_corners[i+1], box_a_corners[i],
                                box_b_corners[j+1], box_b_corners[j],
                                cross_points[cnt])
            if flag:
                poly_center[0] = poly_center[0] + cross_points[cnt, 0]
                poly_center[1] = poly_center[1] + cross_points[cnt, 1]
                cnt += 1

    for k in range(4):
        if check_box2d(box_a, box_b_corners[k]):
            poly_center[0] = poly_center[0] + box_b_corners[k, 0]
            poly_center[1] = poly_center[1] + box_b_corners[k, 1]
            cnt += 1
        if check_box2d(box_b, box_a_corners[k]):
            poly_center[0] = poly_center[0] + box_a_corners[k, 0]
            poly_center[1] = poly_center[1] + box_a_corners[k, 1]
            cnt += 1

    poly_center[0] = poly_center[0] / cnt
    poly_center[1] = poly_center[1] / cnt
    temp = torch.zeros((2, ), device='cpu')
    for j in range(cnt-1):
        for i in range(cnt - 1):
            if (torch.atan2(cross_points[i, 1] - poly_center[1], cross_points[i, 0] - poly_center[0]) >
                torch.atan2(cross_points[i+1, 1] - poly_center[1], cross_points[i+1, 0] - poly_center[0])):
                temp = cross_points[i]
                cross_points[i] = cross_points[i + 1]
                cross_points[i + 1] = temp

    area = 0
    a = torch.zeros((2, ), device='cpu')
    b = torch.zeros((2, ), device='cpu')
    for k in range(cnt-1):
        a[0] = cross_points[k, 0] - cross_points[0, 0]
        a[1] = cross_points[k, 1] - cross_points[0, 1]
        b[0] = cross_points[k+1, 0] - cross_points[0, 0]
        b[1] = cross_points[k+1, 1] - cross_points[0, 1]
        area += (a[0]*b[1] - a[1]*b[0])
    return abs(area)/2


def nms_cpu_process(boxes, keep, thresh):
    threshold = 1e-8
    num_out = 0
    for i in range(boxes.size(0)):
        if keep[i] == 1:
            continue
        for j in range(i+1, boxes.size(0)):
            if keep[j] == 1:
                continue
            sa = boxes[i, 3] * boxes[i, 4]  # dx*dy
            sb = boxes[j, 3] * boxes[j, 4]
            s_overlap = box_overlap(boxes[i], boxes[j])
            iou = s_overlap / max(sa+sb-s_overlap, threshold)
            if iou > thresh:
                keep[j] = 1
                num_out += 1
    return num_out


def nms_cpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    # 对分数按列降序排序(从大到小)，并取出对应索引
    # dim=0 按列排序，dim=1 按行排序，默认 dim=1
    # score传入前已torch.topk排序，所以order是[0 ,1, 2, 3, ...]
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    # 此前已排序，此处无变化
    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))  # 构造一个boxes.size维度的向量  init elem default=0
    # keep：记录保留目标框的下标
    # num_out：返回保留下来的个数
    print('num of box:', boxes.size(0))
    start = datetime.datetime.now()
    num_out = nms_cpu_process(boxes, keep, thresh)
    end = datetime.datetime.now()
    inference_time = (end - start).total_seconds()
    print("nms process 1 sample time ", inference_time, 's')
    return order[keep[:num_out]].contiguous(), None


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None


def nms_normal_gpu(boxes, scores, thresh, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None
