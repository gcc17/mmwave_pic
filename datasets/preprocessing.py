import os
import numpy as np

def sliding_window_mmbody(nframes, radar_path, frame_idx, stride=1):
    path_list = [radar_path]
    for i in range(1, nframes):
        # zero pad the new fname with 0 to make it 3 digits long
        new_frame_idx = f'{int(frame_idx) - i*stride}'
        old_frame_idx = f'{int(frame_idx)}'
        new_path = radar_path.replace(old_frame_idx, new_frame_idx)
        if os.path.exists(new_path):
            path_list.append(new_path)
        else:
            another_new_frame_idx = f'{int(frame_idx) + (nframes-i)*stride}'
            new_path = radar_path.replace(old_frame_idx, another_new_frame_idx)
            if os.path.exists(new_path):
                path_list.append(new_path)
    return path_list


def sliding_window_mmfi(nframes, radar_path, frame_idx, stride=1):
    path_list = [radar_path]
    for i in range(1, nframes):
        # zero pad the new fname with 0 to make it 3 digits long
        new_frame_idx = f'{int(frame_idx) - i*stride:03d}'
        old_frame_idx = f'{int(frame_idx):03d}'
        new_path = radar_path.replace(old_frame_idx, new_frame_idx)
        # print('first', new_path, old_frame_idx, new_frame_idx)
        if os.path.exists(new_path):
            path_list.append(new_path)
        else:
            another_new_frame_idx = f'{int(frame_idx) + (nframes-i)*stride:03d}'
            new_path = radar_path.replace(old_frame_idx, another_new_frame_idx)
            # print('second', new_path, old_frame_idx, new_frame_idx)
            if os.path.exists(new_path):
                path_list.append(new_path)
    return path_list


def cropping(radar_pc, x_range=[-1.67, 1.67], y_range = [-1, 1], z_range=[-1.67, 1.67]):
    r"""
    This function crop the input radar point cloud according to bounding box defined by (x_range, y_range, z_range). 
    Given input radar point cloud, the points within the bounding box will be selected as the output point cloud and the remaining will be removed.
    
    Input: 
        radar_pc (np.ndarray): a frame of PC with shape [N, 5] or [N, 6] depending on mmWave device, where N is the number of points.
    
    Args:
        radar_pc (np.ndarray): Input radar point cloud. Only support float numpy array.
        x_range (list with length 2): [minimum x range, maximum x range]. Default [-1.67, 1.67]. Modify according to mmWave device.
        y_range (list with length 2): [minimum y range, maximum y range]. Default [-1.67, 1.67]. Modify according to mmWave device.
        z_range (list with length 2): [minimum z range, maximum z range]. Default [-1.67, 1.67]. Modify according to mmWave device.

    Return:
        radar_pc [np.ndarray] The returned radar_pc is cropped version of input, of [N', 5] or [N', 6] shape, where N' is the number of points after cropping, not an uniform size.
    """
    radar_pc = radar_pc[np.where((radar_pc[:, 0] <= x_range[1]) & (radar_pc[:, 0] >= x_range[0]))]
    radar_pc = radar_pc[np.where((radar_pc[:,1] <= y_range[1]) & (radar_pc[:,1] >= y_range[0]))]
    radar_pc = radar_pc[np.where((radar_pc[:,2] <= z_range[1]) & (radar_pc[:,2] >= z_range[0]))]

    return radar_pc