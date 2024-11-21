import os 
import numpy as np
import pandas as pd
import yaml
import glob
import torch
import cv2
from torch.utils.data import Dataset
from pointnet2_ops import pointnet2_utils
from .preprocessing import *
import pickle
from tqdm import tqdm

class MMBody(Dataset):
    """
    Args:
        root (str of Path): Path to the dataset.
        split (str): Split of the dataset. Selected from ["train", "test"] 
        test_scenario (str): Only applicable for test split dataset, the selected scene (from 7 test scenes) for testing.
        normalized (bool): Whether normalized data using the ground truth human pose (pelvis location). If True, 
        subtracting all points (radar point clouds, ground truth pose) with the pelvis location.
        device (str): Device to load the data. Needs to be gpu as we need to use pointnet2_ops.
    """
    def __init__(self, root, split, test_scenario='all', normalized=True, device=None, 
                 merge_nframes=5, npoints=200):
        super().__init__()
        self.root = root
        self.split = split
        if test_scenario == "all": 
            self.test_scenario = ["lab1", "lab2", "furnished", "rain", "smoke", "poor_lighting", "occlusion"]
        else: 
            self.test_scenario = test_scenario
        self.normalized = normalized
        self.normalized_center = [None, None, None]
        self.device = device
        self.merge_nframes = merge_nframes  
        self.npoints = npoints
        
        # initialize dataframe
        self.df = pd.DataFrame()
        self.path_df = pd.DataFrame()
        
        if split == "train":
            split_path = os.path.join(self.root, split)
            all_sequences = os.listdir(split_path)
            sequence_id_list = [(int(sequence.split("_")[-1]), sequence) for sequence in all_sequences]
            # sort the sequence_id_list with the first element of each tuple (sequence_id)
            sequence_id_list = sorted(sequence_id_list, key=lambda x: x[0])
            all_sequences = [sequence_id[1] for sequence_id in sequence_id_list]    
            for sequence in all_sequences:
                sequence_path = os.path.join(split_path, sequence)
                radar_path_df = self._load_radar(sequence_path)
                mesh_path_df = self._load_mesh(sequence_path)
                data_path_df = mesh_path_df.set_index(['Sequence', 'Frame']).join(
                    radar_path_df.set_index(['Sequence', 'Frame'])).reset_index(names=['Sequence', 'Frame'])
                self.path_df = pd.concat([self.path_df, data_path_df], ignore_index=True)
            
        else:
            for scenario in self.test_scenario:
                scenario_path = os.path.join(self.root, split, scenario)
                all_sequences = os.listdir(scenario_path)
                sequence_id_list = [(int(sequence.split("_")[-1]), sequence) for sequence in all_sequences]
                # sort the sequence_id_list with the first element of each tuple (sequence_id)
                sequence_id_list = sorted(sequence_id_list, key=lambda x: x[0])
                all_sequences = [sequence_id[1] for sequence_id in sequence_id_list]    
                for sequence in all_sequences:
                    sequence_path = os.path.join(scenario_path, sequence)
                    radar_path_df = self._load_radar(sequence_path)
                    mesh_path_df = self._load_mesh(sequence_path)
                    data_path_df = mesh_path_df.set_index(['Sequence', 'Frame']).join(
                        radar_path_df.set_index(['Sequence', 'Frame'])).reset_index(names=['Sequence', 'Frame'])
                    self.path_df = pd.concat([self.path_df, data_path_df], ignore_index=True)
        
    
    def _load_radar(self, current_path):
        """
        Description: Load the all radar data from .npy to np.array(), concatenate all data
        into dataframe. Sort the dataframe according to the frame.

        Input:
        self: inherent from class
        current_path: indicated the current path of the data to be loaded.
        (Example): ../data/mmBody/train/sequence_9

        Output:
        Return a Tuple of format (sequence, frame, radar data)
        """
        radar_path = []
        sequence = int(current_path.split("/")[-1].split("_")[-1])
        for file in glob.glob(os.path.join(current_path, "radar", "*.npy")):
            frame = int(file.split("/")[-1].split(".")[0].split("_")[1])
            radar_path.append([sequence, frame, file])
        radar_path_df = pd.DataFrame(radar_path, columns=["Sequence", "Frame", "Radar"])
        radar_path_df = radar_path_df.sort_values("Frame", ignore_index=True)

        return radar_path_df
    
    def _load_mesh(self, current_path):
        """
        Description: Load the all mesh data from .npy to np.array(), concatenate all data
        into dataframe. Sort the dataframe according to the frame.

        Input:
        self: inherent from class
        current_path: indicated the current path of the data to be loaded.
        (Example): ../data/mmpose/train/sequence_9

        Output:
        Return a Tuple of format (sequence, frame, mesh data)
        """
        mesh_path = []
        sequence = int(current_path.split("/")[-1].split("_")[-1])
        for file in glob.glob(os.path.join(current_path, "mesh", "*.npz")):
            frame = int(file.split("/")[-1].split(".")[0].split("_")[1])
            mesh_path.append([sequence, frame, file])
        mesh_path_df = pd.DataFrame(mesh_path,columns=["Sequence", "Frame", "Mesh"])
        mesh_path_df = mesh_path_df.sort_values("Frame", ignore_index=True)

        return mesh_path_df
    
        
    def __len__(self):
        return self.path_df.shape[0]
    
    def __getitem__(self, index):
        r'''
        Returns:
            (input, label)[np.ndarray, int]: 
            "input" (np.ndarray): The point clouds from preprocessed radar point clouds. The shape of pc: [frame_together, npoints, 5], the -2 dimension is removed
            "label" (np.ndarray): The ground truth human pose. The shape of pose: [17, 3].
        Example:
            >>> hpe_train_dataset = MMBody(root, split='test', test_scenario=['lab1', 'lab2'], normalized=True, device='cuda')
            >>> index = 9
            >>> sample= har_train_dataset.__getitem__(index)
        '''
        sequence = self.path_df.iloc[index]["Sequence"]
        frame = self.path_df.iloc[index]["Frame"]
        mesh_path = self.path_df.iloc[index]["Mesh"]
        data = {"Sequence": sequence, "Frame": frame}
        
        # Load mesh ground truth
        mesh_data = np.load(mesh_path)
        data["joints"] = mesh_data["joints"]
        if self.normalized:
            self.normalized_center = data["joints"][0]
        
        radar_path = self.path_df.iloc[index]["Radar"]
        radar_path_list = sliding_window_mmbody(self.merge_nframes, radar_path, frame)
        radar_data_list = []
        # print(radar_path_list)
        for path in radar_path_list:
            radar_data = np.load(path)
            if self.normalized:
                radar_data[:, :3] = np.subtract(radar_data[:, :3], self.normalized_center)
                radar_data[:, 3] = radar_data[:, 3] * 1e38
                radar_data[:, 5] = radar_data[:, 5] / 100
            xyzvi_indices = [0,1,2,3,5]
            radar_data = radar_data[:, xyzvi_indices]
            # radar_data = cropping(radar_data)
            radar_data = cropping(radar_data, x_range=[-1.0, 1.0])
            # print("radar_data shape:", radar_data.shape)  # radar_data shape: (423, 5)
            radar_data_list.append(radar_data)
            
        # merge the radar data along the first dimension
        data["Radar"] = np.concatenate(radar_data_list, axis=0)
        # print("data['Radar'] shape:", data["Radar"].shape)    # data['Radar'] shape: (2115, 5)
        
        # downsample the point cloud
        continuous_radar_data = torch.from_numpy(data["Radar"])[:,:3].float().unsqueeze(0).contiguous().to(self.device)
        radar_anchor_idx = pointnet2_utils.furthest_point_sample(continuous_radar_data, self.npoints).squeeze(0).to(torch.device('cpu')).to(torch.int64)
        data["Radar"] = data["Radar"][radar_anchor_idx]
        
        # convert to (data, label) pair
        selected_joints =  [0, 1,4,7, 2,5,8, 6,12,15,24,  16,18,20, 17,19,21]
        label =  torch.from_numpy(data["joints"][selected_joints,:])
        input = torch.from_numpy(data["Radar"])

        return (input, label)
    
    

class MMFi(Dataset):
    """
    Args:
        root (str of Path): Path to the dataset.
        split (str): Split of the dataset. Selected from ["training", "testing"] 
        config (Object): The configuration class object for the MMFi dataset.
    """
    def __init__(self, root, split, config, device=None, 
                 merge_nframes=5, npoints=60):
        super().__init__()
        self.root = root
        self.split = split
        self.device = device
        self.merge_nframes = merge_nframes 
        self.npoints = npoints
        self.subject_action_dict = self.decode_config(config)[split]
        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.load_database()
        self.data_unit = config['data_unit']
        self.data_list = self.load_data()
        
    
    def load_database(self):
        for scene in sorted(os.listdir(self.root)):
            if not os.path.isdir(os.path.join(self.root, scene)):
                continue
            self.scenes[scene] = {}
            for subject in sorted(os.listdir(os.path.join(self.root, scene))):
                if not os.path.isdir(os.path.join(self.root, scene, subject)):
                    continue
                self.scenes[scene][subject] = {}
                self.subjects[subject] = {}
                for action in sorted(os.listdir(os.path.join(self.root, scene, subject))):
                    if not os.path.isdir(os.path.join(self.root, scene, subject, action)):
                        continue
                    self.scenes[scene][subject][action] = {}
                    self.subjects[subject][action] = {}
                    self.actions[action] = {}
                    self.actions[action][scene] = {}
                    self.actions[action][scene][subject] = {}
                    data_path = os.path.join(self.root, scene, subject, action, 'mmwave')
                    self.scenes[scene][subject][action] = data_path
                    self.subjects[subject][action] = data_path
                    self.actions[action][scene][subject] = data_path
                    
    
    def decode_config(self, config):
        all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14',
                        'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
                        'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
        all_actions = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                    'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
        train_subject_action_dict = {}
        val_subject_action_dict = {}
        # Limitation to actions (protocol)
        if config['protocol'] == 'protocol1':  # Daily actions
            actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
        elif config['protocol'] == 'protocol2':  # Rehabilitation actions:
            actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26']
        else:
            actions = all_actions
        # Limitation to subjects and actions (split choices)
        if config['split_to_use'] == 'random_split':
            rs = config['random_split']['random_seed']
            ratio = config['random_split']['ratio']
            for action in actions:
                np.random.seed(rs)
                idx = np.random.permutation(len(all_subjects))
                idx_train = idx[:int(np.floor(ratio*len(all_subjects)))]
                idx_val = idx[int(np.floor(ratio*len(all_subjects))):]
                subjects_train = np.array(all_subjects)[idx_train].tolist()
                subjects_val = np.array(all_subjects)[idx_val].tolist()
                for subject in all_subjects:
                    if subject in subjects_train:
                        if subject in train_subject_action_dict:
                            train_subject_action_dict[subject].append(action)
                        else:
                            train_subject_action_dict[subject] = [action]
                    elif subject in subjects_val:
                        if subject in val_subject_action_dict:
                            val_subject_action_dict[subject].append(action)
                        else:
                            val_subject_action_dict[subject] = [action]
                            
        elif config['split_to_use'] == 'cross_scene_split':
            subjects_train = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                            'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
                            'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']
            subjects_val = ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
            for subject in subjects_train:
                train_subject_action_dict[subject] = actions
            for subject in subjects_val:
                val_subject_action_dict[subject] = actions
        
        elif config['split_to_use'] == 'cross_subject_split':
            subjects_train = config['cross_subject_split']['train_dataset']['subjects']
            subjects_val = config['cross_subject_split']['val_dataset']['subjects']
            for subject in subjects_train:
                train_subject_action_dict[subject] = actions
            for subject in subjects_val:
                val_subject_action_dict[subject] = actions
        
        else:
            subjects_train = config['manual_split']['train_dataset']['subjects']
            subjects_val = config['manual_split']['val_dataset']['subjects']
            actions_train = config['manual_split']['train_dataset']['actions']
            actions_val = config['manual_split']['val_dataset']['actions']
            for subject in subjects_train:
                train_subject_action_dict[subject] = actions_train
            for subject in subjects_val:
                val_subject_action_dict[subject] = actions_val
    
        dataset_config = {
            'train': train_subject_action_dict, 
            'test': val_subject_action_dict
        }
        return dataset_config
    
    def get_scene(self, subject):
        if subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
            return 'E01'
        elif subject in ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']:
            return 'E02'
        elif subject in ['S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']:
            return 'E03'
        elif subject in ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']:
            return 'E04'
        else:
            raise ValueError('Subject does not exist in this dataset.')
    
    def load_data(self):
        data_info = []
        for subject, actions in self.subject_action_dict.items():
            print(subject, actions)
            for action in actions:
                if self.data_unit == 'sequence':
                    data_dict = {
                        'scene': self.get_scene(subject),
                        'subject': subject,
                        'action': action,
                        'gt_path': os.path.join(self.root, self.get_scene(subject), 
                                                subject, action, 'ground_truth.npy'), 
                        'mmwave_path': os.path.join(self.root, self.get_scene(subject), 
                                                    subject, action,'mmwave')
                    }
                    data_info.append(data_dict)
                elif self.data_unit == 'frame':
                    frame_list = sorted(os.listdir(os.path.join(self.root, self.get_scene(subject),
                                                    subject, action,'mmwave')))
                    frame_num = len(frame_list)
                    for idx in range(frame_num):
                        frame_idx = int(frame_list[idx].split('.')[0][5:])-1
                        data_dict = {
                            'scene': self.get_scene(subject),
                            'subject': subject,
                            'action': action,
                            'gt_path': os.path.join(self.root, self.get_scene(subject), 
                                                    subject, action, 'ground_truth.npy'), 
                            'mmwave_path': os.path.join(self.root, self.get_scene(subject), 
                                                        subject, action,'mmwave', frame_list[idx]),
                            'idx': frame_idx
                        }
                        data_info.append(data_dict)
                else:
                    raise ValueError('Data unit should be "sequence" or "frame".')
        return data_info
    
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        '''
        Returns:
            (input, label)[np.ndarray, int]: 
            "input" (np.ndarray): The point clouds from preprocessed radar point clouds. The shape of pc: [npoints, 5], default [50, 5].
            "label" (np.ndarray): The ground truth human pose. The shape of pose: [17, 3].
        '''
        item = self.data_list[idx]
        gt_numpy = np.load(item['gt_path'])
        gt_torch = torch.from_numpy(gt_numpy)
        
        if self.data_unit == 'sequence':
            mmwave_label = gt_torch
            if os.path.isdir(item['mmwave_path']):
                mmwave_data = self.read_dir(item['mmwave_path'])
            else:
                mmwave_data = np.load(f'{item["mmwave_path"]}.npy')
        elif self.data_unit == 'frame':
            mmwave_data = self.read_frame(item['mmwave_path'])
            mmwave_label = gt_torch[item['idx']]
        else:
            raise ValueError('Data unit should be "sequence" or "frame".')
        return mmwave_data, mmwave_label
    
    def read_dir(self, mmwave_dir):
        data = []
        for bin_file in sorted(glob.glob(os.path.join(mmwave_dir, "frame*.bin"))):
            with open(bin_file, 'rb') as f:
                raw_data = f.read()
                data_tmp = np.frombuffer(raw_data, dtype=np.float64)
                data_tmp = data_tmp.reshape(-1, 3)
            data.append(data_tmp)
        return data
    
    def read_frame(self, mmwave_frame_path):
        
        frame_idx = int(mmwave_frame_path.split("/")[-1].split('.')[0][5:])
        radar_data_list = []
        radar_path_list = sliding_window_mmfi(self.merge_nframes, mmwave_frame_path, frame_idx)
        # print(radar_path_list)
        for path in radar_path_list:
            with open(path, 'rb') as f:
                raw_data = f.read()
            radar_data = np.frombuffer(raw_data, dtype=np.float64)
            radar_data = radar_data.copy().reshape(-1, 5)
            radar_data_list.append(radar_data)
            # print(radar_data.shape, radar_data)
        data = np.concatenate(radar_data_list, axis=0)
        if data.shape[0] == 0:
            data = np.zeros((self.npoints, 5))
        elif data.shape[0] < self.npoints:
            data = np.pad(data, ((0, self.npoints-data.shape[0]), (0, 0)), 'edge')
            
        # downsample the point cloud
        continuous_radar_data = torch.from_numpy(data)[:,:3].float().unsqueeze(0).contiguous().to(self.device)
        radar_anchor_idx = pointnet2_utils.furthest_point_sample(continuous_radar_data, self.npoints).squeeze(0).to(torch.device('cpu')).to(torch.int64)
        # print(radar_anchor_idx)
        data = data[radar_anchor_idx]
        
        return data

partitions = (0.8, 0.1, 0.1)
class MiliPointPose(Dataset):
    def __init__(self, raw_data_path, split, device=None, num_keypoints=17, 
                 merge_nframes=5, npoints=30):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.split = split
        self.device = device
        self.num_keypoints = num_keypoints
        self.merge_nframes = merge_nframes
        self.npoints = npoints
        self.data_list = self._process()

    
    @property
    def raw_file_names(self):
        file_names = [i for i in range(19)]
        return [f'{self.raw_data_path}/{i}.pkl' for i in file_names]
    
    def _process(self):
        data_list = []
        for fn in self.raw_file_names:
            with open(fn, 'rb') as f:
                data_slice = pickle.load(f)
            data_list = data_list + data_slice
        num_samples = len(data_list)
        data_list = self.transform_keypoints(data_list)
        
        # get partitions
        train_end = int(partitions[0] * num_samples)
        val_end = train_end + int(partitions[1] * num_samples)
        train_data = data_list[:train_end]
        val_data = data_list[train_end:val_end]
        test_data = data_list[val_end:]
        data_map = {'train': train_data, 'val': val_data, 'test': test_data}
        return data_map[self.split]
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_point = self.pad_frame(self.data_list, idx)
        return data_point, self.data_list[idx]['y']
        
        
    def transform_keypoints(self, data_list):
        if self.num_keypoints == 18:
            return data_list
        elif self.num_keypoints == 17:
            # pelvis and spine coordinates: 1/3 of ((left_hip+right_hip)/2, neck)
            # nose as head, right_eye
            for data in tqdm(data_list, total=len(data_list)):
                kpts = data['y']
                hip_coord = (kpts[8] + kpts[11]) / 2
                neck_coord = kpts[1]
                pelvis_coord = (neck_coord-hip_coord)/3 + hip_coord
                spine_coord = (neck_coord-hip_coord)*2/3 + hip_coord
                head_coord = kpts[0]
                right_eye_coord = kpts[14]
                kpts_new = np.array([
                    pelvis_coord, kpts[11], kpts[12], kpts[13], 
                    kpts[8], kpts[9], kpts[10], 
                    spine_coord, neck_coord, head_coord, right_eye_coord, 
                    kpts[5], kpts[6], kpts[7], 
                    kpts[2], kpts[3], kpts[4] ])
                data['y'] = kpts_new
            return data_list
        else:
            raise ValueError('Number of keypoints should be 17 or 18.')
    
    def pad_frame(self, data_list, i):
        xs = [d['x'] for d in data_list]
        data_point = []
        start_frame_idx = max(0, i-self.merge_nframes)
        end_frame_idx = min(len(xs), start_frame_idx+self.merge_nframes)
        for fi in range(start_frame_idx, end_frame_idx):
            data_point.append(xs[fi])
            print(xs[fi].shape)
        data_point = np.concatenate(data_point, axis=0)
        if data_point.shape[0] < self.npoints:
            print('data_point shape: ', data_point.shape)
            data_point = np.pad(data_point, ((0, self.npoints-data_point.shape[0]), (0, 0)), 'edge')
        continuous_radar_data = torch.from_numpy(data_point)[:,:3].float().unsqueeze(0).contiguous().to(self.device)
        radar_anchor_idx = pointnet2_utils.furthest_point_sample(continuous_radar_data, self.npoints).squeeze(0).to(torch.device('cpu')).to(torch.int64)
        data_point = data_point[radar_anchor_idx]
        return data_point
            

# for milipoint dataset
kp18_names = ['NOSE', 'NECK', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 
            'RIGHT_WRIST', 'LEFT_SHOULDER', 'LEFT_ELBOW',   # [4,5,6]
            'LEFT_WRIST', 'RIGHT_HIP', 'RIGHT_KNEE',        # [7,8,9]
            'RIGHT_ANKLE', 'LEFT_HIP', 'LEFT_KNEE',         # [10,11,12]
            'LEFT_ANKLE', 'RIGHT_EYE', 'LEFT_EYE',          # [13,14,15]
            'RIGHT_EAR', 'LEFT_EAR']  
kp17_names_mmbody = ['pelvis', 'left_hip', 'left_knee', 'left_ankle', 
                     'right_hip', 'right_knee', 'right_ankle', 
                     'spine2', 'neck', 'head', 'right_eye', 
                     'left_shoulder', 'left_elbow', 'left_wrist', 
                     'right_shoulder', 'right_elbow', 'right_wrist']    


class mmPoseNLP(Dataset):
    def __init__(self, gt_data_path,
                 noise_level=0.05, perline_points=5, subset_indices=None):
        super().__init__()
        self.gt_data_path = gt_data_path
        self.noise_level = noise_level
        self.perline_points = perline_points
        self.subset_indices = subset_indices
        self.gt_data = np.load(gt_data_path)
        if self.subset_indices is not None:
            self.gt_data = self.gt_data[self.subset_indices]
    
    def __len__(self):
        return len(self.gt_data)
    
    def __getitem__(self, idx):
        data_points = self.gt_data[idx]
        edges = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], 
                              [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        all_random_points = []
        if isinstance(self.noise_level, list):
            # randomly select one noise level
            cur_noise_level = np.random.choice(self.noise_level)
        else:
            cur_noise_level = self.noise_level
        for edge in edges:
            p1 = data_points[edge[0]]
            p2 = data_points[edge[1]]
            random_points = self.random_point_between_spheres_and_cylinder(p1, p2, cur_noise_level, self.perline_points)
            all_random_points.extend(random_points)
        all_random_points = np.array(all_random_points)
        input = torch.from_numpy(all_random_points).float()
        label = torch.from_numpy(data_points).float()
        return input, label
    
    def random_point_between_spheres_and_cylinder(self, p1, p2, d, num_points=1):
        """
        在两个点的球体和连线圆柱体组合区域内随机生成点。

        参数:
        - p1: 第一个点的坐标 (x1, y1, z1)
        - p2: 第二个点的坐标 (x2, y2, z2)
        - d: 球的半径和圆柱的半径
        - num_points: 生成点的数量
        
        返回:
        - points: 随机点的列表，每个点为 (x, y, z)
        """
        p1 = np.array(p1)
        p2 = np.array(p2)
        points = []
        
        # 计算连线向量和其单位向量
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            line_len = d
            unit_vec = np.array([0, 0, 1])
        else:
            unit_vec = line_vec / line_len
        
        for _ in range(num_points):
            while True:
                # 随机决定生成的点是来自球体还是圆柱体
                if np.random.rand() < 0.5:  # 随机选择一个球体
                    center = p1 if np.random.rand() < 0.5 else p2
                    point = center + np.random.uniform(-1, 1, 3) * d
                    if np.linalg.norm(point - center) <= d:
                        points.append(tuple(point))
                        break
                else:  # 随机选择圆柱体
                    t = np.random.uniform(0, line_len)  # 在线段上的随机位置
                    center = p1 + unit_vec * t
                    # 随机生成圆柱的截面点
                    r = np.random.uniform(0, d)
                    theta = np.random.uniform(0, 2 * np.pi)
                    offset = np.array([r * np.cos(theta), r * np.sin(theta), 0])
                    # 将偏移量旋转到与线段方向一致
                    if np.linalg.norm(line_vec[:2]) == 0:  # 如果线段平行于z轴
                        rot_matrix = np.eye(3)
                    else:
                        z_vec = np.array([0, 0, 1])
                        axis = np.cross(z_vec, unit_vec)
                        axis_len = np.linalg.norm(axis)
                        axis = axis / axis_len if axis_len != 0 else axis
                        angle = np.arccos(np.dot(z_vec, unit_vec))
                        K = np.array([[0, -axis[2], axis[1]],
                                    [axis[2], 0, -axis[0]],
                                    [-axis[1], axis[0], 0]])
                        rot_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
                    offset = np.dot(rot_matrix, offset)
                    point = center + offset
                    if np.linalg.norm(point - center) <= d:
                        points.append(tuple(point))
                        break
        
        return points



if __name__ == "__main__":
    # root = '../../mmBody'
    # mmbody_dataset = MMBody(root, split='train', test_scenario=['lab1', 'lab2'], normalized=True, device='cuda')
    # index = 0
    # input, label = mmbody_dataset.__getitem__(index)
    # print(input.shape)
    # print(label.shape)
    # print(input)
    # print(mmbody_dataset._load_radar(os.path.join(root, 'train', 'sequence_9')))
    
    # root = '../../mmfi/MMFi_dataset/MMFi_Dataset/Unzipfiles'
    # with open('mmfi.yaml', 'r') as fd:
    #     config = yaml.load(fd, Loader=yaml.FullLoader)
    # mmfi_dataset = MMFi(root, split='test', config=config, device='cuda')
    # index = 10
    # input, label = mmfi_dataset.__getitem__(index)
    # print(input.shape)
    # print(label.shape)
    # print(label)
    # print(input)
    
    # raw_data_dir = '../../MiliPoint/data/raw'
    # mili_dataset = MiliPointPose(raw_data_dir, split='train', device='cuda')
    # index = 10
    # input, label = mili_dataset.__getitem__(index)
    # print(input.shape)
    # print(label.shape)
    # print(label)
    # print(input)
    
    train_data_path = os.path.join('merged_data', 'frames1_test.npy')
    pose_dataset = mmPoseNLP(train_data_path)
    input, label = pose_dataset.__getitem__(0)
    print(input.shape)
    print(label.shape)
    print(label)
    print(input)
    