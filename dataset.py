import os
import cv2
import parse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img

from utils import *


class Badminton_Dataset(Dataset):
    def __init__(self, root_dir=['/kaggle/input/shuttlecock-tracknetv2'], split='train', mode='2d', num_frame=3, slideing_step=1, frame_dir=None, debug=False, rate=1.0, seed=42):
        """
        Args:
            root_dir (string): Directory with all the images (default reads from /kaggle/input/shuttlecock-tracknetv2).
            split (string): name of the split or subfolder inside root_dir (e.g., 'Train', 'Val', 'test', or any custom mode).
            Saved index files (.npz) are stored under /kaggle/working.
        """
        self.HEIGHT = 288
        self.WIDTH = 512
        self.mag = 1
        self.sigma = 2.5

        #
        self.rate = rate
        self.seed = seed

        self.root_dirs = root_dir
        #####
        self.split = split
        self.mode = mode
        self.num_frame = num_frame
        self.slideing_step = slideing_step

        # where to save/load generated index files
        self.work_dir = '/kaggle/working/TrackNetv2/TrackNetV2_Dataset'
        os.makedirs(self.work_dir, exist_ok=True)

        npz_name = f'f{self.num_frame}_s{self.slideing_step}_{self.split}.npz'
        npz_path_work = os.path.join(self.work_dir, npz_name)
        

        
        if not os.path.exists(npz_path_work):
            # generate from image folders under input root and save to work dir
            self._gen_frame_files()
        # load from work dir
        data_dict = np.load(npz_path_work, allow_pickle=True)
        
        if debug:
            num_debug = 256
            self.frame_files = data_dict['filename'][:num_debug] # (N, 3)
            self.coordinates = data_dict['coordinates'][:num_debug] # (N, 3, 2)
            self.visibility = data_dict['visibility'][:num_debug] # (N, 3)
        elif frame_dir:
            self.frame_files, self.coordinates, self.visibility = self._gen_frame_unit(frame_dir)
        else:
            self.frame_files = data_dict['filename'] # (N, 3)
            self.coordinates = data_dict['coordinates'] # (N, 3, 2)
            self.visibility = data_dict['visibility'] # (N, 3)

    def _get_rally_dirs(self, rate=1.0, seed=42):
        """
        Collect rally directories depending on self.split.
        Supports arbitrary split names: if a subfolder named self.split exists under root_dir, use it.
        Otherwise, if split == 'test' look for 'Test' folder; else scan all subfolders except 'Test'.
        """
        match_dirs = []

        for root_dir in self.root_dirs:
        
            # If there is a subfolder exactly matching split under root_dir, use that
            split_path = os.path.join(root_dir, self.split)
            if os.path.isdir(split_path):
                base_dirs = [split_path]
                
            else:
                # common case: user requested 'test' but folder named 'Test' (case-sensitive)
                if self.split.lower() == 'test' and os.path.isdir(os.path.join(root_dir, 'Test')):
                    base_dirs = [os.path.join(root_dir, 'Test')]
                else:
                    # fallback: scan all top-level subfolders under root_dir excluding 'Test' (so we don't double-include)
                    base_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                                 if os.path.isdir(os.path.join(root_dir, d)) and d != 'Test' and d!= 'Amateur']
            print('base_dirs: ', base_dirs)

            for base in base_dirs:
                # list_dirs is assumed to return list of match directories under base (from utils)
                try:
                    sub_match_dirs = list_dirs(base)
                except Exception:
                    # if list_dirs is not appropriate, try collecting direct subdirs that contain a 'frame' folder
                    sub_match_dirs = []
                    for d in os.listdir(base):
                        cand = os.path.join(base, d)
                        frame_folder = os.path.join(cand, 'frame')
                        if os.path.isdir(frame_folder):
                            sub_match_dirs.append(frame_folder)
                # extend the global list
                match_dirs.extend(sub_match_dirs)
                
        print('match_dirs: ',match_dirs)
        
        # match_dirs now contains paths like .../matchX/frame
        # we will choose 30% rallies per each match, in seed condition
        rally_dirs = []
        rng = np.random.default_rng(seed)
        
        for match_dir in match_dirs:
            rally_dir = list_dirs(os.path.join(match_dir,'frame'))
            
            k = max(1, int(len(rally_dir) * rate))
            sampled_rallies = list(rng.choice(rally_dir, size=k, replace=False))
            
            rally_dirs.extend(sampled_rallies)

        print('Num of rally_dirs: ', len(rally_dirs))
        return rally_dirs

    def _gen_frame_files(self):
        rally_dirs = self._get_rally_dirs(rate = self.rate, seed = self.seed)
        frame_files = np.array([]).reshape(0, self.num_frame)
        coordinates = np.array([], dtype=np.float32).reshape(0, self.num_frame, 2)
        visibility = np.array([], dtype=np.float32).reshape(0, self.num_frame)

        # Generate input sequences from each rally
        for rally_dir in tqdm(rally_dirs):
            #print('rally_dir', rally_dir)
            # rally_dir expected like .../matchX/frame/<rally_id>  OR could be .../matchX/frame if already at frame dir.
            try:
                match_dir, rally_id = parse.parse('{}/frame/{}', rally_dir)
            except Exception:
                # fallback: try to find parent match_dir
                match_dir = os.path.dirname(os.path.dirname(rally_dir))
                rally_id = os.path.basename(rally_dir)
            csv_file = os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
            try:
                label_df = pd.read_csv(csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)
            except Exception:
                print(f'Label file {csv_file} not found. Skipping rally {rally_dir}.')
                continue
            
            frame_file = np.array([os.path.join(rally_dir, f'{f_id}.png') for f_id in label_df['Frame']])
            
            x, y, vis = np.array(label_df['X']), np.array(label_df['Y']), np.array(label_df['Visibility'])
            if not (len(frame_file) == len(x) == len(y) == len(vis)):
                print(f'Length mismatch in {rally_dir}. Skipping.')
                continue

            # Sliding on the frame sequence
            for i in range(0, len(frame_file)-self.num_frame, self.slideing_step):
                tmp_frames, tmp_coor, tmp_vis = [], [], []
                # Construct a single input sequence
                for f in range(self.num_frame):
                    if os.path.exists(frame_file[i+f]):
                        tmp_frames.append(frame_file[i+f])
                        tmp_coor.append((x[i+f], y[i+f]))
                        tmp_vis.append(vis[i+f])
                    else:
                        break
                    
                if len(tmp_frames) == self.num_frame:
                    assert len(tmp_frames) == len(tmp_coor) == len(tmp_vis)
                    frame_files = np.concatenate((frame_files, [tmp_frames]), axis=0)
                    coordinates = np.concatenate((coordinates, [tmp_coor]), axis=0)
                    visibility = np.concatenate((visibility, [tmp_vis]), axis=0)
        
        # Save to /kaggle/working for reuse
        npz_name = f'f{self.num_frame}_s{self.slideing_step}_{self.split}.npz'
        save_path = os.path.join(self.work_dir, npz_name)
        np.savez(save_path, filename=frame_files, coordinates=coordinates, visibility=visibility)

    def _gen_frame_unit(self, frame_dir):
        frame_files = np.array([]).reshape(0, self.num_frame)
        coordinates = np.array([], dtype=np.float32).reshape(0, self.num_frame, 2)
        visibility = np.array([], dtype=np.float32).reshape(0, self.num_frame)
        
        match_dir, rally_id = parse.parse('{}/frame/{}', frame_dir)
        csv_file = f'{match_dir}/csv/{rally_id}_ball.csv'
        label_df = pd.read_csv(csv_file, encoding='utf8').sort_values(by='Frame')
        frame_file = np.array([f'{frame_dir}/{f_id}.png' for f_id in label_df['Frame']])
        x, y, vis = np.array(label_df['X']), np.array(label_df['Y']), np.array(label_df['Visibility'])
        assert len(frame_file) == len(x) == len(y) == len(vis)

        # Sliding on the frame sequence
        for i in range(0, len(frame_file)-self.num_frame, self.slideing_step):
            tmp_frames, tmp_coor, tmp_vis = [], [], []
            # Construct a single input sequence
            for f in range(self.num_frame):
                if os.path.exists(frame_file[i+f]):
                    tmp_frames.append(frame_file[i+f])
                    tmp_coor.append((x[i+f], y[i+f]))
                    tmp_vis.append(vis[i+f])

            # Append the input sequence
            if len(tmp_frames) == self.num_frame:
                assert len(tmp_frames) == len(tmp_coor) == len(tmp_vis)
                frame_files = np.concatenate((frame_files, [tmp_frames]), axis=0)
                coordinates = np.concatenate((coordinates, [tmp_coor]), axis=0)
                visibility = np.concatenate((visibility, [tmp_vis]), axis=0)
        
        return frame_files, coordinates, visibility

    def _get_heatmap(self, cx, cy, visible):
        if not visible:
            return np.zeros((1, self.HEIGHT, self.WIDTH)) if self.mode == '2d' else np.zeros((1, 1, self.HEIGHT, self.WIDTH))
        x, y = np.meshgrid(np.linspace(1, self.WIDTH, self.WIDTH), np.linspace(1, self.HEIGHT, self.HEIGHT))
        heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
        heatmap[heatmap <= self.sigma**2] = 1.
        heatmap[heatmap > self.sigma**2] = 0.
        heatmap = heatmap * self.mag
        return heatmap.reshape(1, self.HEIGHT, self.WIDTH) if self.mode == '2d' else heatmap.reshape(1, 1, self.HEIGHT, self.WIDTH)

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        frame_file = self.frame_files[idx]
        coors = self.coordinates[idx]
        vis = self.visibility[idx]

        # Get the resize scaler
        h, w, _ = cv2.imread(frame_file[0]).shape
        h_ratio, w_ratio = h / self.HEIGHT, w / self.WIDTH

        # Transform the coordinate
        coors[:, 0] = coors[:, 0] / h_ratio
        coors[:, 1] = coors[:, 1] / w_ratio

        if self.mode == '2d':
            frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
            heatmaps = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)

            for i in range(self.num_frame):
                img = load_img(frame_file[i])
                img = img_to_array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                img = np.moveaxis(img, -1, 0)
                frames = np.concatenate((frames, img), axis=0)
                heatmap = self._get_heatmap(int(coors[i][0]), int(coors[i][1]), vis[i])
                heatmaps = np.concatenate((heatmaps, heatmap), axis=0)        
        else:
            frames = np.array([]).reshape(3, 0, self.HEIGHT, self.WIDTH)
            heatmaps = np.array([]).reshape(1, 0, self.HEIGHT, self.WIDTH)

            for i in range(self.num_frame):
                img = load_img(frame_file[i])
                img = img_to_array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                img = np.moveaxis(img, -1, 0) 
                img = img.reshape(3, 1, self.HEIGHT, self.WIDTH)
                frames = np.concatenate((frames, img), axis=1)
                heatmap = self._get_heatmap(int(coors[i][0]), int(coors[i][1]), vis[i])
                heatmaps = np.concatenate((heatmaps, heatmap), axis=1)
        
        frames /= 255.
        return idx, frames, heatmaps, coors