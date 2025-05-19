"""
    @author: Jay Lago, NIWC Pacific, 55280
    adapted from https://github.com/zhouhaoyi/Informer2020
"""
import os
import pandas as pd
import numpy as np
import pickle
from typing import List
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import matplotlib.lines as mlines
import matplotlib as mpl
plt.rcParams.update({
    'font.family': 'sans-serif',        # or 'sans-serif', 'monospace', etc.
    'font.serif': ['Times New Roman'],  # specify a particular serif font
    'font.style': 'normal',             # or 'italic'
    'font.weight': 'normal',            # or 'normal', 'light', etc.
    'font.size': 14,                    # base font size
    'axes.titlesize': 16,               # title font size
    'axes.labelsize': 14,               # axis label font size
    'xtick.labelsize': 14,              # x-axis tick label font size
    'ytick.labelsize': 14,              # y-axis tick label font size
    'legend.fontsize': 14,              # legend font size
})

DPI = 150
SEED = 1997
np.random.seed(SEED)
torch.manual_seed(SEED)

#------------------------------------------------------------------------------
# PyTorch Dataset
#------------------------------------------------------------------------------
class IonosphereTransformerDataset(Dataset):
    def __init__(
            self,
            data: torch.tensor,
            indices: list,
            enc_seq_len: int,
            dec_seq_len: int,
            pred_seq_len: int,
            pred_cols: List,
            past_cov_cols: List,
            future_cov_cols: List,
            device
        ):
        super().__init__()
        self.device = device
        self.data = data.to(self.device)
        self.indices = indices
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.pred_seq_len = pred_seq_len
        self.past_cov_cols = past_cov_cols
        self.pred_cols = pred_cols
        self.future_cov_cols = future_cov_cols

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # Get indices for the segment start and stop
        start_idx = self.indices[index][0]
        end_idx = self.indices[index][1]
        segment = self.data[start_idx:end_idx]
        
        # Get encoder input - covariates for past only (observations)
        enc_input = segment[:self.enc_seq_len, self.past_cov_cols]

        # Get decoder input - covariates for both past and future (known values)
        if len(self.future_cov_cols)>=1:
            dec_input = segment[self.enc_seq_len:, self.future_cov_cols]
        else:
            dec_input = torch.zeros((self.dec_seq_len, 1))

        # Get targets for computing loss
        targets = segment[self.enc_seq_len:, self.pred_cols]
        
        return enc_input, dec_input, targets


#------------------------------------------------------------------------------
# PyTorch Data Module
#------------------------------------------------------------------------------
class TransformerDataModule():
    def __init__(self, hyp):
        super(TransformerDataModule).__init__()
        self.data_path = hyp['data_path']
        self.device = hyp['device']
        self.model_name = hyp['model_name']
        self.train_list = hyp['train_list']
        self.test_list = hyp['test_list']
        self.num_days = hyp['num_days']
        self.num_sub = hyp['num_sub']
        self.val_split = hyp['val_split']
        self.window_size = hyp['window_size']
        self.overlap = hyp['window_step']
        self.enc_seq_len = hyp['enc_seq_len']
        self.dec_seq_len = hyp['dec_seq_len']
        self.tgt_seq_len = hyp['tgt_seq_len']
        self.batch_size = hyp['batch_size']
        self.num_loader_workers = hyp['num_loader_workers']
        self.accel = hyp['accelerator']
        self.indices_path = hyp['indices_path']
        self.new_indices = hyp['new_indices']
        self.data_cols = hyp['data_cols']
        self.tgt_cols_idx = hyp['tgt_cols_idx']
        self.past_cols_idx = hyp['past_cols_idx']
        self.future_cols_idx = hyp['future_cols_idx']
        self.time_feature_cols = ['sYOS', 'cYOS', 'sDOY', 'cDOY', 'sHOD', 'cHOD', 'sMOH', 'cMOH']

    def prepare_data(self, make_plots=False):
        if os.path.exists(self.data_path):
            df = pd.read_pickle(self.data_path)
            print(f'[INFO] Successfully loaded data from file: {self.data_path}')
        else:
            print(f'[ERROR] Data file not found: {self.data_path}')
            exit()
        # Convert to float32
        cols32 = ['foF2', 'foF1', 'foE', 'hmF2', 'hmF1', 'hmE', 'TEC',
                  'glat', 'glon', 'sza',
                  'IRI_F2_fo', 'IRI_F2_hm', 'IRI_F2_B_bot', 'IRI_F2_B_top',
                  'Kp', 'Ap', 'SSN', 'F107']
        df[cols32] = df[cols32].astype(np.float32)

        # Create time features from timestamps
        df = self._generate_time_features(df)

        # Separate training from testing stations
        self.train_list = pd.read_csv(self.train_list)
        self.test_list = pd.read_csv(self.test_list)
        df_train = df[df['station_id'].isin(self.train_list.ID.values)]
        self.test_data = df[df['station_id'].isin(self.test_list.ID.values)]
        del df

        # Choose random subsample of the segments
        if self.num_sub>0:
            num_unique_segs = df_train['segment'].unique().max()
            rand_segs = np.random.randint(0, num_unique_segs, self.num_sub)
            df_train = df_train[df_train['segment'].isin(rand_segs)]
        
        # Separate validation set using segment identifier
        split_idx = int(self.val_split * df_train['segment'].max())
        unique_ids = df_train['segment'].unique()
        train_set_ids = np.random.choice(unique_ids, split_idx, replace=False)
        val_set_ids = np.setdiff1d(unique_ids, train_set_ids)
        self.train_data = df_train[df_train['segment'].isin(train_set_ids)]
        self.val_data = df_train[df_train['segment'].isin(val_set_ids)]
        
        # split_idx = int(self.val_split * df_train['segment'].max())
        # self.train_data = df_train.loc[df_train['segment']<split_idx]
        # self.val_data = df_train.loc[df_train['segment']>=split_idx]

        # Find start-stop indices for the segments
        if self.new_indices:
            print(f'[INFO] Making new start/stop indices for training set')
            self.train_indices = self._get_segment_start_stop(df=self.train_data)
            print(f'[INFO] Making new start/stop indices for validation set')
            self.val_indices = self._get_segment_start_stop(df=self.val_data)
            print(f'[INFO] Making new start/stop indices for test set')
            self.test_indices = self._get_segment_start_stop(df=self.test_data)
            print(f'[INFO] Saving start/stop indices to file, {self.indices_path}')
            self.segment_indices = {
                'train': self.train_indices,
                'val': self.val_indices,
                'test': self.test_indices
            }
            pickle.dump(self.segment_indices, open(self.indices_path, 'wb'))
        else:
            print(f'[INFO] Using start/stop indices from file, {self.indices_path}')
            self.segment_indices = pickle.load(open(self.indices_path, 'rb'))
            self.train_indices = self.segment_indices['train']
            self.val_indices = self.segment_indices['val']
            self.test_indices = self.segment_indices['test']
        
        # Shuffle the indices (seed should be set in main)
        print(f'[INFO] Number of unique train sequences: {len(self.train_indices)}')
        print(f'[INFO] Number of unique validation sequences: {len(self.val_indices)}')
        print(f'[INFO] Number of unique test sequences: {len(self.test_indices)}')
        np.random.shuffle(self.train_indices)
        np.random.shuffle(self.val_indices)
        np.random.shuffle(self.test_indices)
        
        # k-fold cross validation (not implemented)
        self.train_indices = [self.train_indices]
        self.val_indices = [self.val_indices]
        num_train = len(self.train_data)
        num_val = len(self.val_data)
        num_test = len(self.test_data)
        print(f'[INFO] Number of train points: {num_train}')
        print(f'[INFO] Number of validation points: {num_val}')
        print(f'[INFO] Number of test points: {num_test}')

        # Convert final train data to a tensor
        self.train_df = self.train_data
        self.val_df = self.val_data
        self.test_df = self.test_data
        self.train_data = torch.Tensor(self.train_data[self.data_cols].values)
        self.val_data = torch.Tensor(self.val_data[self.data_cols].values)
        self.test_data = torch.Tensor(self.test_data[self.data_cols].values)

        # Plot station map and the segment splits
        if make_plots:
            self.save_station_map()
            self.save_segment_plot()
            self.save_segment_densities()

    def setup(self, stage: str, fold: int=-1):
        # Creates a new dataset for this fold and returns the appropriate dataloader
        if stage=="fit":
            train_fold = IonosphereTransformerDataset(
                data=self.train_data,
                indices=self.train_indices[fold],
                enc_seq_len=self.enc_seq_len,
                dec_seq_len=self.dec_seq_len,
                pred_seq_len=self.tgt_seq_len,
                pred_cols=self.tgt_cols_idx,
                past_cov_cols=self.past_cols_idx,
                future_cov_cols=self.future_cols_idx,
                device=self.device
            )
            return self.get_train_dataloader(train_fold)
        
        if stage=="validate":
            val_fold = IonosphereTransformerDataset(
                data=self.val_data,
                indices=self.val_indices[fold],
                enc_seq_len=self.enc_seq_len,
                dec_seq_len=self.dec_seq_len,
                pred_seq_len=self.tgt_seq_len,
                pred_cols=self.tgt_cols_idx,
                past_cov_cols=self.past_cols_idx,
                future_cov_cols=self.future_cols_idx,
                device=self.device
            )
            return self.get_val_dataloader(val_fold)
        
        if stage=="test":
            test_fold = IonosphereTransformerDataset(
                data=self.test_data,
                indices=self.test_indices,
                enc_seq_len=self.enc_seq_len,
                dec_seq_len=self.dec_seq_len,
                pred_seq_len=self.tgt_seq_len,
                pred_cols=self.tgt_cols_idx,
                past_cov_cols=self.past_cols_idx,
                future_cov_cols=self.future_cols_idx,
                device=self.device
            )
            return self.get_test_dataloader(test_fold)

    def get_train_dataloader(self, train_data) -> DataLoader:
        if self.accel=='gpu' or self.accel=='mps':
            train_dataloader = DataLoader(
                dataset=train_data,
                batch_size=self.batch_size,
                num_workers=self.num_loader_workers,
                drop_last=True,
            )
        else:
            train_dataloader = DataLoader(
                dataset=train_data,
                batch_size=self.batch_size,
                num_workers=self.num_loader_workers,
                drop_last=True,
            )
        return train_dataloader

    def get_val_dataloader(self, val_data) -> DataLoader:
        if self.accel=='gpu' or self.accel=='mps':
            val_dataloader = DataLoader(
                dataset=val_data,
                batch_size=self.batch_size,
                num_workers=self.num_loader_workers,
                drop_last=True,
            )
        else:
            val_dataloader = DataLoader(
                dataset=val_data,
                batch_size=self.batch_size,
                num_workers=self.num_loader_workers,
                drop_last=True,
            )
        return val_dataloader

    def get_test_dataloader(self, test_data) -> DataLoader:
        if self.accel=='gpu' or self.accel=='mps':
            test_dataloader = DataLoader(
                dataset=test_data,
                batch_size=self.batch_size,
                num_workers=self.num_loader_workers,
                drop_last=True,
            )
        else:
            test_dataloader = DataLoader(
                dataset=test_data,
                batch_size=self.batch_size,
                num_workers=self.num_loader_workers,
                drop_last=True,
            )
        return test_dataloader

    def _get_individual_test_segment_dataset(self):
        test_dataset = IonosphereTransformerDataset(
            data=self.test_data,
            indices=self.test_indices,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            pred_seq_len=self.tgt_seq_len,
            pred_cols=self.tgt_cols_idx,
            past_cov_cols=self.past_cols_idx,
            future_cov_cols=self.future_cols_idx,
            device=self.device
        )
        return test_dataset
    
    def _get_segment_start_stop(self, df: pd.DataFrame) -> List:
        seg_idx_left = 0
        seg_idx_right = self.window_size
        segments = []
        n = len(df)-1
        while seg_idx_right <= len(df)-1:
            station_id_left = df.iloc[seg_idx_left]['station_id']
            station_id_right = df.iloc[seg_idx_right]['station_id']
            seg_id_left = df.iloc[seg_idx_left]['segment']
            seg_id_right = df.iloc[seg_idx_right]['segment']
            if (station_id_left != station_id_right) or (seg_id_left != seg_id_right):
                seg_idx_left += 1
                seg_idx_right += 1
            else:
                segments.append((seg_idx_left, seg_idx_right))
                seg_idx_left += self.overlap
                seg_idx_right += self.overlap
            perc_done = 100*(seg_idx_right/n)
            print(f'{seg_idx_right}/{n}, {perc_done:3.2f}%', end='\r')
        return segments
    
    def _generate_time_features(self, df):
        # Old features
        two_pi = 2*np.pi
        df['mmI'] = np.cos(two_pi * df.index.minute / 60.0)
        df['mmQ'] = np.sin(two_pi * df.index.minute / 60.0)
        df['hhI'] = np.cos(two_pi * df.index.hour / 24.0)
        df['hhQ'] = np.sin(two_pi * df.index.hour / 24.0)
        df['doyI'] = np.cos(two_pi * df.index.dayofyear / 365.0)
        df['doyQ'] = np.sin(two_pi * df.index.dayofyear / 365.0)
        df['yosI'] = np.cos(two_pi * df.index.year / 11.0)
        df['yosQ'] = np.sin(two_pi * df.index.year / 11.0)
        df['timeI'] = df['mmI'] + df['hhI'] + df['doyI'] + df['yosI']
        df['timeQ'] = df['mmQ'] + df['hhQ'] + df['doyQ'] + df['yosQ']
        
        # New features
        timestamps = df['timestamp']
        yos = year_of_solar_cycle(timestamps)
        doy = day_of_year(timestamps)
        hod = hour_of_day(timestamps)
        moh = minute_of_hour(timestamps)
        time_features = np.hstack([yos, doy, hod, moh])
        df[self.time_feature_cols] = time_features
        
        return df
    
    def get_dataframes(self):
        return self.train_df, self.val_df, self.test_df

    def save_station_map(self):
        print('[INFO] Plotting station map')
        train_obs_count = np.ones(len(self.train_list))
        for ii, station in enumerate(self.train_list.ID.values):
            num_obs = len(self.train_df.loc[self.train_df['station_id']==station])
            if num_obs>0:
                train_obs_count[ii] = num_obs
        
        test_obs_count = np.ones(len(self.test_list))
        for ii, station in enumerate(self.test_list.ID.values):
            num_obs = len(self.test_df.loc[self.test_df['station_id']==station])
            if num_obs>0:
                test_obs_count[ii] = num_obs

        lats_train = self.train_list.lat.values
        lons_train = self.train_list.lon.values
        lats_test = self.test_list.lat.values
        lons_test = self.test_list.lon.values
        train_m_size = 1e3*train_obs_count/train_obs_count.max()
        test_m_size = 1e3*test_obs_count/test_obs_count.max()
        # train_m_size = np.log10(train_obs_count)*100
        # test_m_size = np.log10(test_obs_count)*200
        
        # Plot for the training set
        fig_train = plt.figure(figsize=(16, 9))
        ax_train = fig_train.add_subplot(1, 1, 1, projection=crs.PlateCarree())
        ax_train.stock_img()
        ax_train.gridlines(alpha=0.5)
        plt.scatter(
            x=lons_train, y=lats_train, s=train_m_size, marker='o',
            color="blue", alpha=0.75, label='Train/Val', transform=crs.PlateCarree()
        )
        # plt.legend(loc='lower left', fontsize=18)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Training/Validation Set Station Data (2000-2023)')
        plt.savefig(f"./results/{self.model_name}/fig_station_map_{self.num_days}day_train.pdf", dpi=DPI, bbox_inches='tight')
        plt.close()

        # Plot for the test set
        fig_test = plt.figure(figsize=(16, 9))
        ax_test = fig_test.add_subplot(1, 1, 1, projection=crs.PlateCarree())
        ax_test.stock_img()
        ax_test.gridlines(alpha=0.5)
        plt.scatter(
            x=lons_test, y=lats_test, s=test_m_size, marker='s',
            color="red", alpha=0.75, label='Test', transform=crs.PlateCarree()
        )
        # plt.legend(loc='lower left', fontsize=18)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Test Set Station Data (2000-2023)')
        plt.savefig(f"./results/{self.model_name}/fig_station_map_{self.num_days}day_test.pdf", dpi=DPI, bbox_inches='tight')
        plt.close()


    def save_segment_plot(self):
        mpl.rcParams['font.size'] = 24
        mpl.rcParams['font.weight'] = 'bold'
        mpl.rcParams['font.family'] = 'Sans-Serif'

        print('[INFO] Plotting segments')
        plt.figure(figsize=(20, 15))
        plt.plot(self.train_df.index, self.train_df['segment'], 'b.', label='Training')
        plt.plot(self.val_df.index, self.val_df['segment'], 'r.', label='Validation')
        plt.plot(self.test_df.index, self.test_df['segment'], 'g.', label='Testing')
        m1 = mlines.Line2D([], [], color='blue', marker='.', markersize=15, linestyle='None')
        m2 = mlines.Line2D([], [], color='red', marker='.', markersize=15, linestyle='None')
        m3 = mlines.Line2D([], [], color='green', marker='.', markersize=15, linestyle='None')
        plt.legend(handles=[m1, m2, m3], labels=['Training', 'Validation', 'Testing'])
        plt.grid(alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Unique segment ID')
        plt.title('Data splits by segment ID')
        # plt.show()
        plt.savefig(f"./results/{self.model_name}/fig_segment_plot_{self.num_days}day.pdf", dpi=DPI, bbox_inches='tight')
        plt.close()
    
    
    def save_segment_densities(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.lines as mlines
        mpl.rcParams['font.size'] = 24
        mpl.rcParams['font.weight'] = 'bold'
        mpl.rcParams['font.family'] = 'Sans-Serif'

        print('[INFO] Plotting segment densities')
        # Group by date and count segment ids
        train_counts = self.train_df.groupby(self.train_df['timestamp'].dt.year)['segment'].count()
        val_counts = self.val_df.groupby(self.val_df['timestamp'].dt.year)['segment'].count()
        test_counts = self.test_df.groupby(self.test_df['timestamp'].dt.year)['segment'].count()

        plt.figure(figsize=(10, 6))
        plt.semilogy(train_counts.index, train_counts.values, color='blue', linestyle=':', marker='o', label='Training')
        plt.semilogy(val_counts.index, val_counts.values, color='green', linestyle=':', marker='x', label='Validation')
        plt.semilogy(test_counts.index, test_counts.values, color='red', linestyle=':', marker='s', label='Testing')
        plt.xlabel('Date (Year)')
        plt.ylabel('Unique Segments (count)')
        plt.title('Segment ID Counts Over Time')
        # plt.xticks(rotation=45)
        plt.grid(True, which="both", ls="-", alpha=0.75)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./results/{self.model_name}/fig_segment_plot_{self.num_days}day_counts.pdf", dpi=DPI, bbox_inches='tight')
        plt.close()


def year_of_solar_cycle(timestamps, cycle_start='1996-08-01', cycle_years=11):
    cycle_start = pd.Timestamp(cycle_start)
    days_since_start = (timestamps - cycle_start).dt.total_seconds() / (24 * 3600)
    cycles = (days_since_start % (cycle_years * 365.25)) / (cycle_years * 365.25)
    return np.column_stack([
        np.sin(2 * np.pi * cycles),
        np.cos(2 * np.pi * cycles)
    ])

def day_of_year(timestamps):
    days = timestamps.dt.dayofyear
    return np.column_stack([
        np.sin(2 * np.pi * days / 365.25),
        np.cos(2 * np.pi * days / 365.25)
    ])

def hour_of_day(timestamps):
    hours = timestamps.dt.hour + timestamps.dt.minute / 60 + timestamps.dt.second / 3600
    return np.column_stack([
        np.sin(2 * np.pi * hours / 24),
        np.cos(2 * np.pi * hours / 24)
    ])

def minute_of_hour(timestamps):
    minutes = timestamps.dt.minute
    return np.column_stack([
        np.sin(2 * np.pi * minutes / 60),
        np.cos(2 * np.pi * minutes / 60)
    ])

