import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
from torch.utils.data.sampler import SubsetRandomSampler
import random
import shutil

# Define a custom dataset class to load data from HDF5 files
class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        random.shuffle(self.file_list)
        print(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_dir, self.file_list[idx])
        with h5py.File(file_name, 'r') as hf:
            data = hf['data'][:]
            energy = hf['energy'][:]
            cell = hf['cell'][:]
            kpoint = hf['kpt'][:]
            GWenergy = hf['GWenergy'][:]
            charge_density = hf['charge_density'][:]
            super_band = hf['super_band'][:]
        return [abs(data), energy, cell, kpoint, GWenergy, charge_density,super_band] # we ignore phase firstly !!!

class Merge:
    def __init__(self, data_dir, dimension:list=(30,30,30),) -> None:
        self.h5list = CustomDataset(data_dir=data_dir)

        # todo: think about another way to resize data
        self.height = dimension[2]
        self.width  = dimension[0]
        self.length = dimension[1]
        self.figures = [self.h5list[i][0].shape[0] * self.h5list[i][0].shape[1] for i in range(len(self.h5list))] # [n1*k1, n2*k2...]


        f = h5py.File('wfs_merged.h5','w')
        f.create_dataset('data', (np.sum(np.array(self.figures)),1, self.width, self.length,self.height) )
        f.create_dataset('charge_density', (np.sum(np.array(self.figures)),1, self.width, self.length,self.height))
        f.create_dataset('energy', (np.sum(np.array(self.figures)),) )
        f.create_dataset('GWenergy', (np.sum(np.array(self.figures)),))
        f.create_dataset('information', (np.sum(np.array(self.figures)),9+3+1)) # 9 is 3*3 cells information, 3 is kpoints, 1 is quantum number [..-2,-1,1,2...]
        f.create_dataset('charge_density_unique',( len(self.h5list) ,1, self.width, self.length,self.height))

        ######### number of super band 12/02/2023
        self.n_super_band = self.h5list[0][6].shape[2]
        f.create_dataset('super_band', (np.sum(np.array(self.figures)), self.n_super_band,1, self.width, self.length,self.height))
        f.create_dataset('super_band_unique', (len(self.h5list), self.n_super_band ,1, self.width, self.length,self.height))
        ######### number of super band

        start = 0 # index of state for all materials
        for i, nstate in enumerate(self.figures): # loop over different materials
            print('materials: %s'%self.h5list.file_list[i])
            # print('h5:',i)
            wfs_mat = self.h5list[i][0]
            energy =  self.h5list[i][1]
            cell = self.h5list[i][2]
            kpoint = self.h5list[i][3]
            GWenergy = self.h5list[i][4]
            charge_density_mat = self.h5list[i][5]

            ######### number of super band 12/02/2023
            superband_mat = self.h5list[i][6]
            ######### number of super band

            number_band = wfs_mat.shape[1]
            number_kpt  = wfs_mat.shape[0] 

            wfs_mat = wfs_mat.reshape(nstate, 1,wfs_mat.shape[2], wfs_mat.shape[3], wfs_mat.shape[4])
            energy =energy.reshape(nstate)
            GWenergy = GWenergy.reshape(nstate)
            cell = cell.reshape(nstate, 9 )
            kpoint = kpoint.reshape(nstate, 3 )
            charge_density_mat = charge_density_mat.reshape(nstate, 1,charge_density_mat.shape[2], charge_density_mat.shape[3], charge_density_mat.shape[4])
            
            ######### number of super band 12/02/2023
            superband_mat = superband_mat.reshape(nstate, self.n_super_band, 1,superband_mat.shape[3], superband_mat.shape[4], superband_mat.shape[5])
            ######### number of super band


            for j in range(start, start+nstate): # loop over states |nk> in each materials


                i_kn = j -start
                i_n = i_kn % number_band
                i_k = i_kn // number_band
                valence_band = number_band // 2 ### number_band = nelectron

                wfs_state = wfs_mat[i_kn] # (w,l,c)
                charge_state = charge_density_mat[i_kn]
                superband_states = superband_mat[i_kn]
                wfs_max = np.argmax(wfs_state.sum(axis=(0,1)))
                charge_max = np.argmax(charge_state.sum(axis=(0,1,2)))


                f['energy'][j] = energy[i_kn]
                f['GWenergy'][j] = GWenergy[i_kn]
                f['information'][j] = np.hstack( (cell[i_kn], kpoint[i_kn], (i_n+1) - valence_band) )


                wfs_max = wfs_state.shape[3] // 2
                superband_max = superband_states.shape[4] // 2
                charge_max = charge_state.shape[3]//2

                print('state:',j,'/', sum(self.figures), 'wfs_max:', wfs_max)

                for c in range(wfs_max - self.height//2, wfs_max + self.height//2):
                    f['data'][j, 0,:,:,c-wfs_max + self.height//2] = cv2.resize(wfs_state[0,...,c], (self.width, self.length))
                    pass
                for c in range(charge_max - self.height//2, charge_max + self.height//2):
                    f['charge_density'][j, 0,:,:,c - charge_max + self.height//2] = cv2.resize(charge_state[0,...,c], (self.width, self.length))
                    f['charge_density_unique'][i,0,:,:, c - charge_max + self.height//2] = cv2.resize(charge_state[0,...,c], (self.width, self.length))
                    pass
                for c in range(superband_max - self.height//2, superband_max + self.height//2):
                    for n_sp in range(self.n_super_band):
                        f['super_band'][j, n_sp, 0,:,:,c - superband_max + self.height//2] = cv2.resize(superband_states[n_sp,0,:,:,c], (self.width, self.length))
                        f['super_band_unique'][i, n_sp, 0,:,:,c - superband_max + self.height // 2] = cv2.resize(superband_states[n_sp,0,:,:,c], (self.width, self.length))
                    pass

            start += nstate

        f.close()



if __name__ == "__main__":
    '''
    source_dir = "figures"  # 原始文件夹
    target_dir = "figures_random"  # 目标文件夹

    num_files_to_select = 100

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    all_h5_files = [f for f in os.listdir(source_dir) if f.endswith('.hdf5')]

    if num_files_to_select > len(all_h5_files):
        raise ValueError("要选取的文件数量大于可用文件数量")

    selected_files = random.sample(all_h5_files, num_files_to_select)

    for file_name in selected_files:
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)
        shutil.copy(source_file, target_file)
    '''   
    merge = Merge('figures',dimension=[30,30,50])
    
    pass
