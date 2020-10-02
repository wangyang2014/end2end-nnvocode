import os
import random
import pickle
import numpy as np
import librosa as wavelib
import torch as th
from torch.utils.data import DataLoader

def parse_scps(scp_path):
    assert os.path.exists(scp_path)
    scp_dict = dict()
    with open(scp_path, 'r') as f:
        for scp in f.readlines():
            scp_tokens = scp.strip().split()
            if len(scp_tokens) != 2:
                raise RuntimeError(
                    "Error format of context \'{}\'".format(scp))
            key, addr = scp_tokens
            if key in scp_dict:
                raise ValueError("Duplicate key \'{}\' exists!".format(key))
            scp_dict[key] = addr
    return scp_dict

def formatSignal(file,frame_length=320,LPCsize=16):
    wave,_ = wavelib.load(file,sr=None)
    signalLen = wave.shape[0]
    sizeof = signalLen //  frame_length 
    
    wave[1:] = -0.85 * wave[:-1] + wave[1:]
    wave = wave[:sizeof*frame_length]
    wave = wave.reshape([sizeof,frame_length])

    input_date = np.hstack((wave[:-1,:],wave[1:,:]))
    baseline = wave[1:,:]
    N,_ =  baseline.shape
    LPC = np.zeros((N,LPCsize))
    for i in range(N):
        a = wavelib.lpc(baseline[i], LPCsize)
        LPC[i,:] = -1 * a[1:]
    
    Initial_state = input_date[:,frame_length-LPCsize:frame_length]

    return input_date,baseline,LPC,Initial_state

class getSignal(object):
    def __init__(self,wave_scp,frame_length):
        if not os.path.exists(wave_scp):
            raise FileNotFoundError("Could not find file {}".format(wave_scp))
        self.frame_length = frame_length
        self.wave_dict = parse_scps(wave_scp)
        self.wave_keys = [key for key in self.wave_dict.keys()]
    
    def __len__(self):
        return len(self.wave_dict)
    
    def __contains__(self, key):
        return key in self.wave_dict
    
    def __load(self,key):
        return formatSignal(self.wave_dict[key],self.frame_length)
    
    def __iter__(self):
        for key in self.wave_dict:
            yield key, self.__load(key)
    
    def __getitem__(self, key):
        if key not in self.wave_dict:
            raise KeyError("Could not find utterance {}".format(key))
        return self.__load(key)

class Datasets(object):
    def __init__(self,sig_reader):
        self.sig_reader = sig_reader
        self.key_list = sig_reader.wave_keys

    def __len__(self):
        return len(self.sig_reader)
    
    def __getitem__(self, index):
        key = self.key_list[index]
        input_date,baseline,LPC,Initial_state = self.sig_reader[key]
        return [input_date,baseline,LPC,Initial_state] 

def _collate(egs):
    if type(egs) is not list:
        raise ValueError("Unsupported index type({})".format(type(egs)))
    
    input_date =  th.tensor(egs[0][0], dtype=th.float32)
    baseline = th.tensor(egs[0][1], dtype=th.float32)
    LPC = th.tensor(egs[0][2], dtype=th.float32)
    Initial_state = th.tensor(egs[0][3], dtype=th.float32)

    return input_date,baseline,LPC,Initial_state

if __name__ == "__main__":
    path = r"C:\Users\Lala\Desktop\ONN-BLSTM\create_scp\tt_s1.scp"
    sig_reader = getSignal(path,320)
    dataset = Datasets(sig_reader)
    data_loader = DataLoader(dataset, batch_size=1,shuffle=True,
                             sampler=None,drop_last=True,
                            collate_fn=_collate)
    num_batch = 0
    for input_date,baseline in data_loader:
        num_batch += 1 
        infor = input_date
        #print("*"*20)
    print(num_batch)
        
    
    


