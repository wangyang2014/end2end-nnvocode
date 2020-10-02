import os
import torch as th
from model import NNVocode
from torch.utils.data import DataLoader
from dataset import _collate,getSignal,Datasets

def train():
    path = r"C:\Users\Lala\Desktop\ONN-BLSTM\create_scp\tt_s1.scp"
    sig_reader = getSignal(path,320)
    dataset = Datasets(sig_reader)
    data_loader = DataLoader(dataset, batch_size=1,shuffle=True,
                             sampler=None,drop_last=True,
                            collate_fn=_collate)
    
    nnet = NNVocode(7,5,1,3,10,320)
    print(nnet)
    run(nnet,data_loader,0)

def  run(nnet,data_loader,useCuda=1):
    if useCuda:
        nnet = th.nn.DataParallel(nnet)
        nnet.cuda()
    
    optimizier = th.optim.SGD(nnet.parameters(),
                                    lr=1.0e-3,
                                    momentum =0.9,
                                    weight_decay=0.0)
    nnet.train()
    tot_loss = num_batch = 0

    for input_date,baseline,LPC,Initial_state in data_loader:
        num_batch += 1
        optimizier.zero_grad()

        speech,vq_loss = nnet(input_date, baseline,LPC,Initial_state)
        loss = nnet.loss_function(speech,baseline)
        cur_loss = th.mean(loss) + vq_loss * 10
        tot_loss += cur_loss.item()

        cur_loss.backward()
        optimizier.step()
        if num_batch % 50 == 0:
                print("Processed loss {}".format(cur_loss))



if __name__ == "__main__":
    train()