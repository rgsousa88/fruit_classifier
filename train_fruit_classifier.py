import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataloader import *
from model import *
from configParser import ConfigParser
from trainHelper import *

import os, sys
from time import time
from tqdm import tqdm

import argparse

def evalFunc(pred, targets, topk=1):
    assert pred.dim() == 2
    assert targets.dim() == 1
    assert pred.size(0) == targets.size(0)
    
    with torch.no_grad():
        if isinstance(topk, int):
            topk = (topk,)
        
        maxk = max(topk)
        batch_size = targets.size(0)
        
        # Top-k
        _, pred = pred.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        result = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            result[f'top{k}'] = (correct_k / batch_size).item()
        
        return result if len(result) > 1 else result[f'top{topk[0]}']

def main(config):
    trainHelper = TrainHelper(config=config,
                              evalFunc=evalFunc)
    
    
    trainLoader, validLoader = get_data_loaders(basePath=config['basePath'],
                                                input_size=config['input_size'],
                                                batch_size=config['batch_size'])

    start_time = time()
    best_val_acc = 0.0
    MIN_VAL_ACC = 0.70

    for epoch in range(config['num_epochs']):
        tic = time()
        loss_value = 0.0
        n_batches = 0
        acc_value = 0.0

        t_train = tqdm(trainLoader, unit='batch')

        trainHelper.model.train()
        for batch in t_train:
            try:
                loss, metric = trainHelper.trainStep(batch[0], batch[1])
                loss_value += loss
                acc_value += metric
                n_batches += 1
                
                desc = f"Epoch {epoch} Loss {loss_value/n_batches:.4f}"
                if n_batches % 20 == 0:
                    desc = f"{desc} Acc {acc_value/n_batches:.4f} "
                desc = f"{desc} Elapsed Time {time()-tic:.3f}"
                
                t_train.set_description(desc)

            except Exception as e:
                print(e)
                return
        
        trainHelper.scheduler.step()
        train_loss = loss_value / n_batches
        train_acc = acc_value / n_batches

        loss_value = 0.0
        acc_value = 0.0
        n_batches = 0
        t_val = tqdm(validLoader, unit='batch')

        trainHelper.model.eval()
        for batch in t_val:
            try:
                loss,metric = trainHelper.valStep(batch[0], batch[1])
                loss_value += loss
                acc_value += metric
                n_batches += 1

                desc = f"Epoch {epoch} Loss {loss_value/n_batches:.4f}"
                if n_batches % 20 == 0:
                    desc = f"{desc} Acc {acc_value/n_batches:.4f} "
                desc = f"{desc} Elapsed Time {time()-tic:.3f}"
                
                t_val.set_description(desc)

            except Exception as e:
                print(e)
                return
        
        val_loss = loss_value / n_batches
        val_acc = acc_value / n_batches

        if val_acc > best_val_acc and val_acc > MIN_VAL_ACC:
            best_val_acc = val_acc
            trainHelper.saveModel(epoch=epoch, val_loss=val_loss, val_acc=val_acc)

        epochReport = f"Train Loss {train_loss:.4f} Val Loss {val_loss:.4f}\nTrain Acc {train_acc:.4f} Val Acc {val_acc:.4f}"
        print(f"Epoch {epoch} LR {trainHelper.optimizer.param_groups[0]['lr']:.4f} {epochReport}")
        print(f"Elapsed Time {time() - start_time:.4f}")

    print(f"Total time {time() - start_time:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Fruit Classifier')

    parser.add_argument('--config_file', type=str, required=True, help='Seu nome')
    args = parser.parse_args()

    config = ConfigParser(args.config_file).config
    
    main(config)


