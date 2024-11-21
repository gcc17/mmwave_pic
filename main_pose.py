import argparse
import os
import yaml
import torch
import datasets
import models
from utils import dict2namespace, AverageMeter
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from models.loss import mpjpe, p_mpjpe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mmbody_point_transformer.yaml', help='path to config file')
    args = parser.parse_args()
    
    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load dataset
    if config.data.dataset == 'MMBody':
        train_dataset = datasets.__dict__[config.data.dataset](config.data.dataset_root, split='train', device=device)
        test_dataset = datasets.__dict__[config.data.dataset](config.data.dataset_root, split='test', device=device)
    elif config.data.dataset == 'MMFi':
        dataset_config_path = os.path.join('datasets', 'mmfi.yaml')
        with open(dataset_config_path, 'r') as fd:
            dataset_config = yaml.load(fd, Loader=yaml.FullLoader)
        train_dataset = datasets.__dict__[config.data.dataset](config.data.dataset_root, split='train', config=dataset_config, device=device)
        test_dataset = datasets.__dict__[config.data.dataset](config.data.dataset_root, split='test', config=dataset_config, device=device)
    elif config.data.dataset == 'mmPoseNLP':
        train_data_path = os.path.join('datasets', 'merged_data', 'train_data.npy')
        test_data_path = os.path.join('datasets', 'merged_data', 'test_data.npy')
        train_dataset = datasets.__dict__[config.data.dataset](train_data_path, noise_level=[0.05,0.1,0.2])
        test_dataset = datasets.__dict__[config.data.dataset](test_data_path, noise_level=0.25)
        
    print(len(train_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)
    
    # set up model
    model = models.__dict__[config.model.model](input_dim=config.data.radar_input_c, n_p=config.data.num_joints).to(device)
    print(model)
    # set up optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=config.train.learning_rate, weight_decay=config.train.weight_decay, foreach=True)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=config.train.amp)
    criterion = nn.MSELoss()
    
    os.makedirs(config.train.save_model_root, exist_ok=True)
    cur_save_dir = os.path.join(config.train.save_model_root, f'{config.data.dataset}_{config.model.model}')
    os.makedirs(cur_save_dir, exist_ok=True)
    
    best_mpjpe = 1000000
    best_p_mpjpe = 1000000
    for epoch in range(config.train.n_epochs):
        model.train()
        epoch_loss = AverageMeter()
        epoch_mpjpe = AverageMeter()
        epoch_p_mpjpe = AverageMeter()
        with tqdm(total=len(train_loader), desc=f'Train round{epoch}/{config.train.n_epochs}', unit='batch') as pbar:
            for data in train_loader:
                inputs, labels = data
                labels = labels[:,:,:] - labels[:,:1,:]
                inputs = inputs.type(torch.FloatTensor).to(device)
                labels = labels.type(torch.FloatTensor).to(device)
                
                optimizer.zero_grad()
                outputs, feat = model(inputs)
                outputs = outputs.type(torch.FloatTensor).to(device)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                clip_grad_norm_(model.parameters(), config.train.gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                batch_mpjpe = mpjpe(outputs, labels).item()*1e3
                batch_p_mpjpe = p_mpjpe(outputs, labels).item()*1e3 
                epoch_loss.update(loss.item(), inputs.size(0))
                epoch_mpjpe.update(batch_mpjpe, inputs.size(0))
                epoch_p_mpjpe.update(batch_p_mpjpe, inputs.size(0))
                pbar.update(1)
                pbar.set_postfix(**{'loss (batch)': loss.item(), 'mpjpe': batch_mpjpe, 'p-mpjpe': batch_p_mpjpe})
        
        print('Epoch:{}, Loss:{:.9f}, MPJPE:{:.4f}, P-MPJPE:{:.4f}'.format(
            epoch + 1, epoch_loss.avg, epoch_mpjpe.avg, epoch_p_mpjpe.avg))
    
        if epoch % config.train.test_freq == 0:
            model.eval()
            test_loss = AverageMeter()
            test_mpjpe = AverageMeter()
            test_p_mpjpe = AverageMeter()
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data
                    labels = labels[:,:,:] - labels[:,:1,:]
                    inputs = inputs.type(torch.FloatTensor).to(device)
                    labels = labels.type(torch.FloatTensor).to(device)
                    
                    outputs, feat = model(inputs)
                    outputs = outputs.type(torch.FloatTensor).to(device)
                    loss = criterion(outputs, labels)
                    
                    test_loss.update(loss.item(), inputs.size(0))
                    test_mpjpe.update(mpjpe(outputs, labels).item()*1e3, inputs.size(0))
                    test_p_mpjpe.update(p_mpjpe(outputs, labels).item()*1e3, inputs.size(0))
            
            print('Test Loss:{:.9f}, MPJPE:{:.4f}, P-MPJPE:{:.4f}'.format(
                test_loss.avg, test_mpjpe.avg, test_p_mpjpe.avg))
            if test_mpjpe.avg < best_mpjpe:
                best_mpjpe = test_mpjpe.avg
                torch.save(model.state_dict(), os.path.join(cur_save_dir, 
                                                            f'epoch{epoch}_mpjpe{test_mpjpe.avg:.4f}_pmpjpe{test_p_mpjpe.avg:.4f}.pth'))
            if test_p_mpjpe.avg < best_p_mpjpe:
                best_p_mpjpe = test_p_mpjpe.avg
                torch.save(model.state_dict(), os.path.join(cur_save_dir, 
                                                            f'epoch{epoch}_mpjpe{test_mpjpe.avg:.4f}_pmpjpe{test_p_mpjpe.avg:.4f}.pth'))
                
                
                