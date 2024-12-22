import argparse
import os
import yaml
import torch
import numpy as np
import datasets
from datasets import CoresetWrapper
import models
import strategies
from utils import dict2namespace, AverageMeter, get_current_time
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from models.loss import mpjpe, p_mpjpe
from torch.utils.tensorboard import SummaryWriter
import logging
import sys

CURRENT_TIME: str = get_current_time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mmbody_point_transformer.yaml', help='path to config file')
    parser.add_argument('--model_ckpt_path', type=str, default=None, help='Path to the model checkpoint to load.')
    parser.add_argument('--center_mmfi', action='store_true', help='Whether to center the mmfi dataset.')
    parser.add_argument('--random_ratio', type=float, help='The ratio of random samples in the coreset.')
    parser.add_argument('--random_seed', type=int, default=420, help='The random seed for coreset selection.')
    args = parser.parse_args()
    
    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    
    # set up logging
    log_dir = 'random_logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{log_dir}/{CURRENT_TIME}.txt", mode="w", encoding="utf-8")
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(config)
    
    # set up tensorboard
    if isinstance(config.data.train_noise_level, list):
        train_noise_name = '_'.join([str(n) for n in config.data.train_noise_level])
    else:
        train_noise_name = config.data.train_noise_level
    if isinstance(config.data.test_noise_level, list):
        test_noise_name = '_'.join([str(n) for n in config.data.test_noise_level])
    else:
        test_noise_name = config.data.test_noise_level
    select_name = f'ratio{args.random_ratio}_seed{args.random_seed}'
    if not config.data.dataset == 'mmPoseNLP':
        if config.data.dataset == 'MMFi':
            center_name = 'center' if args.center_mmfi else 'original'
            checkpoint_dir = os.path.join(config.model.checkpoint_root_dir, \
                f'{select_name}-{config.data.dataset}-{config.model.model}-{center_name}-{CURRENT_TIME}')
        else:
            checkpoint_dir = os.path.join(config.model.checkpoint_root_dir, \
                f'{select_name}-{config.data.dataset}-{config.model.model}-{CURRENT_TIME}')
    else:
        checkpoint_dir = os.path.join(config.model.checkpoint_root_dir, \
            f'{config.data.dataset}-{config.model.model}-trainnoise{train_noise_name}-testnoise{test_noise_name}-{CURRENT_TIME}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=checkpoint_dir)
    logger.info(f"Writing tensorboard logs to {checkpoint_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load dataset
    if config.data.dataset == 'MMBody':
        train_dataset = datasets.__dict__[config.data.dataset](config.data.dataset_root, split='train', device=device)
        test_dataset = datasets.__dict__[config.data.dataset](config.data.dataset_root, split='test', device=device)
    elif config.data.dataset == 'MMFi':
        dataset_config_path = os.path.join('datasets', 'mmfi.yaml')
        with open(dataset_config_path, 'r') as fd:
            dataset_config = yaml.load(fd, Loader=yaml.FullLoader)
        train_dataset = datasets.__dict__[config.data.dataset](config.data.dataset_root, split='train', config=dataset_config, device=device, move_to_center=args.center_mmfi)
        test_dataset = datasets.__dict__[config.data.dataset](config.data.dataset_root, split='test', config=dataset_config, device=device, move_to_center=args.center_mmfi)
    elif config.data.dataset == 'mmPoseNLP':
        train_data_path = os.path.join('datasets', 'merged_data', 'train_data.npy')
        test_data_path = os.path.join('datasets', 'merged_data', 'test_data.npy')
        train_dataset = datasets.__dict__[config.data.dataset](train_data_path, noise_level=[0.05,0.1,0.2])
        test_dataset = datasets.__dict__[config.data.dataset](test_data_path, noise_level=0.25)
    
    train_coreset = CoresetWrapper(train_dataset)
    # select random samples from coreset
    # random seed
    np.random.seed(args.random_seed)
    random_cnt = int(len(train_dataset) * args.random_ratio)
    random_indices = np.random.choice(len(train_dataset), random_cnt, replace=False)
    logger.info(f"random indices: {random_indices}")
    train_coreset.set_indices(random_indices)
    train_loader = DataLoader(train_coreset, batch_size=config.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)
    
    # set up model
    model = models.__dict__[config.model.model](device=device, input_dim=config.data.radar_input_c, n_p=config.data.num_joints).to(device)
    # print(model)
    # set up optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=config.train.learning_rate, weight_decay=config.train.weight_decay, foreach=True)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=config.train.amp)
    criterion = nn.MSELoss()
    
    best_mpjpe = 1000000
    best_p_mpjpe = 1000000
    select_iter = 0
    for epoch in range(config.train.n_epochs):
        
        if epoch > 0 and epoch % config.train.test_freq == 0:
            model.eval()
            test_loss = AverageMeter()
            test_mpjpe = AverageMeter()
            test_p_mpjpe = AverageMeter()
            with torch.no_grad():
                for data in tqdm(test_loader):
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
            
            logger.info('Test Loss:{:.9f}, MPJPE:{:.4f}, P-MPJPE:{:.4f}'.format(
                test_loss.avg, test_mpjpe.avg, test_p_mpjpe.avg))
            writer.add_scalar('test/loss', test_loss.avg, epoch)
            writer.add_scalar('test/mpjpe', test_mpjpe.avg, epoch)
            writer.add_scalar('test/p-mpjpe', test_p_mpjpe.avg, epoch)
            if test_mpjpe.avg < best_mpjpe:
                best_mpjpe = test_mpjpe.avg
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 
                                                            f'epoch{epoch}_mpjpe{test_mpjpe.avg:.4f}_pmpjpe{test_p_mpjpe.avg:.4f}.pth'))
            if test_p_mpjpe.avg < best_p_mpjpe:
                best_p_mpjpe = test_p_mpjpe.avg
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 
                                                            f'epoch{epoch}_mpjpe{test_mpjpe.avg:.4f}_pmpjpe{test_p_mpjpe.avg:.4f}.pth'))
        
        
        model.train()
        logger.info(f"train data batches {len(train_loader)}")
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
                writer.add_scalar('train/loss', loss.item(), epoch*len(train_loader) + pbar.n)
                writer.add_scalar('train/mpjpe', batch_mpjpe, epoch*len(train_loader) + pbar.n)
                writer.add_scalar('train/p-mpjpe', batch_p_mpjpe, epoch*len(train_loader) + pbar.n)
        
        logger.info('Epoch:{}, Loss:{:.9f}, MPJPE:{:.4f}, P-MPJPE:{:.4f}'.format(
            epoch + 1, epoch_loss.avg, epoch_mpjpe.avg, epoch_p_mpjpe.avg))
        writer.add_scalar('train/epoch_loss', epoch_loss.avg, epoch)
        writer.add_scalar('train/epoch_mpjpe', epoch_mpjpe.avg, epoch)
        writer.add_scalar('train/epoch_p-mpjpe', epoch_p_mpjpe.avg, epoch)
        
        
    writer.close()
    logger.info(f"Training done, best MPJPE: {best_mpjpe:.4f}, best P-MPJPE: {best_p_mpjpe:.4f}")          
        
            
    
    
    