import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.cuda.amp import autocast, GradScaler
from pytorch3dunet.unet3d import model
from load_data import * 
from ResNetmodel import *


def train_model(model, train_loader, val_loader, outputname, optimizer, scheduler, criterion, epochs = 5000):

    # Initialize mixed precision scaler
    scaler = GradScaler()
    val_loss_best = np.inf

    outputfile = open(outputname+'_loss.txt','w')

    for i in range(epochs):
    
        model.train()

        training_loss = 0
        val_loss = 0

        for bn, (inputs, targets) in enumerate(train_loader):
           
            # Move inputs and targets to GPU
            inputs = inputs.to(dtype=torch.float32).cuda()
            targets = targets.to(dtype=torch.float32).cuda()

            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Use mixed precision for forward and backward pass
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()


            # Step optimizer with scaler
            scaler.step(optimizer)
            scaler.update()  # Update scaler after each optimizer step

            training_loss += loss.item()

        
        training_loss /= len(train_loader)
        
        
        # Step the scheduler
        scheduler.step()

        # Evaluation step, avoid gradients
        model.eval()

        with torch.no_grad():
            for val_input, val_output in val_loader:
                # Move val data to GPU only for evaluation
                val_input = val_input.to(dtype=torch.float32).cuda()
                val_output = val_output.to(dtype=torch.float32).cuda()

                # Calculate validating loss
                val_loss += criterion(model(val_input), val_output).item()
        
        val_loss /= len(val_loader)

        # Logging
        print(f"{i+1} {training_loss} {val_loss}", file = outputfile)


        if i%1 == 0:
            print(f'Epoch {i+1} training loss:', training_loss)
            print(f'Epoch {i+1} validating loss:', val_loss)

        # Save model if validating loss improves
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            torch.save(model.state_dict(), os.path.join('./', outputname + '_best.pth'))

        # Periodic checkpointing
        if i % 100 == 0:
            torch.save(model.state_dict(), os.path.join('./', outputname + '_ckp.pth'))

        # Clear cache at the end of the epoch
        torch.cuda.empty_cache()







if __name__ == "__main__":

    bt_size = 32
    epochs = 5
    lr = 0.0001
    scheduler_step_size = 2500
    scheduler_gamma=0.1,
    model_type = 'Resnet'
    outputname = model_type

    #f_maps = 32 #comment out if select Resnet
    n_blocks, ngf = 6, 64 #comment out if select Unet/AttentionUnet

    train_loader, val_loader, test_loader = load_data(bt_size)

    # Select Model (ResNet/U-Net/Attention UNet) to for ion distribution prediction

    if model_type == 'Resnet':
        model = IonResnet(2, 1, ngf = ngf, norm_layer = nn.BatchNorm3d, use_dropout = True, n_blocks = n_blocks).cuda()
    
    elif model_type == 'Unet':
        model = model.UNet3D(in_channels=2, out_channels=1, final_sigmoid=True, f_maps=f_maps, \
                layer_order='gcr', num_groups=8, num_levels=4, is_segmentation=False, \
                conv_padding=1, conv_upscale=2, upsample='default', dropout_prob=0.1, Attention=False).cuda()

    elif model_type == 'AttentionUnet':
        model = model.UNet3D(in_channels = 2, out_channels = 1, final_sigmoid=True, f_maps=f_maps, \
            layer_order='gcr', num_groups=8, num_levels=3, is_segmentation=False, \
            conv_padding=1, conv_upscale=2, upsample='default', dropout_prob=0.1, Attention=True).cuda()

    else:
        print("No such model type available!")

    print(model)

    #if torch.cuda.device_count() > 1:
        #print(f"Using {torch.cuda.device_count()} GPUs!")
        #model = nn.DataParallel(model)  # Wrap the model to use multiple GPUs


    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = StepLR(optimizer, scheduler_step_size, scheduler_gamma)


    train_model(model, train_loader, val_loader, outputname, optimizer, scheduler, criterion, epochs)

