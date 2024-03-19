import os
import datetime
import argparse
import math
import glob
import shutil
from tqdm import tqdm

from pprint import pprint
from omegaconf import OmegaConf, listconfig

import torch 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from scripts.model import Denoiser
from scripts.dataset import TableDataset
from scripts.noise_schedule import NoiseSchedule
from scripts.utils import *


# denoising params
denoise_method_info = {
    # direct_map, add_noise
    "direct_map_once": [True, None],
    "direct_map": [True, None],
    "grad_nonoise": [False, False],
    "grad_noise": [False, True]
}
denoise_methods = list(denoise_method_info.keys())

# see more in adjust_parameters():
## loss parameter
L1_coeff = None

## data generation parameter (for when intersection is not allowed)
pen_siz_scale=0.92 # how much intersection is allowed in the input to denoise


def log(line, filepath=None, toPrint=True):
    if filepath is None: filepath=args['logfile']
    if args['log']:
        with open(filepath, 'a+') as f:
            f.write(f"{line}\n")
    if toPrint:
        print(line)

def load_checkpoint(model, model_fp):
    # Referencing global variables defined in main
    global optimizer
    global start_epoch

    checkpoint = torch.load(model_fp, map_location=args['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
    model = model.to(args['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    log("\n=> loaded checkpoint '{}' (epoch {})" .format(model_fp, checkpoint['epoch']))
    return model


def loss_fn(inp,tar,pad=None):
    """ feat1, feat2: [batch_size, 6 or 14, 2] """
    # if args['predict_absolute_pos']: 
    MSE_loss = torch.nn.MSELoss()
    L1_loss  = torch.nn.L1Loss()

    if args['loss_func']=="MSE": 
        if pad is None:
            return MSE_loss(inp, tar)
        else:
            for scene in range(inp.shape[0]):
                nobject = pad[scene] == 0
                if scene == 0:
                    loss = MSE_loss(inp[scene, nobject, :], tar[scene, nobject, :])
                else:
                    loss += MSE_loss(inp[scene, nobject, :], tar[scene, nobject, :])
            return loss / float(inp.shape[0])
    elif args['loss_func'] == 'MSE+L1':
        if pad is None:
            return MSE_loss(inp, tar) + L1_coeff * L1_loss(inp, tar)
        else:
            for scene in range(inp.shape[0]):
                nobject = pad[scene] == 0
                if scene == 0:
                    loss = MSE_loss(inp[scene, nobject, :], tar[scene, nobject, :]) + L1_coeff * L1_loss(inp[scene, nobject, :], tar[scene, nobject, :])
                else:
                    loss += MSE_loss(inp[scene, nobject, :], tar[scene, nobject, :]) + L1_coeff * L1_loss(inp[scene, nobject, :], tar[scene, nobject, :])
            return loss / float(inp.shape[0])
    

def denoise_1scene(i, mask, cond, un_cond, word, args, method, device, model, total_t, schedule, guidance=None, guidance_scale=0.0, cond_scale_guidance=1.0, no_rotate_class=[4, 5, 7], static_class=[], padding_mask=None, mask_padding_mask=None, steer_away=False, denoise_move_less=False):
    """Deals with actual optimization process and visualization of results.
       Optimize from generated noisy scene to clean position based on network prediction through one of the denoise_methods. In scale [-1,1].
       i: [1, numobj, pos+ang+siz+cla], input scene
       Methods:
        a. direct_map: directly map to predicted position, use [network prediction] as next iteration's input position
        b. direct_map_once: do the above for one iteration
        c. grad_nonoise/noise: slowly move in direction of prediction, potentially add noise between each step
            c.1. nonoise: use [network input + (displacement * step size) ] as next iter's input
            c.2. noise:   use [network input + (displacement * step size)  + noise]  as next iter's input
       Criteria for ending: either reached max_iter or displacement size small enough """
    assert i.shape[0] == 1
    direct_map, add_noise = denoise_method_info[method]
    pos_d, ang_d, siz_d = model.pos_dim, model.ang_dim, model.siz_dim

    nobj, _ = TableDataset.parse_cla(i[0, :, pos_d+ang_d+siz_d:].cpu().numpy())
    clean_cla = torch.argmax(i[:, :, pos_d+ang_d+siz_d:], dim=-1)
    clean_cla[torch.sum(i, dim=-1) == 0] = -1
    clean_static = clean_cla == - 1
    for sc in static_class:
        clean_static = torch.bitwise_or(clean_static, clean_cla == sc - 1)
    clean_translate = clean_static
    clean_rotate = clean_static
    for sc in no_rotate_class:
        clean_rotate = torch.bitwise_or(clean_rotate, clean_cla == sc - 1)
    clean_translate = clean_translate.unsqueeze(-1).repeat(1, 1, pos_d)
    clean_rotate = clean_rotate.unsqueeze(-1).repeat(1, 1, ang_d)
    if guidance is not None:
        bbox_y, bbox_x, bbox_h, bbox_w = guidance
        bbox_mean_small = torch.tensor([bbox_y - bbox_h / 2, bbox_x - bbox_w / 2]).unsqueeze(0).unsqueeze(0).to(device)
        bbox_mean_big = torch.tensor([bbox_y + bbox_h / 2, bbox_x + bbox_w / 2]).unsqueeze(0).unsqueeze(0).to(device)

    traj = [] # (iter, 6, 2)
    # local_max_iter = 1 if method == "direct_map_once" else max_iter # global
    local_max_iter = 1 if method == "direct_map_once" else 35
    conse_break_meets = 0 # counts the consecutive number of times it has met the break condition
    iter_i = 0
    pos_disp_size, ang_disp_pi_size = 1, 1
    for iter_i in range(local_max_iter):  
        step_size =  step_size0 / (1 + step_decay*iter_i) 
        # noise_scale = noise_scale0 * noise_decay**(iter_i // noise_drop_freq) # standard deviation
        # ang_noise_scale = noise_scale/noise_scale0 * ((np.pi/4)/10)  # noise_scale0=0.01 correspond to (np.pi/4)/10

        t = torch.LongTensor([local_max_iter - 1 - iter_i])
        sqrt_posterior_variance_t = np.sqrt(schedule.posterior_variance[t].item()) * 0.1
        # betas_t = schedule.betas[t].item()
        # sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].item()
        # step_size = betas_t / sqrt_one_minus_alphas_cumprod_t * 10.0
        noise_scale = args['train_pos_noise_level_stddev'] * sqrt_posterior_variance_t
        ang_noise_scale = args['train_ang_noise_level_stddev'] * sqrt_posterior_variance_t

        t = t.to(device)

        traj.append(torch.squeeze(i.detach().cpu()).tolist()) # add (6,2) or (14,2)

        ori_i = i.clone()

        p = model(i, padding_mask, mask, mask_padding_mask, t, word=word, condition=cond, un_condition=un_cond, cond_scale_guidance=cond_scale_guidance)  # padding_mask: [batch_size=1, seq_len=maxnumobj]

        # normalize at inference time at every iteration
        if ang_d>0: p[:,:,pos_d:pos_d+ang_d] = torch_normalize(p[:,:,pos_d:pos_d+ang_d]) # i is already normalized

        pos_disp = p[:,:,0:pos_d]-i[:,:,0:pos_d] if args['predict_absolute_pos'] else p[:,:,0:pos_d]
        pos_disp_size = np.linalg.norm(pos_disp.detach().cpu().numpy())

        ang_disp_pi, ang_disp_pi_size = None, 1
        if ang_d>0: 
            ang_disp_pi = torch_angle_between(i[:,:,pos_d:pos_d+ang_d], p[:,:,pos_d:pos_d+ang_d]) if args['predict_absolute_ang'] else torch_angle(p[:,:,pos_d:pos_d+ang_d])
                # ang in [-pi, pi] from input to pred; [batch_size, numobj, 1]
            ang_disp_pi_size = torch.mean(torch.abs(ang_disp_pi)).item()
        
        if ((pos_disp_size < pos_disp_break) and (ang_disp_pi_size < ang_disp_pi_break)): 
            conse_break_meets += 1
            if conse_break_meets >= conse_break_meets_max: break  # break on the max-th time
        else:
            conse_break_meets = 0

        # apply the change/prepare input for next iteration (type=i[:,:,pos_d+ang_d:] unchanged)
        if direct_map: step_size=1 
        if steer_away: pos_disp = get_obstacle_avoiding_displacement_bbox(i, pos_disp, step_size, pos_d, ang_d)
        if denoise_move_less:
            pos_disp -= 0.1 * (pos_disp + i[:,:,0:pos_d] - torch.tensor(traj[0]).to(device).unsqueeze(0)[:, :,0:pos_d])
        if guidance is not None:
            guidance_disp = torch.relu(i[:,:,0:pos_d] + (i[:, :, pos_d+ang_d:pos_d+ang_d+siz_d] + 1) / 2 - bbox_mean_big) - torch.relu(bbox_mean_small - i[:,:,0:pos_d] + (i[:, :, pos_d+ang_d:pos_d+ang_d+siz_d] + 1) / 2)
            pos_disp -= guidance_scale * guidance_disp
        i[:,:,0:pos_d] += pos_disp*step_size 

        if ang_d>0: i[:,:,pos_d:pos_d+ang_d] = torch_rotate_wrapper(i[:,:,pos_d:pos_d+ang_d], ang_disp_pi*step_size) # length preserved (stay normalized)
        
        if add_noise==True and 35 > 0: 
            # NOTE: zero-mean gaussian distributions for noise (std-dev designated by noise scale)
            i[:,:,0:pos_d]+=torch.tensor(np.random.normal(size=(i[:,:,0:pos_d]).shape, loc=0.0, scale=noise_scale)).to(device)
            if ang_d>0: 
                rads = torch.tensor(np.random.normal(size=((i[:,:,pos_d:pos_d+ang_d]).shape[0], (i[:,:,pos_d:pos_d+ang_d]).shape[1], 1), loc=0.0, scale=ang_noise_scale)).to(args['device'])
                i[:,:,pos_d:pos_d+ang_d]= torch_rotate_wrapper(i[:,:,pos_d:pos_d+ang_d], rads).to(args['device'])
        i[:,:,0:pos_d] = torch.where(clean_translate, ori_i[:,:,0:pos_d], i[:,:,0:pos_d])
        i[:,:,pos_d:pos_d+ang_d] = torch.where(clean_rotate, ori_i[:,:,pos_d:pos_d+ang_d], i[:,:,pos_d:pos_d+ang_d])

    traj.append(torch.squeeze(i.detach().cpu()).tolist()) # add (6/14,2/4/6)
    traj_to_return = torch.tensor(traj) # [iter, nobj, pos+ang+sha]
    traj = np.array(traj, dtype=float)
    break_idx = iter_i
    perobj_distmoved = np.mean(np.linalg.norm(traj[-1,:nobj,:pos_d]-traj[0, :nobj, :pos_d], axis=1))  # [nobj, pos_d->None] -> scalar: on avg, how much each obj moved

    return traj_to_return, break_idx, pos_disp_size, ang_disp_pi_size, perobj_distmoved


def denoise_1batch(i, mask, cond, un_cond, word, args, method, device, model, total_t, schedule, guidance=None, guidance_scale=0.0, cond_scale_guidance=1.0, no_rotate_class=[4, 5, 7], static_class=[], padding_mask=None, mask_padding_mask=None, steer_away=False, denoise_move_less=False):
    """Deals with actual optimization process and visualization of results.
       Optimize from generated noisy scene to clean position based on network prediction through one of the denoise_methods. In scale [-1,1].
       i: [n, numobj, pos+ang+siz+cla], input scene
       Methods:
        a. direct_map: directly map to predicted position, use [network prediction] as next iteration's input position
        b. direct_map_once: do the above for one iteration
        c. grad_nonoise/noise: slowly move in direction of prediction, potentially add noise between each step
            c.1. nonoise: use [network input + (displacement * step size) ] as next iter's input
            c.2. noise:   use [network input + (displacement * step size)  + noise]  as next iter's input
       Criteria for ending: either reached max_iter or displacement size small enough """
    direct_map, add_noise = denoise_method_info[method]
    pos_d, ang_d, siz_d = model.pos_dim, model.ang_dim, model.siz_dim

    clean_nobj = torch.sum(torch.sum(i, dim=-1) != 0, dim=1)
    clean_cla = torch.argmax(i[:, :, pos_d+ang_d+siz_d:], dim=-1)
    clean_cla[torch.sum(i, dim=-1) == 0] = -1
    clean_static = clean_cla == - 1
    for sc in static_class:
        clean_static = torch.bitwise_or(clean_static, clean_cla == sc - 1)
    clean_translate = clean_static
    clean_rotate = clean_static
    for sc in no_rotate_class:
        clean_rotate = torch.bitwise_or(clean_rotate, clean_cla == sc - 1)
    clean_translate = clean_translate.unsqueeze(-1).repeat(1, 1, pos_d)
    clean_rotate = clean_rotate.unsqueeze(-1).repeat(1, 1, ang_d)
    if guidance is not None:
        bbox_y, bbox_x, bbox_h, bbox_w = guidance
        bbox_mean_small = torch.tensor([bbox_y - bbox_h / 2, bbox_x - bbox_w / 2]).unsqueeze(0).unsqueeze(0).to(device)
        bbox_mean_big = torch.tensor([bbox_y + bbox_h / 2, bbox_x + bbox_w / 2]).unsqueeze(0).unsqueeze(0).to(device)

    # local_max_iter = 1 if method == "direct_map_once" else max_iter # global
    local_max_iter = 1 if method == "direct_map_once" else 35
    iter_i = 0
    initial = i.detach().cpu()
    for iter_i in range(local_max_iter):  
        step_size =  step_size0 / (1 + step_decay*iter_i) 
        # noise_scale = noise_scale0 * noise_decay**(iter_i // noise_drop_freq) # standard deviation
        # ang_noise_scale = noise_scale/noise_scale0 * ((np.pi/4)/10)  # noise_scale0=0.01 correspond to (np.pi/4)/10

        t = torch.LongTensor([local_max_iter - 1 - iter_i])
        sqrt_posterior_variance_t = np.sqrt(schedule.posterior_variance[t].item()) * 0.1
        # betas_t = schedule.betas[t].item()
        # sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].item()
        # step_size = betas_t / sqrt_one_minus_alphas_cumprod_t * 10.0
        noise_scale = args['train_pos_noise_level_stddev'] * sqrt_posterior_variance_t
        ang_noise_scale = args['train_ang_noise_level_stddev'] * sqrt_posterior_variance_t

        t = t.repeat(i.shape[0])
        t = t.to(device)

        ori_i = i.clone()

        p = model(i, padding_mask, mask, mask_padding_mask, t, word=word, condition=cond, un_condition=un_cond, cond_scale_guidance=cond_scale_guidance)  # padding_mask: [batch_size=1, seq_len=maxnumobj]

        # normalize at inference time at every iteration
        if ang_d>0: p[:,:,pos_d:pos_d+ang_d] = torch_normalize(p[:,:,pos_d:pos_d+ang_d]) # i is already normalized

        pos_disp = p[:,:,0:pos_d]-i[:,:,0:pos_d] if args['predict_absolute_pos'] else p[:,:,0:pos_d]

        ang_disp_pi, ang_disp_pi_size = None, 1
        if ang_d>0: 
            ang_disp_pi = torch_angle_between(i[:,:,pos_d:pos_d+ang_d], p[:,:,pos_d:pos_d+ang_d]) if args['predict_absolute_ang'] else torch_angle(p[:,:,pos_d:pos_d+ang_d])
                # ang in [-pi, pi] from input to pred; [batch_size, numobj, 1]

        # apply the change/prepare input for next iteration (type=i[:,:,pos_d+ang_d:] unchanged)
        if direct_map: step_size=1 
        if steer_away: pos_disp = get_obstacle_avoiding_displacement_bbox(i, pos_disp, step_size, pos_d, ang_d)
        if denoise_move_less:
            pos_disp -= 0.1 * (pos_disp + i[:,:,0:pos_d] - initial.to(device)[:, :,0:pos_d])
        if guidance is not None:
            guidance_disp = torch.relu(i[:,:,0:pos_d] + (i[:, :, pos_d+ang_d:pos_d+ang_d+siz_d] + 1) / 2 - bbox_mean_big) - torch.relu(bbox_mean_small - i[:,:,0:pos_d] + (i[:, :, pos_d+ang_d:pos_d+ang_d+siz_d] + 1) / 2)
            pos_disp -= guidance_scale * guidance_disp
        i[:,:,0:pos_d] += pos_disp*step_size 

        if ang_d>0: i[:,:,pos_d:pos_d+ang_d] = torch_rotate_wrapper(i[:,:,pos_d:pos_d+ang_d], ang_disp_pi*step_size) # length preserved (stay normalized)
        
        if add_noise==True and 35 > 0: 
            # NOTE: zero-mean gaussian distributions for noise (std-dev designated by noise scale)
            i[:,:,0:pos_d]+=torch.tensor(np.random.normal(size=(i[:,:,0:pos_d]).shape, loc=0.0, scale=noise_scale)).to(device)
            if ang_d>0: 
                rads = torch.tensor(np.random.normal(size=((i[:,:,pos_d:pos_d+ang_d]).shape[0], (i[:,:,pos_d:pos_d+ang_d]).shape[1], 1), loc=0.0, scale=ang_noise_scale)).to(device)
                i[:,:,pos_d:pos_d+ang_d]= torch_rotate_wrapper(i[:,:,pos_d:pos_d+ang_d], rads).to(device)
        i[:,:,0:pos_d] = torch.where(clean_translate, ori_i[:,:,0:pos_d], i[:,:,0:pos_d])
        i[:,:,pos_d:pos_d+ang_d] = torch.where(clean_rotate, ori_i[:,:,pos_d:pos_d+ang_d], i[:,:,pos_d:pos_d+ang_d])

    final = i.detach().cpu()
    traj_initial = np.array(initial, dtype=float)
    traj_final = np.array(final, dtype=float)
    perobj_distmoved = np.sum(np.linalg.norm(traj_final[:,:,:pos_d]-traj_initial[:, :, :pos_d], axis=-1), axis=1) / clean_nobj.cpu().numpy()

    return final, perobj_distmoved


def train(epoch, half_batch_numscene, schedule, args, data_type=["YCB_kitchen"], device="cpu"):
    """Variables defined in main"""
    log(f"\n\n------------BEGINNING TRAINING [Time: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}] ------------")

    if args["ema"]:
        ema = EMA(model, decay=0.999)
    
    train_losses, val_losses, val_losses_epoch, min_val_loss = [], [], [], math.inf
    for e in range(start_epoch, start_epoch+epoch): # epoch = batch
        model.train()

        padding_mask = None # variable-length data, kitchen specific
        t = torch.randint(0, schedule.timesteps, (half_batch_numscene * 2,), dtype=torch.long).to(device)
        input, mask, cond, word, labels, padding_mask, mask_padding_mask, _ = dataset.gen_kitchen(half_batch_numscene * 2, t, schedule, data_augment=args["data_augment"], abs_pos=args["predict_absolute_pos"], abs_ang=args["predict_absolute_ang"], same_class=args["same_class"], weigh_by_class=args["train_weigh_by_class"], within_floorplan=args["train_within_floorplan"], no_penetration=args["train_no_penetration"], no_rotate_class=args['no_rotate_class'], static_class=args['static_class'], floorplan_class=args['floorplan_class'], use_emd=args['use_emd'], use_move_less=args['use_move_less'], noise_level_stddev=args['train_pos_noise_level_stddev'], angle_noise_level_stddev=args['train_ang_noise_level_stddev'])
        word = torch.tensor(word).to(device)
        padding_mask = torch.tensor(padding_mask).to(device) # boolean 
        mask_padding_mask = torch.tensor(mask_padding_mask).to(device) # boolean 

        input, mask, labels = torch.tensor(input).float().to(device), torch.tensor(mask).float().to(device), torch.tensor(labels).float().to(device)

        if args["use_text"]:
            if args["text_form"] == "word":
                pred = model(input, padding_mask, mask, mask_padding_mask, t, word=word)  # padding_mask: [batch_size, seq_len=maxnumobj]
            elif args["text_form"] == "sentence":
                cond = model.encode_cond(cond, device)
                un_cond = model.encode_cond(input.shape[0] * [""], device)
                pred = model(input, padding_mask, mask, mask_padding_mask, t, condition=cond, un_condition=un_cond)  # padding_mask: [batch_size, seq_len=maxnumobj]
        else:
            pred = model(input, padding_mask, mask, mask_padding_mask, t)  # padding_mask: [batch_size, seq_len=maxnumobj]
        loss = loss_fn(pred, labels, pad=padding_mask)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args["ema"]:
            ema.update()

        # evaluation + save best-so-far checkpoint
        if (e+1)%250 == 0 or e==start_epoch+epoch-1:
            with torch.no_grad():
                if args["ema"]:
                    ema.apply_shadow()
                model.eval()
                val_padding_mask = None # variable-length data, kitchen specific
                val_t = torch.randint(0, schedule.timesteps, (half_batch_numscene * 2,), dtype=torch.long).to(device)
                val_input, val_mask, val_cond, val_word, val_labels, val_padding_mask, val_mask_padding_mask, val_scenepaths = dataset.gen_kitchen(half_batch_numscene * 2, val_t, schedule, data_partition="test", abs_pos=args["predict_absolute_pos"], abs_ang=args["predict_absolute_ang"], same_class=args["same_class"], weigh_by_class=args["denoise_weigh_by_class"], within_floorplan=args["denoise_within_floorplan"], no_penetration=args["denoise_no_penetration"], no_rotate_class=args['no_rotate_class'], static_class=args['static_class'], floorplan_class=args['floorplan_class'], use_emd=args['use_emd'], use_move_less=args['use_move_less'], noise_level_stddev=args['train_pos_noise_level_stddev'], angle_noise_level_stddev=args['train_ang_noise_level_stddev'])
                val_word = torch.tensor(val_word).to(device)
                val_padding_mask = torch.tensor(val_padding_mask).to(device) # boolean
                val_mask_padding_mask = torch.tensor(val_mask_padding_mask).to(device) # boolean

                val_input, val_mask, val_labels = torch.tensor(val_input).float().to(device), torch.tensor(val_mask).float().to(device), torch.tensor(val_labels).float().to(device)
                
                if args["use_text"]:
                    if args["text_form"] == "word":
                        val_pred = model(val_input, val_padding_mask, val_mask, val_mask_padding_mask, val_t, word=val_word)  # val_padding_mask: [batch_size, seq_len=maxnumobj]
                    elif args["text_form"] == "sentence":
                        val_cond = model.encode_cond(val_cond, device)
                        val_un_cond = model.encode_cond(val_input.shape[0] * [""], device)
                        val_pred = model(val_input, val_padding_mask, val_mask, val_mask_padding_mask, val_t, condition=val_cond, un_condition=val_un_cond)  # val_padding_mask: [batch_size, seq_len=maxnumobj]
                else:
                    val_pred = model(val_input, val_padding_mask, val_mask, val_mask_padding_mask, val_t)  # val_padding_mask: [batch_size, seq_len=maxnumobj]
                val_loss = loss_fn(val_pred, val_labels, pad=val_padding_mask)
                val_losses.append(val_loss.item())
                val_losses_epoch.append(e)
                log(f"Epoch {e+1}: train loss = {loss.item()}, val loss = {val_loss.item()}")

                if val_loss.item() < min_val_loss:
                    min_val_loss = val_loss.item()
                    for old_best_checkpoint in glob.glob(os.path.join(args["logsavedir"], "best*")):  os.remove(old_best_checkpoint) 
                    state = { 'epoch': e + 1, 
                              'model_state_dict': model.state_dict(), 
                              'optimizer_state_dict': optimizer.state_dict(),
                              'loss': loss }
                    minvalloss_model_fp = os.path.join(args["logsavedir"], (f"best_{e+1}iter_valloss" + "{0:.6f}.pt".format(val_loss.item())) )
                    torch.save(state, minvalloss_model_fp)
                    log(f"          Current best: saved to {minvalloss_model_fp}")

                if args["ema"]:
                    ema.restore()
            
        # plot loss
        if (e>0 and (e+1)%1000 == 0) or e==start_epoch+epoch-1:
            fig = plt.figure(figsize=(10, 6))
            train, = plt.plot(list(range(len(train_losses))), train_losses, label = "Training Loss")
            val, = plt.plot( val_losses_epoch, val_losses, label = "Validation Loss")
            plt.figlegend(handles=[train, val], loc='upper right')
            plt.gca().set(xlabel='Training Epoch', ylabel=f'Loss', title=f"Training and Validation Loss vs. Training Epoch:\n{args['logsavedir']}")
            # if e>1000: plt.gca().set(ylim=[0,0.1])
            if e>1000: plt.gca().set(ylim=[0,max(train_losses)])
            plt.savefig(os.path.join(args["logsavedir"], 'TrainValLoss_Epoch.jpg'), dpi=300)
            plt.close(fig)
    
        # save checkpoint
        if  (e>0 and (e+1)%5000 == 0) or e==start_epoch+epoch-1:
            if args["ema"]:
                ema.apply_shadow()
            state = { 'epoch': e + 1, 
                      'model_state_dict': model.state_dict(), 
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': loss }
            torch.save(state, os.path.join(args["logsavedir"], f"{e+1}iter.pt"))

            with torch.no_grad():
                model.eval()
                val_padding_mask = None # variable-length data, kitchen specific
                val_t = torch.randint(0, schedule.timesteps, (1,), dtype=torch.long).to(device)
                val_input, val_mask, val_cond, val_word, _, val_padding_mask, val_mask_padding_mask, val_scenepaths = dataset.gen_kitchen(1, val_t, schedule, data_partition="test", abs_pos=args["predict_absolute_pos"], abs_ang=args["predict_absolute_ang"], same_class=args["same_class"], weigh_by_class=args["denoise_weigh_by_class"], within_floorplan=args["denoise_within_floorplan"], no_penetration=args["denoise_no_penetration"], no_rotate_class=args['no_rotate_class'], static_class=args['static_class'], floorplan_class=args['floorplan_class'], use_emd=args['use_emd'], use_move_less=args['use_move_less'], noise_level_stddev=args['train_pos_noise_level_stddev'], angle_noise_level_stddev=args['train_ang_noise_level_stddev'])
                val_word = torch.tensor(val_word).to(device)
                val_input, val_padding_mask = torch.tensor(val_input).float().to(device), torch.tensor(val_padding_mask).to(device)
                val_mask, val_mask_padding_mask = torch.tensor(val_mask).float().to(device), torch.tensor(val_mask_padding_mask).to(device)
                
                if args["use_text"]:
                    if args["text_form"] == "word":
                        traj_to_return, _, _, _, _ = denoise_1scene(val_input, val_mask, None, None, val_word, args, "grad_noise", device, model, val_t, schedule, cond_scale_guidance=args["denoise_cond_scale_guidance"], no_rotate_class=args['no_rotate_class'], static_class=args['static_class'], padding_mask=val_padding_mask, mask_padding_mask=val_mask_padding_mask, steer_away=args["denoise_steer_away"], denoise_move_less=args["denoise_move_less"])
                    elif args["text_form"] == "sentence":
                        val_cond = model.encode_cond(val_cond, device)
                        val_un_cond = model.encode_cond(val_input.shape[0] * [""], device)
                        traj_to_return, _, _, _, _ = denoise_1scene(val_input, val_mask, val_cond, val_un_cond, None, args, "grad_noise", device, model, val_t, schedule, cond_scale_guidance=args["denoise_cond_scale_guidance"], no_rotate_class=args['no_rotate_class'], static_class=args['static_class'], padding_mask=val_padding_mask, mask_padding_mask=val_mask_padding_mask, steer_away=args["denoise_steer_away"], denoise_move_less=args["denoise_move_less"])
                else:
                    traj_to_return, _, _, _, _ = denoise_1scene(val_input, val_mask, None, None, None, args, "grad_noise", device, model, val_t, schedule, cond_scale_guidance=args["denoise_cond_scale_guidance"], no_rotate_class=args['no_rotate_class'], static_class=args['static_class'], padding_mask=val_padding_mask, mask_padding_mask=val_mask_padding_mask, steer_away=args["denoise_steer_away"], denoise_move_less=args["denoise_move_less"])

            target_bbox = [(bbox_y, bbox_x, bbox_h + 1, bbox_w + 1, bbox_type.int().item()) for bbox_y, bbox_x, _, _, bbox_h, bbox_w, bbox_type in val_mask[0].cpu()]
            fig = plt.figure()
            def update(tr):
                plt.cla()
                visual_noise(val_scenepaths[0], traj_to_return[tr], val_padding_mask[0], target_bbox=target_bbox)
            ani = animation.FuncAnimation(fig, update, frames=traj_to_return.shape[0])
            ani.save(os.path.join(args["logsavedir"], f'dynamic_Epoch{e+1}.gif'))
            plt.close(fig)

            if args["ema"]:
                ema.restore()

        if (e>0 and (e+1)%10000 == 0) or e==start_epoch+epoch-1:
            if args["ema"]:
                ema.apply_shadow()
            noise_levels = [args['train_pos_noise_level_stddev']]
            angle_noise_levels = [args['train_ang_noise_level_stddev']]
            denoise_meta(args, [model], schedule, numscene=10, numtrials=5, use_methods=[True, False, True, True], data_type=args['data_type'], device=args['device'],
                            noise_levels=noise_levels, angle_noise_levels=angle_noise_levels)
            if args["ema"]:
                ema.restore()
    
    log(f"------------FINISHED WITH TRAINING [Time: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}]------------\n")
    

def denoise_meta(args, models, schedule, numscene=10, numtrials=5, use_methods=[True, True, True, True], data_type=["YCB_kitchen"], device="cpu", 
                 noise_levels=[0.25], angle_noise_levels=[np.pi/4]):
    """ Denoise for each noise level, each model, each denoise method.

        models: list of model to run the denoising process with.
        numscene: number of scene for each noise level
        numtrials: number of trial for each scene
        use_methods: whether to use each of the globally defined denoise_methods ("direct_map_once", "direct_map", "grad_nonoise", "grad_noise")
        use_sameset: use same set of scenes for all noise levels, all models, all denosing methods
        save_results: write denoising data to npz files, including scenepaths, initial scene, trajectories, statistics, etc
    """
    log(f"\n### DENOISE")

    data_partition = "test"
    name_group = ["squre", "circle", "horizon", "vertical", "container", "0", "uniform", "symmetry", "3", "all"]
    total_iters = len(dataset.data_test) // numscene
    if len(dataset.data_test) % numscene > 0:
        total_iters += 1

    all_emd2gts, all_ang2gts, all_perobj_distmoveds, all_iou25s, all_iou50s = [], [], [], [], [] # [n_nl, n_model*n_method=4]
    for noise_i in range(len(noise_levels)):
        noise_emd2gts, noise_ang2gts, noise_perobj_distmoveds = np.zeros((numtrials, len(denoise_methods), 10)), np.zeros((numtrials, len(denoise_methods), 10)), np.zeros((numtrials, len(denoise_methods), 10))
        noise_iou25s, noise_iou50s = np.zeros((numtrials, len(denoise_methods), 10)), np.zeros((numtrials, len(denoise_methods), 10))
        noise_scene_group, noise_obj_group = np.zeros((1, 1, 10)), np.zeros((1, 1, 10))
        nl, anl = noise_levels[noise_i], angle_noise_levels[noise_i] # from global variables
        log(f"\n\n------ Denoise: noise level={nl}, angle noise level={round(anl/np.pi*180)} ")

        for iters in tqdm(range(total_iters)):
            if iters == total_iters - 1:
                nums = len(dataset.data_test) % numscene
                if nums == 0:
                    nums = numscene
            else:
                nums = numscene
            random_idx = np.arange(iters * numscene, iters * numscene + nums)
            for trials in range(numtrials):
                t = torch.randint(0, schedule.timesteps, (1,), dtype=torch.long).repeat(nums)
                de_padding_mask, de_scenepaths = None, None # variable-length data, 3dfront specific
                de_input, de_mask, de_cond, de_word, de_label, de_padding_mask, de_mask_padding_mask, de_scenepaths = dataset.gen_kitchen(nums, t, schedule, random_idx=random_idx, data_partition=data_partition, use_emd=args['use_emd'], use_move_less=args['use_move_less'], # random_idx: same set of scenes everytime
                                                                                                                            abs_pos=args["predict_absolute_pos"], abs_ang=args["predict_absolute_ang"],
                                                                                                                            noise_level_stddev=nl, angle_noise_level_stddev=anl,
                                                                                                                            no_rotate_class=args["no_rotate_class"], static_class=args["static_class"], same_class=[], floorplan_class=args["floorplan_class"], 
                                                                                                                            weigh_by_class = args['denoise_weigh_by_class'], within_floorplan = args['denoise_within_floorplan'], no_penetration = args['denoise_no_penetration'], 
                                                                                                                            pen_siz_scale=pen_siz_scale)

                de_padding_mask = torch.tensor(de_padding_mask).to(device) # boolean
                de_mask_padding_mask = torch.tensor(de_mask_padding_mask).to(device) # boolean
                de_input, de_label = torch.tensor(de_input).float().to(device), torch.tensor(de_label).float().to(device) # [nums*2,nobj,d] - first half is clean
                de_mask, de_word = torch.tensor(de_mask).float().to(device), torch.tensor(de_word).to(device)

                # denoising
                for model_i in range(len(models)):
                    global model
                    model = models[model_i] # gloabl variable used in denoise_1scene
                    
                    model.eval()
                    with torch.no_grad():
                        for method_i in range(len(denoise_methods)): # 4 [direct_map_once, direct_map, grad_nonoise, grad_noise]
                            if not use_methods[method_i]: continue
                            trajs_g, perobj_distmoveds_g, gt_g, ini_g = [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []]
                            trajs_l, perobj_distmoveds_l, gt_l, ini_l = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
                            final_state, perobj_distmoved = denoise_1batch(de_input.detach().clone(), de_mask, None, None, de_word, args, denoise_methods[method_i], de_input.device, model, t, schedule, cond_scale_guidance=1.0, guidance=None, guidance_scale=0.0, no_rotate_class=args["no_rotate_class"], static_class=args["static_class"], padding_mask=de_padding_mask, mask_padding_mask=de_mask_padding_mask, steer_away=args["denoise_steer_away"], denoise_move_less=args["denoise_move_less"])
                            for scene_i in range(de_input.shape[0]):
                                trajs_g[de_word[scene_i, 0]].append(final_state[scene_i].cpu().numpy()) # append [niter, nobj, pos+ang+siz+cla] # traj[0] is initial, traj[-1] is final
                                perobj_distmoveds_g[de_word[scene_i, 0]].append(perobj_distmoved[scene_i]) # append a scalar -> [nscene,]
                                gt_g[de_word[scene_i, 0]].append(de_label[scene_i].cpu().numpy())
                                ini_g[de_word[scene_i, 0]].append(de_input[scene_i].cpu().numpy())
                                trajs_l[de_word[scene_i, 1]].append(final_state[scene_i].cpu().numpy()) # append [niter, nobj, pos+ang+siz+cla] # traj[0] is initial, traj[-1] is final
                                perobj_distmoveds_l[de_word[scene_i, 1]].append(perobj_distmoved[scene_i]) # append a scalar -> [nscene,]
                                gt_l[de_word[scene_i, 1]].append(de_label[scene_i].cpu().numpy())
                                ini_l[de_word[scene_i, 1]].append(de_input[scene_i].cpu().numpy())

                            for glo in range(len(trajs_g)):
                                if len(trajs_g[glo]) > 0:
                                    trajs, perobj_distmoveds, gds, inis = np.array(trajs_g[glo]), np.array(perobj_distmoveds_g[glo]), np.array(gt_g[glo]), np.array(ini_g[glo]) 
                                    emd2gt, ang2gt, iou_25, iou_50, numobj = dist_2_gt(trajs, gds, dataset, use_emd=True) # trajs: normalized 
                                    noise_emd2gts[trials][method_i][glo] += emd2gt * trajs.shape[0]
                                    noise_ang2gts[trials][method_i][glo] += ang2gt * trajs.shape[0]
                                    # noise_perobj_distmoveds[trials][method_i][glo] += np.sum(perobj_distmoveds)
                                    noise_scene_group[0, 0, glo] += trajs.shape[0]
                                    noise_iou25s[trials][method_i][glo] += iou_25
                                    noise_iou50s[trials][method_i][glo] += iou_50
                                    noise_obj_group[0, 0, glo] += numobj
                                    noise_emd2gts[trials][method_i][-1] += emd2gt * trajs.shape[0]
                                    noise_ang2gts[trials][method_i][-1] += ang2gt * trajs.shape[0]
                                    # noise_perobj_distmoveds[trials][method_i][-1] += np.sum(perobj_distmoveds)
                                    noise_scene_group[0, 0, -1] += trajs.shape[0]
                                    noise_iou25s[trials][method_i][-1] += iou_25
                                    noise_iou50s[trials][method_i][-1] += iou_50
                                    noise_obj_group[0, 0, -1] += numobj

                                    emd2gt, _, _, _, _ = dist_2_gt(trajs, inis, dataset, use_emd=True)
                                    noise_perobj_distmoveds[trials][method_i][glo] += emd2gt * trajs.shape[0]
                                    noise_perobj_distmoveds[trials][method_i][-1] += emd2gt * trajs.shape[0]
                            
                            for loc in range(len(trajs_l)):
                                if len(trajs_l[loc]) > 0:
                                    trajs, perobj_distmoveds, gds, inis = np.array(trajs_l[loc]), np.array(perobj_distmoveds_l[loc]), np.array(gt_l[loc]), np.array(ini_l[loc]) 
                                    emd2gt, ang2gt, iou_25, iou_50, numobj = dist_2_gt(trajs, gds, dataset, use_emd=True) # trajs: normalized 
                                    noise_emd2gts[trials][method_i][loc+5] += emd2gt * trajs.shape[0]
                                    noise_ang2gts[trials][method_i][loc+5] += ang2gt * trajs.shape[0]
                                    # noise_perobj_distmoveds[trials][method_i][loc+5] += np.sum(perobj_distmoveds)
                                    noise_scene_group[0, 0, loc+5] += trajs.shape[0]
                                    noise_iou25s[trials][method_i][loc+5] += iou_25
                                    noise_iou50s[trials][method_i][loc+5] += iou_50
                                    noise_obj_group[0, 0, loc+5] += numobj
                                    emd2gt, _, _, _, _ = dist_2_gt(trajs, inis, dataset, use_emd=True)
                                    noise_perobj_distmoveds[trials][method_i][loc+5] += emd2gt * trajs.shape[0]

        noise_scene_group = noise_scene_group / (numtrials * len(models) * sum(use_methods))
        noise_obj_group = noise_obj_group / (numtrials * len(models) * sum(use_methods))
        noise_emd2gts = noise_emd2gts / noise_scene_group.astype(float)
        noise_ang2gts = noise_ang2gts / noise_scene_group.astype(float)
        noise_perobj_distmoveds = noise_perobj_distmoveds / noise_scene_group.astype(float)
        noise_iou25s = noise_iou25s / noise_obj_group.astype(float)
        noise_iou50s = noise_iou50s / noise_obj_group.astype(float)

        # noise_emd2gts = noise_emd2gts / float(len(dataset.data_test))
        # noise_ang2gts = noise_ang2gts / float(len(dataset.data_test))
        # noise_perobj_distmoveds = noise_perobj_distmoveds / float(len(dataset.data_test))
        for method_i in range(len(denoise_methods)): # 4 [direct_map_once, direct_map, grad_nonoise, grad_noise]
            if not use_methods[method_i]: continue
            for glo_loc in range(noise_scene_group.shape[-1]):
                if noise_scene_group[0, 0, glo_loc] == 0: continue
                log(f"{name_group[glo_loc]}: ")
                log(f"emd to ground truth = {np.mean(noise_emd2gts[:, method_i, glo_loc])}+{np.std(noise_emd2gts[:, method_i, glo_loc])}")
                log(f"ang to ground truth = {np.mean(noise_ang2gts[:, method_i, glo_loc])}+{np.std(noise_ang2gts[:, method_i, glo_loc])}")
                log(f"mean per-scene perobj_distmoved = {np.mean(noise_perobj_distmoveds[:, method_i, glo_loc])}+{np.std(noise_perobj_distmoveds[:, method_i, glo_loc])}")
                log(f"IoU 25 = {np.mean(noise_iou25s[:, method_i, glo_loc])}+{np.std(noise_iou25s[:, method_i, glo_loc])}")
                log(f"IoU 50 = {np.mean(noise_iou50s[:, method_i, glo_loc])}+{np.std(noise_iou50s[:, method_i, glo_loc])}")
            log("\n")
        log(f"number of each scene: {noise_scene_group}")
        all_emd2gts.append(np.mean(noise_emd2gts[:, :, -1], axis=0).tolist())
        all_ang2gts.append(np.mean(noise_ang2gts[:, :, -1], axis=0).tolist())
        all_perobj_distmoveds.append(np.mean(noise_perobj_distmoveds[:, :, -1], axis=0).tolist())
        all_iou25s.append(np.mean(noise_iou25s[:, :, -1], axis=0).tolist())
        all_iou50s.append(np.mean(noise_iou50s[:, :, -1], axis=0).tolist())

    log(f"\nall_emd2gts={all_emd2gts}") # to print with the commas
    log(f"all_ang2gts={all_ang2gts}")
    log(f"all_perobj_distmoveds={all_perobj_distmoveds}")
    log(f"all_iou25s={all_iou25s}") # to print with the commas
    log(f"all_iou50s={all_iou50s}")
    log(f"noise_levels={noise_levels}\nangle_noise_levels={angle_noise_levels}")
    log(f"denoise_methods={denoise_methods}\nuse_methods={use_methods}")


def adjust_parameters(train=True):
    if train:
        args['learning_rate'] = 1e-4 * args['train_batch_size']/128
        
        log(f"\nNumber of GPUs (torch.cuda.device_count()) = {gpucount}, torch.cuda.is_available()={torch.cuda.is_available()}\n") 
        if gpucount >= 1: 
            args['train_batch_size'] *= gpucount # break down first dim (batch_size)
            args['learning_rate'] *= gpucount

        pprint(args)
        pprint(args, stream=open(args['logfile'], 'w'))
        log("\n")
        
        if args['loss_func'] == "MSE+L1": 
            global L1_coeff
            # default:
            L1_coeff = 0.07 # sqrt(stabalized MSE=0.005) for kitchen ->  ~ 0.005 + sqrt(0.005)^2 (equal weights)

            log(f"loss parameters:\n  L1_coeff={L1_coeff}")

    global max_iter, step_size0, step_decay, noise_scale0, noise_decay, noise_drop_freq, pos_disp_break, ang_disp_pi_break, conse_break_meets_max
        # step_size =  step_size0 / (1 + step_decay*iter_i) 
        # noise_scale = noise_scale0 * noise_decay**(iter_i // noise_drop_freq)
    # default
    pos_disp_break, ang_disp_pi_break, conse_break_meets_max = 0.01, 0.005, 3 # 0.3 degree for angle, break on the 3rd time
    max_iter = 1500
    step_size0, step_decay = 0.1, 0.005 # roughly step_size0 / step_decay*iteration: 0.067 at 100 iter
    noise_scale0, noise_decay, noise_drop_freq = 0.01, 0.9, 10  # multiply by noise_decay=0.9 every noise_drop_freq=2 iterations: 0.005 at iter 50   

    if train:
        log(f"\ndenoising parameters:")
        log(f"  max_iter={max_iter}, step_size0={step_size0}, step_decay={step_decay}; noise_scale0={noise_scale0}, noise_decay={noise_decay}, noise_drop_freq={noise_drop_freq}")
        log(f"  pos_disp_break={pos_disp_break}, ang_disp_pi_break={ang_disp_pi_break}, conse_break_meets_max={conse_break_meets_max}; pen_siz_scale={pen_siz_scale}")


def logsavedir_from_args(args):
    """  Construct name of directory (logsavedir) to save results to. """
    folderprefix = "logs"
    if args['train']:
        if args['predict_absolute_pos']==1 and args['predict_absolute_ang']==1: predtype='abs'
        elif args['predict_absolute_pos']==0 and args['predict_absolute_ang']==0: predtype='rel'
        elif args['predict_absolute_pos']==0 and args['predict_absolute_ang']==1: predtype='relposabsang'
        elif args['predict_absolute_pos']==1 and args['predict_absolute_ang']==0: predtype='absposrelang'
        branch = '1branch' if args['use_two_branch']==0 else '2branch'

        cons = f"_{'T' if args['train_weigh_by_class'] else 'F'}{'T' if args['train_within_floorplan'] else 'F'}{'T' if args['train_no_penetration'] else 'F'}{'T' if args['use_position'] else 'F'}{'T' if args['use_time'] else 'F'}{'T' if args['use_text'] else 'F'}_"
        return os.path.join(f"{folderprefix}", f"{'_'.join(args['data_type'])}", f"{args['timestamp']}_{cons}") # _{args['loss_func']}
    
    elif args['train']==0:
        numiterk = os.path.split(args['model_path'])[-1].split(".")[0]
        cons = f"{'T' if args['denoise_weigh_by_class'] else 'F'}{'T' if args['denoise_within_floorplan'] else 'F'}{'T' if args['denoise_no_penetration'] else 'F'}{'T' if args['use_position'] else 'F'}{'T' if args['use_time'] else 'F'}{'T' if args['use_text'] else 'F'}"
        return os.path.join(args['model_path'].split("/")[0], args['model_path'].split("/")[1], f"Infer_{args['timestamp']}_{cons}_{numiterk}")
    

def initialize_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--log", type = int, default=1, help="If 1, write to log file in addition to printing to stdout.")

    parser.add_argument("--train", type = int, default=1, help="Binary flag for train versus inference.")
    parser.add_argument("--ema", type = int, default=0, help="Whether adopt ema trick.")
    parser.add_argument("--model_type", type = str, default="Transformer", # only matters for train
                        choices = ["Transformer"] )
    parser.add_argument("--use_two_branch", type = int, default=0, help="Transformer-specific; if 0, uses 1 branch for position and angle; else use 2 separate branches.")
    parser.add_argument("--predict_absolute_pos", type = int, default=1, help="Absolute position prediction (versus relative)")
    parser.add_argument("--predict_absolute_ang", type = int, default=1, help="Absolute angle/orientation prediction (versus relative)")
   
    ## for train==1:
    parser.add_argument("--resume_train", type = int, default=0, help="Training specific. If 1, resume training from model_path passed in.")
    parser.add_argument("--train_epoch", type = int, default=500000, help="Training specific. Number of epoches to train for.")
    parser.add_argument("--train_batch_size", type = int, default=128, help="Training specific. Each item in batch is a scene.") 
    parser.add_argument("--loss_func", type = str, default="MSE+L1", choices = ["MSE", "MSE+L1"]) 
    parser.add_argument("--train_pos_noise_level_stddev", type = float, default=0.1,  # 0.25 for table chair experiments
                        help="Training specific. Standard deviation of the Gaussian distribution from which the training data position noise level (stddev of actual noise distribution) is drawn")
    parser.add_argument("--train_ang_noise_level_stddev", type = float, default=np.pi/12, # np.pi/4 for table chair experiments
                        help="Training specific. Standard deviation of the Gaussian distribution from which the training data angle noise level (stddev of actual noise distribution) is drawn")
    parser.add_argument("--use_position", type = int, default=1, help="Condition mode. If 1, use position embedding for objects.")
    parser.add_argument("--use_time", type = int, default=1, help="Condition mode. If 1, use time embedding.")
    parser.add_argument("--use_text", type = int, default=1, help="Condition mode. If 1, use text embedding.")
    parser.add_argument("--timesteps", type = int, default=1000, help="total time steps to denoise")
    parser.add_argument("--time_form", type = str, default="prepend", help="how to introduce time embedding [prepend, concat, add]")
    parser.add_argument("--text_form", type = str, default="word", help="how to introduce text embedding [word, sentence]")
    ### for kitchen
    parser.add_argument("--train_weigh_by_class", type = int, default=0, help="Training specific. kitchen specific. If 1, in training data generation, objects with higher volume will move less.") 
    parser.add_argument("--train_within_floorplan", type = int, default=1, help="Training specific. kitchen specific. If 1, in training data generation, objects must be within boundary of floor plan.") 
    parser.add_argument("--train_no_penetration", type = int, default=0, help="Training specific. kitchen specific. If 1, in training data generation, objects cannot intersect one another (or only minimally).") 

    ## for (train==1 && resume_train==1) or (train==0):
    parser.add_argument("--model_path", type = str, help="If train==1 && resume_train==1, serves as the model path from which to resume training; otherwise, serves as the model to use for inference.")

    ## for train==0
    parser.add_argument("--stratefied_mass_denoising", type = int, default=1, help="Inference specific. If 1, will do inference with stratefied noise levels.")
    ### for kitchen
    parser.add_argument("--denoise_weigh_by_class", type = int, default=0, help="Inference specific. kitchen specific. If 1, in inference data generation, objects with higher volume will move less.") 
    parser.add_argument("--denoise_within_floorplan", type = int, default=1, help="Inference specific. kitchen specific. If 1, in inference data generation, objects must be within boundary of floor plan.") 
    parser.add_argument("--denoise_no_penetration", type = int, default=0, help="Inference specific. kitchen specific. If 1, in inference data generation, objects cannot intersect one another (or only minimally).") 
    parser.add_argument("--denoise_steer_away", type = int, default=0, help="Inference specific. kitchen specific. If 1, during inference, attempts to steer objects away from one another to avoid penetration.") 
    parser.add_argument("--denoise_cond_scale_guidance", type = float, default=1.0, help="Inference specific. kitchen specific. The cond scale of guidance.") 
    parser.add_argument("--denoise_move_less", type = int, default=0, help="Inference specific. Closest configuration.") # whether to adopt closest configuration in the denoising process

    ## data:
    parser.add_argument("--data_type", type = str, nargs='+', default=["YCB_kitchen"]) # choices=["Kinect_image", "YCB_kitchen", "YCB_Inpainted"])
    ### for kitchen
    parser.add_argument("--use_emd", type = int, default=1, help="kitchen specific. Earthmover's distance.") # whether to use earthmover distance assignment or original scene for noisy label
    parser.add_argument("--use_move_less", type = int, default=1, help="kitchen specific. Closest configuration.") # whether to adopt closest configuration in the data generation process
    
    # parser.add_argument("--dataset_path",  type = str, default="data/YCB_kitchen_data", help="where is the dataset.") 
    parser.add_argument("--dataset_split",  type = str, default="folder", help="how to split train and test set.") 
    parser.add_argument("--data_augment", type=int,  default=1, help="times to augment each data")
    # parser.add_argument("--num_class", type=int,  default=8, help="total categories considered.")
    # parser.add_argument("--mask_types", type=int,  default=1, help="total mask categories considered.")
    # parser.add_argument("--floorplan_class", type=int,  default=1, help="category of inpant mask that restricts the boundary.")
    # parser.add_argument("--no_rotate_class", type=int, nargs='+',  default=[4, 5, 7], help="categories not considering rotating when rearrange.")
    parser.add_argument("--static_class", type=int, nargs='+',  default=[], help="categories keeping static when rearrange.")
    parser.add_argument("--same_class", type=int, nargs='+', action='append',  default=[], help="treated as the same class when calculate earth moving distance.")

    args = parser.parse_args() #  'argparse.Namespace'
    return vars(args) #'dict'


if __name__ == "__main__":
    args = initialize_parser() # global variable
    args["timestamp"] = datetime.datetime.now().strftime("%m%d_%H%M%S_%f")[:-3] # trim 3/6 digits of ms, used to be str(int(time.time()))
    args["logsavedir"] = logsavedir_from_args(args)
    args["logfile"] = os.path.join(args['logsavedir'], "log.txt")
    args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['learning_rate'] = 1e-4
    for index, data_type in enumerate(args['data_type']):
        if index == 0:
            total_dataset_config = OmegaConf.load(f"configs/{data_type}.yaml")
            for key in total_dataset_config:
                if isinstance(total_dataset_config[key], str):
                    total_dataset_config[key] = [total_dataset_config[key]]
        else:
            dataset_config = OmegaConf.load(f"configs/{data_type}.yaml")
            for key in total_dataset_config:
                if isinstance(dataset_config[key], listconfig.ListConfig):
                    total_dataset_config[key].extend(dataset_config[key])
                    total_dataset_config[key] = list(set(total_dataset_config[key]))
                elif isinstance(dataset_config[key], str):
                    total_dataset_config[key] += [dataset_config[key]]
                else:
                    total_dataset_config[key] = max(total_dataset_config[key], dataset_config[key])
    args.update(total_dataset_config)

    if not os.path.exists(args['logsavedir']): os.makedirs(args['logsavedir'])
    gpucount = torch.cuda.device_count()
    
    adjust_parameters(train=args["train"])
    
    files = ['scripts/dataset.py', 'scripts/model.py', 'scripts/utils.py', 'scripts/noise_schedule.py', 'train.py'] + [f"configs/{data_type}.yaml" for data_type in args["data_type"]]
    for f in files:
        shutil.copy(f, os.path.join(args['logsavedir'], f.split('/')[-1]))
    files = [f"{dataset_path}/split.json" for dataset_path in args["dataset_path"]]
    for f in files:
        shutil.copy(f, os.path.join(args['logsavedir'], f"{f.split('/')[-2]}_split.json"))

    maxnfpoc, nfpbpn  = None, None # to be overriden
    dataset = TableDataset(path=args["dataset_path"], split=args["dataset_split"], num_class=args["num_class"])
    n_obj, pos_d, ang_d, siz_d, cla_d, invsha_d, maxnfpoc, nfpbpn = dataset.maxnobj, dataset.pos_dim, dataset.ang_dim, dataset.siz_dim, dataset.cla_dim, 0, dataset.maxnfpoc, dataset.nfpbpn # shape = size+class
    log(f"args['data_type']={'_'.join(args['data_type'])}: n_obj={n_obj}, pos_d={pos_d}, ang_d={ang_d}, siz_d={siz_d}, cla_d={cla_d}, invsha_d={invsha_d}\n")

    schedule = NoiseSchedule(timesteps=args["timesteps"])

    input_d = pos_d + ang_d + siz_d + cla_d + invsha_d
    out_d = pos_d + ang_d
    sha_code = True if input_d > pos_d else False
    subtract_f = True

    transformer_config = None
    transformer_config = {"pos_dim": pos_d, "ang_dim": ang_d, "siz_dim": siz_d, "cla_dim": cla_d, 
                            # "ang_initial_d": 128, "siz_initial_unit": None, "cla_initial_unit": [128, 128],
                            "ang_initial_d": 128, "siz_initial_unit": [64, 16], "cla_initial_unit": [128, 128],
                            "all_initial_unit": [512, 512], "final_lin_unit": [256, out_d], 
                            "use_two_branch": args['use_two_branch'], "pe_numfreq": 32, "pe_end": 128, 
                            "use_position": args['use_position'], "use_time": args['use_time'], "use_text": args['use_text'], 
                            "type_dim": 8, "max_position": n_obj, "position_dim": 8, "mask_types": args["mask_types"], 
                            "time_form": args['time_form'], "time_emb_dim": 80, 
                            "text_form": args['text_form'], "vocab_size": [4, 4], "version": "ViT-B/32", "cond_dim": 512}
    log(f"args['model_type']={args['model_type']}: {transformer_config}\n")
    model = Denoiser(**transformer_config)
        
    if gpucount > 1: model = torch.nn.DataParallel(model)

    model = model.to(args['device'])
    log(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    
    if args['train']:
        start_epoch = 0
        if args['resume_train']: model=load_checkpoint(model, args['model_path']) # updates model, optimizer, start_epoch
        
        train(args['train_epoch'], int(args['train_batch_size']/2), schedule=schedule, args=args, data_type=args['data_type'], device=args['device'])

    else:
        pprint(args)
        pprint(args, stream=open(args['logfile'], 'w'))
        log("\n")

        # loading models
        model=load_checkpoint(model, args['model_path'])  # updates model, optimizer, start_epoch
        models = [model]
        # model_epoch = args['train_epoch'] if args['train'] else args['model_path'].split("/")[-1][:-7]
        # model_names = [f"train{model_epoch}"]

        
        # inference/denoising
        ## 1. Mass inference/denoising
        if args['stratefied_mass_denoising']:
            # noise_levels = [0.25]
            # angle_noise_levels = [np.pi/4] # [np.pi/24, np.pi/12, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi]  # 15, 30, 45, 60, 90, 180 
            noise_levels = [1.0]
            angle_noise_levels = [np.pi/3] # [np.pi/24, np.pi/12, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi]  # 15, 30, 45, 60, 90, 180
            denoise_meta(args, models, schedule, numscene=10, numtrials=5, use_methods=[True, False, True, True], data_type=args['data_type'], device=args['device'],
                         noise_levels=noise_levels, angle_noise_levels=angle_noise_levels)

    log(f"\nTime: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}")
    log(f"### Done (results saved to {args['logfile']})\n")