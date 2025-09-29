import torch
import torch.nn.functional as F
from environment.do_action import gumbel_select_action
from utils.torch_metrics import calculate_differences, calculate_entropy
from utils.transform_utils import scale
import numpy as np
import torch.nn as nn
from typing import Tuple




def update_networks(
        actor, critic,
        actor_target, critic_target,
        actor_opt, critic_opt,
        batch_s, batch_a_onehot, batch_r,
        batch_s_n, batch_d,
        batch_mask_c, batch_mask_n,
        gamma              = 0.99,
        tau                = 1e-3,
        target_update_freq = 1,
        global_step_ctr    = None,
        budget             = None
):
    
 
    with torch.no_grad():
        logits_next = actor_target(batch_s_n)
        _, g_n      = gumbel_select_action(logits_next, 1, 1, batch_mask_n, budget)
        q_next         = critic_target(batch_s_n, g_n)
        y              = batch_r + gamma * (1 - batch_d) * q_next

    q_pred       = critic(batch_s, batch_a_onehot)
    critic_loss  = F.mse_loss(q_pred, y)

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    for p in critic.parameters():                      
        p.requires_grad_(False)

    logits_cur    = actor(batch_s)
    _, g_cur      = gumbel_select_action(logits_cur, 1, 1,batch_mask_c, budget)
    q_for_actor   = critic(batch_s, g_cur)
    actor_loss    = -q_for_actor.mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    for p in critic.parameters():                      
        p.requires_grad_(True)

    if global_step_ctr is not None:
        global_step_ctr += 1
        if global_step_ctr % target_update_freq == 0:
            soft_update(actor_target,  actor,  tau)
            soft_update(critic_target, critic, tau)
    return global_step_ctr




def update_inputs(replay_buffer, batch_size, resolution, device):
   
    batch_s, batch_a, batch_r, batch_s_n, batch_d, batch_mask_c, batch_mask_n = replay_buffer.sample(batch_size)
       
    if isinstance(batch_a, np.ndarray):
        batch_a = torch.from_numpy(batch_a)
    oh = F.one_hot(batch_a[:, 0], num_classes=resolution).float()
    batch_a_onehot = oh.view(batch_a.size(0), -1)
    
    batch_r = torch.FloatTensor(batch_r).unsqueeze(1)
    batch_s = torch.FloatTensor(batch_s)
    batch_s_n = torch.FloatTensor(batch_s_n)
    batch_d = torch.tensor(batch_d, dtype=torch.float32).unsqueeze(1)
    batch_mask_c = torch.tensor(batch_mask_c, dtype=torch.bool)
    batch_mask_n = torch.tensor(batch_mask_n, dtype=torch.bool)
    
    
    batch_s       = batch_s.to(device)
    batch_a_onehot= batch_a_onehot.to(device)
    batch_r       = batch_r.to(device)
    batch_s_n     = batch_s_n.to(device)
    batch_d       = batch_d.to(device)
    batch_mask_c  = batch_mask_c.to(device)
    batch_mask_n  = batch_mask_n.to(device)
    
    return batch_s, batch_a_onehot, batch_r, batch_s_n, batch_d, batch_mask_c, batch_mask_n


def update_reward(base_logits, next_logits,base_probs, next_probs,slice_info, batch_ssim_base, batch_ssim):
    
    device = base_logits.device
    label = slice_info['label'].to(device).long()
    w = slice_info['ce_weights'].to(device).float().mean(dim=0)
    criterion = nn.CrossEntropyLoss(weight=w, reduction='none').to(device)
    
    
    with torch.no_grad():
        base_scores = criterion(base_logits, label)
        next_scores = criterion(next_logits, label)
    ce_improvement = base_scores - next_scores  

    ssim_improvement = calculate_differences(batch_ssim, batch_ssim_base, device=device)
    base_uncertainty = calculate_entropy(base_probs)   
    next_uncertainty = calculate_entropy(next_probs)   
    uncertainty_reduction = base_uncertainty - next_uncertainty

    k1 = 10 * ce_improvement
    k2 = 100 * ssim_improvement
    k3 = 10 * uncertainty_reduction

    return (k1 + k2 + k3).unsqueeze(-1)



def update_env(
    masked_kspace,
    mask_next,
    recon_engine,
    infer_model,
    args,
    used_budget = 0,
    use_update_flag = 0
):
    
    with torch.no_grad():
        if use_update_flag == 1:
            mask_n = mask_next.view(-1, 1, 1, mask_next.size(1), 1)
        else:
            mask_n=mask_next
        recon_u = recon_engine.reconstruct(masked_kspace, mask_n)    

        inp3 = scale(recon_u).to(args.device)
        outputs, logits = infer_model(inp3)


    batch_done = None
    if use_update_flag == 1:
        batch_done,used_budget = update_flag(
            used_budget      = used_budget,
            step_budget      = args.budget,
            resolution       = args.resolution,
            center_fraction  = args.center_fractions[0],
            acceleration     = args.accelerations[0],
            batch_size       = args.batch_size,
            device           = args.device,
        )

    return outputs, logits, inp_out, recon_u, batch_done, used_budget


def update_flag(
        used_budget: int,
        step_budget: int,
        resolution: int,
        center_fraction: float,
        acceleration: float,
        batch_size: int,
        device: torch.device, 
) -> Tuple[torch.Tensor, int]:
    

    
    used_budget += step_budget
    centre_cols  = int(round(resolution * center_fraction))    
    total_budget = int(round(resolution / acceleration))       

    if used_budget + centre_cols >= total_budget:
        batch_done  = True            
        used_budget = 0               
    else:
        batch_done  = False           

    batch_done = torch.full(
        (batch_size, 1),
        fill_value=batch_done,
        dtype=torch.bool,
        device=device if device is not None else torch.device("cpu")
    )

    return batch_done, used_budget


def update_kspace(kspace, masked_kspace, mask, to_acquire):
    num_coils = kspace.size(1) 
    
    for sl in range(len(to_acquire)):
        columns = to_acquire[sl]
        for col in columns:
            index = 0  
            mask[sl][index][:, col.item(), :] = 1
            for coil in range(num_coils):
                masked_kspace[sl][coil][:, col.item(), :] = kspace[sl][coil][:, col.item(), :]
    return masked_kspace 


def soft_update(target, source, tau=0.001):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
