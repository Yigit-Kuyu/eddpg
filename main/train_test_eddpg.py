import random
import torch
import numpy as np
from utils.torch_metrics import  calculate_final, calculate_psnr_ssim_batch
from utils.helpers import get_training_logger, load_yaml
from environment.do_action import select_action, select_action_test
from environment.update import update_env,update_kspace, update_networks, update_inputs, update_reward
from environment.inital import reset, initialization, initialization_test


def process(args, epoch, infer_model, actor, loader, partition,logger, replay_buffer, critic, actor_target, critic_target, actor_optimizer, critic_optimizer, tracker= None, recon_engine=None):
    
    num_iters = len(loader)
    batch_ssims = np.zeros(num_iters)
    batch_psnrs = np.zeros(num_iters)
    calls_by_step = np.zeros(num_iters, dtype=int)        
    global_ctr  = 0
   
    for it, data in enumerate(loader): 
        print(f"In batch: {partition} Epoch {epoch}, Iteration {it+1}/{num_iters}")   
        kspace, masked_kspace, mask, gt, fname, slice_info = data 

        kspace_base = kspace.to(args.device) 
        masked_kspace_base = masked_kspace.to(args.device) 
        mask_base = mask.to(args.device) 
        
        probs_base, logits_base, state_base, recons_base, _, _ = update_env(
                masked_kspace     = masked_kspace_base,
                mask_next         = mask_base,
                recon_engine      = recon_engine,
                infer_model       = infer_model,
                args              = args
                )
        
        batch_ssim_base,_, _,_=calculate_psnr_ssim_batch(recons_base, gt)
        
        for r_step in range(args.r_steps):  
            print(f"{partition} Epoch {epoch}, Iteration {it+1}/{num_iters}, R-step {r_step+1}/{args.r_steps}")
            used_budget, probs_old, logits_old, kspace_old, mask_old, masked_kspace_old, state = reset(
                probs_base,
                logits_base,
                kspace_base,
                mask_base,
                masked_kspace_base,
                state_base,
                args.device
            )
        
            for _ in range(args.a_steps):  

                if  args.do_train==1:
                    actions, mask_next=select_action(args, actor, state, mask_old.squeeze().bool().clone(), 1, args.budget)
                else:
                    actions, mask_next = select_action_test(actor, state, mask_old.squeeze().bool().clone(), args.budget)

                masked_kspace = update_kspace(kspace_old, masked_kspace_old, mask_old, actions)

                
                outputs, logits, state_next, recons, batch_done, used_budget = update_env(
                masked_kspace     = masked_kspace,
                mask_next         = mask_next,
                recon_engine      = recon_engine,
                infer_model       = infer_model,
                args              = args,
                used_budget       = used_budget,
                use_update_flag   = 1,
                )
                
                batch_ssim,_, avg_ssim_batch, avg_psnr_batch =calculate_psnr_ssim_batch(recons, gt)   
                
                if partition=='Train' and args.do_train==1:
                    action_rewards = update_reward(logits_old, logits, probs_old, outputs, slice_info, batch_ssim_base, batch_ssim)

                    replay_buffer.push(
                    state.cpu().numpy(), 
                    actions.cpu().numpy(), 
                    action_rewards.cpu().numpy(), 
                    state_next.cpu().numpy(),    
                    batch_done.cpu().numpy(),   
                    mask_old.squeeze().bool().clone().cpu().numpy(), 
                    mask_next.cpu().numpy()        
                    )

                    if  len(replay_buffer) > args.start_minibatch and args.do_train==1 and partition == 'Train':
                        
                        batch_s, batch_a_onehot, batch_r, batch_s_n, batch_d, batch_mask_c, batch_mask_n = update_inputs(replay_buffer, args.minibatch_size, args.resolution, args.device)

                        global_ctr=update_networks(
                        actor, critic,
                        actor_target, critic_target,
                        actor_optimizer, critic_optimizer,
                        batch_s, batch_a_onehot, batch_r,
                        batch_s_n, batch_d,
                        batch_mask_c, batch_mask_n,
                        gamma              = 0.99,
                        tau                = 1e-3,
                        target_update_freq = args.target_update_freq,
                        global_step_ctr    = global_ctr,
                        budget=args.budget
                        )
                  
                state      = state_next          
                mask_old   = mask_next.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
                masked_kspace_old = masked_kspace
                logits_old = logits              
                probs_old  = outputs
                if batch_done.any():
                    break 
        if r_step==args.r_steps-1:
                if partition=='Val' or partition=='Test':
                    batch_ssims[it]  += avg_ssim_batch
                    batch_psnrs[it]  += avg_psnr_batch
                    calls_by_step[it] += 1 
                    tracker.update(outputs, slice_info['label'].to(args.device).long())

    
    if partition=='Val' or partition=='Test':
        calculate_final(batch_ssims,batch_psnrs, calls_by_step,tracker,
                        logger, partition, epoch, actor, critic,actor_target, 
                        critic_target,actor_optimizer, critic_optimizer, args)

    return actor,critic,actor_target,critic_target,actor_optimizer,critic_optimizer 


def train_and_eval(args): 
    print("=" * 50)
    print("STARTING EDDPG TRAINING")
    print("=" * 50)
         
    logger_val = get_training_logger(args.policy_model_checkpoint)
    print("Initializing models and data loaders...")
    (critic, actor_target, critic_target, optimizer_actor, optimizer_critic,
     tracker, recon_engine,
     infer_model, actor,
     train_loader, dev_loader,
     replay_buffer) = initialization(args)

    print(f"Training will run for {args.num_epochs} epochs")
    print(f"Train loader has {len(train_loader)} batches")
    print(f"Validation loader has {len(dev_loader)} batches")
    
    for epoch in range(0, args.num_epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{args.num_epochs} ---")
          
        actor,critic,actor_target,critic_target,optimizer_actor,optimizer_critic = process(args, epoch, infer_model, actor, train_loader, 'Train',None, replay_buffer, 
                                                                        critic=critic,
                                                                        actor_target=actor_target,
                                                                        critic_target=critic_target,
                                                                        actor_optimizer=optimizer_actor,
                                                                        critic_optimizer=optimizer_critic,
                                                                        tracker=tracker,
                                                                        recon_engine=recon_engine)

        _,_,_,_,_,_ = process(args, epoch, infer_model, actor, dev_loader, 'Val',logger_val, replay_buffer,
                                                                        critic=critic,
                                                                        actor_target=actor_target,
                                                                        critic_target=critic_target,
                                                                        actor_optimizer=optimizer_actor,
                                                                        critic_optimizer=optimizer_critic,
                                                                        tracker=tracker,
                                                                        recon_engine=recon_engine)


def test(args):

    logger_test = get_training_logger(args.policy_model_checkpoint)
    actor, test_loader, recon_engine, infer_model, tracker = initialization_test(args)
    process(args, 0, infer_model, actor, test_loader, 'Test',logger_test,replay_buffer=None,critic=None, actor_target=None, critic_target=None, actor_optimizer=None, critic_optimizer=None, tracker=tracker, recon_engine=recon_engine)
    

if __name__ == '__main__':
    args = load_yaml("config/config.yaml")
        
    if args.seed != 0: 
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)

    if args.do_train:
        train_and_eval(args) 
    else:
        test(args)
    
    
