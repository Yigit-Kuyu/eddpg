import torch
import torch.optim as optim
import pathlib
import os
from data_modules.critic_mod import Critic
from data_modules.recon_mod import KneeRecon
from environment.buffer_p import PercentileBasedReplayBuffer
from utils.load_utils import load_infer_model, load_policy_model, create_data_loader, load_policy_model_test
from utils.torch_metrics import EpochMetricTracker


def initialization(args):

    train_loader = create_data_loader(args, "train", shuffle=True)
    dev_loader   = create_data_loader(args, "val",   shuffle=False)


    actor         = load_policy_model(args)
    actor_target  = load_policy_model(args)
    critic        = Critic()
    critic_target = Critic()

    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    optimizer_actor  = optim.Adam(actor.parameters(),  lr=args.lr)
    optimizer_critic = optim.Adam(critic.parameters(), lr=args.lr)


    tracker = EpochMetricTracker(task="binary", device=args.device)


    recon_engine = KneeRecon(
        ckpt_path          = args.recons_model_checkpoint,
        device             = args.device,
        num_cascades       = args.num_cascades,
        num_adj_slices     = args.num_adj_slices,
        n_feat0            = args.n_feat0,
        feature_dim        = args.feature_dim,
        prompt_dim         = args.prompt_dim,
        sens_n_feat0       = args.sens_n_feat0,
        sens_feature_dim   = args.sens_feature_dim,
        sens_prompt_dim    = args.sens_prompt_dim,
        len_prompt         = args.len_prompt,
        prompt_size        = args.prompt_size,
        n_enc_cab          = args.n_enc_cab,
        n_dec_cab          = args.n_dec_cab,
        n_skip_cab         = args.n_skip_cab,
        n_bottleneck_cab   = args.n_bottleneck_cab,
    )

    _, infer_model = load_infer_model(args)
    infer_model = infer_model.to(args.device)


    rb_cap = args.replay_buffer_size
    replay_buffer = PercentileBasedReplayBuffer(
        high_capacity = rb_cap // 2,
        low_capacity  = rb_cap // 2,
        percentile    = 75,
        args          = args,
    )


    actor            = actor.to(args.device)
    actor_target     = actor_target.to(args.device)
    critic           = critic.to(args.device)
    critic_target    = critic_target.to(args.device)
    recon_engine.model = recon_engine.model.to(args.device)

    if args.data_parallel: 
        device_ids = list(range(torch.cuda.device_count()))
        infer_model     = torch.nn.DataParallel(infer_model,     device_ids=device_ids)
        actor           = torch.nn.DataParallel(actor,           device_ids=device_ids)
        critic          = torch.nn.DataParallel(critic,          device_ids=device_ids)
        actor_target    = torch.nn.DataParallel(actor_target,    device_ids=device_ids)
        critic_target   = torch.nn.DataParallel(critic_target,   device_ids=device_ids)
        recon_engine.model = torch.nn.DataParallel(recon_engine.model, device_ids=device_ids)

   
    return (
        critic,
        actor_target,
        critic_target,
        optimizer_actor,
        optimizer_critic,
        tracker,
        recon_engine,
        infer_model,
        actor,
        train_loader,
        dev_loader,
        replay_buffer,
    )

def initialization_test(args):

    checkpoint_file = os.path.join(args.policy_model_checkpoint, "model_final.pth")   
    actor=load_policy_model_test(pathlib.Path(checkpoint_file), args, key='actor_state_dict')
    test_loader   = create_data_loader(args, "test",   shuffle=False)
    _, infer_model = load_infer_model(args)
    recon_engine = KneeRecon(
        ckpt_path          = args.recons_model_checkpoint,
        device             = args.device,
        num_cascades       = args.num_cascades,
        num_adj_slices     = args.num_adj_slices,
        n_feat0            = args.n_feat0,
        feature_dim        = args.feature_dim,
        prompt_dim         = args.prompt_dim,
        sens_n_feat0       = args.sens_n_feat0,
        sens_feature_dim   = args.sens_feature_dim,
        sens_prompt_dim    = args.sens_prompt_dim,
        len_prompt         = args.len_prompt,
        prompt_size        = args.prompt_size,
        n_enc_cab          = args.n_enc_cab,
        n_dec_cab          = args.n_dec_cab,
        n_skip_cab         = args.n_skip_cab,
        n_bottleneck_cab   = args.n_bottleneck_cab,
    )

    tracker = EpochMetricTracker(task="binary", device=args.device)

    actor            = actor.to(args.device)
    recon_engine.model = recon_engine.model.to(args.device)
    infer_model = infer_model.to(args.device)

    return actor, test_loader, recon_engine, infer_model, tracker


def reset(
    probs_base,
    logits_base,
    kspace_base,
    mask_base,
    masked_kspace_base,
    state_base,
    device
):
    
    
    used_budget = 0
    probs_old, logits_old, kspace_old, mask_old, masked_kspace_old, state = [
        x.clone() for x in (
            probs_base,
            logits_base,
            kspace_base,
            mask_base,
            masked_kspace_base,
            state_base,
        )
    ]

    state = state.to(device)
    return (
        used_budget,
        probs_old,
        logits_old,
        kspace_old,
        mask_old,
        masked_kspace_old,
        state
    )

