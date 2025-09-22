import torch


def gumbel_select_action(logits, tau, noise_scale, available_mask, k=1): 

    fill_value    = torch.finfo(logits.dtype).min
    masked_logits = torch.where(available_mask, logits, fill_value)

    gumbel_noise  = -torch.log(-torch.log(torch.rand_like(masked_logits) + 1e-20) + 1e-20) * noise_scale
    gumbel_logits = (masked_logits + gumbel_noise) / tau        
    probs         = torch.softmax(gumbel_logits, dim=-1)   
   
    _, actions    = torch.topk(gumbel_logits, k=k, dim=-1)  
    one_hot       = torch.zeros_like(probs).scatter_(-1, actions, 1.0)

    st_actions    = one_hot + probs - probs.detach()

    return actions, st_actions


def select_action(args, actor, state, selected_columns, tau, num_columns_to_select):


    logits = actor(state)

    available_mask = ~selected_columns  
    available_counts = available_mask.sum(dim=1)  

    different_shapes_flag = (available_counts.min() != available_counts.max())
    if different_shapes_flag:
            print("Warning: Available column counts differ across batch elements:", available_counts)

    if (available_mask.sum(dim=1) < num_columns_to_select).any():
        raise ValueError("Not enough available columns to sample the required number of unique indices.")
    
    discrete_actions, _ = gumbel_select_action(
        logits        = logits,
        tau           = tau,        
        noise_scale   = 1.0,               
        available_mask= available_mask,
        k             = num_columns_to_select,
    )
    
    selected_columns_updated = selected_columns.clone()
    batch_size = selected_columns.shape[0]
    for b in range(batch_size):
            selected_columns_updated[b, discrete_actions[b]] = True

    return discrete_actions, selected_columns_updated

def select_action_test(actor, state, selected_columns, num_columns_to_select):

    with torch.no_grad():
        
        state = state.to(next(actor.parameters()).device)
        logits = actor(state)

        selected_columns = selected_columns.to(logits.device)
        available_mask = ~selected_columns 
        
        masked_logits = torch.where(available_mask, logits, torch.full_like(logits, -1e6))
        
        _, action = torch.topk(masked_logits, k=num_columns_to_select, dim=1)  
        
        selected_columns_updated = selected_columns.clone()
        batch_size = selected_columns.shape[0]
        for b in range(batch_size):
            selected_columns_updated[b, action[b]] = True
        
    return action, selected_columns_updated