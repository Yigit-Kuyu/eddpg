import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchmetrics import Accuracy, AUROC, Recall, Specificity, AveragePrecision
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pathlib import Path


class EpochMetricTracker:
    
    def __init__(self, task: str = "binary", device: torch.device = "cpu"):
        self.device = device
        self.accuracy    = Accuracy(task=task).to(device)
        self.recall      = Recall(task=task).to(device)
        self.specificity = Specificity(task=task).to(device)
        self.auprc       = AveragePrecision(task=task).to(device)
        self.auroc       = AUROC(task=task).to(device)
        self._all = [
            self.accuracy, self.recall, self.specificity,
            self.auprc, self.auroc,
        ]

    @torch.inference_mode()
    def update(self, probs: torch.Tensor, labels: torch.Tensor) -> None:
        probs  = probs.to(self.device)
        labels = labels.to(self.device).long()

        
        preds = torch.argmax(probs, dim=1)
        self.accuracy.update(preds, labels)
        self.recall.update(preds, labels)
        self.specificity.update(preds, labels)

        
        probs_pos = probs[:, 1]
        self.auprc.update(probs_pos, labels)
        self.auroc.update(probs_pos, labels)

    def compute(self, reset: bool = True) -> dict:
        out = {
            "accuracy":    self.accuracy.compute().item(),
            "recall":      self.recall.compute().item(),
            "specificity": self.specificity.compute().item(),
            "auprc":       self.auprc.compute().item(),
            "auc":         self.auroc.compute().item(),
        }
        if reset:
            self.reset()
        return out

    def reset(self) -> None:
        for m in self._all:
            m.reset()


def calculate_entropy(outputs): 
    log_probs = torch.log2(outputs + 1e-8)  
    return -torch.sum(outputs * log_probs, dim=-1)


def calculate_differences(list1: list, list2: list, device=None):

    if not list1 or not list2:
        print("Warning: One or both lists are empty.")
        return 0.0
    if len(list1) != len(list2):
        raise ValueError("Input lists must have the same length.")

   
    differences = [x - y for x, y in zip(list1, list2)]
    differences_tensor = torch.tensor(differences, device=device)

    return differences_tensor

def calculate_psnr_ssim_batch(tensor_image_masked, target):
   
    
    target_complex=target[..., 0] + 1j * target[..., 1] 
    target_rss = torch.sqrt(torch.sum(torch.abs(target_complex)**2, dim=1))  
    batch_ssim = []
    batch_psnr = []

    recon_rss=tensor_image_masked 
    
    for recon, tgt in zip(recon_rss, target_rss):
        recon_np = recon.cpu().numpy()
        tgt_np   =   tgt.cpu().numpy()
  
    dr = float(target_rss.cpu().numpy().max()) 
           
    s = ssim(tgt_np, recon_np, data_range=dr, channel_axis=None)
    p = psnr(tgt_np, recon_np, data_range=dr)

    batch_ssim.append(s)
    batch_psnr.append(p)

    avg_ssim = float(np.mean(batch_ssim))
    avg_psnr = float(np.mean(batch_psnr))
    
    return batch_ssim, batch_psnr, avg_ssim, avg_psnr



def calculate_final(
    batch_ssims,
    batch_psnrs,
    calls_by_step,
    tracker,
    logger,
    partition,
    epoch,
    actor,
    critic,
    actor_target,
    critic_target,
    actor_optimizer,
    critic_optimizer,
    args
):
    
    
    batch_ssims = np.array(batch_ssims, dtype=float)
    batch_psnrs = np.array(batch_psnrs, dtype=float)
    ssims_full  = np.zeros_like(batch_ssims)
    psnrs_full  = np.zeros_like(batch_psnrs)

    valid_steps = calls_by_step > 0
    ssims_full[valid_steps] = batch_ssims[valid_steps] / calls_by_step[valid_steps]
    psnrs_full[valid_steps] = batch_psnrs[valid_steps] / calls_by_step[valid_steps]

    last_valid = np.where(valid_steps)[0][-1]
    ssims      = ssims_full[: last_valid + 1]
    psnrs      = psnrs_full[: last_valid + 1]

    
    stats              = tracker.compute()
    recall             = stats["recall"]
    specificity        = stats["specificity"]
    auc                = stats["auc"]
    balanced_accuracy  = 0.5 * (recall + specificity)

    
    logger.info(
        f"Batch Iterations End: Partition {partition}: "
        f"Epoch {epoch}: "
        f"AUC={auc:.4f}, "
        f"Recall={recall:.4f}, "
        f"Specificity={specificity:.4f}, "
        f"SSIM={ssims.mean():.4f}, "
        f"PSNR={psnrs.mean():.4f}, "
        f"SSIM_std={ssims.std():.4f}, "
        f"PSNR_std={psnrs.std():.4f}, "
        f"Balanced_Accuracy={balanced_accuracy:.4f}"
    )

    
    if args.do_train:
        save_path = Path(args.policy_model_checkpoint) / "model_final.pth"
        torch.save({
            'epoch':                          epoch,
            'actor_state_dict':               actor.state_dict(),
            'critic_state_dict':              critic.state_dict(),
            'actor_target_state_dict':        actor_target.state_dict(),
            'critic_target_state_dict':       critic_target.state_dict(),
            'actor_optimizer_state_dict':     actor_optimizer.state_dict(),
            'critic_optimizer_state_dict':    critic_optimizer.state_dict(),
        }, save_path)



