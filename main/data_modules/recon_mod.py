import fastmri.data.transforms as T
from types import SimpleNamespace
from typing import List
import torch
from utils.load_utils import  load_varnet_model
from utils.promptmr import PromptMR


class KneeRecon:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda:0",
        num_adj_slices: int = 1,
        num_cascades: int = 12,
        n_feat0: int = 48,
        feature_dim: list[int] = None,
        prompt_dim: list[int] = None,
        sens_n_feat0: int = 24,
        sens_feature_dim: list[int] = None,
        sens_prompt_dim: list[int] = None,
        use_checkpoint: bool = False,
        


        len_prompt: List[int] = [5, 5, 5],
        prompt_size: List[int] = [64, 32, 16],
        n_enc_cab: List[int] = [2, 3, 3],
        n_dec_cab: List[int] = [2, 2, 3],
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        mask_center: bool = True,
        low_mem: bool = False,
    ):
        
        self.device = device

        
        self.model = PromptMR(
            num_cascades      = num_cascades,
            num_adj_slices    = num_adj_slices,
            n_feat0           = n_feat0,
            feature_dim       = feature_dim,
            prompt_dim        = prompt_dim,
            sens_n_feat0      = sens_n_feat0,
            sens_feature_dim  = sens_feature_dim,
            sens_prompt_dim   = sens_prompt_dim,
            use_checkpoint    = False,
            no_use_ca         = True,
            len_prompt = len_prompt,
            prompt_size = prompt_size,
            n_enc_cab = n_enc_cab,
            n_dec_cab =n_dec_cab,
            n_skip_cab =n_skip_cab,
            n_bottleneck_cab = n_bottleneck_cab

            

        ).to(self.device).eval()

        sd = torch.load(ckpt_path, map_location="cpu",weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
            sd = {k.replace("promptmr.",""): v for k,v in sd.items()}

        model_state = self.model.state_dict()
        filtered_sd = {k: v for k, v in sd.items() if k in model_state}
 
        self.model.load_state_dict(filtered_sd, strict=False)

    def reconstruct(self, masked_kspace: torch.Tensor, mask: torch.Tensor):
       
        B, C, H, W, _ = masked_kspace.shape

        batch = SimpleNamespace(
            masked_kspace = masked_kspace,
            mask          = mask,
            crop_size     = (H, W),
            slice_num     = torch.zeros(B, dtype=torch.int32),
            fname         = ["" for _ in range(B)],
        )

        output, _, _ = load_varnet_model(batch, self.model, self.device)
        return output