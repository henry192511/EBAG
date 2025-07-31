import torch
import numpy as np
import scipy
import sys
from tqdm import tqdm
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import scipy
import os
import multiprocessing as mp

n_cpu = int(os.cpu_count() * 0.8)

def single_gpd_fit(name, data):
    return name, scipy.stats.genpareto.fit(data)

class Score_Base:
    def __init__(self, data, FVs):
        print("Score init")
        self.Score_params = {}

        print("Starting norm")
        mask_for_gpd_fit = torch.randperm(data.shape[0])
        data_shuffled = data[mask_for_gpd_fit]
        FVs_shuffled = FVs[mask_for_gpd_fit]
        data_normed = self.norm(data_shuffled, FVs_shuffled)

        print("Start GPD fit")
        self.gpd_mp_fit(data_normed)

    def gpd_mp_fit(self, data):
        max_values, gt = torch.max(data, dim=1)

        pool = mp.Pool(processes=1)
        jobs = []
        for class_id in np.unique(gt.cpu().numpy()):
            class_data = data[gt == class_id][:, class_id]
            job = pool.apply_async(func=single_gpd_fit, args=(class_id, class_data.cpu().numpy()))
            jobs.append(job)

        for job in jobs:
            class_id, gpd_param = job.get()
            shape, loc, scale = gpd_param
            self.Score_params[class_id] = (shape, loc, scale)

        pool.close()
        pool.join()
        return

    def ReScore(self, data, FVs):
        logits = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        FVs = torch.nan_to_num(FVs, nan=0.0, posinf=0.0, neginf=0.0)

        normd_data = self.norm(logits, FVs)
        max_data, pred = torch.max(logits, dim=1)

        rescored_data = torch.zeros(data.shape[0], device=data.device)
        for class_id in range(data.shape[1]):
            class_mask = pred == class_id
            if class_id not in self.Score_params or class_mask.sum() == 0:
                rescored_data[class_mask] = 0.0
                continue

            shape, loc, scale = self.Score_params[class_id]
            class_scores = normd_data[class_mask, class_id].cpu().numpy()
            class_scores = np.nan_to_num(class_scores, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                rescored_class_data = scipy.stats.genpareto.cdf(class_scores, shape, loc=loc, scale=scale)
                rescored_data[class_mask] = torch.from_numpy(rescored_class_data).float().to(data.device)
            except Exception as e:
                print(f"[ERROR] GPD CDF failed for class {class_id}: {e}")
                rescored_data[class_mask] = 0.0

        return rescored_data

    def norm(self, logits, FVs):
        raise NotImplementedError("Base norm must be overridden")

class Score(Score_Base):
    def __init__(self, data, FVs):
        self.Gaus_dict = self.Gaus_gen(data, FVs)
        super().__init__(data, FVs)

    def Gaus_gen(self, logits, FV):
        preds = torch.max(logits, dim=1).indices
        classes = torch.unique(preds).long().tolist()

        class_models = {}
        eps = 1e-6
        print("Generating gaussian models")
        for c in tqdm(classes):
            select_class_FVs = FV[preds == c]
            if select_class_FVs.numel() == 0:
                continue

            mean = torch.mean(select_class_FVs, dim=0)
            std = torch.std(select_class_FVs, dim=0)
            std = torch.where(std < eps, torch.full_like(std, eps), std)
            mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
            std = torch.nan_to_num(std, nan=eps, posinf=eps, neginf=eps)

            class_models[c] = (mean, std)

        return class_models

    def norm(self, logits: torch.Tensor, FVs: torch.Tensor) -> torch.Tensor:
            import torch.nn.functional as F
            beta = 1.0
            normalized = torch.zeros_like(logits)
            for c, (mean, std) in self.Gaus_dict.items():
                class_mask = (logits.argmax(dim=1) == c)
                if not class_mask.any():
                    continue
                Z = torch.abs((FVs[class_mask] - mean) / std)
                diff = Z.sum(dim=1).clamp(min=1e-6)
                raw_logit = logits[class_mask, c]             
                sp = F.softplus(raw_logit)                    
                normalized[class_mask, c] = sp * diff.pow(-beta)
            return normalized

    def update(self, logits: torch.Tensor, FVs: torch.Tensor):
        preds = torch.max(logits, dim=1).indices
        new_classes = torch.unique(preds).tolist()

        eps = 1e-6
        for c in new_classes:
            mask_c = preds == c
            fv_c = FVs[mask_c]        # (Nc, D)
            logits_c = logits[mask_c] # (Nc, C)

            if fv_c.size(0) == 0:
                print(f"[WARN] No samples for class {c} during update, skipping.")
                continue

            mean = fv_c.mean(dim=0)
            std = fv_c.std(dim=0)
            std = torch.where(std < eps, torch.full_like(std, eps), std)

            mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
            std = torch.nan_to_num(std, nan=eps, posinf=eps, neginf=eps)
            self.Gaus_dict[c] = (mean, std)

            try:
                normed_c = self.norm(logits_c, fv_c)[:, c].cpu()
                normed_c = torch.nan_to_num(normed_c, nan=0.0, posinf=0.0, neginf=0.0)
                if not torch.isfinite(normed_c).all():
                    print(f"[ERROR] Non-finite normed_c for class {c}, skipping GPD fit.")
                    continue

                shape, loc, scale = scipy.stats.genpareto.fit(normed_c.numpy())
                self.Score_params[c] = (shape, loc, scale)
            except Exception as e:
                print(f"[EXCEPTION] Fitting GPD failed for class {c}: {e}")

choices = {
    "Score": Score,
}

if __name__ == '__main__':
    print("Score module")
