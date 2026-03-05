import os
from time import perf_counter

import numpy as np
import torch
import pnpflow.image_generation.models.utils as mutils
import pnpflow.utils as utils
from PIL import Image

class FLOWER(object):

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = args.method

    def model_forward(self, x, t):
        if self.args.model == "ot" or  self.args.model == "flow_indp":
            return self.model(x, t)

        elif self.args.model == "rectified":
            model_fn = mutils.get_model_fn(self.model, train=False)
            t_ = t[:, None, None, None]
            v = model_fn(x.type(torch.float), t * 999)
            return v

    def _debug_enabled(self):
        return bool(getattr(self.args, "save_step_debug", False))

    def _debug_stride(self):
        return max(1, int(getattr(self.args, "step_debug_stride", 1)))

    def _save_step_debug_image(self, x, batch, count, step_name, delta):
        if not self._debug_enabled():
            return
        # Save first sample of the batch for fast debugging and readability.
        x_vis = utils.postprocess(x[:1].detach().cpu().clone(), self.args)[0]
        x_vis = torch.clamp(x_vis, 0.0, 1.0)
        x_vis = x_vis.permute(1, 2, 0).numpy()
        if x_vis.shape[-1] == 1:
            x_vis = x_vis[..., 0]
            x_uint8 = (x_vis * 255.0).astype(np.uint8)
            img = Image.fromarray(x_uint8, mode="L")
        else:
            x_uint8 = (x_vis * 255.0).astype(np.uint8)
            img = Image.fromarray(x_uint8)

        debug_dir = os.path.join(
            self.args.save_path_ip,
            "debug_steps",
            f"delta_t={delta:.6f}",
            f"batch{batch:03d}",
            step_name,
        )
        os.makedirs(debug_dir, exist_ok=True)
        img.save(os.path.join(debug_dir, f"iter{count:06d}.png"))

    def grad_datafit(self, x, y, H, H_adj):
        if self.args.noise_type == 'gaussian':
            return H_adj(H(x) - y) / (self.args.sigma_noise**2)
        elif self.args.noise_type == 'laplace':
            return H_adj(2*torch.heaviside(H(x)-y, torch.zeros_like(H(x)))-1)/self.args.sigma_noise
        else:
            raise ValueError('Noise type not supported')

    def interpolation_step(self, x, t):
        return t * x + torch.randn_like(x) * (1 - t)

    def denoiser(self, x, t):
        v = self.model_forward(x, t)
        return x + (1 - t.view(-1, 1, 1, 1)) * v


    def BtB(self, x, H, Ht, lam, idx=None):    
        BtBD =(Ht(H(x))) / (self.args.sigma_noise**2) + x / lam
        return BtBD
    
    def cg(self, b, x0=None, lam=1, max_iter=100, eps=1e-5, H=lambda x: x, Ht=lambda x: x, dims=(1, 2, 3)):
        
        A = lambda x: self.BtB(x, H, Ht, lam)
        if x0 is None:
            x0 = torch.zeros_like(b, device=b.device, dtype=b.dtype)
        
        x = x0.clone() 
        r = b - A(x)

        p = r.clone()
        r_norm = r_norm_old = (r ** 2).sum(dim=dims, keepdim=True)  

        with torch.no_grad():
            for i in range(max_iter):

                BTBp = A(p)
                alpha = r_norm / ((p * BTBp).sum(dim=dims, keepdim=True))

                x = x + alpha * p
                r_norm_old = r_norm.clone()
                r = r - alpha * BTBp

                r_norm = (r ** 2).sum(dim=dims, keepdim=True) 
                if r_norm.sqrt().all() < eps:
                    break
                beta = r_norm / (r_norm_old)
                p = r + beta * p

        return x, i
    
    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        H = degradation.H
        H_adj = degradation.H_adj
        self.args.sigma_noise = sigma_noise
        num_samples = self.args.num_samples
        steps, delta = self.args.steps, 1 / self.args.steps
        if self.args.noise_type == 'gaussian':
            pass
        else:
            raise ValueError('Noise type not supported')

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):
            
            (clean_img, labels) = next(loader)
            self.args.batch = batch

            if self.args.noise_type == 'gaussian':
                noisy_img = H(clean_img.clone().to(self.device))
                torch.manual_seed(batch)
                noisy_img += torch.randn_like(noisy_img) * sigma_noise
            elif self.args.noise_type == 'laplace':
                noisy_img = H(clean_img.clone().to(self.device))
                noise = torch.distributions.laplace.Laplace(
                    torch.zeros_like(noisy_img), sigma_noise * torch.ones_like(noisy_img)).sample().to(self.device)
                noisy_img += noise
            else:
                raise ValueError('Noise type not supported')

            noisy_img, clean_img = noisy_img.to(
                self.device), clean_img.to('cpu')

            # intialize the image with the adjoint operator
            
            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)

            with torch.no_grad():
                if self.args.compute_time:
                    time_counter_1 = perf_counter()
                x_avg = torch.zeros_like(clean_img, device=self.device)
                for _ in range(num_samples):
                    
                    x =  torch.randn_like(clean_img, device=self.device)
                    ones = torch.ones(len(x), device=self.device) 
                    for count, iteration in enumerate(range(int(steps))):
                        
                        
                        t = delta * iteration    
                        
                        sigma_r = (1 - t) / np.sqrt(t**2 + (1-t)**2)

                        x_hat_1 = x + (1 - t) * self.model_forward(x, ones * t)
                        if count % self._debug_stride() == 0:
                            self._save_step_debug_image(
                                x_hat_1, batch, count, "step1", delta)
                        
                        lam = sigma_r ** 2
                        b = H_adj(noisy_img) / (self.args.sigma_noise**2) + x_hat_1 / lam
                        x_star, _ = self.cg(b, x_hat_1, lam, max_iter=50, eps=1e-5, H=H, Ht=H_adj)
                        if count % self._debug_stride() == 0:
                            self._save_step_debug_image(
                                x_star, batch, count, "step2", delta)

                        z0 =  torch.randn_like(x, device=self.device)
                        
                        estimated_iso_cov = 1 - t - delta

                        x = (t + delta) * x_star + estimated_iso_cov * z0
                        if count % self._debug_stride() == 0:
                            self._save_step_debug_image(
                                x, batch, count, "step3", delta)
                                            
                    x_avg += x
                x = x_avg / num_samples  

                if self.args.compute_time:
                    torch.cuda.synchronize()
                    time_counter_2 = perf_counter()
                    time_per_batch += time_counter_2 - time_counter_1


            if self.args.compute_memory:
                dict_memory = {}
                dict_memory["batch"] = batch
                dict_memory["max_allocated"] = torch.cuda.max_memory_allocated(
                    self.device)
                utils.save_memory_use(dict_memory, self.args)

            if self.args.compute_time:
                dict_time = {}
                dict_time["batch"] = batch
                dict_time["time_per_batch"] = time_per_batch
                utils.save_time_use(dict_time, self.args)

            if self.args.save_results:
                restored_img = x.detach().clone()
                utils.save_images(clean_img, noisy_img, restored_img,
                                  self.args, H_adj, iter='final')
                utils.compute_psnr(clean_img, noisy_img,
                                   restored_img, self.args, H_adj, iter=iteration)
                utils.compute_ssim(
                    clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                utils.compute_lpips(clean_img, noisy_img,
                                    restored_img, self.args, H_adj, iter=iteration)

        if self.args.save_results:
            utils.compute_average_psnr(self.args)
            utils.compute_average_ssim(self.args)
            utils.compute_average_lpips(self.args)
        if self.args.compute_memory:
            utils.compute_average_memory(self.args)
        if self.args.compute_time:
            utils.compute_average_time(self.args)

    def should_save_image(self, iteration, steps):
        return iteration % (steps // 10) == 0

    def run_method(self, data_loaders, degradation, sigma_noise, H_funcs=None):
        
        # Construct the save path for results
        folder = utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)

        # Create the directory if it doesn't exist
        os.makedirs(self.args.save_path_ip, exist_ok=True)

        # Solve the inverse problem
        self.solve_ip(
            data_loaders[self.args.eval_split], degradation, sigma_noise, H_funcs)
