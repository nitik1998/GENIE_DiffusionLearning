import torch
import torch.nn.functional as F

class DDPM:
    """Core DDPM Scheduler for training and sampling sparse manifolds."""
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    def add_noise(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
        
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.to(a.device))
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)

    def compute_loss(self, x_0, loss_type="l1"):
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=self.device).long()
        x_t, noise = self.add_noise(x_0, t)
        noise_pred = self.model(x_t, t)
        if loss_type == "l1":
            return F.l1_loss(noise_pred, noise)
        if loss_type == "mse":
            return F.mse_loss(noise_pred, noise)
        raise ValueError(f"Unsupported loss_type: {loss_type}")
        
    @torch.no_grad()
    def sample(self, shape):
        self.model.eval()
        x_t = torch.randn(shape, device=self.device)
        ndim = len(shape)  # 2 for latent vectors, 4 for images
        extra_dims = (1,) * (ndim - 1)  # (1,) for 2D, (1,1,1) for 4D
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            noise_pred = self.model(x_t, t)
            
            alpha_t = self.alphas[t].reshape(-1, *extra_dims)
            alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, *extra_dims)
            beta_t = self.betas[t].reshape(-1, *extra_dims)
            
            if i > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
                
            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred) + torch.sqrt(beta_t) * noise
        return x_t
