import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return h

class SimpleUNet(nn.Module):
    """DDPM U-Net tailored for 125x125 jet images (internally padded to 128x128)."""
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.ReLU()
        )
        self.inc = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.down1 = Block(base_channels, base_channels, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(base_channels, base_channels*2, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = Block(base_channels*2, base_channels*4, time_emb_dim)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bot1 = Block(base_channels*4, base_channels*8, time_emb_dim)
        
        self.up1 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, 2, 1)
        self.res1 = Block(base_channels*8, base_channels*4, time_emb_dim)
        
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1)
        self.res2 = Block(base_channels*4, base_channels*2, time_emb_dim)
        
        self.up3 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1)
        self.res3 = Block(base_channels*2, base_channels, time_emb_dim)
        
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        
        h, w = x.shape[2:]
        pad_h = (128 - h) if h < 128 else 0
        pad_w = (128 - w) if w < 128 else 0
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
            
        x1 = self.inc(x)
        d1 = self.down1(x1, t)
        p1 = self.pool1(d1)
        
        d2 = self.down2(p1, t)
        p2 = self.pool2(d2)
        
        d3 = self.down3(p2, t)
        p3 = self.pool3(d3)
        
        b = self.bot1(p3, t)
        
        u1 = self.up1(b)
        u1 = torch.nn.functional.interpolate(u1, size=d3.shape[2:])
        u1 = torch.cat([u1, d3], dim=1)
        r1 = self.res1(u1, t)
        
        u2 = self.up2(r1)
        u2 = torch.nn.functional.interpolate(u2, size=d2.shape[2:])
        u2 = torch.cat([u2, d2], dim=1)
        r2 = self.res2(u2, t)
        
        u3 = self.up3(r2)
        u3 = torch.nn.functional.interpolate(u3, size=d1.shape[2:])
        u3 = torch.cat([u3, d1], dim=1)
        r3 = self.res3(u3, t)
        
        out = self.outc(r3)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
        return out
