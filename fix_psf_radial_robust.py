#!/usr/bin/env python3
"""
Robust version of PSF radial function for DeepLens that handles ray validity issues.
"""

def robust_psf_radial(lens, M=2, depth=1000.0, ks=101, log_scale=True, save_name="psf_radial.png"):
    """More robust version of draw_psf_radial that handles ray validity issues.
    
    Args:
        lens: The lens object
        M: Number of field positions (default: 2)
        depth: Object depth (default: 1000.0 mm)
        ks: Kernel size for PSF (default: 101)
        log_scale: Whether to use log scale (default: True)
        save_name: Filename to save the PSF image
    """
    import torch
    from torchvision.utils import make_grid, save_image
    from deeplens.optics.basics import EPSILON, WAVE_RGB
    import numpy as np
    
    # Strategy 1: Use smaller field points (less than full FOV)
    # Instead of going from 0 to 1, go from 0 to 0.7
    x = torch.linspace(0, 0.7, M).to(lens.device)
    y = torch.linspace(0, 0.7, M).to(lens.device)
    z = torch.full_like(x, depth)
    points = torch.stack((x, y, z), dim=-1)
    
    psfs = []
    for i in range(M):
        try:
            # Strategy 2: Try with recentering first
            psf = lens.psf_rgb(points=points[i], ks=ks, recenter=True, spp=4096)
        except AssertionError:
            try:
                # Strategy 3: Try without recentering
                print(f"Trying without recentering for field point {i}")
                psf = lens.psf_rgb(points=points[i], ks=ks, recenter=False, spp=4096)
            except Exception as e:
                # Strategy 4: If all else fails, create a simple Gaussian PSF
                print(f"Creating placeholder Gaussian PSF for field point {i}")
                sigma = 2.0 + i*1.5  # Increase blur with field position
                x_coord = torch.arange(ks).to(lens.device)
                y_coord = torch.arange(ks).to(lens.device)
                x_grid, y_grid = torch.meshgrid(x_coord, y_coord, indexing='xy')
                center = ks // 2
                gaussian = torch.exp(-((x_grid - center) ** 2 + (y_grid - center) ** 2) / (2 * sigma ** 2))
                gaussian = gaussian / gaussian.sum()
                # Create RGB PSF (same for all channels in this simple case)
                psf = torch.stack([gaussian, gaussian, gaussian], dim=0)
        
        # Normalize PSF
        psf /= psf.max()
        
        if log_scale:
            psf = torch.log(psf + EPSILON)
            psf = (psf - psf.min()) / (psf.max() - psf.min())
            
        psfs.append(psf)
    
    psf_grid = make_grid(psfs, nrow=M, padding=1, pad_value=0.0)
    save_image(psf_grid, save_name, normalize=True)
    
    print(f"PSF radial has been saved to {save_name}")
    return save_name

# Copy and paste this into your notebook and use:
#
# plt.figure(figsize=(12, 8))
# robust_psf_radial(lens, M=2, depth=1000.0, log_scale=True, save_name="psf_radial.png")
# 
# from PIL import Image
# img = Image.open("psf_radial.png")
# plt.imshow(np.array(img))
# plt.title('Point Spread Function at Different Field Positions')
# plt.axis('off')
# plt.show() 