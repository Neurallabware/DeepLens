#!/usr/bin/env python3
"""
Fix for the DeepLens PSF radial function that uses 'center' instead of 'recenter'.
"""

def fixed_psf_radial(lens, M=3, depth=float('inf'), ks=101, log_scale=True, save_name="psf_radial.png"):
    """Modified version of draw_psf_radial that fixes the parameter mismatch.
    
    Args:
        lens: The lens object
        M: Number of field positions (default: 3)
        depth: Object depth (default: infinity)
        ks: Kernel size for PSF (default: 101)
        log_scale: Whether to use log scale (default: True)
        save_name: Filename to save the PSF image
    """
    import torch
    from torchvision.utils import make_grid, save_image
    from deeplens.optics.basics import EPSILON
    
    x = torch.linspace(0, 1, M).to(lens.device)
    y = torch.linspace(0, 1, M).to(lens.device)
    z = torch.full_like(x, depth)
    points = torch.stack((x, y, z), dim=-1)
    
    psfs = []
    for i in range(M):
        # Fix: Use recenter instead of center
        psf = lens.psf_rgb(points=points[i], ks=ks, recenter=True, spp=4096)
        psf /= psf.max()
        
        if log_scale:
            psf = torch.log(psf + EPSILON)
            psf = (psf - psf.min()) / (psf.max() - psf.min())
            
        psfs.append(psf)
    
    psf_grid = make_grid(psfs, nrow=M, padding=1, pad_value=0.0)
    save_image(psf_grid, save_name, normalize=True)
    
    print(f"PSF radial has been saved to {save_name}")
    return save_name

if __name__ == "__main__":
    print("Run this in your notebook:")
    print("""
# Add this function to your notebook
def fixed_psf_radial(lens, M=3, depth=float('inf'), ks=101, log_scale=True, save_name="psf_radial.png"):
    import torch
    from torchvision.utils import make_grid, save_image
    from deeplens.optics.basics import EPSILON
    
    x = torch.linspace(0, 1, M).to(lens.device)
    y = torch.linspace(0, 1, M).to(lens.device)
    z = torch.full_like(x, depth)
    points = torch.stack((x, y, z), dim=-1)
    
    psfs = []
    for i in range(M):
        # Fix: Use recenter instead of center
        psf = lens.psf_rgb(points=points[i], ks=ks, recenter=True, spp=4096)
        psf /= psf.max()
        
        if log_scale:
            psf = torch.log(psf + EPSILON)
            psf = (psf - psf.min()) / (psf.max() - psf.min())
            
        psfs.append(psf)
    
    psf_grid = make_grid(psfs, nrow=M, padding=1, pad_value=0.0)
    save_image(psf_grid, save_name, normalize=True)
    
    # Display the image
    from PIL import Image
    import matplotlib.pyplot as plt
    
    img = Image.open(save_name)
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title('Point Spread Function at Different Field Positions')
    plt.axis('off')
    plt.show()
    
# Use the fixed function
plt.figure(figsize=(12, 8))
fixed_psf_radial(lens, M=3, depth=float('inf'), log_scale=True, save_name="psf_radial.png")
"""
    ) 