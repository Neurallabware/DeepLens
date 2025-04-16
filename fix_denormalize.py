import torch

def safe_denormalize(tensor):
    """Safely denormalize a tensor by ensuring it has a batch dimension.
    Works with both 3D and 4D tensors.
    """
    original_dim = tensor.dim()
    if original_dim == 3:
        # Add batch dimension if missing
        tensor = tensor.unsqueeze(0)
    
    # Now tensor should be 4D [B, C, H, W]
    mean = torch.zeros_like(tensor)
    std = torch.zeros_like(tensor)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    
    result = tensor * std + mean
    
    # Return in the original dimension format
    if original_dim == 3:
        result = result.squeeze(0)
        
    return result

# Create a helper function for visualizing images in the notebook
def vis_sample(img_org, img_render, img_rec, loss=None, epoch=None, batch=None):
    """
    Safely visualize sample images using matplotlib.
    Works with both 3D and 4D tensors.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    if img_org.dim() == 4:
        # If it's a batch, take the first image
        img = safe_denormalize(img_org[0])
    else:
        # Otherwise use as is
        img = safe_denormalize(img_org)
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    plt.title('Original')
    plt.axis('off')
    
    # Rendered image
    plt.subplot(132)
    if img_render.dim() == 4:
        img = safe_denormalize(img_render[0])
    else:
        img = safe_denormalize(img_render)
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    plt.title('Rendered')
    plt.axis('off')
    
    # Recovered image
    plt.subplot(133)
    if img_rec.dim() == 4:
        img = safe_denormalize(img_rec[0])
    else:
        img = safe_denormalize(img_rec)
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    plt.title('Recovered')
    plt.axis('off')
    
    if loss is not None and epoch is not None and batch is not None:
        plt.suptitle(f"Epoch {epoch+1}, Batch {batch}, Loss: {loss:.4f}")
        
    plt.tight_layout()
    plt.show()

print("Created safe_denormalize and vis_sample functions for handling tensor dimension issues") 