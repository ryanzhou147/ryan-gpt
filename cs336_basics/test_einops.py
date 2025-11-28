# import torch
# import matplotlib.pyplot as plt
# from skimage import data
# import numpy as np
# from einops import rearrange

# # Load a sample RGB image and downsample
# orig_img = data.astronaut()  # (512, 512, 3)
# orig_img_small = torch.tensor(orig_img[::8, ::8, :], dtype=torch.float32)  # 32x32
# h, w, c = orig_img_small.shape
# channels_last = orig_img_small.unsqueeze(0)  # (1, 32, 32, 3)

# # Random linear transformation
# B = torch.randn(h*w, h*w)

# # Use einops to flatten and move channels first
# channels_first_flat = rearrange(channels_last, "b h w c -> b c (h w)")

# # Apply linear transformation
# channels_first_flat_transformed = channels_first_flat @ B.T

# # Rearrange back to original shape
# channels_last_transformed = rearrange(channels_first_flat_transformed, "b c (h w) -> b h w c", h=h, w=w)

# # Visualize original vs scrambled
# plt.figure(figsize=(8,4))

# plt.subplot(1,2,1)
# plt.imshow(orig_img_small.numpy().astype(np.uint8))
# plt.title("Original Image")
# plt.axis("off")

# plt.subplot(1,2,2)
# scrambled = channels_last_transformed[0].detach().numpy()
# scrambled = np.clip(scrambled, 0, 255)
# plt.imshow(scrambled.astype(np.uint8))
# plt.title("Scrambled Image (after B.T)")
# plt.axis("off")

# plt.show()
