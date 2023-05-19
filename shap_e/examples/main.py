import os
import torch
import trimesh

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_mesh
from shap_e.util.image_util import load_image
from shap_e.rendering.ply_util import write_ply

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
model = load_model('image300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

image = load_image("test1.png")
# image2 = load_image("test2.png")
# image3 = load_image("test3.png")
# image = [image1, image2, image3]

# 이미지 미리보기
# import numpy as np
# from PIL import Image

# image_np = np.array(image)
# Image.fromarray(image_np)

output_folder = "plys"  # 원하는 폴더 이름으로 변경
os.makedirs(output_folder, exist_ok=True)  # 폴더가 없으면 생성

guidance_scale = 3.3

latents = sample_latents(
    batch_size=1,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(images=image),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
    )

mesh = decode_latent_mesh(xm, latents).tri_mesh()

# Get the vertex colors
red_channel = mesh.vertex_channels['R']
green_channel = mesh.vertex_channels['G']
blue_channel = mesh.vertex_channels['B']

mesh = trimesh.Trimesh(vertices=mesh.verts, faces=mesh.faces)

# Add the vertex colors to the PLY file
mesh.visual.vertex_colors = np.column_stack((red_channel, green_channel, blue_channel))

file_name = f'scale_{guidance_scale}.ply'
file_path = os.path.join(output_folder, file_name)
mesh.export(file_path)