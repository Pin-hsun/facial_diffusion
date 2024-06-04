import glob
import os
from PIL import Image

# root = '/home/gloria/projects/facial_diffusion/experiments'
# dataset = 'CelebAHQmask'
# destination = 'compare_celebahq_hybrid'
# folder_paths = ['test_celebahq_hybrid_XY_condTrans', 'test_celebahq_hybrid_XY_condGT']
# folder_paths = [f'{root}/{dataset}/{folder}/compare' for folder in folder_paths]
# img_paths = [glob.glob(folder +  '/*') for folder in folder_paths]
#
# for i in range(500):
#     # image_names = []
#
#     # Load the images
#     # images = [Image.open(f"{folder}/{image_name}") for folder in folder_paths]
#     images = [Image.open(img_paths[folder][i]) for folder in range(len(folder_paths))]
#
#     # Get total width and maximum height (for horizontal concatenation)
#     max_width = max(img.width for img in images)
#     total_height = sum(img.height for img in images)
#
#     # Create a new blank image with the correct size
#     concatenated_image = Image.new('RGB', (max_width, total_height))
#
#     # Paste the images into the new image
#     y_offset = 0
#     for img in images:
#         concatenated_image.paste(img, (0, y_offset))
#         y_offset += img.height
#
#     # Save or display the concatenated image
#     os.makedirs(os.path.join(root, destination), exist_ok=True)
#     concatenated_image.save(os.path.join(root, destination, str(i) + '.jpg'))

root = '/home/gloria/projects/facial_diffusion/experiments'
destination = 'all'
# dataset = 'CelebAHQmask'
# folder_paths = ['test_celebahq_hybrid_XY_condGT', 'test_celebahq_hybrid_XY_condTrans',
#                 'test_celebahq_smile256_au_condGT', 'test_celebahq_smile256_au_condTrans']
# dataset = 'AffectNet'
# folder_paths = ['test_affectnet_hybrid_down4_XY_condGT', 'test_affectnet_hybrid_down4_XY_condTrans',
#                 'test_affectnet_au_XY_condGT', 'test_affectnet_au_XY_condTrans']
dataset = 'CelebA'
folder_paths = ['test_celeba_smile128_hybrid_nosmile_conGT', 'test_celeba_smile128_hybrid_nosmile_conTrans',
                'test_celeba_smile128_au_nosmile_conGT', 'test_celeba_smile128_au_nosmile_conTrans']
GT_path = f'{root}/{dataset}/{folder_paths[0]}/GT'
folder_paths = [GT_path] + [f'{root}/{dataset}/{folder}/Out' for folder in folder_paths]
img_paths = [glob.glob(folder +  '/*') for folder in folder_paths]
print(folder_paths)

for i in range(500):
    images = [Image.open(img_paths[folder][i]) for folder in range(len(folder_paths))]

    # Get total width and maximum height (for horizontal concatenation)
    sum_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new blank image with the correct size
    concatenated_image = Image.new('RGB', (sum_width, max_height))

    # Paste the images into the new image
    x_offset = 0
    for img in images:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save or display the concatenated image
    os.makedirs(os.path.join(root, dataset, destination), exist_ok=True)
    concatenated_image.save(os.path.join(root, dataset, destination, str(i) + '.jpg'))
    # concatenated_image.show()
