import glob
import os

from PIL import Image

# Define the directories containing the images
root = '/home/gloria/projects/facial_diffusion/experiments'
destination = 'compare_affectnet_au'
folder_paths = ['test_affectnet_au_YY_condGT_maskAU','test_affectnet_au_YY_condTrans_maskAU',
                'test_affectnet_au_XY_condGT_maskAU', 'test_affectnet_au_XY_condTrans_maskAU']
folder_paths = [f'{root}/{folder}/compare' for folder in folder_paths]
img_paths = [glob.glob(folder +  '/*') for folder in folder_paths]

for i in range(500):
    # image_names = []

    # Load the images
    # images = [Image.open(f"{folder}/{image_name}") for folder in folder_paths]
    images = [Image.open(img_paths[folder][i])for folder in range(4)]

    # Get total width and maximum height (for horizontal concatenation)
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)

    # Create a new blank image with the correct size
    concatenated_image = Image.new('RGB', (max_width, total_height))

    # Paste the images into the new image
    y_offset = 0
    for img in images:
        concatenated_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Save or display the concatenated image
    os.makedirs(os.path.join(root, destination), exist_ok=True)
    concatenated_image.save(os.path.join(root, destination, str(i) + '.jpg'))
    # concatenated_image.show()
