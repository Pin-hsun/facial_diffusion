import os

from PIL import Image

# Define the directories containing the images
root = '/home/glory/projects/Palette-Image-to-Image-Diffusion-Models/experiments'
destination = 'compare_celebahq_smile256'
folder_paths = ['test_celebahq_smile256_au_YY',
                'test_celebahq_smile256_au_condTrans', 'test_celebahq_smile256_au_condGT']
folder_paths = [f'{root}/{folder}/compare' for folder in folder_paths]

for i in range(500):
    image_name = str(i) + '.jpg'  # The common filename of images you want to concatenate

    # Load the images
    images = [Image.open(f"{folder}/{image_name}") for folder in folder_paths]

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
