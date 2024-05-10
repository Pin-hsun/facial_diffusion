from PIL import Image

# Define the directories containing the images
root = '/media/ziyi/glory/logs_pin/diffusion'
folder_paths = ['celeba_smileAU_onnosmile', 'celeba_smileAU_onXY(240418_acgan)_orimask',
                'celeba_smileAU_onXY(240418_acgan)_Neumask', 'celeba_smileAU_onXY(240418_acgan)_XYmask']
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
    concatenated_image.save(root+'/compare/'+ str(i) + '.jpg')
    # concatenated_image.show()
