import PIL.Image as Image
import os
import numpy as np
import matplotlib.pyplot as plt


prj = 'test_celebahq_smile256_au_condGT'
text_dict = {
    'input': 'GT_X',
    'cond': 'GT_X',
    'mask': 'GT_X',
}
INPUT_PATH = f'/home/glory/projects/Palette-Image-to-Image-Diffusion-Models/experiments/{prj}/results/test/0'
OUTPUT_PATH = f'/home/glory/projects/Palette-Image-to-Image-Diffusion-Models/experiments/{prj}/compare'


def draw_comparison(GT_path: str,
                    img_path: str,
                    mask_path: str,
                    wfp: str,
                    text_dict: dict,
                    cond_path = None,
                    ):
    os.makedirs(os.path.dirname(wfp), exist_ok=True)

    GT = Image.open(GT_path)
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    GT = np.array(GT)
    img = np.array(img)
    mask = np.array(mask)
    if cond_path:
        cond_img = Image.open(cond_path)
        cond_img = cond_img.crop((2, 2, 258, 258))
        cond_img = np.array(cond_img)

    diff = np.abs(img - GT)
    # diff = img - GT
    diff = np.max(diff, axis=2)
    diff = np.where(diff > 100, 255, 0)

    ncols = 4
    nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    fontsize = 7

    i = 0
    axes[0, i].imshow(GT)
    axes[0, i].text(
        3, 40, "Input: " + text_dict['input'],
        fontsize=fontsize,
        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
    )
    i += 1

    axes[0, i].imshow(cond_img)
    axes[0, i].text(
        3, 40, "Condition: " + text_dict['cond'],
        fontsize=fontsize,
        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
    )
    i += 1

    axes[0, i].imshow(mask)
    axes[0, i].text(
        3, 40, "Mask: " + text_dict['mask'],
        fontsize=fontsize,
        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
    )
    i += 1

    axes[0, i].imshow(img)
    axes[0, i].text(
        3, 40, "output",
        fontsize=fontsize,
        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
    )

    # axes[0, i].imshow(GT)
    # axes[0, i].imshow(diff, alpha=0.3)
    # axes[0, i].text(
    #     3, 40, "diff",
    #     fontsize=fontsize,
    #     bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
    # )

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    # plt.show()

    plt.savefig(wfp, pad_inches=0, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    GT_paths = []
    image_paths = []
    mask_paths = []
    cond_path = []
    for dirpath, dirnames,filenames in os.walk(INPUT_PATH):
        for file in filenames:
            if file.endswith('.jpg') and file.startswith('Out'):
                image_filepath = os.path.join(dirpath, file)
                image_paths.append(image_filepath)
            if file.endswith('.jpg') and file.startswith('GT'):
                image_filepath = os.path.join(dirpath, file)
                GT_paths.append(image_filepath)
            if file.endswith('.jpg') and file.startswith('Mask'):
                image_filepath = os.path.join(dirpath, file)
                mask_paths.append(image_filepath)
            if file.endswith('.jpg') and file.startswith('Process'):
                image_filepath = os.path.join(dirpath, file)
                cond_path.append(image_filepath)
    image_paths.sort()
    GT_paths.sort()
    mask_paths.sort()
    cond_path.sort()
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # for i in range(len(image_paths)):

    for i in range(500):
        try:
            draw_comparison(GT_path = GT_paths[i],
                            img_path = image_paths[i],
                            mask_path = mask_paths[i],
                            wfp = OUTPUT_PATH + f'/{i}.jpg',
                            text_dict=text_dict,
                            cond_path = cond_path[i]
                            )
        except:
            print(f"Error in num {i} : {GT_paths[i]}")
            continue

    # for i in range(500):
    #     GT = INPUT_PATH + f'/GT_{i}.jpg'
    #     img = INPUT_PATH + f'/Out_{i}.jpg'
    #     mask = INPUT_PATH + f'/Mask_{i}.jpg'
    #     try:
    #         draw_comparison(GT, img, mask, OUTPUT_PATH + f'/{i}.jpg')
    #     except:
    #         print(f"Error in {i}")
    #         continue
