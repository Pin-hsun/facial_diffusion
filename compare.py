import PIL.Image as Image
import os
import numpy as np
import matplotlib.pyplot as plt


prj = 'celeba_smileAU_onnosmile'
INPUT_PATH = f'/media/ziyi/glory/logs_pin/diffusion/{prj}/results/test/0'
OUTPUT_PATH = f'/media/ziyi/glory/logs_pin/diffusion/{prj}/compare'


def draw_comparison(GT_path: str,
                    img_path: str,
                    mask_path: str,
                    wfp: str,
                    ):
    os.makedirs(os.path.dirname(wfp), exist_ok=True)

    GT = Image.open(GT_path)
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    GT = np.array(GT)
    img = np.array(img)
    mask = np.array(mask)

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
        3, 40, "GT",
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
    i += 1

    axes[0, i].imshow(mask)
    axes[0, i].text(
        3, 40, "mask",
        fontsize=fontsize,
        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
    )
    i += 1

    axes[0, i].imshow(GT)
    axes[0, i].imshow(diff, alpha=0.3)
    axes[0, i].text(
        3, 40, "diff",
        fontsize=fontsize,
        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
    )

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
    image_paths.sort()
    GT_paths.sort()
    mask_paths.sort()
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # for i in range(len(image_paths)):
    for i in range(500):
        draw_comparison(GT_paths[i], image_paths[i], mask_paths[i] ,OUTPUT_PATH + f'/{i}.jpg')
