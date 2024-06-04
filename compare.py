import shutil
import PIL.Image as Image
import os
import numpy as np
import matplotlib.pyplot as plt
from flip_rate import ResNetClassifier, flip_rate
from cleanfid import fid
from models.metric import inception_score
from core.base_dataset import BaseDataset


def draw_comparison(GT_path: str,
                    img_path: str,
                    mask_path: str,
                    wfp: str,
                    text_dict: dict,
                    img_size,
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
        if img_size == 128:
            cond_img = cond_img.crop((2, 2, 130, 130))
        elif img_size == 256:
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
        3, 15, "Input: " + text_dict['input'],
        fontsize=fontsize,
        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
    )
    i += 1

    axes[0, i].imshow(cond_img)
    axes[0, i].text(
        3, 15, "Condition: " + text_dict['cond'],
        fontsize=fontsize,
        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
    )
    i += 1

    axes[0, i].imshow(mask)
    axes[0, i].text(
        3, 15, "Mask: " + text_dict['mask'],
        fontsize=fontsize,
        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
    )
    i += 1

    axes[0, i].imshow(img)
    axes[0, i].text(
        3, 15, "output",
        fontsize=fontsize,
        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
    )

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(wfp, pad_inches=0, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    dataset = "CelebAHQmask"
    img_size = 256
    root = "/home/gloria/projects/facial_diffusion/experiments"
    prj = 'test_celebahq_hybrid_XY_condTrans'
    text_dict = {
        'input': 'Ox',
        'cond': 'Gxy',
        'mask': 'center',
    }

    INPUT_PATH = os.path.join(root, prj, 'results/test/0')
    OUTPUT_PATH = os.path.join(root, prj, 'compare')
    os.makedirs(os.path.join(root, prj, "Out"), exist_ok=True)
    os.makedirs(os.path.join(root, prj, "GT"), exist_ok=True)
    GT_paths = []
    image_paths = []
    mask_paths = []
    cond_path = []
    for dirpath, dirnames,filenames in os.walk(INPUT_PATH):
        for file in filenames:
            if file.endswith('.jpg') and file.startswith('Out'):
                name = file.split('_')[-1]
                image_filepath = os.path.join(dirpath, file)
                shutil.copy(image_filepath, os.path.join(root, prj, "Out", name))
                image_paths.append(image_filepath)
            if file.endswith('.jpg') and file.startswith('GT'):
                name = file.split('_')[-1]
                image_filepath = os.path.join(dirpath, file)
                GT_paths.append(image_filepath)
                shutil.copy(image_filepath, os.path.join(root, prj, "GT", name))
            if file.endswith('.jpg') and file.startswith('Mask'):
                image_filepath = os.path.join(dirpath, file)
                mask_paths.append(image_filepath)
            if file.endswith('.jpg') and file.startswith('Cond'):
                image_filepath = os.path.join(dirpath, file)
                cond_path.append(image_filepath)
    image_paths.sort()
    GT_paths.sort()
    mask_paths.sort()
    cond_path.sort()
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # for i in range(len(image_paths)):

    flip_rate = flip_rate(dataset, image_paths)
    fid_score = fid.compute_fid(os.path.join(root, prj, "GT"), os.path.join(root, prj, "Out"))
    # is_mean, is_std = inception_score(BaseDataset(os.path.join(root, prj, "Out")), cuda=True, batch_size=8, resize=True, splits=10)

    print("Flip rate:", flip_rate)
    print('FID: {}'.format(fid_score))
    # print('IS:{} {}'.format(is_mean, is_std))

    for i in range(500):
        id = GT_paths[i].split('/')[-1].split('_')[-1]
        draw_comparison(GT_path = GT_paths[i],
                        img_path = image_paths[i],
                        mask_path = mask_paths[i],
                        wfp = OUTPUT_PATH + '/' + id,
                        text_dict=text_dict,
                        img_size=img_size,
                        cond_path = cond_path[i]
                        )
