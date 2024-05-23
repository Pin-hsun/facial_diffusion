import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
try:
    from util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)
    from util.au_mask import facial_mask
except:
    from data.util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)
    from data.util.au_mask import facial_mask

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
GT_IMG_SIZE = {
    "CelebA": 128,
    "CelebAHQmask": 256,
    "AffectNet": 224
}
TRANSFORMED_IMG_SIZE = {
    "CelebA": 128,
    "CelebAHQmask": 256,
    "AffectNet": 256
}

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_npy_file(filename):
    return filename.endswith('.npy')

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def make_landmark_dataset(dir):
    if os.path.isfile(dir):
        landmark = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        landmark = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_npy_file(fname):
                    path = os.path.join(root, fname)
                    landmark.append(path)

    return landmark

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            # mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
            mask = bbox2mask(self.image_size, (h // 4 + 20, w // 4, h // 2 , w // 2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)

class InpaintAUDataset(data.Dataset):
    def __init__(self, dataset, data_root, GT, condition, landmark, mask_config={},
                 data_len=-1, image_size=256, loader=pil_loader):
        imgs = make_dataset(os.path.join(data_root, GT, "images"))
        cond_imgs = make_dataset(os.path.join(data_root, condition))
        lds = make_landmark_dataset(os.path.join(data_root, landmark,'landmark'))

        if landmark == 'transformed':
            self.ld_size = TRANSFORMED_IMG_SIZE[dataset]
        else:
            self.ld_size = GT_IMG_SIZE[dataset]

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
            self.lds = lds[:int(data_len)]
            self.cond_imgs = cond_imgs[:int(data_len)]
        else:
            self.imgs = imgs
            self.lds = lds
            self.cond_imgs = cond_imgs

        self.tfs = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.cond_tfs = transforms.Compose([
            transforms.Resize((image_size // 8, image_size // 8)),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.ld_tfs = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = [image_size, image_size]
        self.condition = condition
        self.dataset = dataset
        assert self.dataset in ['CelebA', 'CelebAHQmask', 'AffectNet']

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask(index)
        if self.mask_mode == 'au':
            mask = self.ld_tfs(mask)
        if self.condition == None:
            cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        else:
            cond_path = self.cond_imgs[index]
            cond_image = self.cond_tfs(self.loader(cond_path))
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, index=None):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox(img_shape=self.image_size,
                                                          max_bbox_shape=(self.image_size[0]//2, self.image_size[1]//2)))
        elif self.mask_mode == 'center':
            h, w = self.image_size
            if self.dataset == 'AffectNet':
                start, length = int(h//7), int(h//1.4)
                mask = bbox2mask(self.image_size, (start, start, length, length))
            elif self.dataset == 'CelebA' or self.dataset == 'CelebAHQmask':
                start, length = int(h//4), int(h//2)
                mask = bbox2mask(self.image_size, (start + 20, start, length , length))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            h, w = self.image_size
            if self.dataset == 'AffectNet':
                start, length = int(h//7), int(h//1.4)
                regular_mask = bbox2mask(self.image_size, (start, start, length, length))
            elif self.dataset == 'CelebA' or self.dataset == 'CelebAHQmask':
                start, length = int(h//4), int(h//2)
                regular_mask = bbox2mask(self.image_size, (start + 20, start, length , length))
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'au':
            ld = np.load(self.lds[index]).tolist()
            mask = facial_mask(ld, [self.ld_size, self.ld_size])
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

if __name__ == "__main__":
    import sys
    sys.path.append('/home/glory/projects/Palette-Image-to-Image-Diffusion-Models')
    dataset = InpaintAUDataset(dataset="AffectNet" ,data_root="/media/ExtHDD02/AffectNet/Neutral/test", GT="GT",
                    condition="transformed", landmark="GT", image_size= 256, mask_config={'mask_mode':'au'})
    print(len(dataset))
    for i in range(10):
        mask_img = dataset[i]['mask_image'].permute(1,2,0).numpy()
        cond_img = dataset[i]['cond_image'].permute(1,2,0).numpy()
        gt_img = dataset[i]['gt_image'].permute(1,2,0).numpy()
        mask_img = (mask_img + 1) / 2
        plt.imsave('tmp/' + str(i) + 'mask.jpg', mask_img)
        cond_img = (cond_img + 1) / 2
        plt.imsave('tmp/' + str(i) + 'cond.jpg', cond_img)
        plt.close()
        # plt.imsave('tmp/' + str(i) + 'gt.jpg', gt_img)
        # plt.close()
