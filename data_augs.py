import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from TransformLayer import ColorJitterLayer


def random_crop(imgs, out=84):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    h, w, c = imgs.shape
    n = 1
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    w11 = int(w1)
    h11 = int(h1)
    cropped = np.empty((out, out,c), dtype=imgs.dtype)
        
    cropped = imgs[h11:h11 + out, w11:w11 + out,:]
    return cropped


def grayscale(imgs):
    # imgs: b x c x h x w
    #device = imgs.device
    h, w, c = imgs.shape
    b = 1
    frames = c // 3
    out = imgs.copy()
    
    #imgs = imgs.view([frames,3,h,w])
    imgs = imgs[:, :, 0] * 0.2989 + imgs[:, :, 1] * 0.587 + imgs[:, :, 2] * 0.114
    out[:,:,0] = imgs
    out[:, :, 1] = imgs
    out[:, :, 2] = imgs

    

    # assert len(imgs.shape) == 3, imgs.shape
    #imgs = imgs[:, :, None, :, :]
    #imgs = imgs * torch.ones([1, 1, 3, 1, 1], dtype=imgs.dtype).float() # broadcast tiling
    return out

def random_grayscale(images,p=.3):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: cpu or cuda
        returns torch.tensor
    """
    #device = images.device
    #in_type = images.type()
    images = images * 255.

    # images: [B, C, H, W]
    h, w, channels = images.shape
    bs = 1
    #images = images.to(device)
    gray_images = grayscale(images)

    return gray_images

# random cutout
# TODO: should mask this 

def random_cutout(imgs, min_cut=10,max_cut=30):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        min / max cut: int, min / max size of cutout 
        returns np.array
    """

    h, w, c = imgs.shape
    n = 1
    w1 = np.random.randint(min_cut, max_cut, n)
    w11 = int(w1)
    h1 = np.random.randint(min_cut, max_cut, n)
    h11 = int(h1)
    cutouts = np.empty((h, w,c), dtype=imgs.dtype)
    cutouts[h11:h11 + h11, w11:w11 + w11,0] = 0

    return cutouts

def random_cutout_color(imgs, min_cut=10,max_cut=30):
    """
        args:
        imgs: shape (B,C,H,W)
        out: output size (e.g. 84)
    """

    h, w, c = imgs.shape
    n = 1
    w1 = np.random.randint(min_cut, max_cut, n)
    w11 = int(w1)
    h1 = np.random.randint(min_cut, max_cut, n)
    h11 = int(h1)
    #cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    rand_box = np.random.randint(0, 255, size=(c,n)) / 255.

    cut_img = imgs.copy()
    a = rand_box.reshape(-1,1,1)
    b =  cut_img[h11:h11 + h11, w11:w11 + w11].shape[:2]
        # add random box
    d = cut_img[h11:h11 + h11, w11:w11 + w11, :]
    temp = np.tile(
            rand_box.reshape(-1,1,1),
    cut_img[h11:h11 + h11, w11:w11 + w11].shape[:2])
    cut_img[h11:h11 + h11, w11:w11 + w11, 0] =temp[0,:,:]
    cut_img[h11:h11 + h11, w11:w11 + w11, 1] = temp[1, :, :]
    cut_img[h11:h11 + h11, w11:w11 + w11, 2] = temp[2, :, :]
    #cutouts[i] = cut_img
    return cut_img

# random flip

def random_flip(images,p=.2):
    
        #args:
        #imgs: torch.tensor shape (B,C,H,W)
        #device: cpu or gpu, 
        #p: prob of applying aug,
        #returns torch.tensor
  
    # images: [B, C, H, W]

    h, w, channels = images.shape
    bs = 1
    
    #images = images.to(device)

    flipped_images = np.flip(images)
    
    rnd = np.random.uniform(0., 1., size=(channels,))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1] #// 3
    images = images.view(*flipped_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
    
    mask = mask.type(images.dtype)
    mask = mask[:, :, None, None]
    
    out = mask * flipped_images + (1 - mask) * images

    out = out.view([ h, w,bs])
    return out

# random rotation

def random_rotation(images,p=.3):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: str, cpu or gpu, 
        p: float, prob of applying aug,
        returns torch.tensor
    """
    #device = images.device
    # images: [B, C, H, W]
    #bs, channels, h, w = images.shape
    h, w, channels = images.shape
    bs = 1
    #images = images.to(device)

    rot90_images = np.rot90(images,1,[0,1])
    rot180_images = np.rot90(images,2,[0,1])
    rot270_images = np.rot90(images,3,[0,1])
    
    rnd = np.random.uniform(0., 1., size=(256,))
    rnd_rot = np.random.randint(1, 4, size=(256,))
    mask = rnd <= p
    mask = rnd_rot * mask
    mask = torch.from_numpy(mask)
    
    frames = 256
    masks = [torch.zeros_like(mask) for _ in range(4)]
    for i,m in enumerate(masks):
        m[torch.where(mask==i)] = 1
        m = m[:, None] * torch.ones([frames])#.type(mask.dtype).type(images.dtype)
        m = m[:,:,None]*torch.ones([3])
        masks[i] = m


    
    
    out = masks[0] * images + masks[1] * rot90_images + masks[2] * rot180_images + masks[3] * rot270_images

    out = out.view([h, w, 3])
    return out


# random color

    






def random_translate(imgs, size=300, return_random_idxs=False, h1s=None, w1s=None):
    h, w, c = imgs.shape
    n = 1
    assert size >= h and size >= w
    outs = np.zeros((size, size,c ), dtype=imgs.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    h1s = int(h1s)
    w1s = int(w1s)
    outs[h1s:h1s + h, w1s:w1s + w,:] = imgs
    if return_random_idxs:  # So can do the same to another set of imgs.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs


def no_aug(x):
    return x


if __name__ == '__main__':
    import time 
    from tabulate import tabulate
    def now():
        return time.time()
    def secs(t):
        s = now() - t
        tot = round((1e5 * s)/60,1)
        return round(s,3),tot

    x = np.load('data_sample.npy',allow_pickle=True)
    x = np.concatenate([x,x,x],1)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.from_numpy(x)
    x = x.float() / 255.

    # crop
    t = now()
    random_crop(x.cpu().numpy(),64)
    s1,tot1 = secs(t)
    # grayscale 
    t = now()
    random_grayscale(x,p=.5)
    s2,tot2 = secs(t)
    # normal cutout 
    t = now()
    random_cutout(x.cpu().numpy(),10,30)
    s3,tot3 = secs(t)
    # color cutout 
    t = now()
    random_cutout_color(x.cpu().numpy(),10,30)
    s4,tot4 = secs(t)
    # flip 
    t = now()
    random_flip(x,p=.5)
    s5,tot5 = secs(t)
    # rotate 
    t = now()
    random_rotation(x,p=.5)
    s6,tot6 = secs(t)
    # rand conv 
    t = now()
    random_convolution(x)
    s7,tot7 = secs(t)
    # rand color jitter 
    t = now()
    random_color_jitter(x)
    s8,tot8 = secs(t)
    
    print(tabulate([['Crop', s1,tot1], 
                    ['Grayscale', s2,tot2], 
                    ['Normal Cutout', s3,tot3], 
                    ['Color Cutout', s4,tot4], 
                    ['Flip', s5,tot5], 
                    ['Rotate', s6,tot6], 
                    ['Rand Conv', s7,tot7], 
                    ['Color Jitter', s8,tot8]], 
                    headers=['Data Aug', 'Time / batch (secs)', 'Time / 100k steps (mins)']))

