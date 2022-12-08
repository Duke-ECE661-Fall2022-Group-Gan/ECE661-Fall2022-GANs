"""
Metrics for evaluate GAN generated samples
"""

import torch
import torch.nn.functional as F
from math import exp
from torch import nn
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy

"""
MS-SSIM
"""
def gaussian(window_size, sigma):
    """
    compute one-dimensional Gaussian Distribution vector
    """
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    """
    create a Gaussian kernel
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def get_ssim(img1, img2, window_size=11, window=None, val_range=None):
    """
    calculate the SSIM value
    use the equation(6) in Wang et al.,2003b
    :return: SSIM value and luminance, contrast, structure
    """
    # get the value range
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (N, C, H, W) = img1.size()
    if window is None:
        real_size = min(window_size, H, W)
        window = create_window(real_size, channel=C).to(img1.device)
    # means
    mu1 = F.conv2d(img1, window, padding=padd, groups=C)
    mu2 = F.conv2d(img2, window, padding=padd, groups=C)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    # variances(x) = Var(X)=E[X^2]-E[X]^2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=C) - mu1_mu2
    # parameters
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    C3 = C2/2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    # get results
    luminance = torch.mean((2*mu1_mu2+C1)/(mu1_sq+mu2_sq+C2))
    contrast = torch.mean(v1 / v2)
    structure = torch.mean((sigma12+C3)/(torch.sqrt(sigma1_sq)*torch.sqrt(sigma2_sq)+C3))
    ssim = torch.mean(((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2))
    return ssim, luminance, contrast, structure


def get_ms_ssim(img1, img2, window_size=11, val_range=None):
    """
    calculate the MS-SSIM value
    use the equation(7) in Wang et al.,2003b
    :return: MS-SSIM value
    """
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    M = weights.size()[0]
    m_lu = []
    m_cs = []
    m_st = []
    for i in range(M):
        ssim, lu,cs,st= get_ssim(img1, img2, window_size=window_size, val_range=val_range)
        m_lu.append(lu)
        m_cs.append(cs)
        m_st.append(st)
        # scale img
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    m_lu = torch.stack(m_lu)
    m_cs = torch.stack(m_cs)
    m_st = torch.stack(m_st)
    # calculate three components
    l = abs(m_lu[-1])**(weights[-1]*M)
    c = abs(m_cs) ** weights
    s = abs(m_st) ** weights
    res = l*torch.prod(c * s)  # multiply all elements
    return res


"""
Inception score
"""
def get_inception_score(imgs, batch_size=64, resize=True, splits=10):
    """
    calculate the Inception Score of the generated images
    :param imgs: Torch dataset of (3xHxW) numpy images normalized in [-1, 1]
    :param batch_size: batch size for feeding into inception_v3
    :param resize: resize to 229 if size is smaller than 229
    :param splits: # slits (different splits could lead to different IS)
    :return: IS, std
    """
    device = torch.device("cuda:0")
    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader and inception model
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    upsample = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)

    # Get predictions using pre-trained inception_v3 model
    def get_prediction(x):
        if resize:
            x = upsample(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    preds = np.zeros((N, 1000))
    for i, batch in enumerate(dataloader, 0):
        batch = batch[0].to(device)
        batch_size_i = batch.size()[0]
        preds[i * batch_size:i * batch_size + batch_size_i] = get_prediction(batch)

    # split the whole data into several parts
    split_scores = []
    for k in range(splits):
        split_part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        # compute marginal probability
        p_y = np.mean(split_part, axis=0)
        scores = []
        for i in range(split_part.shape[0]):
            # compute conditional probability
            p_yx = split_part[i, :]
            scores.append(entropy(p_yx, p_y))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)