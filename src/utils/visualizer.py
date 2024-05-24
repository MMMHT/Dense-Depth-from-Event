import cv2
import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm


def tensor_to_disparity_image(tensor_data):


    if len(tensor_data.size()) > 4:
        raise  NotImplementedError
    if len(tensor_data.size()) == 4:
        tensor_data = tensor_data[0].squeeze(0)
    elif len(tensor_data.size()) == 3:
        tensor_data = tensor_data[0]
    elif len(tensor_data.size()) == 2:
        pass
    # print(tensor_data.size())
    assert len(tensor_data.size()) == 2
    assert (tensor_data >= 0.0).all().item()

    disparity_image = Image.fromarray(np.asarray(tensor_data * 256.0).astype(np.uint16))

    return disparity_image

def tensor_to_flow_image(flow ):
    flow[torch.isnan(flow)] = 0
    x_flow, y_flow = flow[0, 0], flow[0, 1]
    max_magnitude = max(torch.max(x_flow).item(), torch.max(y_flow).item())
    x_flow_normalized = (x_flow / max_magnitude) * 255
    y_flow_normalized = (y_flow / max_magnitude) * 255
    angle = np.arctan2(y_flow_normalized.cpu().numpy(), x_flow_normalized.cpu().numpy())
    magnitude = np.sqrt(np.square(x_flow_normalized.cpu().numpy()) + np.square(y_flow_normalized.cpu().numpy()))
    hsv = np.zeros((x_flow.shape[0], x_flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (angle + np.pi) * 180 / (2 * np.pi)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    flow_image = Image.fromarray(flow_image)
    return flow_image
    # cv2.imshow('gt_flow', flow_image)


def tensor_to_disparity_image_paper_vis(tensor_data, pathsave):
    # make sure the dim is 3
    if len(tensor_data.size()) == 4:
        tensor_data = tensor_data[0].squeeze(0)
    elif len(tensor_data.size()) == 3:
        pass
    elif len(tensor_data.size()) == 2:
        tensor_data = tensor_data.unsqueeze(0)
    assert len(tensor_data.size()) == 3
    assert (tensor_data >= 0.0).all().item()

    assert isinstance(tensor_data, torch.Tensor)

    valid_mask = tensor_data > 0
    valid_mask = valid_mask.permute(1, 2, 0).repeat(1, 1, 3).numpy()
    tensor_data = (tensor_data - tensor_data.min()) / (tensor_data.max() - tensor_data.min()) * 255
    img_mapped = cv2.applyColorMap(tensor_data[0].numpy().astype('uint8'), cv2.COLORMAP_MAGMA)
    img_mapped[~valid_mask] = 200

    cv2.imwrite(pathsave, img_mapped.astype('uint8'))


def tensor_to_disparity_magma_image(tensor_data, vmax=None, mask=None):
    if len(tensor_data.size()) == 4:
        tensor_data = tensor_data[0].squeeze(0)
    elif len(tensor_data.size()) == 3:
        tensor_data = tensor_data[0]
    elif len(tensor_data.size()) == 2:
        pass
    assert len(tensor_data.size()) == 2
    assert (tensor_data >= 0.0).all().item()

    numpy_data = np.asarray(tensor_data)

    if vmax is not None:
        numpy_data = numpy_data * 255 / vmax
        numpy_data = np.clip(numpy_data, 0, 255)

    zero_mask = numpy_data==0

    numpy_data = numpy_data.astype(np.uint8)
    numpy_data_magma = cv2.applyColorMap(numpy_data, cv2.COLORMAP_MAGMA)
    numpy_data_magma = cv2.cvtColor(numpy_data_magma, cv2.COLOR_BGR2RGB)

    numpy_data_magma[zero_mask] = [0 ,0 ,0]

    if mask is not None:
        assert tensor_data.size() == mask.size()
        numpy_data_magma[~mask] = [255, 255, 255]

    disparity_image = Image.fromarray(numpy_data_magma)

    return disparity_image
