import torch


def ev2grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations。
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """

    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    with torch.no_grad():
        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32).flatten()
        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events[-1, 0]
        first_stamp = events[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
        ts = torch.from_numpy(events[:, 0])
        xs = torch.from_numpy(events[:, 1]).long()
        ys = torch.from_numpy(events[:, 2]).long()
        pols = torch.from_numpy(events[:, 3]).float()
        # pols[pols == 0] = 1  # polarity should be +1 / -1
        pols = 1
        #

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = pols * (1.0 - dts.float())

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        valid_indices &= xs < width
        valid_indices &= ys < height
        valid_indices &= xs >= 0
        valid_indices &= ys >= 0

        index_check = xs[valid_indices] + ys[valid_indices] * width + tis_long[valid_indices] * width * height
        invalid_mask = index_check > height * width
        if invalid_mask.sum() > 0:
            print(index_check[invalid_mask])

        voxel_grid.put_(index=xs[valid_indices] + ys[valid_indices] * width + tis_long[valid_indices] * width * height,
                        source=vals_left[valid_indices], accumulate=True)


        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid


def time_occupied_grid(events, num_bins, width, height):
    """
    Build a voxel grid, on pixel can only be occupied once ( the last appeared event in the grid)

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations。
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """

    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    with torch.no_grad():
        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32).flatten()
        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events[-1, 0]
        first_stamp = events[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        events[:, 0] = (events[:, 0] - first_stamp) / deltaT
        ts = torch.from_numpy(events[:, 0])
        xs = torch.from_numpy(events[:, 1]).long()
        ys = torch.from_numpy(events[:, 2]).long()
        pols = torch.from_numpy(events[:, 3]).float()
        # pols[pols == 0] = 1  # polarity should be +1 / -1
        pols = 1
        #

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = pols *  dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        valid_indices &= xs < width
        valid_indices &= ys < height
        valid_indices &= xs >= 0
        valid_indices &= ys >= 0

        index_check = xs[valid_indices] + ys[valid_indices] * width + tis_long[valid_indices] * width * height
        invalid_mask = index_check > height * width
        if invalid_mask.sum() > 0:
            print(index_check[invalid_mask])

        voxel_grid.put_(index=xs[valid_indices] + ys[valid_indices] * width + tis_long[valid_indices] * width * height,
                        source=vals_left[valid_indices], accumulate=False)


        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid


def events_to_timestamp_image(events, num_bins,
                              width, height, clip_out_of_range=True,
                              interpolation='bilinear', padding=False):
    """
    Method to generate the average timestamp images from 'Zhu19, Unsupervised Event-based Learning
    of Optical Flow, Depth, and Egomotion'. This method does not have known derivative.
    Parameters
    ----------
    xs : list of event x coordinates
    ys : list of event y coordinates
    ts : list of event timestamps
    ps : list of event polarities
    device : the device that the events are on
    sensor_size : the size of the event sensor/output voxels
    clip_out_of_range: if the events go beyond the desired image size,
       clip the events to fit into the image
    interpolation: which interpolation to use. Options=None,'bilinear'
    padding: if bilinear interpolation, allow padding the image by 1 to allow events to fit:
    Returns
    -------
    img_pos: timestamp image of the positive events
    img_neg: timestamp image of the negative events
    """

    # nomalize the timestamp to avoid double precision issue
    ts = events[:, 0]- events[0, 0]

    xn = events[:, 1]
    yn = events[:, 2]
    pn = events[:, 3]

    sensor_size = (height, width)

    xt, yt, ts, pt = torch.from_numpy(xn), torch.from_numpy(yn), torch.from_numpy(ts), torch.from_numpy(pn)
    # xs, ys, ts, ps = xt, yt, ts, pt
    xs, ys, ts, ps = xt.float(), yt.float(), ts.float(), pt.float()
    zero_v = torch.tensor([0.])
    ones_v = torch.tensor([1.])

    if padding:
        img_size = (sensor_size[0] + 1, sensor_size[1] + 1)
    else:
        img_size = sensor_size

    mask = torch.ones(xs.size())
    if clip_out_of_range:
        clipx = img_size[1] if interpolation is None and padding == False else img_size[1] - 1
        clipy = img_size[0] if interpolation is None and padding == False else img_size[0] - 1
        mask = torch.where(xs >= clipx, zero_v, ones_v) * torch.where(ys >= clipy, zero_v, ones_v)

    #  diminish the polarity of the event
    pn[pn<=0] = 1

    pos_events_mask = torch.where(ps > 0, ones_v, zero_v)
    # neg_events_mask = torch.where(ps <= 0, ones_v, zero_v)
    normalized_ts = (ts - ts[0]) / (ts[-1] - ts[0] + 1e-6)
    pxs = xs.floor()
    pys = ys.floor()
    dxs = xs - pxs
    dys = ys - pys
    pxs = (pxs * mask).long()
    pys = (pys * mask).long()
    masked_ps = ps * mask

    pos_weights = normalized_ts * pos_events_mask
    # neg_weights = normalized_ts * neg_events_mask
    img_pos = torch.zeros(img_size)
    img_pos_cnt = torch.ones(img_size)
    img_neg = torch.zeros(img_size)
    img_neg_cnt = torch.ones(img_size)

    interpolate_to_image(pxs, pys, dxs, dys, pos_weights, img_pos)
    # interpolate_to_image(pxs, pys, dxs, dys, pos_events_mask, img_pos_cnt)
    # interpolate_to_image(pxs, pys, dxs, dys, neg_weights, img_neg)
    # interpolate_to_image(pxs, pys, dxs, dys, neg_events_mask, img_neg_cnt)

    img_pos, img_pos_cnt = img_pos.unsqueeze(0), img_pos_cnt

    # delete the wierd accumulation in the (0,0) and it surroundings
    img_pos[:, 0:2, 0:2] = 0

    # img_pos_cnt[img_pos_cnt == 0] = 1
    # img_neg, img_neg_cnt = img_neg.unsqueeze(0), img_neg_cnt
    # img_neg_cnt[img_neg_cnt == 0] = 1
    return img_pos #, img_neg  # /img_pos_cnt, img_neg/img_neg_cnt


def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    """
    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)
    return img