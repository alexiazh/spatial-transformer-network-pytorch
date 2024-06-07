import torch

"""
Input
--------
- in_feature_map: Tensor of shape (B, H, W, C), 
                or [num_batch, height, width, num_channels].
Input image or output feature map from the previous layer.

- theta: Tensor of shape (B, 6). 
Transformation matrix. Output of the localization network.

- out_dims: Tuple of two ints (H, W).
Dimensions of the network output.

Output
--------
- out_feature_map: Tensor of shape (B, H, W, C).
Transformed input feature map. 

Notes
--------
To initialize the network to the identity transform, init `theta` to:
    identity = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    identity = identity.flatten()
    theta = tf.Variable(initial_value=identity)

References
--------
[1]  'Spatial Transformer Networks', Jaderberg et. al. https://arxiv.org/abs/1506.02025
[2]  https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py
[3]  https://github.com/LijieFan/tvnet/blob/master/spatial_transformer.py
[4]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
"""
class SpatialTransformer():

    def __init__(self, out_dims=None):
        self.out_dims = out_dims

    def forward(self, in_feature_map, theta):
        # get input dimensions
        B = in_feature_map.size(0)
        H = in_feature_map.size(1)
        W = in_feature_map.size(2)

        # reshape theta to (B, 2, 3)
        theta = theta.view(B, 2, 3)

        # generate grids of same size or upsample/downsample if specified
        if self.out_dims:
            out_H = self.out_dims[0]
            out_W = self.out_dims[1]
        else:
            out_H = H
            out_W = W
        
        batch_grids = self.affine_grid_generator(out_H, out_W, theta)

        x_s = batch_grids[:, 0, :, :]
        y_s = batch_grids[:, 1, :, :]

        # sample input with grid to get output
        out_feature_map = self.bilinear_sampler(in_feature_map, x_s, y_s)

        return out_feature_map


    def _get_pixel_value(self, img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a 4D tensor image.

        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B * H * W)
        - y: flattened tensor of shape (B * H * W)

        Returns
        -------
        - output: tensor of shape (B, H, W, C)
        """
        B, H, W, C = img.size()
        batch_idx = torch.arange(0, B).view(B, 1, 1).expand(B, H, W)
        indices = torch.stack([batch_idx, y, x], dim=3)
        return img[indices[:, :, :, 0], indices[:, :, :, 1], indices[:, :, :, 2], :]


    def affine_grid_generator(self, h, w, theta):
        """
        Performs affine transformation to generate a grid of coordinates
        in the input image corresponding to each pixel of the output image.

        Input
        --------
        - h: torch.float32
        Output height of gridl; used to downsample or upsample.

        - w: torch.float32
        Output width of grid; used to downsample or upsample.

        - theta: Tensor of shape (B, 2, 3).
        Affine transform matrices. 
        For each image in the batch, we use 6 theta parameters of 
        shape (2, 3) to define the transformation T.

        Returns
        --------
        - normalized grid (-1, 1) of shape (B, 2, H, W).
        The 2nd dim represents a pair of coordinates: (x, y), 
        which are the sampling points of the original image 
        for each point in the target image.
        """
        B = theta.size(0)

        # create normalized 2D grid
        x = torch.linspace(-1.0, 1.0, w)
        y = torch.linspace(-1.0, 1.0, h)
        x_t, y_t = torch.meshgrid(x, y, indexing='xy')  # (h, w)

        # flatten
        x_t_flat = x_t.reshape(-1)  # (h * w)
        y_t_flat = y_t.reshape(-1)  # (h * w)

        # reshape to [x_t, y_t, 1] - (homogeneous form)
        ones = torch.ones_like(x_t_flat)
        sampling_grid = torch.stack([x_t_flat, y_t_flat, ones])  # (3, h*w)

        # repeat grid num_batch times
        sampling_grid = sampling_grid.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, h*w)

        # transform the sampling grid - batch multiply
        batch_grids = torch.bmm(theta, sampling_grid)  # (B, 2, h*w)
        batch_grids = batch_grids.view(B, 2, h, w)  # (B, 2, h, w)

        return batch_grids


    def bilinear_sampler(self, img, x, y):
        """
        Performs bilinear sampling of the input images according to the
        normalized coordinates provided by the sampling grid.

        Input
        -----
        - img: Tensor of size (B, H, W, C)
        Bacth of images.

        - x, y: Output of affine grid generator

        Returns
        -------
        - output: Tensor of size (B, H, W, C)
        Interpolated images according to grids. Same size as grid.
        """
        H, W = img.size(1), img.size(2)
        max_y = H - 1
        max_x = W - 1

        # rescale x and y to [0, W-1/H-1]
        x = 0.5 * ((x + 1.0) * float(max_x - 1))
        y = 0.5 * ((y + 1.0) * float(max_y - 1))

        # grab 4 nearest corner points for each (x_i, y_i)
        x0 = torch.floor(x).long()  # used as indices in _get_pixel_value()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        # clip to range [0, H-1/W-1] to not violate img boundaries
        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        # get pixel value at corner coords
        Ia = self._get_pixel_value(img, x0, y0)
        Ib = self._get_pixel_value(img, x0, y1)
        Ic = self._get_pixel_value(img, x1, y0)
        Id = self._get_pixel_value(img, x1, y1)

        # recast as float for delta calculation
        x0 = x0.float()
        x1 = x1.float()
        y0 = y0.float()
        y1 = y1.float()

        # calculate deltas
        wa = (x1 - x) * (y1 - y)  # (B, H, W)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # add dimension for addition
        wa = wa.unsqueeze(3)  # (B, H, W, 1)
        wb = wb.unsqueeze(3)
        wc = wc.unsqueeze(3)
        wd = wd.unsqueeze(3)

        output = wa * Ia + wb * Ib + wc * Ic + wd * Id  # (B, H, W, C)

        return output
