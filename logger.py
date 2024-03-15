"""For logging loss values and generated images"""

import logging
import numpy as np
import torch.nn.functional as F

from skimage.draw import disk
import imageio


class Logger:
    def __init__(self, log_path, log_filename):
        self.logger = self.get_logger(log_path, log_filename)
        self.log_path = log_path
        self.batch_losses = []
        self.loss_names = None

        self.visualizer = Visualizer(num_kp=5, num_rows=5,
                                     draw_border=True, colormap='gist_rainbow')
  
    def get_logger(self, log_path, log_filename):
        logger = logging.getLogger()
        logger.handlers.clear()

        logger.setLevel(logging.INFO)    
        # Logging to a file
        file_handler = logging.FileHandler(f"{log_path}/{log_filename}")
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

        return logger    
        
    def log_batch_loss(self, losses):
        if self.loss_names is None:
            self.loss_names = list(losses.keys())
        self.batch_losses.append(list(losses.values()))

    def log_epoch_loss(self, epoch):
        loss_mean = np.array(self.batch_losses).mean(axis=0)

        loss_string = "; ".join([f"{name}-{value:.3f}" for name, value in zip(self.loss_names, loss_mean)])
        loss_string = "Epoch:" + str(epoch) + ' losses: ' + loss_string

        self.logger.info(loss_string)
        self.batch_losses = []
    
    def log_vis_images(self, source, driving, output, epoch):
        image = self.visualizer.visualize(source, driving, output)
        filename = f"{self.log_path}/visualizations/{epoch}-vis.png"
        imageio.imsave(filename, image)


class Visualizer:
    def __init__(self, num_kp=5, num_rows=5, draw_border=True, colormap='gist_rainbow'):
        self.num_kp = num_kp #num of keypoints to plot on source & driving image
        self.num_rows = num_rows #num of data points to visualize
        self.draw_border = draw_border
        self.colormap = colormap

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = disk(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, source, driving, out):
        images = []

        #source image with keypoints
        source = source.cpu().numpy()[:self.num_rows]
        source = np.transpose(source, [0, 2, 3, 1])
        kp_source = out['kp_source']['value'].cpu().numpy()[:self.num_rows]        
        images.append((source, kp_source))

        #driving image with keypoints
        driving = driving.cpu().numpy()[:self.num_rows]
        driving = np.transpose(driving, [0, 2, 3, 1])
        kp_driving = out['kp_driving']['value'].cpu().numpy()[:self.num_rows]
        images.append((driving, kp_driving))

        ## Occlusion map  
        occlusion_map = out['occlusion_map'].cpu().repeat(1, 3, 1, 1)
        occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()[:self.num_rows]
        occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
        images.append(occlusion_map)

        #generated driving image
        prediction = out['prediction'].cpu().numpy()[:self.num_rows]
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)        

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
