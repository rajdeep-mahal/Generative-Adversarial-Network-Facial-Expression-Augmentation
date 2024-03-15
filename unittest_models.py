"""Testing code for all the models"""

import torch

from models.hourglass import Hourglass
from models.keypoint_detector import KPDetector
from models.dense_motion_net import DenseMotionNetwork
from models.generator import OcclusionAwareGenerator
from models.discriminator import MultiScaleDiscriminator
from models.model import GeneratorModel, DiscriminatorModel


if __name__ == "__main__":
   
   #Test Hourglass
   sample_input = torch.randn((1, 3, 256, 256))
   hourglass = Hourglass(block_expansion=32, in_features=3, num_blocks=3, max_features=256)
   print(hourglass(sample_input).shape)

   #Test KPDetector
   kp_detector = KPDetector(block_expansion=32, num_kp=10, num_channels=3, max_features=256,
                 num_blocks=5, temperature=0.1, estimate_jacobian=True, scale_factor=0.25,
                 single_jacobian_map=False, pad=0)
   out_kp = kp_detector(sample_input)
   print(out_kp['value'].shape)
   print(out_kp['jacobian'].shape)

   #Test DenseMotionNetwork
   dense_motion_net = DenseMotionNetwork(block_expansion=64, num_blocks=5, max_features=1024,
                                         num_kp=10, num_channels=3, estimate_occlusion_map=True,
                                         scale_factor=0.25)
   source_image = torch.randn((1, 3, 256, 256))
   kp_driving = out_kp
   kp_source = out_kp
   out = dense_motion_net(source_image, kp_driving, kp_source)
   print(out['occlusion_map'].shape)

   #Test OcclusionAwareGenerator
   dense_motion_params = {'block_expansion': 64, 'max_features': 1024,
                          'num_blocks': 5, 'scale_factor': 0.25}
   generator = OcclusionAwareGenerator(num_channels=3, num_kp=10, block_expansion=64,
                                       max_features=512, num_down_blocks=2, num_bottleneck_blocks=6,
                                       estimate_occlusion_map=True, dense_motion_params=dense_motion_params, estimate_jacobian=True)
   out_gen = generator(source_image, kp_driving, kp_source)
   print(out_gen['prediction'].shape)
   
   #Test MultiScaleDiscriminator
   discriminator = MultiScaleDiscriminator(scales=[1], block_expansion=32, max_features=512,
                                           num_blocks=4, sn=True)
   
   #Test GeneratorModel
   x = {"source": source_image, "driving": source_image}
   train_params = {"scales": [1, 0.5, 0.25, 0.125], "loss_weights":{
      "generator_gan": 0,
      "discriminator_gan": 1,
      "feature_matching": [10, 10, 10, 10],
      "perceptual": [10, 10, 10, 10, 10],
      "equivariance_value": 10,
      "equivariance_jacobian": 10}, "transform_params": {
      "sigma_affine": 0.05,
      "sigma_tps": 0.005,
      "points_tps": 5}}
   
   generator_model = GeneratorModel(kp_detector, generator, discriminator, train_params)
   loss_values, generated = generator_model(x)
   print(generated['prediction'].shape)
   print(loss_values)
   
   #Test DiscriminatorModel
   discriminator_model = DiscriminatorModel(kp_detector, generator, discriminator, train_params)
   loss_values = discriminator_model(x, generated)
   print(loss_values)
