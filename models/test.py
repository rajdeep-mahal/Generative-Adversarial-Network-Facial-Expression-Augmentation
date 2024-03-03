"""Testing code for all the models"""

import torch

from hourglass import Encoder
from hourglass import Decoder
from hourglass import Hourglass

from keypoint_detector import KPDetector

if __name__ == "__main__":
   
   #Test Hourglass
   sample_input = torch.randn((1, 3, 256, 256))
   encoder = Encoder(block_expansion=32, in_features=3, num_blocks=3, max_features=256)
   decoder = Decoder(block_expansion=32, in_features=3, num_blocks=3, max_features=256)
   hourglass = Hourglass(block_expansion=32, in_features=3, num_blocks=3, max_features=256)
   print(hourglass(sample_input).shape)

   #Test KPDetector
   kp_detector = KPDetector(block_expansion=32, num_kp=10, num_channels=3, max_features=256,
                 num_blocks=5, temperature=0.1, estimate_jacobian=True, scale_factor=0.25,
                 single_jacobian_map=False, pad=0)
   out = kp_detector(sample_input)
   print(out['value'].shape)
   print(out['jacobian'].shape)
