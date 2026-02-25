
import torch
import monai
from detection.feature_extractor_wrapper import FeatureExtractorWrapper
from monai.networks.nets import resnet50
from monai.apps.detection.networks.retinanet_network import resnet_fpn_feature_extractor

def check_strides():
    spatial_dims = 3
    # Same backbone config as train_luna16.json
    conv1_t_stride = [2, 2, 1]
    conv1_t_size = [7, 7, 7]
    backbone = resnet50(
        spatial_dims=spatial_dims,
        n_input_channels=1,
        conv1_t_stride=conv1_t_stride,
        conv1_t_size=conv1_t_size
    )
    
    # Same feature extractor base config
    # returned_layers=[1, 2, 3]
    feature_extractor_base = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=spatial_dims,
        returned_layers=[1, 2, 3]
    )
    
    wrapper = FeatureExtractorWrapper(feature_extractor_base)
    
    # Input size from config [192, 192, 80]
    input_shape = (1, 1, 192, 192, 80)
    dummy_input = torch.randn(input_shape)
    
    print(f"Input shape: {input_shape}")
    
    wrapper.eval()
    with torch.no_grad():
        outputs = wrapper(dummy_input)
        
    print(f"Output keys: {outputs.keys()}")
    for k, v in outputs.items():
        print(f"Feature Map {k} shape: {v.shape}")
        # Calculate stride
        stride_h = input_shape[2] / v.shape[2]
        stride_w = input_shape[3] / v.shape[3]
        stride_d = input_shape[4] / v.shape[4]
        print(f"  -> Calculated Stride: ({stride_h}, {stride_w}, {stride_d})")

if __name__ == "__main__":
    check_strides()
