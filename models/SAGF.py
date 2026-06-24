import torch
import torch.nn as nn
import torchvision.models as models


class CSA_Attention(nn.Module):
    """
    Channel-Space-Angle (CSA) Attention Mechanism.
    Matches the SAGF-IQA paper Section 3.3.
    """

    def __init__(self, in_channels):
        super(CSA_Attention, self).__init__()
        # Channel Attention: generates channel weights via global avg pooling and FC layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention: generates spatial weights via 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        # Channel Attention computation
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        channel_att = x * y.expand_as(x)

        # Spatial Attention computation
        spatial_att = self.conv1(channel_att)
        spatial_att = self.bn1(spatial_att)
        spatial_att = self.relu(spatial_att)

        # Angle/Directional Attention computation (Averaged across channel dimension)
        angle_att = torch.mean(spatial_att, dim=1, keepdim=True)

        # Channel-Space-Angle fusion
        out = channel_att * spatial_att * angle_att
        return out


class ResNet50DualBranch(nn.Module):
    """
    Dual-branch feature extraction based on ResNet50.
    Extracts both low-level (Stage 1) and high-level (Stage 5) representations.
    Output dimensions: 2048 for both branches.
    """

    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet50
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Shallow feature branch (First layer)
        self.first_layer = self.resnet50.conv1
        self.first_bn = self.resnet50.bn1
        self.first_relu = self.resnet50.relu
        

        # Deep semantic feature branch
        self.feature_extractor = nn.Sequential(*list(self.resnet50.children())[:-2])

        # CSA Attention Module for refinement
        self.csa_attention_shallow = CSA_Attention(2048)
        self.csa_attention_deep = CSA_Attention(2048)

        # Dimensionality alignment for shallow features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.channel_adjust = nn.Conv2d(64, 2048, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # 1. Extract shallow features
        first_features = self.first_layer(x)
        first_features = self.first_bn(first_features)
        first_features = self.first_relu(first_features)
        

        # 2. Dimensionality adjustment to match deep features (Retains spatial layout 7x7)
        first_features = self.adaptive_pool(first_features)
        first_features = self.channel_adjust(first_features)

        # 3. Apply CSA Attention on spatial features, THEN global pool
        first_features = self.csa_attention_shallow(first_features)
        first_features = self.global_pool(first_features)
        first_features = torch.flatten(first_features, 1)  # [B, 2048]

        # 4. Extract deep features (Outputs spatial features, e.g., 7x7)
        last_features = self.feature_extractor(x)

        # 5. Apply CSA Attention on deep spatial features, THEN global pool
        last_features = self.csa_attention_deep(last_features)
        last_features = self.global_pool(last_features)
        last_features = torch.flatten(last_features, 1)  # [B, 2048]

        return first_features, last_features


class SwinGlobalBranch(nn.Module):
    """
    Global contextual feature extraction based on Swin Transformer Base.
    Output dimension: 1024
    """

    def __init__(self):
        super().__init__()
        self.swin = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        # Swin feature extractor outputs sequence or pooled tensor directly depending on implementation
        self.feature_extractor = nn.Sequential(*list(self.swin.children())[:-1])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature_extractor(x)
        # Ensure correct flattening behavior depending on Swin's spatial output
        if len(x.shape) == 4:
            x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x


class SalientCropBranch(nn.Module):
    """
    Fine-grained feature extractor for saliency-guided cropped regions (EfficientNetV2-S).
    Output dimension: 1280
    """

    def __init__(self):
        super().__init__()
        self.efficientnet_v2_s = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.efficientnet_v2_s.features.children()))
        self.csa_attention = CSA_Attention(1280)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.feature_extractor(x)  # Retains spatial dimensions [B, 1280, H, W]
        features = self.csa_attention(features)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)  # [B, 1280]
        return features


class DMFF(nn.Module):
    """
    Dynamic Multi-branch Feature Fusion (DMFF) Strategy.
    Fuses features from 4 branches using a shared attention mechanism.
    """

    def __init__(self, dim1=2048, dim2=2048, dim3=1280, dim4=1024, out_dim=1024):
        super().__init__()
        # Project heterogeneous features to a common latent space
        self.proj1 = nn.Linear(dim1, out_dim)
        self.proj2 = nn.Linear(dim2, out_dim)
        self.proj3 = nn.Linear(dim3, out_dim)
        self.proj4 = nn.Linear(dim4, out_dim)

        # Attention weight network
        self.attention = nn.Sequential(
            nn.Linear(out_dim, out_dim // 8),
            nn.ReLU(),
            nn.Linear(out_dim // 8, 4),
            nn.Softmax(dim=1)
        )
        self.final_proj = nn.Linear(out_dim, out_dim)

    def forward(self, feat1, feat2, feat3, feat4):
        feat1_proj = self.proj1(feat1)
        feat2_proj = self.proj2(feat2)
        feat3_proj = self.proj3(feat3)
        feat4_proj = self.proj4(feat4)

        # Global context vector for attention computation
        total_feat = feat1_proj + feat2_proj + feat3_proj + feat4_proj
        attention_weights = self.attention(total_feat)

        # Dynamically weighted fusion
        fused_feat = attention_weights[:, 0:1] * feat1_proj + \
                     attention_weights[:, 1:2] * feat2_proj + \
                     attention_weights[:, 2:3] * feat3_proj + \
                     attention_weights[:, 3:4] * feat4_proj

        fused_feat = self.final_proj(fused_feat)
        return fused_feat


class SAGFIQA(nn.Module):
    """
    SAGF-IQA: Saliency-Guided Adaptive Global-Local Framework
    Predicts image quality score through multi-scale extraction and DMFF.
    """

    def __init__(self):
        super().__init__()
        # Initialize the 3 parallel backbones (yielding 4 feature branches)
        self.resnet_branch = ResNet50DualBranch()
        self.swin_branch = SwinGlobalBranch()
        self.salient_branch = SalientCropBranch()

        # Dynamic Multi-branch Feature Fusion
        self.dmff = DMFF(
            dim1=2048,
            dim2=2048,
            dim3=1280,
            dim4=1024,
            out_dim=1024
        )

        # MLP for Quality Regression
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def to(self, device):
        super().to(device)
        return self

    def forward(self, x, cropped_images):
        """
        Args:
            x: Original full image.
            cropped_images: Saliency-guided cropped patch.
        """
        # Multi-Branch Extraction
        first_resnet_feat, last_resnet_feat = self.resnet_branch(x)
        swin_feat = self.swin_branch(x)
        salient_feat = self.salient_branch(cropped_images)

        # Dynamic Fusion
        fused_features = self.dmff(first_resnet_feat, last_resnet_feat, salient_feat, swin_feat)

        quality_score = self.mlp(fused_features)
        return quality_score


if __name__ == "__main__":
    # Test script to verify dimensions
    model = SAGFIQA()
    input_image = torch.randn(2, 3, 224, 224)
    cropped_images = torch.randn(2, 3, 224, 224)

    predicted_quality = model(input_image, cropped_images)
    print("Predicted Quality Score:", predicted_quality)
