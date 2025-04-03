import os
import io
import base64
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from huggingface_hub import hf_hub_download
import timm
from transformers import ViTModel, ViTFeatureExtractor

app = Flask(__name__, static_folder='static')
CORS(app)

# Cấu hình
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tên mô hình và repo tương ứng
HF_MODELS = {
    'cnn_resnet18': 'hai-minh-son/cnn-attention-resnet18-fas-model',
    'cnn_mobilenetv2': 'hai-minh-son/cnn-attention-mobilenetv2-fas-model',
    'vit_base': 'hai-minh-son/vit-base-fas-model',
    'vit_tiny': 'hai-minh-son/vit-tiny-fas-model',
    'gan': 'hai-minh-son/gan-fas-model'
}

# Cache các mô hình đã tải
loaded_models = {}

# Định nghĩa kiến trúc mô hình
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(attention)
        attention_map = self.sigmoid(attention)
        return x * attention_map

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class CNNWithAttention(nn.Module):
    def __init__(self, model_name="resnet18", pretrained=True, use_attention=True):
        super(CNNWithAttention, self).__init__()
        self.model_name = model_name.lower()
        
        if self.model_name == "resnet18":
            from torchvision import models
            backbone = models.resnet18(pretrained=pretrained)
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            
            self.use_attention = use_attention
            if use_attention:
                self.cbam1 = CBAM(64)
                self.cbam2 = CBAM(128)
                self.cbam3 = CBAM(256)
                self.cbam4 = CBAM(512)
            
            self.feature_dim = 512
                
        elif self.model_name == "mobilenetv2":
            from torchvision import models
            backbone = models.mobilenet_v2(pretrained=pretrained)
            self.features = backbone.features
            
            self.use_attention = use_attention
            if use_attention:
                self.cbam1 = CBAM(24)
                self.cbam2 = CBAM(32)
                self.cbam3 = CBAM(96)
                self.cbam4 = CBAM(1280)
            
            self.feature_dim = 1280
        
        else:
            raise ValueError(f"Không hỗ trợ model {model_name}. Chỉ hỗ trợ 'resnet18' hoặc 'mobilenetv2'")
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.feature_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        if self.model_name == "resnet18":
            x = self.features[:4](x)
            
            x = self.features[4](x)
            if self.use_attention:
                x = self.cbam1(x)
            
            x = self.features[5](x)
            if self.use_attention:
                x = self.cbam2(x)
                
            x = self.features[6](x)
            if self.use_attention:
                x = self.cbam3(x)
                
            x = self.features[7](x)
            if self.use_attention:
                x = self.cbam4(x)
        
        elif self.model_name == "mobilenetv2":
            x = self.features[:3](x)
            if self.use_attention:
                x = self.cbam1(x)
                
            x = self.features[3:7](x)
            if self.use_attention:
                x = self.cbam2(x)
                
            x = self.features[7:14](x)
            if self.use_attention:
                x = self.cbam3(x)
                
            x = self.features[14:](x)
            if self.use_attention:
                x = self.cbam4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ViTForFAS(nn.Module):
    def __init__(self, pretrained_model_name="google/vit-base-patch16-224", freeze_base=True):
        super(ViTForFAS, self).__init__()
        
        if 'timm/' in pretrained_model_name:
            self.model_type = 'timm'
            timm_model_name = pretrained_model_name.replace('timm/', '')
            self.vit_model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
            
            if 'tiny' in pretrained_model_name:
                hidden_size = 192
            else:
                hidden_size = 768
        else:
            self.model_type = 'hf'
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_model_name)
            self.vit_model = ViTModel.from_pretrained(pretrained_model_name)
            hidden_size = self.vit_model.config.hidden_size
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256 if hidden_size > 256 else hidden_size),
            nn.LayerNorm(256 if hidden_size > 256 else hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256 if hidden_size > 256 else hidden_size, 1)
        )
        
    def forward(self, x):
        if self.model_type == 'timm':
            features = self.vit_model(x)
        else:
            outputs = self.vit_model(x)
            features = outputs.last_hidden_state[:, 0]
        
        features = self.dropout(features)
        logits = self.classifier(features)
        
        return logits

class Discriminator(nn.Module):
    def __init__(self, img_size=64):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        
        self.cbam = CBAM(128)
        
        ds_size = img_size // 16
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )
        
        self.aux_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        features = self.model(img)
        features = self.cbam(features)
        features = features.view(features.shape[0], -1)
        
        validity = self.adv_layer(features)
        label = self.aux_layer(features)
        
        return validity, label

def get_transform():
    """Trả về biến đổi chuẩn hóa hình ảnh"""
    import torchvision.transforms as transforms
    
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model(model_name):
    """Tải mô hình từ Hugging Face Hub nếu chưa có trong cache"""
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    try:
        # Xác định repo_id từ model_name
        repo_id = HF_MODELS.get(model_name)
        if repo_id is None:
            return None, f"Không tìm thấy repo_id cho mô hình {model_name}"
        
        print(f"Đang tải mô hình từ {repo_id}...")
        
        # Tải file pytorch_model.bin từ Hugging Face Hub
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename="pytorch_model.bin"
        )
        
        # Khởi tạo mô hình tùy theo loại
        if model_name == 'cnn_resnet18':
            model = CNNWithAttention(model_name="resnet18", pretrained=False, use_attention=True).to(DEVICE)
        elif model_name == 'cnn_mobilenetv2':
            model = CNNWithAttention(model_name="mobilenetv2", pretrained=False, use_attention=True).to(DEVICE)
        elif model_name == 'vit_base':
            model = ViTForFAS(pretrained_model_name="google/vit-base-patch16-224", freeze_base=False).to(DEVICE)
        elif model_name == 'vit_tiny':
            model = ViTForFAS(pretrained_model_name="timm/vit_tiny_patch16_224.augreg_in21k", freeze_base=False).to(DEVICE)
        elif model_name == 'gan':
            model = Discriminator(img_size=IMAGE_SIZE).to(DEVICE)
        else:
            return None, f"Không hỗ trợ mô hình {model_name}"
        
        # Tải trọng số
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.eval()
        
        # Lưu vào cache
        loaded_models[model_name] = model
        
        print(f"Đã tải thành công mô hình {model_name}")
        return model, None
    
    except Exception as e:
        return None, f"Lỗi khi tải mô hình {model_name}: {str(e)}"

def predict_image(model, model_name, image):
    """Dự đoán xem ảnh là thật hay giả mạo"""
    try:
        transform = get_transform()
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            if model_name == 'gan':
                _, output = model(input_tensor)
            else:
                output = model(input_tensor)
            
            # Chuyển đổi logits thành xác suất
            if model_name == 'gan':
                probability = output.item()
            else:
                probability = torch.sigmoid(output).item()
            
            # Xác định nhãn: 1 = real, 0 = spoof
            prediction = 1 if probability >= 0.5 else 0
            
            result = {
                "prediction": "real" if prediction == 1 else "spoof",
                "confidence": probability if prediction == 1 else 1 - probability,
                "score": probability
            }
            
            return result, None
    
    except Exception as e:
        return None, f"Lỗi khi dự đoán: {str(e)}"

@app.route('/')
def index():
    """Trả về trang chủ"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint để nhận ảnh và trả về kết quả dự đoán"""
    try:
        # Lấy dữ liệu từ request
        data = request.json
        model_name = data.get('model', 'vit_base')
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "Không tìm thấy dữ liệu ảnh"}), 400
        
        # Xử lý dữ liệu ảnh (base64)
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Tải mô hình
        model, error = load_model(model_name)
        if error:
            return jsonify({"error": error}), 500
        
        # Dự đoán
        result, error = predict_image(model, model_name, image)
        if error:
            return jsonify({"error": error}), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Lỗi khi xử lý yêu cầu: {str(e)}"}), 500

@app.route('/api/models', methods=['GET'])
def api_models():
    """Trả về danh sách các mô hình có sẵn"""
    models = [
        {
            "id": "vit_base",
            "name": "ViT Base",
            "description": "Vision Transformer Base model"
        },
        {
            "id": "vit_tiny",
            "name": "ViT Tiny",
            "description": "Vision Transformer Tiny model (nhẹ hơn)"
        },
        {
            "id": "cnn_resnet18",
            "name": "CNN ResNet18",
            "description": "CNN với ResNet18 backbone và cơ chế Attention"
        },
        {
            "id": "cnn_mobilenetv2",
            "name": "CNN MobileNetV2",
            "description": "CNN với MobileNetV2 backbone và cơ chế Attention (nhẹ hơn)"
        },
        {
            "id": "gan",
            "name": "GAN Discriminator",
            "description": "Discriminator từ GAN được huấn luyện cho face anti-spoofing"
        }
    ]
    
    return jsonify(models)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True) 