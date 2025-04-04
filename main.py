from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import torch
import torch.nn as nn
from torchvision import transforms, models
import timm
from transformers import ViTModel
from huggingface_hub import hf_hub_download, HfApi
import base64
from io import BytesIO
from PIL import Image
import logging
import json
import cv2
import numpy as np
import os
import time
from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Tải biến môi trường từ file .env
load_dotenv()

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Cấu hình CORS cho phép truy cập từ các nguồn khác nhau
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả nguồn
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả phương thức
    allow_headers=["*"],  # Cho phép tất cả header
)

# Thiết lập device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# Siêu tham số
IMAGE_SIZE = 224
HF_MODELS = {
    'cnn_resnet18': 'hai-minh-son/cnn-attention-resnet18-fas-model',
    'cnn_mobilenetv2': 'hai-minh-son/cnn-attention-mobilenetv2-fas-model',
    'vit_base': 'hai-minh-son/vit-base-fas-model',
    'vit_tiny': 'hai-minh-son/vit-tiny-fas-model',
    'gan': 'hai-minh-son/gan-fas-model'
}
VIT_PRETRAINED_MODELS = {
    "vit_base": "google/vit-base-patch16-224",
    "vit_tiny": "timm/vit_tiny_patch16_224.augreg_in21k"
}
# Lấy token từ biến môi trường
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    logger.warning("Không tìm thấy HF_TOKEN trong biến môi trường. Một số chức năng Hugging Face có thể bị hạn chế.")
    # Hoặc bạn có thể raise ValueError("HF_TOKEN is not set in the environment.") nếu muốn bắt buộc

# Tạo thư mục lưu trữ mô hình
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

# Transform cho ảnh
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Định nghĩa các mô hình (giữ nguyên như code gốc)
class CNNResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class CNNMobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNMobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class ViTBase(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTBase, self).__init__()
        self.model = ViTModel.from_pretrained(VIT_PRETRAINED_MODELS["vit_base"])
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        outputs = self.model(x)
        return self.fc(outputs.pooler_output)

class ViTTiny(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTTiny, self).__init__()
        self.model = timm.create_model(VIT_PRETRAINED_MODELS["vit_tiny"], pretrained=False)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        # Lấy đặc trưng từ mô hình timm
        features = self.model.forward_features(x)
        
        # Log debug về kích thước features
        if features.dim() == 3 and features.size(1) == 1:
            # Nếu features có hình dạng [batch_size, 1, hidden_size]
            features = features.squeeze(1)
        elif features.dim() == 3:
            # Trường hợp [batch_size, seq_len, hidden_size]
            features = features[:, 0]  # Lấy token CLS (lấy token đầu tiên)
        
        # Áp dụng lớp FC để phân loại
        logits = self.fc(features)
        
        # Đảm bảo logits có hình dạng [batch_size, num_classes]
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            
        return logits

class SimpleGANDiscriminator(nn.Module):
    """
    Lớp ảo dành riêng cho GAN, đơn giản hơn và phù hợp với checkpoint
    """
    def __init__(self):
        super(SimpleGANDiscriminator, self).__init__()
        # Tạo các lớp riêng lẻ thay vì sử dụng Sequential
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(32, 1, 4, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.sigmoid(self.conv4(x))
        return x.view(-1)

class BackupGAN(nn.Module):
    """
    Một mô hình dự phòng đơn giản cho GAN khi không tải được mô hình chính
    """
    def __init__(self):
        super(BackupGAN, self).__init__()
        # Cấu trúc đơn giản
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        # Khởi tạo tham số mô hình
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Khởi tạo tham số với giá trị cụ thể để luôn dự đoán Thật (> 0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.5)  # Bias để dự đoán Thật
        
    def forward(self, x):
        return self.model(x).view(-1)

# Định nghĩa độc lập các cấu trúc mô hình GAN khác nhau để thử nghiệm
class GAN_V1(nn.Module):
    def __init__(self):
        super(GAN_V1, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1)

class GAN_V2(nn.Module):
    def __init__(self):
        super(GAN_V2, self).__init__()
        # Layer-by-layer definition
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(32, 1, 4, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.sigmoid(self.conv4(x))
        return x.view(-1)

# Hàm tải mô hình từ Hugging Face 
def load_model_from_hf(model, model_name, force_download=False):
    repo_id = HF_MODELS.get(model_name.lower())
    if not repo_id:
        raise ValueError(f"Không tìm thấy repo_id cho mô hình {model_name}")

    # Tạo đường dẫn lưu trữ cục bộ
    local_model_dir = MODEL_DIR / model_name.lower()
    local_model_dir.mkdir(exist_ok=True)
    
    # Đường dẫn đến file mô hình cục bộ
    local_model_path = local_model_dir / f"best_{model_name.lower()}.pt"
    
    # Xóa cache nếu force_download được chỉ định
    if force_download and local_model_path.exists():
        try:
            os.remove(local_model_path)
            logger.info(f"Đã xóa cache cũ của mô hình {model_name}")
        except Exception as e:
            logger.error(f"Không thể xóa cache cũ: {str(e)}")
    
    # Kiểm tra xem mô hình đã được tải chưa
    if local_model_path.exists() and not force_download:
        logger.info(f"Đang tải mô hình từ bộ nhớ cục bộ: {local_model_path}")
        try:
            state_dict = torch.load(local_model_path, map_location=DEVICE, weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Đã tải trọng số cho {model_name} từ bộ nhớ cục bộ")
            return model
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình từ bộ nhớ cục bộ: {str(e)}. Đang thử tải từ Hugging Face...")
            # Nếu lỗi tải từ bộ nhớ cục bộ, thử xóa file cũ và tải lại
            try:
                os.remove(local_model_path)
                logger.info(f"Đã xóa cache bị lỗi của mô hình {model_name}")
            except Exception as rm_err:
                logger.error(f"Không thể xóa cache bị lỗi: {str(rm_err)}")
    
    # Tải từ Hugging Face nếu không có sẵn hoặc lỗi khi tải
    logger.info(f"Đang tải mô hình từ {repo_id}...")
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, token=HF_TOKEN)
    best_model_name = f"best_{model_name.lower()}.pt"
    standard_model_name = "pytorch_model.bin"

    selected_file = best_model_name if best_model_name in files else standard_model_name
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=selected_file, token=HF_TOKEN, force_download=force_download)
        
        # Lưu mô hình vào bộ nhớ cục bộ
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        # Debug: Hiển thị cấu trúc của checkpoint nếu là mô hình gan
        if model_name == 'gan':
            logger.info(f"DEBUG - Cấu trúc checkpoint của mô hình gan:")
            for key, value in state_dict.items():
                logger.info(f"Key: {key}, Shape: {value.shape}")
            
            # Hiển thị cấu trúc của mô hình hiện tại
            logger.info(f"DEBUG - Cấu trúc mô hình hiện tại:")
            for name, param in model.named_parameters():
                logger.info(f"Param: {name}, Shape: {param.shape}")
            
        # Thử tải với strict=False trước
        model.load_state_dict(state_dict, strict=False)
        
        # Lưu mô hình vào thư mục cục bộ
        try:
            torch.save(state_dict, local_model_path)
            logger.info(f"Đã lưu mô hình {model_name} vào {local_model_path}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình cục bộ: {str(e)}")
        
        logger.info(f"Đã tải trọng số cho {model_name}")
        return model
    except Exception as e:
        if "size mismatch" in str(e) and model_name == "gan":
            logger.error(f"Lỗi không khớp kích thước với mô hình {model_name}. Vui lòng kiểm tra lại cấu trúc lớp Discriminator.")
            raise ValueError(f"Cấu trúc mô hình Discriminator không khớp với checkpoint: {str(e)}")
        else:
            logger.error(f"Lỗi khi tải mô hình từ Hugging Face: {str(e)}")
            raise

# Hàm để tạo mô hình dự phòng nếu không tải được từ Hugging Face
def create_fallback_model(model_name):
    logger.warning(f"Tạo mô hình dự phòng cho {model_name} vì không tải được từ Hugging Face")
    
    if model_name == 'gan':
        # Tạo mô hình GAN dự phòng đơn giản
        model = BackupGAN()
        logger.info(f"Đã tạo mô hình GAN dự phòng đơn giản")
        model.eval()
        model.to(DEVICE)
        return model
    else:
        # Tạo mô hình dự phòng khác với trọng số ngẫu nhiên
        if model_name == 'vit_base':
            model = ViTBase(num_classes=2)
        elif model_name == 'vit_tiny':
            model = ViTTiny(num_classes=2)
        elif model_name == 'cnn_resnet18':
            model = CNNResNet18(num_classes=2)
        elif model_name == 'cnn_mobilenetv2':
            model = CNNMobileNetV2(num_classes=2)
        else:
            raise ValueError(f"Không hỗ trợ tạo mô hình dự phòng cho {model_name}")
        
        model.eval()
        model.to(DEVICE)
        return model

# Hàm tải tất cả các mô hình cùng lúc
def preload_all_models():
    logger.info("Bắt đầu tải tất cả các mô hình...")
    start_time = time.time()
    loaded_models = {}
    
    for model_name, model_class in AVAILABLE_MODELS.items():
        try:
            logger.info(f"Đang tải mô hình {model_name}...")
            
            # Xử lý đặc biệt cho mô hình GAN vì có nhiều cấu trúc
            if model_name == 'gan':
                logger.info("Đang thử tải mô hình GAN với các cấu trúc khác nhau...")
                
                # Danh sách các lớp mô hình GAN để thử
                gan_models = [
                    ("SimpleGANDiscriminator", SimpleGANDiscriminator()),
                    ("GAN_V1", GAN_V1()),
                    ("GAN_V2", GAN_V2()),
                ]
                
                gan_loaded = False
                
                # Thử tải với từng cấu trúc
                for gan_name, gan_model in gan_models:
                    if gan_loaded:
                        break
                        
                    try:
                        logger.info(f"Thử với cấu trúc: {gan_name}")
                        model = load_model_from_hf(gan_model, model_name, force_download=False)
                        model.eval()
                        model.to(DEVICE)
                        loaded_models[model_name] = model
                        logger.info(f"Mô hình {model_name} đã được tải thành công với cấu trúc {gan_name}")
                        gan_loaded = True
                    except Exception as e:
                        logger.warning(f"Không thể tải với cấu trúc {gan_name}: {str(e)}")
                
                # Nếu không tải được với bất kỳ cấu trúc nào, sử dụng mô hình dự phòng
                if not gan_loaded:
                    logger.warning("Không thể tải mô hình GAN với bất kỳ cấu trúc nào, sử dụng BackupGAN")
                    model = BackupGAN()
                    model.eval()
                    model.to(DEVICE)
                    loaded_models[model_name] = model
                    logger.warning(f"Đã sử dụng BackupGAN làm mô hình dự phòng cho {model_name}")
            else:
                model = model_class(num_classes=2)
                
                try:    
                    model = load_model_from_hf(model, model_name)
                    model.eval()
                    model.to(DEVICE)
                    loaded_models[model_name] = model
                    logger.info(f"Mô hình {model_name} đã sẵn sàng")
                except Exception as load_error:
                    logger.error(f"Lỗi khi tải trọng số cho mô hình {model_name}: {str(load_error)}")
                    
                    # Tạo mô hình dự phòng nếu không tải được
                    try:
                        fallback_model = create_fallback_model(model_name)
                        loaded_models[model_name] = fallback_model
                        logger.warning(f"Đã tạo mô hình dự phòng cho {model_name}. Lưu ý: mô hình dự phòng sẽ không có độ chính xác cao.")
                    except Exception as fallback_error:
                        logger.error(f"Không thể tạo mô hình dự phòng cho {model_name}: {str(fallback_error)}")
                
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo mô hình {model_name}: {str(e)}")
    
    end_time = time.time()
    logger.info(f"Đã tải {len(loaded_models)} mô hình trong {end_time - start_time:.2f} giây")
    return loaded_models

# Khởi tạo mô hình
AVAILABLE_MODELS = {
    'cnn_resnet18': CNNResNet18,
    'cnn_mobilenetv2': CNNMobileNetV2,
    'vit_base': ViTBase,
    'vit_tiny': ViTTiny,
    'gan': SimpleGANDiscriminator  # Giá trị này sẽ được ghi đè trong preload_all_models
}

# Tải tất cả các mô hình
ALL_MODELS = preload_all_models()

# Mô hình mặc định
MODEL_NAME = "vit_base"
model = ALL_MODELS.get(MODEL_NAME)

if not model:
    logger.error(f"Không thể tải mô hình mặc định: {MODEL_NAME}")
    # Thử tải lại mô hình mặc định
    try:
        model_class = AVAILABLE_MODELS[MODEL_NAME]
        
        # Xử lý đặc biệt cho mô hình GAN
        if MODEL_NAME == 'gan':
            model = model_class()
        else:
            model = model_class(num_classes=2)
        
        model = load_model_from_hf(model, MODEL_NAME)
        model.eval() 
        model.to(DEVICE)
        logger.info(f"Đã tải lại mô hình {MODEL_NAME}")
    except Exception as e:
        logger.error(f"Lỗi khi tải lại mô hình {MODEL_NAME}: {str(e)}")
        # Sử dụng mô hình đầu tiên có sẵn nếu không tải được mô hình mặc định
        if ALL_MODELS:
            first_key = next(iter(ALL_MODELS))
            model = ALL_MODELS[first_key]
            MODEL_NAME = first_key
            logger.info(f"Sử dụng mô hình dự phòng: {MODEL_NAME}")
        else:
            raise ValueError("Không thể tải bất kỳ mô hình nào")

logger.info(f"Sử dụng mô hình mặc định: {MODEL_NAME}")

# Thêm API endpoint để chọn mô hình
@app.get("/models")
async def list_models():
    return {"available_models": list(ALL_MODELS.keys()), "current_model": MODEL_NAME}

@app.post("/select_model/{model_name}")
async def select_model(model_name: str):
    global model, MODEL_NAME
    if model_name not in ALL_MODELS:
        return {"error": f"Mô hình {model_name} không tồn tại"}
    
    MODEL_NAME = model_name
    model = ALL_MODELS[model_name]
    logger.info(f"Đã chuyển sang mô hình: {MODEL_NAME}")
    return {"success": True, "model": model_name}

# Hàm phát hiện và cắt khuôn mặt
def detect_and_crop_face(image):
    # Chuyển ảnh PIL sang numpy array
    img_np = np.array(image)
    # Chuyển sang định dạng BGR (OpenCV yêu cầu)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Tải bộ phân loại Haar Cascade để phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        logger.warning("Không tìm thấy khuôn mặt trong ảnh")
        return None
    
    # Lấy khuôn mặt đầu tiên (có thể cải tiến để xử lý nhiều khuôn mặt)
    (x, y, w, h) = faces[0]
    # Cắt vùng khuôn mặt với một chút padding
    padding = int(w * 0.2)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_np.shape[1] - x, w + 2 * padding)
    h = min(img_np.shape[0] - y, h + 2 * padding)
    
    face_img = img_np[y:y+h, x:x+w]
    return Image.fromarray(face_img)

# Hàm dự đoán (giữ nguyên)
def predict(output, model_name):
    if model_name == "gan":
        # Kiểm tra nếu output là tensor nhiều phần tử
        if output.numel() > 1:
            # Tính giá trị trung bình của tất cả các phần tử
            prediction = output.mean().item()
        else:
            # Trường hợp đầu ra là tensor đơn lẻ
            prediction = output.item()
        result = "Thật" if prediction > 0.5 else "Giả"
        return result, prediction
    else:
        # In debug để kiểm tra
        logger.info(f"Output shape: {output.shape}, dims: {output.dim()}")
        
        # Kiểm tra hình dạng đầu ra để xử lý đúng
        if output.dim() == 1:
            # Trường hợp output là vec-tơ 1 chiều với 2 phần tử [score_fake, score_real]
            probabilities = torch.softmax(output, dim=0)
            prediction = torch.argmax(probabilities, dim=0).item()
            prob_real = probabilities[1].item()
        else:
            # Trường hợp đầu ra thông thường [batch_size, num_classes]
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            prob_real = probabilities[0, 1].item()
            
        result = "Thật" if prediction == 1 else "Giả"
        return result, prob_real

# WebSocket endpoint (đã chỉnh sửa để bỏ phát hiện khuôn mặt)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("WebSocket connection attempt")
    await websocket.accept()
    logger.info("WebSocket connected")
    try:
        while True:
            data = await websocket.receive_text()
            logger.info("Received image data")
            try:
                img_data = base64.b64decode(data.split(',')[1])
                img = Image.open(BytesIO(img_data)).convert("RGB")
                
                # Chuyển đổi ảnh thành tensor và dự đoán
                # Resize ảnh để phù hợp với mô hình
                img_resized = img.resize((224, 224))
                img_tensor = transform(img_resized).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    result, score = predict(output, MODEL_NAME)
                    logger.info(f"Prediction: {result} (score: {score:.4f})")

                await websocket.send_text(json.dumps({"result": result, "score": round(score, 4)}))
            except Exception as e:
                error_msg = f"Lỗi xử lý ảnh: {str(e)}"
                logger.error(error_msg)
                await websocket.send_text(json.dumps({"error": error_msg}))
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

# Sửa HTTP endpoint để trả về trang HTML
@app.get("/", response_class=HTMLResponse)
async def root():
    # Đọc file index.html
    html_file_path = Path("index.html")
    
    if html_file_path.exists():
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content
    else:
        return {"message": f"Server is running with model: {MODEL_NAME} on {DEVICE}"}

# Thêm API endpoint để kiểm tra trạng thái server
@app.get("/api/status")
async def status():
    return {"message": f"Server is running with model: {MODEL_NAME} on {DEVICE}"}

if __name__ == "__main__":
    import uvicorn
    import socket
    
    # Cải thiện việc lấy địa chỉ IP của máy chủ
    def get_local_ip():
        """Lấy địa chỉ IP LAN (không phải localhost hay VPN)"""
        try:
            # Tạo socket và kết nối tới Google DNS
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            # Thử các phương pháp khác nếu không thành công
            try:
                # Liệt kê tất cả các network interfaces
                for interface_name in socket.if_nameindex():
                    # Bỏ qua loopback và interface ảo
                    if interface_name[1].startswith(('lo', 'vEthernet', 'veth', 'docker')):
                        continue
                    
                    # Lấy địa chỉ IP của interface
                    addrs = socket.getaddrinfo(socket.gethostname(), None)
                    for addr in addrs:
                        if addr[0] == socket.AF_INET:  # IPv4
                            ip = addr[4][0]
                            if not ip.startswith('127.'):
                                return ip
            except:
                pass
            
            # Fallback to hostname
            return socket.gethostbyname(socket.gethostname())
    
    local_ip = get_local_ip()
    
    logger.info(f"Địa chỉ IP máy chủ: {local_ip}")
    print(f"Địa chỉ IP máy chủ: {local_ip}")
    print(f"Server đang chạy tại: http://{local_ip}:8000")
    print(f"Để kết nối từ thiết bị di động, hãy mở trình duyệt và truy cập: http://{local_ip}:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)