# Ứng dụng Face Anti-Spoofing (FAS)

Ứng dụng Face Anti-Spoofing (FAS) là một hệ thống sử dụng trí tuệ nhân tạo để phân biệt khuôn mặt thật và khuôn mặt giả (ảnh, video, mặt nạ) trong thời gian thực thông qua webcam.

## Tính năng

- Phát hiện khuôn mặt trong thời gian thực
- Phân loại khuôn mặt thật/giả với nhiều mô hình khác nhau
- Giao diện web thân thiện với người dùng
- Hỗ trợ thiết bị di động thông qua mạng LAN
- Tự động tải và lưu trữ mô hình cục bộ

## Các mô hình được hỗ trợ

- Vision Transformer Base (vit_base)
- Vision Transformer Tiny (vit_tiny)
- CNN ResNet18 (cnn_resnet18)
- CNN MobileNetV2 (cnn_mobilenetv2)
- GAN Discriminator (gan)

## Cài đặt

### Yêu cầu

- Python 3.7 hoặc cao hơn
- Các gói phụ thuộc được liệt kê trong `requirements.txt`

### Các bước cài đặt

1. Clone repository hoặc giải nén mã nguồn:

2. Cài đặt các gói phụ thuộc:

Cài torch torchvision numpy pillow qua [trang chủ PyTorch](https://pytorch.org/) thay vì qua file `requirements.txt` để đảm bảo tương thích với GPU.

Sau đó, cài đặt các gói còn lại bằng lệnh:

```cmd
pip install -r requirements.txt
```

3. Khởi chạy ứng dụng:

Thêm file `.env` vào thư mục gốc của dự án với nội dung sau:

```env
HF_TOKEN=<token_huggingface>
```

Sau đó, chạy ứng dụng bằng lệnh:

```cmd
python main.py
```

1. Khi ứng dụng khởi động, mô hình sẽ được tải về cục bộ trong thư mục `models/`. Quá trình này có thể mất một chút thời gian trong lần chạy đầu tiên.

## Sử dụng

1. Sau khi khởi động, ứng dụng sẽ hiển thị địa chỉ IP của máy chủ trong terminal.

2. Truy cập ứng dụng từ trình duyệt:

   - Trên máy chủ: `http://localhost:8000`
   - Từ thiết bị khác trong cùng mạng LAN: `http://<địa_chỉ_IP>:8000`

3. Cho phép trình duyệt truy cập camera khi được hỏi.

4. Ứng dụng sẽ tự động phát hiện khuôn mặt và phân loại thật/giả.

5. Bạn có thể chọn mô hình khác từ danh sách thả xuống để thay đổi mô hình phát hiện.

## Giải quyết sự cố

- Nếu gặp lỗi kết nối WebSocket, hãy kiểm tra xem bạn đã truy cập đúng địa chỉ IP của máy chủ chưa.
- Nếu camera không hoạt động, hãy đảm bảo trình duyệt của bạn đã được cấp quyền truy cập camera.
- Nếu không phát hiện được khuôn mặt, hãy đảm bảo khuôn mặt của bạn nằm trong khung hình và có đủ ánh sáng.

## Giấy phép

© 2025 All rights reserved
