document.addEventListener("DOMContentLoaded", function () {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const captureBtn = document.getElementById("captureBtn");
  const switchCameraBtn = document.getElementById("switchCameraBtn");
  const modelSelect = document.getElementById("modelSelect");
  const predictionResults = document.getElementById("predictionResults");
  const prediction = document.getElementById("prediction");
  const confidenceValue = document.getElementById("confidenceValue");
  const confidenceText = document.getElementById("confidenceText");
  const spinner = document.getElementById("spinner");
  const progressContainer = document.getElementById("progressContainer");
  const progressBar = document.getElementById("progressBar");
  const progressText = document.getElementById("progressText");
  const messageContainer = document.getElementById("messageContainer");

  let currentStream = null;
  let currentFacingMode = "user"; // mặc định là camera trước
  let firstTimeLoad = true;

  // Kiểm tra xem thiết bị có hỗ trợ getUserMedia không
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showMessage("Trình duyệt của bạn không hỗ trợ getUserMedia API!", "error");
    disableControls();
    return;
  }

  // Tải danh sách mô hình từ server
  async function loadModels() {
    try {
      const response = await fetch("/api/models");
      const models = await response.json();

      // Xóa tất cả options hiện tại
      modelSelect.innerHTML = "";

      // Thêm các options mới
      models.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = `${model.name} - ${model.description}`;
        modelSelect.appendChild(option);
      });
    } catch (error) {
      console.error("Không thể tải danh sách mô hình:", error);
    }
  }

  // Khởi tạo camera
  async function initCamera() {
    try {
      if (currentStream) {
        stopCamera();
      }

      const constraints = {
        video: {
          facingMode: currentFacingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      };

      currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = currentStream;

      if (firstTimeLoad) {
        await loadModels();
        firstTimeLoad = false;
      }

      enableControls();
    } catch (error) {
      console.error("Lỗi khi truy cập camera:", error);
      showMessage(
        "Không thể truy cập camera. Vui lòng cấp quyền truy cập camera cho trang web này.",
        "error"
      );
      disableControls();
    }
  }

  function stopCamera() {
    if (currentStream) {
      currentStream.getTracks().forEach((track) => track.stop());
      currentStream = null;
    }
  }

  function disableControls() {
    captureBtn.disabled = true;
    switchCameraBtn.disabled = true;
    modelSelect.disabled = true;
  }

  function enableControls() {
    captureBtn.disabled = false;
    switchCameraBtn.disabled = false;
    modelSelect.disabled = false;
  }

  function showMessage(text, type = "info") {
    const messageElement = document.createElement("div");
    messageElement.className = `message ${type}`;
    messageElement.textContent = text;

    messageContainer.innerHTML = "";
    messageContainer.appendChild(messageElement);

    setTimeout(() => {
      messageElement.style.opacity = "0";
      setTimeout(() => {
        if (messageContainer.contains(messageElement)) {
          messageContainer.removeChild(messageElement);
        }
      }, 500);
    }, 5000);
  }

  function showLoading(isLoading) {
    if (isLoading) {
      spinner.style.display = "block";
      disableControls();
    } else {
      spinner.style.display = "none";
      enableControls();
    }
  }

  function showProgress(show, progress = 0, text = "") {
    if (show) {
      progressContainer.style.display = "block";
      progressBar.style.width = `${progress}%`;
      progressText.textContent = text;
    } else {
      progressContainer.style.display = "none";
    }
  }

  async function captureAndPredict() {
    if (!currentStream) {
      showMessage("Không có camera được kích hoạt!", "error");
      return;
    }

    // Cấu hình canvas
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Chụp ảnh từ video
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Lấy dữ liệu hình ảnh dưới dạng base64
    const imageData = canvas.toDataURL("image/jpeg");

    // Chuẩn bị dữ liệu để gửi lên server
    const selectedModel = modelSelect.value;
    const requestData = {
      model: selectedModel,
      image: imageData,
    };

    // Hiển thị trạng thái đang tải
    showLoading(true);
    showProgress(true, 10, "Đang xử lý...");

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });

      showProgress(true, 50, "Đang phân tích...");

      const result = await response.json();

      if (result.error) {
        showMessage(`Lỗi: ${result.error}`, "error");
        showProgress(false);
        showLoading(false);
        return;
      }

      showProgress(true, 90, "Đang hiển thị kết quả...");

      // Hiển thị kết quả
      predictionResults.style.display = "block";

      if (result.prediction === "real") {
        prediction.innerHTML = '<span class="real">THẬT</span>';
        confidenceValue.style.backgroundColor = "var(--success-color)";
      } else {
        prediction.innerHTML = '<span class="spoof">GIẢ MẠO</span>';
        confidenceValue.style.backgroundColor = "var(--danger-color)";
      }

      const confidencePercent = Math.round(result.confidence * 100);
      confidenceValue.style.width = `${confidencePercent}%`;
      confidenceText.textContent = `Độ tin cậy: ${confidencePercent}%`;

      showProgress(true, 100, "Hoàn thành!");
      setTimeout(() => showProgress(false), 500);
    } catch (error) {
      console.error("Lỗi khi gửi yêu cầu:", error);
      showMessage(
        "Không thể kết nối đến server. Vui lòng thử lại sau.",
        "error"
      );
    } finally {
      showLoading(false);
    }
  }

  // Xử lý sự kiện nút chụp ảnh
  captureBtn.addEventListener("click", captureAndPredict);

  // Xử lý sự kiện đổi camera (trước/sau)
  switchCameraBtn.addEventListener("click", function () {
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    initCamera();
  });

  // Khởi tạo camera khi trang được tải
  initCamera();

  // Xử lý sự kiện khi người dùng rời trang
  window.addEventListener("beforeunload", stopCamera);
});
