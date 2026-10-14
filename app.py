import json
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# إعداد شكل الصفحة
st.set_page_config(
    page_title="Weather AI Classifier",
    page_icon="🌦️",
    layout="wide"
)

# تحديد الجهاز المستخدم للتنبؤ
device = torch.device("cpu")

# بناء نفس معمارية الـ CNN التي استخدمناها في التدريب
class WeatherCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# قراءة أسماء الفئات من الملف
with open("classes.json", "r") as file:
    class_names = json.load(file)

# إنشاء الموديل بنفس عدد الفئات
model = WeatherCNN(len(class_names))

# تحميل أوزان الموديل المدرب
state_dict = torch.load(
    "weather_cnn.pth",
    map_location=device
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# تجهيز الصورة بنفس طريقة الـ Validation والـ Test
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# تصميم الواجهة
st.title("🌦️ Weather AI Classifier")
st.write(
    "Upload a weather image and let the CNN model "
    "predict its weather category."
)

st.caption("AI-powered image classification | PyTorch CNN")

uploaded_file = st.file_uploader(
    "Upload a weather image",
    type=["jpg", "jpeg", "png", "webp"]
)

# تنفيذ التنبؤ بعد رفع الصورة
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:
        if st.button("Predict Weather", type="primary"):
            # تحويل الصورة إلى Tensor
            input_tensor = transform(image).unsqueeze(0).to(device)

            # تنفيذ التنبؤ بدون حساب Gradients
            with torch.no_grad():
                outputs = model(input_tensor)

                # تحويل النتائج إلى احتمالات
                probabilities = torch.softmax(outputs, dim=1)

                # استخراج الفئة صاحبة أعلى احتمال
                confidence, predicted_index = torch.max(
                    probabilities, dim=1
                )

            predicted_class = class_names[predicted_index.item()]
            confidence_percent = confidence.item() * 100

            st.success(f"Predicted Weather: {predicted_class}")
            st.metric(
                "Model Confidence",
                f"{confidence_percent:.2f}%"
            )

            st.subheader("Class Probabilities")

            # عرض احتمال كل فئة
            for index, class_name in enumerate(class_names):
                probability = probabilities[0][index].item()

                st.write(
                    f"{class_name}: {probability * 100:.2f}%"
                )

                st.progress(probability)

st.divider()

st.caption(
    "Note: Predictions are model estimates and may be incorrect."
)