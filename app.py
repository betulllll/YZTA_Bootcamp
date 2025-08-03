import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

@st.cache(allow_output_mutation=True)
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

st.title("Beyin MRI Görüntü Analizi")
st.write("Lütfen bir MRI görüntüsü yükleyin, model anormallik varsa tespit etsin.")

uploaded_file = st.file_uploader("MRI görüntüsü yükleyin (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # PIL -> NumPy
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görüntü", use_column_width=True)

    image_np = np.array(image)
    results = model(image_np)
    result_img = results[0].plot()
    st.image(result_img, caption="Tespit Sonucu", use_column_width=True)

    # 📋 Etiket ve güven skoru göster
    st.subheader("📝 Tespit Özeti")
    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        label = model.names[class_id]
        st.write(f"👉 Sınıf: `{label}` | Confidence: **{confidence:.2f}**")

        # İsteğe bağlı geri bildirim mesajı
        feedback = {
        "glioma": "⚠️ Glioma türü bir tümör tespit edildi. Bu genellikle beyin dokusundan kaynaklanan ciddi bir tümördür. En kısa sürede nöroloji ya da onkoloji uzmanına başvurulması önerilir.",
        "meningioma": "ℹ️ Meningioma tespit edildi. Bu genellikle iyi huylu olsa da beyin zarı üzerinde gelişen bir tümördür. Tedavi ve izlem için uzman değerlendirmesi önemlidir.",
        "pituitary": "ℹ️ Hipofiz bezi (pituitary) tümörü tespit edildi. Hormon dengesi üzerinde etkili olabilir. Endokrinoloji uzmanı tarafından değerlendirilmesi önerilir.",
        "no tumor": "✅ Herhangi bir tümör belirtisi tespit edilmedi. Sonuç normal görünmektedir."
    }

        st.success(feedback.get(label, "🧾 Ek bilgi bulunamadı."))
else:
    st.info("Lütfen bir MRI görüntüsü yükleyin.")
