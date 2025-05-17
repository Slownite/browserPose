# BrowserPose: Real-Time Action Recognition in the Browser

**BrowserPose** is a full-stack, privacy-preserving, browser-based action recognition tool. It uses the [X3D-M](https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html#x3d) model exported from PyTorch and runs inference directly in the browser using [ONNX.js](https://github.com/microsoft/onnxruntime/tree/main/js).

No backend. No server. Just JavaScript, ONNX, and your webcam.

---

## 📁 Project Structure

```
├── export_X3DM.py              # Script to export PyTorch X3D-M model to ONNX
├── flake.nix, flake.lock       # Nix environment files
├── test_onnx_inference.py      # Local test script for ONNX runtime validation
├── public/
│   ├── index.html              # Tailwind-styled frontend UI
│   ├── app.js                  # JS logic: webcam, preprocessing, ONNX.js inference
│   ├── labels.json             # Class index → action label mapping
│   └── model/
│       └── x3d_m_softmax.onnx  # ONNX model (quantized with softmax)
└── venv/                       # Python virtual environment (not required for browser runtime)
```

---

## 🚀 Features

- 🎥 Real-time webcam capture
- 🎞️ 16-frame buffer for action sequences
- 🧠 Inference with ONNX.js + WebAssembly
- 🧱 Clean UI with Tailwind CSS (Material-style layout)
- ⚡ Fully client-side — no server or data upload

---

## 📦 Setup & Run (Local)

> ONNX.js requires the page to be served via HTTP(S), not `file://`.

### 1. Install Python environment (for export/testing)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # If you have one
```

### 2. Export the model (if needed)
```bash
python export_X3DM.py
```

### 3. Serve frontend from `/public`
```bash
cd public
python -m http.server
```

Then open: [http://localhost:8000](http://localhost:8000)

---

## 🧪 Test ONNX Model (Optional)

Run `test_onnx_inference.py` to validate your exported model using ONNX Runtime before deploying to browser:

```bash
python test_onnx_inference.py
```

---

## 🧠 Model Details

- Model: X3D-M from `facebookresearch/pytorchvideo`
- Input shape: `[1, 3, 16, 256, 256]`
- Output: `[1, N]` logits → softmax
- Quantized for in-browser performance

---

## 🔧 Customization

### ➕ Replace model
- Replace `public/model/x3d_m_softmax.onnx` with any `[1, 3, T, H, W]` video model
- Adjust `NUM_FRAMES`, width/height in `app.js` if needed

### 🏷 Customize labels
- Edit `public/labels.json` to match your model’s classes:
```json
{
  "0": "waving",
  "1": "drinking",
  ...
}
```

---

## ✨ UI Preview

The UI is styled using Tailwind CSS with a card-based layout, large webcam preview, and material-style prediction feedback.

---

## 📄 License

MIT — feel free to reuse and modify.

---

## 🙋‍♂️ Author

**Samuel Diop**  
Ph.D. Researcher in Computer Vision and Deep Learning  
[Website](http://samueldiop.com) · [GitHub](https://github.com/Slownite) · [LinkedIn](https://www.linkedin.com/in/samuel-diop)
