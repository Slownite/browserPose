import onnxruntime as ort
import numpy as np
session = ort.InferenceSession("x3d_m_simplified.onnx")
input_array = np.random.randn(1, 3, 16, 256, 256).astype(np.float32)
outputs = session.run(None, {"input": input_array})
print(outputs)

