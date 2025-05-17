import torch
import torch.nn.functional as F

class X3DWithSoftmax(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        logits = self.base(x)
        return F.softmax(logits, dim=1)
model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
model = X3DWithSoftmax(model)
model.eval()

# Dummy input: [batch_size, channels, time, height, width]
dummy_input = torch.randn(1, 3, 16, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    "x3d_m.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=11
)
