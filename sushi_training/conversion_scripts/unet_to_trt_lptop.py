#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import tensorrt as trt
import pycuda.autoinit  # initializes CUDA driver

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, rate, pooling=True):
        super(Encoder, self).__init__()
        self.pooling = pooling
        self.bn = nn.BatchNorm2d(in_channels)
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.drop = nn.Dropout(rate)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.bn(x)
        x = nn.ReLU()(self.c1(x))
        x = self.drop(x)
        x = nn.ReLU()(self.c2(x))
        if self.pooling:
            y = self.pool(x)
            return y, x
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, rate):
        super(Decoder, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.cT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.drop = nn.Dropout(rate)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip_x):
        x = self.bn(x)
        x = nn.ReLU()(self.cT(x))
        x = torch.cat([x, skip_x], dim=1)
        x = nn.ReLU()(self.c1(x))
        x = self.drop(x)
        x = nn.ReLU()(self.c2(x))
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.initial = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc1 = Encoder(64, 64, 0.1)
        self.enc2 = Encoder(64, 128, 0.1)
        self.enc3 = Encoder(128, 256, 0.2)
        self.enc4 = Encoder(256, 512, 0.2)
        self.enc5 = Encoder(512, 512, 0.3, pooling=False)

        self.dec1 = Decoder(512, 512, 512, 0.2)
        self.dec2 = Decoder(512, 256, 256, 0.2)
        self.dec3 = Decoder(256, 128, 128, 0.1)
        self.dec4 = Decoder(128, 64, 64, 0.1)

        self.final = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.ReLU()(self.initial(x))
        p1, c1 = self.enc1(x)
        p2, c2 = self.enc2(p1)
        p3, c3 = self.enc3(p2)
        p4, c4 = self.enc4(p3)
        e = self.enc5(p4)

        d1 = self.dec1(e, c4)
        d2 = self.dec2(d1, c3)
        d3 = self.dec3(d2, c2)
        d4 = self.dec4(d3, c1)

        return torch.sigmoid(self.final(d4))


def convert_to_onnx(model, input_shape, onnx_path):
    model.eval()
    dummy_input = torch.randn(input_shape).cuda()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        opset_version=12,
        input_names=["input"],
        output_names=["output"]
    )
    print(f"✅ ONNX model saved at: {onnx_path}")


def convert_to_tensorrt_py(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    # Set workspace memory limit to 1 GiB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print("⏳ Building TensorRT engine in Python...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build serialized engine")

    # Optional: Deserialize to verify
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError("Failed to deserialize engine")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        print(f"✅ TensorRT engine saved at: {engine_path}")


def main():
    # Paths
    model_path = 'water_segmentation_model.pth'
    onnx_path = 'water_segmentation_model.onnx'
    engine_path = 'water_segmentation_model.engine'

    # Load PyTorch model
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    # Convert to ONNX
    convert_to_onnx(model, (1, 3, 256, 256), onnx_path)

    # Build TRT engine in Python
    convert_to_tensorrt_py(onnx_path, engine_path)


if __name__ == '__main__':
    main()

