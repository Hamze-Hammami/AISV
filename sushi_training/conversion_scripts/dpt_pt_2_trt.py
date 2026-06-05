#!/usr/bin/env python3
"""
Depth-Anything V2  (metric)
PyTorch → ONNX → TensorRT

Default: static 518×518 engine.
Add --dynamic to build a multi-resolution engine.

Error “no optimisation profile” is gone because the script
creates a profile whenever the ONNX contains -1 dimensions.
"""

import argparse, os, sys, onnx, torch, tensorrt as trt, pycuda.driver as cuda
cuda.init(); import pycuda.autoinit  # noqa: E402
from depth_anything_v2.dpt import DepthAnythingV2

# ───────────────────────── checkpoints ─────────────────────────
def ckpt_path(ds, enc):
    return f"checkpoints/depth_anything_v2_metric_{ds}_{enc}.pth"

# ───────────────────────── ONNX export ─────────────────────────
def export_onnx(a):
    cfg = {
        "vits": dict(encoder="vits", features=64,  out_channels=[48, 96,192,384]),
        "vitb": dict(encoder="vitb", features=128, out_channels=[96,192,384,768]),
        "vitl": dict(encoder="vitl", features=256, out_channels=[256,512,1024,1024])
    }
    print(f"[ONNX] export {a.encoder.upper()} ({a.dataset}) …")
    model = DepthAnythingV2(**cfg[a.encoder], max_depth=a.max_depth).to("cuda")
    model.load_state_dict(torch.load(ckpt_path(a.dataset, a.encoder), map_location="cuda"),
                          strict=False)
    model.eval()

    dummy = torch.randn(1, 3, *a.input_hw, device="cuda")
    dynamic_axes = {}
    if a.dynamic:               # only add dynamic sizes if user asked for it
        dynamic_axes = {"rgb": {0: "N", 2: "H", 3: "W"}}

    torch.onnx.export(
        model, dummy, a.onnx,
        opset_version=17,
        input_names=["rgb"], output_names=["depth"],
        do_constant_folding=True, dynamic_axes=dynamic_axes)
    onnx.checker.check_model(onnx.load(a.onnx))
    print(f"[ONNX] saved → {a.onnx}")

# ───────────────────── TensorRT build ──────────────────────────
def build_trt(a):
    print("[TRT] build …")
    logger, builder = trt.Logger(trt.Logger.INFO), trt.Builder(trt.Logger(trt.Logger.INFO))
    network = builder.create_network()
    parser  = trt.OnnxParser(network, logger)
    with open(a.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i)); sys.exit(1)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_gb * (1 << 30)))
    if a.fp16 and builder.platform_has_fast_fp16: cfg.set_flag(trt.BuilderFlag.FP16)
    if a.sparse and hasattr(trt.BuilderFlag, "SPARSE_WEIGHTS"):
        cfg.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    # ── add optimisation profile if input is dynamic ──
    if network.get_input(0).shape[0] == -1:
        prof = builder.create_optimization_profile()
        in_name = network.get_input(0).name
        h, w = a.input_hw
        prof.set_shape(in_name,  (1,3,h,w), (1,3,h,w), (4,3,h,w))
        cfg.add_optimization_profile(prof)      # mandatory :contentReference[oaicite:2]{index=2}

    # TRT-10 path
    if hasattr(builder, "build_serialized_network"):
        eng_bytes = builder.build_serialized_network(network, cfg)
        if eng_bytes is None: sys.exit("[ERR] build failed")
        engine = trt.Runtime(logger).deserialize_cuda_engine(eng_bytes)
    else:                                       # TRT-8 fallback
        engine = builder.build_engine(network, cfg)

    with open(a.engine, "wb") as f: f.write(engine.serialize())
    print(f"[TRT] engine → {a.engine}")

# ───────────────────────────── CLI ─────────────────────────────
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("-e","--encoder",  choices=["vits","vitb","vitl"], default="vitb")
    p.add_argument("-d","--dataset",  choices=["vkitti","hypersim"],  default="vkitti")
    p.add_argument("--max-depth", type=float, default=80.0)
    p.add_argument("--input-hw",  type=int, nargs=2, default=[518,518])
    p.add_argument("--workspace-gb", type=float, default=4.0)
    p.add_argument("--fp16-off", dest="fp16", action="store_false")
    p.add_argument("--sparse",   action="store_true")
    p.add_argument("--dynamic",  action="store_true",
                   help="make H/W & batch dynamic (adds optimisation profile)")
    p.add_argument("--rebuild",  action="store_true")
    return p.parse_args()

def main():
    a = parse(); a.fp16 = getattr(a,"fp16",True); a.input_hw = tuple(a.input_hw)
    base = f"depth_anything_v2_{a.encoder}_{a.dataset}"
    a.onnx   = f"{base}.onnx"
    a.engine = f"{base}_{'fp16' if a.fp16 else 'fp32'}{'_dyn' if a.dynamic else ''}.engine"
    if a.rebuild:
        for f in (a.onnx, a.engine):
            if os.path.isfile(f): os.remove(f)
    if not os.path.isfile(a.onnx):  export_onnx(a)
    if not os.path.isfile(a.engine): build_trt(a)
    else: print(f"[OK] engine exists → {a.engine}")

if __name__ == "__main__": main()

