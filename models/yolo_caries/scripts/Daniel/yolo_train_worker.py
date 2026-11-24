"""Train one Ultralytics run on one GPU. Started by yolo_oral_detector.ipynb (section 5), one process per GPU.

    python yolo_train_worker.py <job.json>

The notebook owns every setting: it writes job.json (pretrained weights, dataset yaml, Ultralytics train arguments,
run folder) and sets CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES so this process sees exactly one GPU (device 0).
A run whose last.pt is not marked finished (Ultralytics sets epoch = -1 when it strips a finished checkpoint)
resumes from it, so an interrupted sweep continues where it stopped.

AMD GPUs (ROCm): on the Radeon AI PRO R9700 with PyTorch 2.14 + ROCm 7.2, float16 depthwise / grouped convolutions
return wrong values for some shapes (e.g. 64 channels at 80x80 px: 107% forward error and NaN gradients against a
float64 CPU reference; float32 is exact; ordinary float16 convolutions and BatchNorm are fine to 0.1%). Ultralytics'
AMP check catches this and silently falls back to float32 training, which is 2.5-3x slower. bfloat16 is no way out:
it gives NaN losses from the first step. So on ROCm this worker runs grouped convolutions in float32 inside the float16
autocast region (they are a small share of the compute), which makes the AMP check pass, and keeps the per-epoch
validation in float32 (Ultralytics would run it in plain float16, where the grouped-convolution bug is not patched).
The MuSGD optimizer (Ultralytics' automatic choice for long runs) orthogonalises its updates in bfloat16; on ROCm that
step runs in float32 too (a precaution: it is a few small matrix products per step).

NaN guard (every GPU): the first yolo12m_v2 run (batch 32) went NaN at the start of epoch 3 and never recovered, and
Ultralytics' own NaN recovery missed it because it only checks the last batch's loss and the val fitness (which was
0, not NaN). Here the recovery also fires when the epoch's mean training loss is NaN/Inf: it reloads last.pt and
redoes the epoch, and after 3 failed attempts the run stops with an error instead of early-stopping on a broken model.
"""
import json
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


def rocm_fp16_fix(args):
    if not torch.version.hip or args.get("amp", True) is not True:
        return args
    conv_forward = torch.nn.Conv2d.forward

    def grouped_conv_fp32(self, x):
        if self.groups > 1 and torch.is_autocast_enabled("cuda"):
            with torch.autocast("cuda", enabled=False):
                return conv_forward(self, x.float())
        return conv_forward(self, x)

    torch.nn.Conv2d.forward = grouped_conv_fp32
    import ultralytics.engine.validator as V
    call = V.BaseValidator.__call__

    def fp32_validation(self, trainer=None, model=None, **kwargs):
        if trainer is None:
            return call(self, trainer, model, **kwargs)
        amp, trainer.amp = trainer.amp, False      # the validator runs in fp16 when trainer.amp is set
        try:
            return call(self, trainer, model, **kwargs)
        finally:
            trainer.amp = amp

    V.BaseValidator.__call__ = fp32_validation

    import ultralytics.optim.muon as M

    def newtonschulz_fp32(G, eps=1e-7):   # zeropower_via_newtonschulz5 with float32 instead of bfloat16
        X = G.reshape(-1, G.size(-2), G.size(-1)).float()
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
        if G.size(-2) > G.size(-1):
            X = X.transpose(-2, -1)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(5):
            A = X @ X.transpose(-2, -1)
            B = torch.baddbmm(A, A, A, beta=b, alpha=c)
            X = torch.baddbmm(X, B, X, beta=a)
        if G.size(-2) > G.size(-1):
            X = X.transpose(-2, -1)
        return X.reshape(G.shape)

    M.zeropower_via_newtonschulz5 = newtonschulz_fp32
    print("ROCm: float16 AMP with grouped convolutions in float32; validation and MuSGD orthogonalisation in float32",
          flush=True)
    return args


def finite_loss(tloss):
    # Ultralytics 8.4 keeps the epoch's mean loss as a dict {loss name: value}; older versions as one tensor
    values = tloss.values() if isinstance(tloss, dict) else [tloss]
    return all(torch.isfinite(torch.as_tensor(v)).all() for v in values)


def nan_guard():
    import ultralytics.engine.trainer as T
    handle = T.BaseTrainer._handle_nan_recovery

    def recover_on_epoch_mean(self, epoch):
        if self.tloss is not None and not finite_loss(self.tloss):
            self.loss = torch.tensor(float("nan"))   # make Ultralytics' check see what the epoch mean shows
        return handle(self, epoch)

    T.BaseTrainer._handle_nan_recovery = recover_on_epoch_mean


def main(job_file):
    job = json.loads(Path(job_file).read_text())
    run = Path(job["project"]) / job["name"]
    last, best = run / "weights" / "last.pt", run / "weights" / "best.pt"
    args = rocm_fp16_fix(job["args"])
    nan_guard()
    os.chdir(job["cwd"])          # Ultralytics drops yolo11n.pt (AMP check) and fonts into the working folder
    if last.exists() and torch.load(last, map_location="cpu", weights_only=False).get("epoch", -1) != -1:
        print(f"[{job['name']}] resuming from {last}", flush=True)
        YOLO(str(last)).train(resume=True)
    elif not best.exists():
        print(f"[{job['name']}] training {job['model']} on {job['data']}", flush=True)
        YOLO(job["model"]).train(data=job["data"], project=job["project"], name=job["name"], exist_ok=True, **args)
    print(f"[{job['name']}] finished", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
