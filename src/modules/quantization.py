import torch

def signed_quantization(x, s, qmax):
        qmax = qmax.to("cuda")
        x_affine = x / s
        qmin = -qmax
        x_int = torch.clamp(torch.round(x_affine),min = qmin, max = qmax)
        return x_int

def unsigned_quantization(x, s, zpn, qmax):
        x_affine = x / s - zpn
        x_int = torch.clamp(torch.round(x_affine), 0, qmax)
        return x_int