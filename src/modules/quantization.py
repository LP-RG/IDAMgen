import torch

def signed_quantization(x, s, qmax):
        qmax = qmax.to("cuda")
        x_affine = x / s
        qmin = -qmax
        x_int = torch.clamp(torch.round(x_affine),min = qmin, max = qmax)
        return x_int

def unsigned_quantization(x, s, zpn, qmin, qmax):
        torch.clamp(x, qmin, qmax)
        x_affine = x / s - zpn
        x_int = (torch.round(x_affine))
        return x_int