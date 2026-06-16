import torch
import numpy as np
from torch import nn
import modules.observers as observers
import modules.functions as functions
import modules.quantization as quantization


# Conv Type list:
# - 1 : standard convolution
# - 2 : quantized convolution no error
# - 3 : quantized convolution error STE gradient
# - 4 : quantized convolution error aware gradient

#TODO Rendere parametrico anche matrice di approx_mult
class Conv2d_custom(nn.Conv2d):
    def __init__(self,channel_in,
                channel_out,
                kernel_size,
                stride,
                padding,
                bias,
                conv_type,
                bit_width,
                multiplier_matrix,
                signed = False,
                name = None):
        
        super().__init__(channel_in,channel_out,kernel_size,stride,padding,bias = bias)
        self.activation_observer = observers.MovingAverageMinMaxObserver(q_level="L", out_channels=None)
        self.weight_observer = observers.MinMaxObserver(q_level="L", out_channels=None)
        self.eps = torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32)
        self.activation_quant_max = torch.tensor(((1 << bit_width) - 1), dtype=torch.float32)
        self.weight_quant_max = torch.tensor(((1 << bit_width) - 1), dtype=torch.float32)
        self.signed = signed
        self.bit_width = bit_width
        self.multiplier_matrix = multiplier_matrix

        self.name = name
        self.conv_type = conv_type
        if(conv_type == 1):
            self.conv2d_op = None
        elif(conv_type == 2):
            self.conv2d_op = functions.QuantizedConv2d
        elif(conv_type == 3):
            self.conv2d_op = functions.ApproxConv2dSTE
        elif(conv_type == 4):
            self.conv2d_op = functions.ApproxConv2d
        elif(conv_type == 5):
            self.conv2d_op = functions.StatsQuantizedConv2d
        else:
            raise(NotImplementedError) 
        

    def forward(self, input):
        if(self.conv_type == 1):
            return nn.functional.conv2d(input=input, 
                                        weight=self.weight,
                                        bias=self.bias,
                                        stride=self.stride,
                                        padding=self.padding)
        #Handling forward operatation for getting heat maps
        if(self.training and self.conv_type == 5):
            self.conv2d_op = functions.QuantizedConv2d
        elif(not self.training and (self.conv_type == 5)):
            self.conv2d_op = functions.StatsQuantizedConv2d
        if(self.conv2d_op == None):
            return nn.functional.conv2d(input=input, 
                                        weight=self.weight,
                                        bias=self.bias,
                                        stride=self.stride,
                                        padding=self.padding)
        #Updating min max of the Observes
        if(self.training):
            self.activation_observer(input)
            self.activation_scale = torch.max((self.activation_observer.max_val - self.activation_observer.min_val) / self.activation_quant_max, self.eps)
            self.activation_zp_neg = torch.round(self.activation_observer.min_val / self.activation_scale)
            self.weight_observer(self.weight)
            self.weight_scale = torch.max((self.weight_observer.max_val - self.weight_observer.min_val) / self.weight_quant_max, self.eps)
            self.weight_zp_neg = torch.round(self.weight_observer.min_val / self.weight_scale)
        #Quantizing inputs and weights
        if(self.signed):
            input_int = quantization.signed_quantization(input, self.activation_scale, self.activation_quant_max)
            weight_int = quantization.signed_quantization(self.weight, self.weight_scale, self.weight_quant_max)
        else:
            input_int = quantization.unsigned_quantization(input, self.activation_scale, self.activation_zp_neg, self.activation_quant_max)
            weight_int = quantization.unsigned_quantization(self.weight, self.weight_scale, self.weight_zp_neg, self.weight_quant_max)

        
        return self.conv2d_op.apply(input,
                                    self.weight,
                                    input_int,
                                    weight_int,
                                    self.bias, 
                                    self.stride, 
                                    self.padding,
                                    self.activation_scale,
                                    self.weight_scale,
                                    self.activation_zp_neg,
                                    self.weight_zp_neg,
                                    self.signed,
                                    self.bit_width,
                                    self.name,
                                    self.multiplier_matrix) 

