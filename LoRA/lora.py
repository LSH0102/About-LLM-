import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class LoRALinear(nn.Linear):

    def __init__(self,
                 # nn.Linear parameters
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 # LoRA parameters
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                ) -> None:
        
        #TODO: Initialize the inherited class, nn.linear 
        
        super().__init__(in_features=in_features, out_features=out_features,bias=bias,device=device)

        self.has_weights_merged = False
        if lora_rank > 0:
            self.lora_dropout = lora_dropout

            self.lora_scaling = lora_alpha/lora_rank
            

            #TODO: Fill in the "..."
            self.lora_A = torch.zeros(size=(lora_rank,in_features),device=device)
                                                                     
            self.lora_B = torch.zeros(size=(out_features,lora_rank),device=device)

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            #TODO: Initialize both lora_A and lora_B with torch.nn.init. Refer to the paper to see how each is initialize
            #Hint: lora_A is initialized using kaiming_uniform_ using negative slope (a) as math.sqrt(5)
            self.lora_A = torch.nn.Parameter(nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5)))
            self.lora_B = torch.nn.Parameter(self.lora_B)
            self.lora_A.requires_grad=False
            self.lora_B.requires_grad=False
            

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #TODO: return input after the forward pass
        #TODO: Remember to use dropout on the input before multiplying with lora_B and lora_A if the weights are not merged
        if self.is_lora():
                
            if self.has_weights_merged==False:
                drop=nn.Dropout(self.lora_dropout)
                
                return super().forward(input)+self.lora_scaling*torch.matmul(torch.matmul(input,self.lora_A.T),self.lora_B.T)
            else:
                return super().forward(input)
        else:
            if self.bias!=None:
                return torch.matmul(input, self.weight.T)+self.bias
            else:
                return torch.matmul(input, self.weight.T)

    def train(self, mode: bool = True) -> "LoRALinear":
        #TODO: Set the linear layer into train mode
        #Hint: Make sure to demerge LORA matrices if already merged
        if self.is_lora():
            if self.has_weights_merged:
                self.weight.data=self.weight.data-torch.matmul(self.lora_B.data,self.lora_A.data)*self.lora_scaling
                self.has_weights_merged=False
        
            if mode==True:
                for param in self.parameters():
                    param.requires_grad = True
            return self
        else:
            if mode==True:
                for param in self.parameters():
                    param.requires_grad = True
            return self

    def eval(self) -> "LoRALinear":
        #TODO: Set the linear layer into eval mode
        #Hint: Make sure to merge LORA matrices if already demerged
        if self.is_lora():
            if self.has_weights_merged==False:
                self.weight.data=self.weight.data+torch.matmul(self.lora_B.data,self.lora_A.data)*self.lora_scaling
                self.has_weights_merged=True
        self.training=False
        
        return self
    
    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
        return out

def mark_only_lora_as_trainable(model: nn.Module) -> nn.Module:
    #TODO: Loop through parameters and mark some as trainable. Which ones should these be?
    #Hint: How do you mark a parameter as trainable (or not trainable)?
    for name, module in model.named_modules():
        
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad=True
            module.lora_B.requires_grad=True
            module.weight.requires_grad=False
            if hasattr(model, 'bias'):
                model.bias.requires_grad=False
        else:
            for param in module.parameters():
                param.requires_grad = False
    return model
