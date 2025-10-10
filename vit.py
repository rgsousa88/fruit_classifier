import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttetionHead(nn.Module):
    def __init__(self, dim:int = 64, input_dim:int=512):
        super().__init__()
        self.dim = dim
        self.input_dim = input_dim
        self.scale = 1.0 / math.sqrt(dim)

        self.Wq = nn.Parameter(torch.rand(input_dim, dim), requires_grad=True)
        self.Wk = nn.Parameter(torch.rand(input_dim, dim), requires_grad=True)
        self.Wv = nn.Parameter(torch.rand(input_dim, dim), requires_grad=True)

    def forward(self, x):
        Q = torch.matmul(x, self.Wq)
        K = torch.matmul(x, self.Wk)
        V = torch.matmul(x, self.Wv)

        V_proj = self.scale * torch.matmul(Q, torch.transpose(K,1,2))
        att = torch.matmul(V_proj.softmax(dim=0), V)

        return att
    
class NormBlock(nn.Module):
    def __init__(self,):
        super().__init__()
        self.gamma = nn.Parameter(torch.rand(1), requires_grad=True)
        self.beta = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x):
        std_val, mean_val = torch.std_mean(x)
        y = self.gamma * ((x - mean_val)/std_val) + self.beta
        return y
    
class EncoderLayer(nn.Module):
    def __init__(self, n_head_att:int = 8, input_dim:int = 512, att_dim:int = 64, hidden_dim:int = 2048):
        super().__init__()
        self.n_heads = n_head_att
        self.att_dim = att_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.att_layers = nn.ModuleList([AttetionHead(dim=self.att_dim, input_dim=self.input_dim) for i in range(self.n_heads)])
        self.norm_block_1 = NormBlock()
        self.norm_block_2 = NormBlock()

        self.feed_foward = nn.Sequential(nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(in_features=self.hidden_dim, out_features=self.input_dim),
                                         nn.ReLU())
        
    def forward(x, self):
        x = x.squeeze()
        att_out = torch.concatenate([self.att_layers[i](x) for i in range(self.n_heads)], dim=2)
        y = att_out + x
        y = self.norm_block_1(y)
        y = self.feed_foward(y) + y
        y = self.norm_block_2(y)

        return y
        
class VisionTransformer(nn.Module):
    def __init__(self, n_patches:int = 16, n_head_att:int = 8, input_dim:int = 512, hidden_dim:int=2048):
        super().__init__()
        self.n_heads = n_head_att
        self.dim = input_dim
        self.n_patches = n_patches
        self.att_dim = input_dim // n_head_att
        self.ff_hidden_dim = hidden_dim
        self.patche_per_dim = int(math.sqrt(self.n_patches))

        self.input_embedding = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3),
                                             nn.Conv2d(16,64, kernel_size=3),
                                             nn.Conv2d(64,self.dim,kernel_size=1),
                                             nn.AdaptiveAvgPool2d(output_size=(1,1)))
        
        self.pos_embedding = nn.Parameter(torch.rand(self.dim,1,1))

        self.encoder = EncoderLayer(n_head_att=self.n_heads,
                                    input_dim=self.dim,
                                    att_dim=self.att_dim,
                                    hidden_dim=self.ff_hidden_dim)
        
    def split_in_patches(self, x):
        b,c,h,w = x.shape
        patch_h = h // self.patche_per_dim
        patch_w = w // self.patche_per_dim

        patches = x.view(b,c,patch_h, patch_w,-1)
        return patches
    
    def encoding_patch(self, x):
        return self.input_embedding(x) + self.pos_embedding
    
    def forward(self, x):
        patches = self.split_in_patches(x)
        input_enc = torch.stack([self.encoding_patch(patches[:,:,:,:,i]) for i in range(self.n_patches)], dim=1)
        output_enc = self.encoder(input_enc)

        return output_enc
    
class ViTClassifier(nn.Module):
    def __init__(self, n_classes:int = 5, input_size=224, n_patches:int = 16, n_heads:int = 8, emb_dim:int = 512):
        super().__init__()
        self.n_classes = n_classes
        self.input_size = input_size
        self.n_patches = n_patches
        self.emb_dim = emb_dim
        self.flatten_size = self.n_patches * self.emb_dim
        
        assert self.input_size % int(math.sqrt(n_patches)) == 0

        self.vit = VisionTransformer(n_patches=self.n_patches,
                                     n_head_att=n_heads,
                                     input_dim=self.emb_dim)
        
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=self.flatten_size, out_features=self.emb_dim // 2),
                                        nn.ReLU(),
                                        nn.Linear(in_features=self.emb_dim//2, out_features=n_classes))
        
    def forward(self, x):
        x_encod = self.vit(x)
        y = self.classifier(x_encod)

        return y
 




                                            
