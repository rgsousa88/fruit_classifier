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

        self.Wq = nn.Linear(in_features=input_dim, out_features=dim)
        self.Wk = nn.Linear(in_features=input_dim, out_features=dim)
        self.Wv = nn.Linear(in_features=input_dim, out_features=dim)

    def forward(self, x):        
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        att = torch.matmul(F.softmax(scores, dim=-1), V)
        
        return att
    
class NormBlock(nn.Module):
    def __init__(self, dim:int=512, eps:float = 1e-10):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        original_shape = x.shape
        needs_reshape = False
        
        if x.dim() == 5:
            x = x.squeeze(-1).squeeze(-1)
            needs_reshape = True
        
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        x_normalized = (x - mean) / (std + self.eps)
        y = self.gamma * x_normalized + self.beta

        if needs_reshape:
            y = y.unsqueeze(-1).unsqueeze(-1)
        
        return y
    
class EncoderLayer(nn.Module):
    def __init__(self, n_head_att:int = 8, input_dim:int = 512, att_dim:int = 64, hidden_dim:int = 2048):
        super().__init__()
        self.n_heads = n_head_att
        self.att_dim = att_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.att_layers = nn.ModuleList([AttetionHead(dim=self.att_dim, input_dim=self.input_dim) for i in range(self.n_heads)])
        self.att_projection = nn.Linear(n_head_att * att_dim, input_dim) 
        
        self.norm_block_1 = NormBlock(input_dim)
        self.norm_block_2 = NormBlock(input_dim)

        self.feed_forward = nn.Sequential(nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim),
                                         nn.GELU(),
                                         nn.Linear(in_features=self.hidden_dim, out_features=self.input_dim),
                                         nn.GELU())
        
    def forward(self, x):
        # Norm inputs
        x_flat = x.squeeze(-1).squeeze(-1)
        x_norm = self.norm_block_1(x_flat)
        
        # Pass inputs to multi-head attention
        head_outputs = []
        for head in self.att_layers:
            head_outputs.append(head(x_norm))
        
        att_concat = torch.cat(head_outputs, dim=-1)  # (b, p, n_heads * att_dim)
        att_out = self.att_projection(att_concat)
        x1 = att_out + x_flat  # (b, p, dim)
        
        # Norm 2
        x1_norm = self.norm_block_2(x1)
        
        # Feed forward
        y = self.feed_forward(x1_norm)  # (b, p, dim)
        
        # Residual connection 2
        y = y + x1  # (b, p, dim)
        
        # Restore original shape
        y = y.unsqueeze(-1).unsqueeze(-1)

        return y
        
class VisionTransformer(nn.Module):
    def __init__(self, n_patches:int = 64, n_encoders:int = 4, n_head_att:int = 8, input_dim:int = 512, hidden_dim:int=2048):
        super().__init__()
        self.n_encoders = n_encoders
        self.n_heads = n_head_att
        self.dim = input_dim
        self.n_patches = n_patches
        self.att_dim = input_dim // n_head_att
        self.ff_hidden_dim = hidden_dim
        self.patches_per_dim = int(math.sqrt(self.n_patches))

        self.input_embedding = nn.Sequential(nn.Conv2d(3,16, kernel_size=3),
                                             nn.ReLU(),
                                             nn.Conv2d(16,64, kernel_size=3),
                                             nn.ReLU(),
                                             nn.Conv2d(64,self.dim,kernel_size=1),
                                             nn.ReLU(),
                                             nn.AdaptiveAvgPool2d(output_size=(1,1)))
        
        self.class_token = nn.Parameter(torch.rand(1,1,self.dim,1,1))
        
        self.pos_embedding = nn.Parameter(torch.rand(self.n_patches+1,self.dim,1,1))

        encoders_layers = []
        for i in range(self.n_encoders):
            encoders_layers.append(EncoderLayer(n_head_att=self.n_heads,
                                    input_dim=self.dim,
                                    att_dim=self.att_dim,
                                    hidden_dim=self.ff_hidden_dim))
        
        self.encoder = nn.Sequential(*encoders_layers)
    
    def split_in_patches(self, x):
        b, c, h, w = x.shape
        patch_h = h // self.patches_per_dim
        patch_w = w // self.patches_per_dim
        
        patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        
        patches = patches.contiguous().view(b, c, self.n_patches, patch_h, patch_w)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        
        return patches
        
    def encoding_patch(self, x):
        embedding = self.input_embedding(x)
        return embedding
    
    def forward(self, x):
        n_batches = x.shape[0]
        
        # Input Patches spliting
        patches = self.split_in_patches(x)

        # Patches encoding
        input_enc = torch.stack([self.encoding_patch(patches[i]) for i in range(n_batches)], dim=0)

        # Appending Class Token
        class_token = self.class_token.repeat(n_batches,1,1,1,1)
        input_enc = torch.cat((class_token, input_enc), dim=1)

        # Adding Pos. Embedding
        input_enc = input_enc + self.pos_embedding
        
        #Encoding with MHA
        output_enc = self.encoder(input_enc)

        return output_enc
    
class ViTClassifier(nn.Module):
    def __init__(self, n_classes:int = 5, input_size=224, n_patches:int = 16, n_encoders:int = 4, n_heads:int = 8, emb_dim:int = 512):
        super().__init__()
        self.n_classes = n_classes
        self.input_size = input_size
        self.n_encoders = n_encoders
        self.n_heads = n_heads
        self.n_patches = n_patches
        self.emb_dim = emb_dim
        self.flatten_size = (1 + self.n_patches) * self.emb_dim
        
        assert self.input_size % int(math.sqrt(n_patches)) == 0

        self.vit = VisionTransformer(n_encoders=self.n_encoders,
                                     n_patches=self.n_patches,
                                     n_head_att=self.n_heads,
                                     input_dim=self.emb_dim)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(emb_dim // 2, emb_dim // 4),
            nn.GELU(), 
            nn.Dropout(0.2),
            nn.Linear(emb_dim // 4, n_classes)
        )
        
    def forward(self, x):
        # Passing through Transformer encoding
        x_encod = self.vit(x)

        #Performing classification with all token
        x_encod = x_encod.squeeze(-1).squeeze(-1)  # (b, n_patches+1, emb_dim)
        
        # Extracting class token
        class_token = x_encod[:, 0, :]
        
        # Classifing
        y = self.classifier(class_token)

        return y
 




                                            
