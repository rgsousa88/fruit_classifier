import torch
import torch.nn as nn
import math

class ComplexExponentialPositionalEncoding(nn.Module):
    """
    Positional encoding usando exponenciais complexas que codifica:
    1. Informações de posição (i, j) em diferentes frequências
    2. Distância do ponto à origem (módulo)
    
    Para uso em Vision Transformers (ViT)
    """
    
    def __init__(self, embedding_dim, max_seq_len=1000, alpha=0.1, base_freq=10000.0):
        """
        Args:
            embedding_dim: Dimensionalidade do embedding de posição
            max_seq_len: Comprimento máximo da sequência (para pré-computação)
            alpha: Fator de escala para a componente de distância
            base_freq: Frequência base para as exponenciais
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.alpha = alpha
        self.base_freq = base_freq
        
        # Garantir que embedding_dim seja par
        assert embedding_dim % 2 == 0, "embedding_dim deve ser par"
        
        # Pré-calcular frequências
        self.freqs_i = self._compute_frequencies(embedding_dim // 2, base_freq)
        self.freqs_j = self._compute_frequencies(embedding_dim // 2, base_freq * 2)  # Frequência diferente para j
        
        # Buffers para serem incluídos no state_dict
        self.register_buffer('freqs_i_buf', self.freqs_i)
        self.register_buffer('freqs_j_buf', self.freqs_j)
        
    def _compute_frequencies(self, dim, base_freq):
        """Calcula as frequências para as exponenciais complexas"""
        freqs = torch.zeros(dim)
        for k in range(dim):
            freqs[k] = 1.0 / (base_freq ** (2 * k / dim))
        return freqs
    
    def _compute_distance(self, i, j):
        """Calcula a distância Euclidiana normalizada"""
        # Normaliza i e j para o intervalo [0, 1]
        i_norm = i / self.max_seq_len
        j_norm = j / self.max_seq_len
        return torch.sqrt(i_norm**2 + j_norm**2)
    
    def forward(self, height, width):
        """
        Gera encoding posicional para um grid de tamanho height x width
        
        Args:
            height: Altura do grid (número de patches na vertical)
            width: Largura do grid (número de patches na horizontal)
            
        Returns:
            Tensor de shape (height, width, embedding_dim)
        """
        # Criar grid de posições
        positions_i = torch.arange(height).float()
        positions_j = torch.arange(width).float()
        
        # Calcular componentes complexas para i e j
        encoding_i = torch.zeros(height, width, self.embedding_dim // 2, 2)
        encoding_j = torch.zeros(height, width, self.embedding_dim // 2, 2)
        
        for i in range(height):
            for j in range(width):
                # Componente para i (frequência base)
                angle_i = positions_i[i] * self.freqs_i
                encoding_i[i, j, :, 0] = torch.cos(angle_i)  # Parte real
                encoding_i[i, j, :, 1] = torch.sin(angle_i)  # Parte imaginária
                
                # Componente para j (frequência dobrada)
                angle_j = positions_j[j] * self.freqs_j
                encoding_j[i, j, :, 0] = torch.cos(angle_j)  # Parte real
                encoding_j[i, j, :, 1] = torch.sin(angle_j)  # Parte imaginária
                
                # Adicionar componente de distância (como fator de amplitude)
                distance = self._compute_distance(positions_i[i], positions_j[j])
                encoding_i[i, j] *= (1 + self.alpha * distance)
                encoding_j[i, j] *= (1 + self.alpha * distance)
        
        # Combinar componentes (concatenação real/imaginária)
        encoding_i_flat = encoding_i.view(height, width, -1)
        encoding_j_flat = encoding_j.view(height, width, -1)
        
        # Soma das componentes (f(i,j) = componente_i + componente_j)
        positional_encoding = encoding_i_flat + encoding_j_flat
        
        return positional_encoding
    
    def forward_learnable(self, height, width):
        """
        Versão com parâmetros aprendíveis para as frequências
        """
        if not hasattr(self, 'learnable_freqs_i'):
            self.learnable_freqs_i = nn.Parameter(torch.randn(self.embedding_dim // 2) * 0.02)
            self.learnable_freqs_j = nn.Parameter(torch.randn(self.embedding_dim // 2) * 0.02)
        
        # Similar ao forward, mas usando parâmetros aprendíveis
        positions_i = torch.arange(height).float()
        positions_j = torch.arange(width).float()
        
        encoding_i = torch.zeros(height, width, self.embedding_dim // 2, 2)
        encoding_j = torch.zeros(height, width, self.embedding_dim // 2, 2)
        
        for i in range(height):
            for j in range(width):
                angle_i = positions_i[i] * self.learnable_freqs_i
                encoding_i[i, j, :, 0] = torch.cos(angle_i)
                encoding_i[i, j, :, 1] = torch.sin(angle_i)
                
                angle_j = positions_j[j] * self.learnable_freqs_j
                encoding_j[i, j, :, 0] = torch.cos(angle_j)
                encoding_j[i, j, :, 1] = torch.sin(angle_j)
                
                distance = self._compute_distance(positions_i[i], positions_j[j])
                encoding_i[i, j] *= (1 + self.alpha * distance)
                encoding_j[i, j] *= (1 + self.alpha * distance)
        
        encoding_i_flat = encoding_i.view(height, width, -1)
        encoding_j_flat = encoding_j.view(height, width, -1)
        
        positional_encoding = encoding_i_flat + encoding_j_flat
        
        return positional_encoding


# Exemplo de uso em uma ViT
class ViTWithComplexPositionalEncoding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, dim=768, depth=8):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        
        self.patch_embedding = nn.Linear(self.patch_dim, dim)
        self.positional_encoding = ComplexExponentialPositionalEncoding(
            embedding_dim=dim, 
            max_seq_len=self.num_patches
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=8),
            num_layers=depth
        )
        
        self.classifier = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        # Dividir imagem em patches
        b, c, h, w = x.shape
        p = self.patch_size
        n_h, n_w = h // p, w // p
        
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(b, n_h * n_w, -1)
        
        # Embedding de patches
        patch_embeddings = self.patch_embedding(patches)
        
        # Adicionar positional encoding
        pos_encoding = self.positional_encoding(n_h, n_w)
        pos_encoding = pos_encoding.view(1, n_h * n_w, -1).expand(b, -1, -1).requires_grad_(False)
        
        cls_tokens = self.cls_token.expand(b, -1, -1)

        embeddings = patch_embeddings + pos_encoding.to(patch_embeddings.get_device())
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)

        output = self.transformer(embeddings)
        
        # Classificação (usando token [CLS] ou média)
        cls_output = output[:, 0, :]  # Assumindo que o primeiro token é [CLS]
        return self.classifier(cls_output)