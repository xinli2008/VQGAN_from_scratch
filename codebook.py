import torch
import torch.nn as nn

class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        """
        Perform codebook forward
        Input:
            z: torch.Tensor, [b, c, h, w]
        Output:
            z_q: 量化后的z
            min_encoding_indices: z在codebook中最小距离的索引
            loss: codrbook的loss
        """
        z = z.permute(0, 2, 3, 1).contiguous()   # [b, h, w, c]
        b, h, w, c = z.shape

        # reshaping embedding weight and z for broadcasting
        embedding = self.embedding.weight.data
        k, _ = embedding.shape

        embedding_broadcast = embedding.view(1, k, c, 1, 1)
        z_broadcast = z.view(b, 1, c, h, w)

        # NOTE: 量化过程中选择最近邻的过程: 计算每个编码特征向量和所有离散的codebook嵌入向量之间的距离, 并找到距离最近的codebook向量
        distance = torch.sum((embedding_broadcast - z_broadcast) **2, dim = 2)
        min_encoding_indices = torch.argmin(distance, dim = 1)

        # NOTE: 找到了索引index后, 然后根据index去codebook中取元素
        z_q = self.embedding(min_encoding_indices).view(z.shape)  # [b, h, w, c]

        # 计算关于codebook的loss
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta* torch.mean((z_q - z.detach()) ** 2)

        # NOTE: 我们在回顾一下VQGAN的第一个阶段forward的过程:
        # encoder -> codebook -> decoder
        # 但是, 使用codebook得到zq, 这个过程是不可微的, 即无法直接通过zq计算反向传播的梯度。也就是说此时的zq和encoder之间反向传播已经断开了。
        # 所以, 使用下面的代码, 一方面可以让让z_q赋值得到正确的信息。另一方面, 使用detach方法后, 控制z_q - z不计算梯度, 只有编码器部分的z有梯度。
        # 因此, 模型可以正确反向传播, 并且更新encoder的参数。
        z_q = z + (z_q - z).detach()   # [b, h, w, c]

        z_q = z_q.permute(0, 3, 1, 2)  # [b, c, h, w]

        return z_q, min_encoding_indices, loss