# This project uses the structure of MUSE (https://github.com/facebookresearch/MUSE)

import torch
from torch import nn
import torch.nn.functional as F

from .utils import load_embeddings, normalize_embeddings


class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = params.emb_dim_autoenc
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)  

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.emb_dim = params.emb_dim
        self.bottleneck_dim = params.emb_dim_autoenc
        self.l_relu = params.l_relu
        
        self.encoder = nn.Linear(self.emb_dim, self.bottleneck_dim)
        self.leakyRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.encoder(x)
        if self.l_relu == 1:
            x = self.leakyRelu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.emb_dim = params.emb_dim
        self.bottleneck_dim = params.emb_dim_autoenc
        
        self.decoder = nn.Linear(self.bottleneck_dim, self.emb_dim)
        self.leakyRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.decoder(x)
        return x

def build_model(params, with_dis):
    """
    Build all components of the model.
    """
    # source embeddings
    src_dico, _src_emb = load_embeddings(params, source=True) 
    params.src_dico = src_dico
    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(_src_emb)

    # target embeddings
    if params.tgt_lang:
        tgt_dico, _tgt_emb = load_embeddings(params, source=False)
        params.tgt_dico = tgt_dico
        tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
        tgt_emb.weight.data.copy_(_tgt_emb)
    else:
        tgt_emb = None

    # mapping
    mapping_G = nn.Linear(params.emb_dim_autoenc, params.emb_dim_autoenc, bias=False)
    mapping_F = nn.Linear(params.emb_dim_autoenc, params.emb_dim_autoenc, bias=False)
    if getattr(params, 'map_id_init', True):
        mapping_G.weight.data.copy_(torch.diag(torch.ones(params.emb_dim_autoenc)))
        mapping_F.weight.data.copy_(torch.diag(torch.ones(params.emb_dim_autoenc)))
    
    # discriminator
    discriminator_A = Discriminator(params) if with_dis else None
    discriminator_B = Discriminator(params) if with_dis else None

    # autoencoder
    encoder_A = Encoder(params)
    decoder_A = Decoder(params)
    encoder_B = Encoder(params)
    decoder_B = Decoder(params)

    # cuda
    if params.cuda:
        src_emb.cuda()
        if params.tgt_lang:
            tgt_emb.cuda()
        mapping_G.cuda()
        mapping_F.cuda()
        if with_dis:
            discriminator_A.cuda()
            discriminator_B.cuda()
        encoder_A.cuda()
        decoder_A.cuda()
        encoder_B.cuda()
        decoder_B.cuda()

    # normalize embeddings
    params.src_mean = normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    if params.tgt_lang:
        params.tgt_mean = normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)

    return src_emb, tgt_emb, mapping_G, mapping_F, discriminator_A, discriminator_B, encoder_A, decoder_A, encoder_B, decoder_B
