from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
import clip

class FixedPositionalEncoding(nn.Module):
    def __init__(self, numfreq, end=10):
        super().__init__()
        """for each frequency we have num_coord*2 (cos and sin) terms"""
        exb = torch.linspace(0, numfreq-1, numfreq) / (numfreq-1) #  [0, ..., 15]/15
        self.sigma = torch.pow(end, exb).view(1, -1)  # (1x16)
        # for end=10: [[1=10^(0/15), 1.1659=10^(1/15)..., 10=10^(15/15)]] # geometric with factor 10^(1/15)
            # tensor([[ 1.0000,  1.1659,  1.3594,  1.5849,  1.8478,  2.1544,  2.5119,  2.9286,
            #           3.4145,  3.9811,  4.6416,  5.4117,  6.3096,  7.3564,  8.5770, 10.0000]])
            # divide above tensor by 2 -> frequency (num of periods in [0,1])
        self.sigma = torch.pi * self.sigma # (1x16)
        # NOTE: have the first sigma term be pi ( so that cos(pi*norm_ang) = cos(theta) )

        # ORIGINAL:
            # [0, 2, ..., 30] / 32
            # [1=10^(0/32), 1.1548=10^(2/32) ..., 8.6596=10^(30/32)] # geometric with factor 10^(2/32)
            # 2pi(10^(2/32))^0, 2pi(10^(2/32))^1, ..., 2pi(10^(2/32))^15
            # tensor([[ 6.2832,  7.2557,  8.3788,  9.6756, 11.1733, 12.9027, 14.8998, 17.2060,
            #          19.8692, 22.9446, 26.4960, 30.5971, 35.3329, 40.8019, 47.1172, 54.4101]])

    def forward(self, x):
        # x: B x N x 2
        B,N,_ = x.shape # _=2/4
        x = x.unsqueeze(-1)  # B x N x 2/4 x 1
        return torch.cat([
            torch.sin(x * self.sigma.to(x.device)),
            torch.cos(x * self.sigma.to(x.device))
        ], dim=-1).reshape(B,N,-1)  # B x N x 2/4 x (16+16) --> B x N x 64/128


class Transformer(nn.Module):
    def __init__(self, d_model= 512, nhead= 8, num_encoder_layers= 6,
                 dim_feedforward= 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, x, padding_mask=None):
        # for mod in self.encoder.layers:
        #     attn = mod.self_attn(x, x, x,
        #                    key_padding_mask=padding_mask,
        #                    need_weights=True)[1]
        #     x = mod(x, src_key_padding_mask=padding_mask)
        #     print(attn[:, 5:, 5:])
        # batch x sequence x feature (N, S, E)
        return self.encoder(x, src_key_padding_mask=padding_mask)
        # src_key_padding_mask: (N,S)
            # If a BoolTensor is provided, positions with True not allowed to attend while False values will be unchanged: [False ..., True, ...]
    

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Denoiser(nn.Module):
    def __init__(self,
                    ## choices
                    use_position = True, 
                    use_time = True,
                    use_text = True, 
                    ## type
                    type_dim = 8, 
                    max_position = 12, 
                    position_dim = 8, 
                    ## time
                    time_form = "prepend", # ["prepend", "concat", "add"]
                    time_emb_dim = 80, 
                    ## condition encoder
                    text_form = "word", # ["word", "sentence"]
                    vocab_size = [4, 4], 
                    version = "ViT-B/32",
                    cond_dim = 512,
                    ## encoder
                    # mask
                    mask_types = 3,
                    # property
                    cla_dim = 8,
                    siz_dim = 2,
                    # state
                    pos_dim = 2, 
                    ang_dim = 2, 
                    # structure
                    ang_initial_d = 128, 
                    siz_initial_unit = [64, 16], 
                    cla_initial_unit = [64], 
                    all_initial_unit = [512], 
                    final_lin_unit = [256, 4],
                    pe_numfreq= 16, 
                    pe_end=128,
                    ## transformer
                    transformer_heads = 8,
                    transformer_layers = 2,
                    dim_feedforward= 2048, 
                    dropout = 0.1, 
                    layer_norm_eps = 1e-5, 
                    batch_first = True,
                    ## predictor
                    use_two_branch = False
                    ):
        super().__init__()

        self.pos_dim, self.ang_dim, self.siz_dim, self.cla_dim = pos_dim, ang_dim, siz_dim, cla_dim
        all_initial_unit = [ e for e in all_initial_unit] # make copy as changed below
        transformer_width = all_initial_unit[-1]

        types = 2
        if use_text:
            types += 1
        self.use_position = use_position
        if use_position:
            self.position_embeddings = torch.nn.Embedding(max_position, position_dim)
        self.use_time = use_time
        if use_time:
            if time_form == "prepend":
                types += 1
            if time_form == "concat":
                all_initial_unit[-1] -= time_emb_dim
        
        self.types = types
        self.type_embeddings = torch.nn.Embedding(types, type_dim)
        all_initial_unit[-1] -= type_dim

        self.use_text = use_text
        self.use_two_branch = use_two_branch
        self.numoftoken = 0
        if use_text: 
            self.text_form = text_form
            if text_form == "word":
                self.word_embeddings = torch.nn.ModuleList([torch.nn.Embedding(v_size + 1, all_initial_unit[-1]) for v_size in vocab_size])
                self.vocab_size = vocab_size
                self.numoftoken = len(vocab_size)
            elif text_form == "sentence":
                self.condition_encoder, _ = clip.load(version, device="cpu")
                for param in self.condition_encoder.parameters():
                    param.requires_grad = False
                self.condition_mlp = torch.nn.Linear(cond_dim, all_initial_unit[-1])
                self.numoftoken = 1

        self.activation = torch.nn.LeakyReLU() # ReLU()
        self.pe = FixedPositionalEncoding(pe_numfreq, pe_end) 

        if use_time:
            self.time_form = time_form
            if time_form == "prepend" or time_form == "add":
                self.time_embeddings = nn.Sequential(
                    SinusoidalPositionEmbeddings(time_emb_dim),
                    nn.Linear(time_emb_dim, time_emb_dim),
                    nn.GELU(),
                    nn.Linear(time_emb_dim, all_initial_unit[-1]),
                )
            elif time_form == "concat":
                self.time_embeddings = nn.Sequential(
                    SinusoidalPositionEmbeddings(time_emb_dim),
                    nn.Linear(time_emb_dim, time_emb_dim),
                    nn.GELU(),
                    nn.Linear(time_emb_dim, time_emb_dim),
                )

        # 1a. pos (B x nmask x pos_d) -> (B x nmask x 2*pe_numfreq*(pos_dim)=128)

        # 1b. siz: (B x nmask x siz_d) -> (B x nmask x siz_initial_feat_d=128)
        if siz_initial_unit is None: # Positional encoding on size
            self.siz_initial_mask = None
            siz_initial_feat_d_mask = 2*pe_numfreq*siz_dim 
        else: # assume nonempty
            self.siz_initial_mask = [torch.nn.Linear(siz_dim, siz_initial_unit[0])]
            for i in range(1, len(siz_initial_unit)):
                self.siz_initial_mask.append(torch.nn.Linear(siz_initial_unit[i-1], siz_initial_unit[i]))
            self.siz_initial_mask = torch.nn.ModuleList(self.siz_initial_mask)
            siz_initial_feat_d_mask = siz_initial_unit[-1]

        # 1c. cla: (B x nmask x cla_d) -> (B x nmask x cla_initial_unit[-1]=128)
        self.cla_initial_mask = [torch.nn.Embedding(mask_types + 1, cla_initial_unit[0])]
        for i in range(1, len(cla_initial_unit)):
            self.cla_initial_mask.append(torch.nn.Linear(cla_initial_unit[i-1], cla_initial_unit[i]))
        self.cla_initial_mask = torch.nn.ModuleList(self.cla_initial_mask)

        # 2a. (B x nmask x initial_feat_input_d=128*5) -> (B x nmask x transformer_input_d) 
        initial_feat_input_d_mask = 2*pe_numfreq*(pos_dim) + siz_initial_feat_d_mask + cla_initial_unit[-1]
        self.all_initial_mask = [torch.nn.Linear(initial_feat_input_d_mask, all_initial_unit[0])]
        for i in range(1, len(all_initial_unit)):
            self.all_initial_mask.append(torch.nn.Linear(all_initial_unit[i-1], all_initial_unit[i]))
        self.all_initial_mask = torch.nn.ModuleList(self.all_initial_mask)
        
        # 1a. pos (B x nobj x pos_d) -> (B x nobj x 2*pe_numfreq*(pos_dim)=128)

        # 1b: ang: (B x nobj x ang_d//2) -> (B x nobj x 2*pe_numfreq*(ang_dim//2) -> 128)
        self.ang_initial = torch.nn.Linear(2*pe_numfreq*ang_dim//2, ang_initial_d)

        # 1c. siz: (B x nobj x siz_d) -> (B x nobj x siz_initial_feat_d=128)
        if siz_initial_unit is None: # Positional encoding on size
            self.siz_initial = None
            siz_initial_feat_d = 2*pe_numfreq*siz_dim 
        else: # assume nonempty
            self.siz_initial = [torch.nn.Linear(siz_dim, siz_initial_unit[0])]
            for i in range(1, len(siz_initial_unit)):
                self.siz_initial.append(torch.nn.Linear(siz_initial_unit[i-1], siz_initial_unit[i]))
            self.siz_initial = torch.nn.ModuleList(self.siz_initial)
            siz_initial_feat_d = siz_initial_unit[-1]

        # 1d. cla: (B x nobj x cla_d) -> (B x nobj x cla_initial_unit[-1]=128)
        self.cla_initial = [torch.nn.Linear(cla_dim, cla_initial_unit[0])]
        for i in range(1, len(cla_initial_unit)):
            self.cla_initial.append(torch.nn.Linear(cla_initial_unit[i-1], cla_initial_unit[i]))
        self.cla_initial = torch.nn.ModuleList(self.cla_initial)

        # 2a. (B x nobj x initial_feat_input_d=128*5) -> (B x nobj x transformer_input_d) 
        if use_position:
            all_initial_unit[-1] -= position_dim
        initial_feat_input_d = 2*pe_numfreq*(pos_dim) + ang_initial_d + siz_initial_feat_d + cla_initial_unit[-1]
        self.all_initial = [torch.nn.Linear(initial_feat_input_d, all_initial_unit[0])]
        for i in range(1, len(all_initial_unit)):
            self.all_initial.append(torch.nn.Linear(all_initial_unit[i-1], all_initial_unit[i]))
        self.all_initial = torch.nn.ModuleList(self.all_initial)

        # 3. (B x nobj(+1) x transformer_input_d) ->  (B x nobj(+1) x transformer_input_d) 
        self.transformer = Transformer(transformer_width, transformer_heads, transformer_layers, dim_feedforward, dropout, layer_norm_eps, batch_first)

        # 4. (B x nobj(+1) x transformer_input_d) -> (B x nobj x out_dim=pos_dim+ang_dim)
        if not self.use_two_branch:
            final_lin = [torch.nn.Linear(transformer_width, final_lin_unit[0])]
            for i in range(1, len(final_lin_unit)):
                final_lin.append(torch.nn.Linear(final_lin_unit[i-1], final_lin_unit[i]))
            self.final_lin = torch.nn.ModuleList(final_lin)
        else:
            final_lin_pos = [torch.nn.Linear(transformer_width, final_lin_unit[0])]
            final_lin_ang = [torch.nn.Linear(transformer_width, final_lin_unit[0])]
            for i in range(1, len(final_lin_unit)-1):
                final_lin_pos.append(torch.nn.Linear(final_lin_unit[i-1], final_lin_unit[i]))
                final_lin_ang.append(torch.nn.Linear(final_lin_unit[i-1], final_lin_unit[i]))
            final_lin_pos.append(torch.nn.Linear(final_lin_unit[-2], self.pos_dim))
            final_lin_ang.append(torch.nn.Linear(final_lin_unit[-2], self.ang_dim))

            self.final_lin_pos = torch.nn.ModuleList(final_lin_pos)
            self.final_lin_ang = torch.nn.ModuleList(final_lin_ang)

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # self.initialize_parameters()

    # def initialize_parameters(self):
    #     proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
    #     attn_std = self.transformer.width ** -0.5
    #     fc_std = (2 * self.transformer.width) ** -0.5
    #     for block in self.transformer.resblocks:
    #         nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
    #         nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
    #         nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
    #         nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    @property
    def dtype(self):
        return self.ang_initial.weight.dtype
    
    def encode_cond(self, cond, device):
        with torch.no_grad():
            cond = self.condition_encoder.encode_text(clip.tokenize(cond).to(device))
            cond = cond / cond.norm(dim=1, keepdim=True)
        return cond.detach()

    def forward(self, x, padding_mask, inpaint_mask, padding_inpant, t, condition = None, un_condition = None, word = None, cond_scale_guidance = 1.0):
        """ x           : [batch_size, maxnumobj, pos+ang+siz+cla]
            padding_mask: [batch_size, maxnumobj], for nn.TransformerEncoder (False: not masked, True=masked, not attended to)
            inpaint_mask: [batch_size, maxnummask, pos+ang+siz+1]
            padding_inpant: [batch_size, maxnummask], for nn.TransformerEncoder (False: not masked, True=masked, not attended to)
            t           : [batch_size, ]
            condition          : [batch_size, cond]
            un_condition          : [batch_size, cond]
            word          : [batch_size, len(word)]
        """

        # print(f"TransformerWrapper: x.shape={x.shape}")
        # 1a. pos (B x nmask x pos_d) -> (B x nmask x 2*pe_numfreq*(pos_dim)=128)
        pos_mask = self.pe(inpaint_mask[:, :, :self.pos_dim])

        # 1b. siz: (B x nmask x siz_d) ->  (B x nmask x siz_initial_feat_d=128)
        siz_mask = inpaint_mask[:,:, self.pos_dim+self.ang_dim : self.pos_dim+self.ang_dim+self.siz_dim]
        if self.siz_initial_mask is None:
            siz_mask = self.pe(siz_mask) # [B, nmask, pe_terms=2*2freq]
        else:
            for i in range(len(self.siz_initial_mask)-1):
                siz_mask = self.activation(self.siz_initial_mask[i](siz_mask)) # changes last dim
            siz_mask = self.siz_initial_mask[-1](siz_mask)
            
        # 1c. cla: (B x nmask x cla_d) ->  (B x nmask x cla_initial_unit[-1]=128)
        cla_mask =  inpaint_mask[:,:, self.pos_dim+self.ang_dim+self.siz_dim].to(torch.long)
        for i in range(len(self.cla_initial_mask)-1):
            cla_mask = self.activation(self.cla_initial_mask[i](cla_mask)) # [B, nmask, cla_feat_d]
        cla_mask = self.cla_initial_mask[-1](cla_mask)

        # 2a. (B x nmask x initial_feat_input_d=128*5) -> (B x nmask x transformer_input_d=d_model(-type)) 
        initial_feat_mask = torch.cat([pos_mask, siz_mask, cla_mask], dim=-1)
        for i in range(len(self.all_initial_mask)-1):
            initial_feat_mask = self.activation(self.all_initial_mask[i](initial_feat_mask)) 
        initial_feat_mask = self.all_initial_mask[-1](initial_feat_mask) # [B, nmask, 512(-type)]

        # if initial_feat_mask.shape[1] > 1:
        #     # 1: floorplan_class
        #     floor_index = torch.argmax((inpaint_mask[:, :, self.pos_dim+self.ang_dim+self.siz_dim] == 1).int(), dim=-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, initial_feat_mask.shape[-1])
        #     uncond_inpaint_mask = torch.gather(initial_feat_mask, dim=1, index=floor_index).repeat(1, initial_feat_mask.shape[1], 1)
        #     if self.training:
        #         initial_feat_mask = uncond_inpaint_mask + (torch.rand((x.shape[0], 1, 1)).to(x.device) > 0.3).int() * (initial_feat_mask - uncond_inpaint_mask)
        #     else:
        #         initial_feat_mask = uncond_inpaint_mask + cond_scale_guidance * (initial_feat_mask - uncond_inpaint_mask)

        type_index_mask = torch.zeros((initial_feat_mask.shape[0], ), dtype=torch.long).to(x.device)
        type_embed_mask = self.type_embeddings(type_index_mask).unsqueeze(1).repeat(1, initial_feat_mask.shape[1], 1) # [B, initial_feat_mask, type]

        # 1a. pos (B x nobj x pos_d) -> (B x nobj x 2*pe_numfreq*(pos_dim)=128)
        pos = self.pe(x[:, :, :self.pos_dim])

        # 1b: ang: (B x nobj x ang_d//2) -> (B x nobj x 2*pe_numfreq*(ang_dim//2) -> 128)
        ang_rad_pi = torch.unsqueeze(torch.atan2(x[:,:,self.pos_dim+1], x[:,:,self.pos_dim]), 2) # [B, numobj, 1], [-pi, pi]
        ang = self.pe(ang_rad_pi/torch.pi) # in [-1, 1]
        ang = self.ang_initial(ang)

        # 1c. siz: (B x nobj x siz_d) ->  (B x nobj x siz_initial_feat_d=128)
        siz = x[:,:, self.pos_dim+self.ang_dim : self.pos_dim+self.ang_dim+self.siz_dim]
        if self.siz_initial is None:
            siz = self.pe(siz) # [B, nobj, pe_terms=2*2freq]
        else:
            for i in range(len(self.siz_initial)-1):
                siz = self.activation(self.siz_initial[i](siz)) # changes last dim
            siz = self.siz_initial[-1](siz)
            
        # 1d. cla: (B x nobj x cla_d) ->  (B x nobj x cla_initial_unit[-1]=128)
        cla =  x[:,:, self.pos_dim+self.ang_dim+self.siz_dim:self.pos_dim+self.ang_dim+self.siz_dim+self.cla_dim]
        for i in range(len(self.cla_initial)-1):
            cla = self.activation(self.cla_initial[i](cla)) # [B, nobj, cla_feat_d]
        cla = self.cla_initial[-1](cla)

        # 2a. (B x nobj x initial_feat_input_d=128*5) -> (B x nobj x transformer_input_d=d_model(-type)) 
        initial_feat = torch.cat([pos, ang, siz, cla], dim=-1)
        for i in range(len(self.all_initial)-1):
            initial_feat = self.activation(self.all_initial[i](initial_feat)) 
        initial_feat = self.all_initial[-1](initial_feat) # [B, nobj, 512(-pos-type)]
        if self.use_position:
            position_index = torch.arange(initial_feat.shape[1], dtype=torch.long, device=x.device).unsqueeze(0).repeat(initial_feat.shape[0], 1)
            position_embed = self.position_embeddings(position_index)
            initial_feat = torch.cat((initial_feat, position_embed), dim=-1) # [B, nobj, 512(-type)]
        type_index = torch.ones((initial_feat.shape[0], ), dtype=torch.long).to(x.device)
        type_embed = self.type_embeddings(type_index).unsqueeze(1).repeat(1, initial_feat.shape[1], 1) # [B, initial_feat, type]

        padding_mask = torch.cat([padding_inpant, padding_mask], dim=1)
        initial_feat = torch.cat([initial_feat_mask, initial_feat], dim=1)
        type_embed = torch.cat([type_embed_mask, type_embed], dim=1)

        # 2b. (B x cond) -> (B x 1 x transformer_input_d=d_model(-type))
        if self.use_text:
            if self.text_form == "word":
                assert len(self.word_embeddings) == word.shape[-1]
                word_cond = []
                for num_word, w in enumerate(self.word_embeddings):
                    cond = w(word[:, [num_word]])
                    un_cond = w(torch.LongTensor([self.vocab_size[num_word]]).unsqueeze(0).repeat(word.shape[0], 1).to(x.device))
                    if self.training:
                        cond = un_cond + (torch.rand((x.shape[0], 1, 1)).to(x.device) > 0.3).int() * (cond - un_cond)
                    else:
                        cond = un_cond + cond_scale_guidance * (cond - un_cond)
                    word_cond.append(cond)
                cond = torch.cat(word_cond, dim=1)
            elif self.text_form == "sentence":
                if un_condition is None:
                    un_cond = self.encode_cond(x.shape[0] * [""], x.device)
                else:
                    un_cond = un_condition
                if condition is None:
                    cond = un_cond
                else:
                    cond = condition
                un_cond = self.condition_mlp(un_cond)
                cond = self.condition_mlp(cond)
                if self.training:
                    cond = un_cond + (torch.rand((x.shape[0], 1)).to(x.device) > 0.3).int() * (cond - un_cond)
                else:
                    cond = un_cond + cond_scale_guidance * (cond - un_cond)
                cond = cond.unsqueeze(1)
            padding_mask = torch.cat([torch.zeros_like(padding_mask[:, :self.numoftoken]), padding_mask], dim=1)
            initial_feat = torch.cat([cond, initial_feat], dim=1) # [B, numoftoken+initial_feat, 512(-type)]
            type_index = 2 * torch.ones((initial_feat.shape[0], ), dtype=torch.long).to(x.device)
            type_embed = torch.cat([self.type_embeddings(type_index).unsqueeze(1).repeat(1, cond.shape[1], 1), type_embed], dim=1) # [B, numoftoken+initial_feat, type]

        # 2c. time embedding
        if self.use_time:
            time_embed = self.time_embeddings(t)  # B, dim
            time_embed = time_embed.unsqueeze(1) # B, 1, dim
            if self.time_form == "prepend":
                padding_mask = torch.cat([torch.zeros_like(padding_mask[:, [0]]), padding_mask], dim=1)
                initial_feat = torch.cat([time_embed, initial_feat], dim=1) # [B, 1+numoftoken+initial_feat, 512(-type)]
                type_index = (self.types - 1) * torch.ones((initial_feat.shape[0], ), dtype=torch.long).to(x.device)
                type_embed = torch.cat([self.type_embeddings(type_index).unsqueeze(1), type_embed], dim=1) # [B, 1+numoftoken+initial_feat, type]
            elif self.time_form == "concat":
                time_embed = time_embed.repeat(1, initial_feat.shape[1], 1)
                initial_feat = torch.cat([initial_feat, time_embed], dim=-1) # [B, 1+numoftoken+initial_feat, 512(-type)]
            elif self.time_form == "add":
                time_embed = time_embed.repeat(1, initial_feat.shape[1], 1)
                initial_feat = time_embed + initial_feat # [B, numoftoken+initial_feat, 512(-type)]

        initial_feat = torch.cat((initial_feat, type_embed), dim=-1) # [B, numoftoken+nmask+nobj, transformer_input_d-type+type = transformer_input_d]

        # 3. (B x nobj(+1) x transformer_input_d) ->  (B x nobj(+numoftoken) x transformer_input_d) 
        trans_out = self.transformer(initial_feat, padding_mask=padding_mask)
        if self.use_time and self.time_form == "prepend":
            trans_out = trans_out[:, 1+self.numoftoken+inpaint_mask.shape[1]:, :]
        else:
            trans_out = trans_out[:, self.numoftoken+inpaint_mask.shape[1]:, :]
        
        if not self.use_two_branch:
            for i in range(len(self.final_lin)-1):
                trans_out = self.activation(self.final_lin[i](trans_out))
            out = self.final_lin[-1](trans_out) # (B x nobj x out_dim) 
            
            if out.shape[2] == 1: return torch.sigmoid(out) # classifier (classificaiton token is the last entry along dim1)
            else: return out
        else:
            pos_out = trans_out
            for i in range(len(self.final_lin_pos)-1):
                pos_out = self.activation(self.final_lin_pos[i](pos_out))
            pos_out = self.final_lin_pos[-1](pos_out) # (B x nobj x pos_dim) 

            ang_out = trans_out
            for i in range(len(self.final_lin_ang)-1):
                ang_out = self.activation(self.final_lin_ang[i](ang_out))
            ang_out = self.final_lin_ang[-1](ang_out) # (B x nobj x ang_dim) 
            
            return torch.cat([pos_out, ang_out], dim=2) #(B x nobj x pos_dim+ang_dim) 