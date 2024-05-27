import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict
import scipy.io as sio


from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from .clip_text import clip
from .clip_text.simple_tokenizer import SimpleTokenizer as _Tokenizer
import tqdm
_tokenizer = _Tokenizer()
import numpy as np
import copy
import clip.clip as clip_ori

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def load_clip_to_cpu_ori():
    backbone_names=['RN50','RN101','ViT-B/32','ViT-B/16','ViT-L/14','ViT-L/14@336px']
    dims=[1024,1024,512,512,768,768]
    ind=4
    print(backbone_names[ind])
    url = clip_ori._MODELS[backbone_names[ind]]
    model_path = clip_ori._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model,dims[ind]

CUSTOM_TEMPLATES_ori = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of an aircraft {}.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    #"EuroSAT": "a photo of a {}.",
    "StanfordCars": "a photo of a {}, a type of car.",
    "Food101": "a photo of a {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

CUSTOM_TEMPLATES = {
    "OxfordPets": " {}.",
    "OxfordFlowers": " {}.",
    "FGVCAircraft": " {}.",
    "DescribableTextures": " {}.",
    "EuroSAT": " {}.",
    "StanfordCars": " {}.",
    "Food101": " {}.",
    "SUN397": " {}.",
    "Caltech101": " {}.",
    "UCF101": " {}.",
    "ImageNet": "{}.",
    "ImageNetSketch": "{}.",
    "ImageNetV2": "{}.",
    "ImageNetA": "{}.",
    "ImageNetR": "{}.",
}





class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, prompt_projections, weight, tokenized_prompts,flag=False):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if flag:
            x = self.transformer(x)
        else:
            counter=0
            outputs = self.transformer.resblocks([x,prompt_projections,weight,counter])
            x = outputs[0]            

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                    *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                    "in_proj_bias", "bias_k", "bias_v"
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model,device):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            print("use given words to initialize context vectors")
            temp = 'a photo of a'
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

            ctx_vectors_src = embedding[0, 1 : 1 + n_ctx, :]

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        
        #prompt_prefix="X X a photo of a "
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        
        
        ctx_vectors_vis = torch.empty(6, 768, dtype=dtype)
        nn.init.normal_(ctx_vectors_vis, std=0.02)
        self.ctx_visual = nn.Parameter(ctx_vectors_vis) 
        
        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()

        temp = CUSTOM_TEMPLATES_ori[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        vis_dim = clip_model.visual.output_dim
        self.meta_net_ = PromptAttention()
        self.prompt_projections = _get_clones(self.meta_net_, 12)
        convert_weights(self.prompt_projections)

        self.visual_meta_net_ = PromptAttention(d_model_visual=768)
        self.visual_prompts = _get_clones(self.visual_meta_net_, 12)
        convert_weights(self.visual_prompts)

        classnames = [name.replace("_", " ") for name in classnames]
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        temp =prompt_prefix+temp
        
        prompts = [temp.format(c) for c in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.prev_ctx=None
        
        self.VPT_fc = nn.Linear(512, self.n_cls, bias=False)

    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompt = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompt, self.prompt_projections, self.ctx_visual, self.visual_prompts

class PromptAttention(nn.Module):
    def __init__(self,d_model_visual=512):
        super().__init__()
        n_head_visual = d_model_visual//64
        self.promt_attention_visual = nn.MultiheadAttention(d_model_visual, n_head_visual)
        self.ln_pa_1_visual = LayerNorm(d_model_visual)
        self.ln_pa_1_token = LayerNorm(d_model_visual)

        self.mlp_pa_visual = nn.Sequential(OrderedDict([
            ("c_fc1", nn.Linear(d_model_visual,256,bias=False)),
            ("gelu1", QuickGELU()),
            ("c_proj", nn.Linear(256, d_model_visual,bias=False))
            #("gelu1", QuickGELU())
        ]))

        self.ln_pa_2_visual = LayerNorm(d_model_visual)

        self.scale1 = nn.Parameter(0.75*torch.ones(1))
        self.scale2 = nn.Parameter(0.75*torch.ones(1))

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
              
    def forward(self, visual_context, token_context):
        visual_context = self.scale1.half()*visual_context + (1-self.scale1.half())*self.promt_attention_visual(self.ln_pa_1_visual(token_context),self.ln_pa_1_visual(visual_context),self.ln_pa_1_visual(visual_context),need_weights=False)[0] 
        visual_context = self.scale2.half()*visual_context + (1-self.scale2.half())*self.mlp_pa_visual(self.ln_pa_2_visual(visual_context))

        return visual_context

    
class PromptAttention_Visual(nn.Module):
    def __init__(self,d_model_visual=512):
        super().__init__()
        n_head_visual = d_model_visual//64
        self.promt_attention_visual = nn.MultiheadAttention(d_model_visual, n_head_visual)
        self.ln_pa_1_visual = LayerNorm(d_model_visual)
        self.ln_pa_1_token = LayerNorm(d_model_visual)

        self.mlp_pa_visual = nn.Sequential(OrderedDict([
            ("c_fc1", nn.Linear(d_model_visual,128)),
            ("gelu1", QuickGELU()),
            ("c_proj", nn.Linear(128, d_model_visual))
            #("gelu1", QuickGELU())
        ]))

        self.ln_pa_2_visual = LayerNorm(d_model_visual)

        self.scale1 = nn.Parameter(0.9*torch.ones(1))
        self.scale2 = nn.Parameter(0.9*torch.ones(1))

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
              
    def forward(self, visual_context, token_context):
        visual_context = self.scale1.half()*visual_context + (1-self.scale1.half())*self.promt_attention_visual(self.ln_pa_1_visual(token_context),self.ln_pa_1_visual(visual_context),self.ln_pa_1_visual(visual_context),need_weights=False)[0] 
        visual_context = self.scale2.half()*visual_context + (1-self.scale2.half())*self.mlp_pa_visual(self.ln_pa_2_visual(visual_context))

        return visual_context

from scipy.optimize import linear_sum_assignment
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model,device):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model,device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.text_features
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.domain_sim = -1
        self.domain_sim_src = -1
        self.weight = cfg.TRAINER.COOP.W
        self.ind_ = 0    
        self.n_ctx = cfg.TRAINER.COOP.N_CTX
 
    def forward(self, image, label=None,feat_flag=False,proto=None):
        #############################################################
        text_features_old = self.ori_embedding
        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        #############################################################
        
        prompts,prompt_projections,visual_ctx, visual_prompts = self.prompt_learner()
        with torch.no_grad():
            image_features_fixed = self.image_encoder(image.type(self.dtype))
        image_features_fixed = image_features_fixed / image_features_fixed.norm(dim=-1, keepdim=True) 
        
        if feat_flag:
            return image_features_fixed
        image_features = self.image_encoder(image.type(self.dtype),visual_ctx,visual_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
        image_features = image_features + image_features_fixed
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        
        text_features = self.text_encoder(prompts, prompt_projections,self.n_ctx,tokenized_prompts.detach()) 
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features + text_features_old
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale.detach() * image_features @ text_features.t()

        if self.prompt_learner.training:
            logits_fixed = 10*F.linear(F.normalize(image_features_fixed, p=2, dim=-1), F.normalize(self.prompt_learner.VPT_fc.weight, p=2, dim=-1))
            logits_ = 10*F.linear(F.normalize(image_features, p=2, dim=-1), F.normalize(self.prompt_learner.VPT_fc.weight, p=2, dim=-1))
            loss_ = F.cross_entropy(logits_, label)+F.cross_entropy(logits_fixed, label)
   
            score= cos(text_features,text_features_old)
            score  = 1.0-torch.mean(score)
            loss = F.cross_entropy(logits, label)+8.0*score

            visual_dist= cos(image_features_fixed,image_features)
            visual_dist  = 1.0-torch.mean(visual_dist)
            visual_dist = 6.0*visual_dist
            loss += visual_dist
            loss += 0.5*loss_

            return logits, loss
        else:
            return logits


@TRAINER_REGISTRY.register()
class TCP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(classnames)
        self.n_cls = len(classnames)
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model,self.device)
        self.w = cfg.TRAINER.COOP.W

        print("Turning off gradients in both the image and the text encoder")

        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name and "VPT" not in name:# and "VPT_fc" in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        
        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            #self.model.text_encoder = nn.DataParallel(self.model.text_encoder)
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        #self.proto=None
        self.proto = self.center_feature()
        self.model.prompt_learner.VPT_fc.weight.data=self.proto
        
    def center_feature(self):
        self.set_model_mode("eval")
        data_loader = self.train_loader_x
        embedding_list=[]
        label_list=[]
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            with torch.no_grad():
                image_feature = self.model(input,feat_flag=True)
            embedding_list.append(image_feature)
            label_list.append(label)
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        class_num = len(torch.unique(label_list))
        proto_list = []
        for class_index in range(class_num):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        proto = torch.stack(proto_list, dim=0)
        proto = proto / proto.norm(dim=-1, keepdim=True)
        #self.proto = proto
        return proto


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output,loss = self.model(image, label,proto=self.proto)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            #self.proto = self.center_feature()
            #self.model.prompt_learner.VPT_fc.weight.data=self.proto
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    #def model_inference(self, input):
    #    return self.model(input)


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(names)

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]
            if "VPT_fc.weight" in state_dict:
                del state_dict["VPT_fc.weight"]
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
