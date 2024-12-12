import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from loralib.utils import mark_only_lora_as_trainable, apply_lora
from loralib import layers as lora_layers

from typing import Dict

from loralib.layers import LoRALayer, PlainMultiheadAttentionLoRA

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

class LoRACLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.classnames = classnames
        self.list_lora_layers = apply_lora(cfg,clip_model)
        self.lora_encoder = cfg.TRAINER.LoRA.ENCODER
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model.transformer
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.template = cfg.TRAINER.LoRA.CTX_INIT
        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
            
    def forward(self, images):
        template = self.template[0]
        texts = [template.format(classname.replace('_', ' ')) for classname in self.classnames]
        if self.lora_encoder == 'text' or self.lora_encoder == 'both':
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                texts = clip.tokenize(texts).cuda()
                class_embeddings = self.clip_model.encode_text(texts)
            text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
            
        if self.lora_encoder == 'vision' or self.lora_encoder == 'both':
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = self.clip_model.encode_image(images)
        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = self.clip_model.encode_image(images)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        
        cosine_similarity = self.logit_scale.exp() * image_features @ text_features.t()        
        return cosine_similarity
    def get_lora_parameters(self, bias='none'):
        params = []
        for name, param in self.clip_model.named_parameters():
            if bias == 'none':
                if 'lora_' in name:
                    params.append(param)
            elif bias == 'all':
                if 'lora_' in name or 'bias' in name:
                    params.append(param)
            elif bias == 'lora_only':
                if 'lora_' in name:
                    params.append(param)
                    bias_name = name.split('lora_')[0] + 'bias'
                    if bias_name in self.clip_model.state_dict():
                        bias_param = dict(self.clip_model.named_parameters())[bias_name]
                        params.append(bias_param)
            else:
                raise NotImplementedError
        return params
@TRAINER_REGISTRY.register()
class LoRA(TrainerX):
    """Low Rank Adaptation(LoRA).
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.LoRA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.LoRA.PREC == "fp32" or cfg.TRAINER.LoRA.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building LoRA CLIP")
        self.model = LoRACLIP(cfg, classnames, clip_model)
        
        print("Turning off gradients")
        mark_only_lora_as_trainable(self.model)

        # check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.list_lora_layers, cfg.MODEL.INIT_WEIGHTS)
        
        self.model.to(self.device)
        self.optim = build_optimizer(self.model.list_lora_layers, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("LoRA_layers", self.model.list_lora_layers, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.LoRA.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        #device_ids = [i for i in range(2,8)]
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.LoRA.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

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

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
