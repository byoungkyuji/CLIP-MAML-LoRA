import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from loralib.utils import mark_only_lora_as_trainable, apply_lora
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
class MLoRA(TrainerX):
    """Trainer for MAML with LoRA."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.inner_lr = cfg.TRAINER.MAML.INNER_LR
        self.num_inner_steps = cfg.TRAINER.MAML.NUM_INNER_STEPS
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MLoRA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.MLoRA.PREC == "fp32" or cfg.TRAINER.MLoRA.PREC == "amp":
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
        """
        Override forward_backward for meta-learning tasks using LoRA parameters only.
        """
        support_set, query_set = self.parse_batch_train(batch)

        # Inner loop adaptation (on support set)
        support_images, support_labels = support_set["img"].to(self.device), support_set["label"].to(self.device)
        lora_params = [p.clone().detach().requires_grad_() for p in self.model.get_lora_parameters(bias='none')]

        for _ in range(self.cfg.TRAINER.MAML.NUM_INNER_STEPS):
            support_output = self._forward_with_params(support_images, lora_params)
            support_loss = F.cross_entropy(support_output, support_labels)
            grads = torch.autograd.grad(support_loss, lora_params, create_graph=True)
            lora_params = [p - self.cfg.TRAINER.MAML.INNER_LR * g for p, g in zip(lora_params, grads)]

        # Outer loop optimization (on query set)
        query_images, query_labels = query_set["img"].to(self.device), query_set["label"].to(self.device)
        query_output = self._forward_with_params(query_images, lora_params)
        query_loss = F.cross_entropy(query_output, query_labels)

        self.optim.zero_grad()
        query_loss.backward()
        self.optim.step()

        return {"query_loss": query_loss.item()}


    def adapt_parameters(self, support_set):
        """
        Perform inner loop adaptation on support set.
        support_set: Support data (images and labels).
        Returns:
            Adapted parameters for the task.
        """
        support_images, support_labels = support_set["img"].to(self.device), support_set["label"].to(self.device)
        task_params = [p.clone().detach().requires_grad_() for p in self.model.parameters()]

        for _ in range(self.num_inner_steps):
            support_output = self.model.forward_with_params(support_images, task_params)
            support_loss = F.cross_entropy(support_output, support_labels)

            grads = torch.autograd.grad(support_loss, task_params, create_graph=True)
            task_params = [p - self.inner_lr * g for p, g in zip(task_params, grads)]

        return task_params

    def compute_query_loss(self, query_set, task_params):
        """
        Compute the loss on the query set using adapted parameters.
        query_set: Query data (images and labels).
        task_params: Adapted parameters from the inner loop.
        Returns:
            Query loss.
        """
        query_images, query_labels = query_set["img"].to(self.device), query_set["label"].to(self.device)
        query_output = self.model.forward_with_params(query_images, task_params)
        query_loss = F.cross_entropy(query_output, query_labels)
        return query_loss

    def _forward_with_params(self, images, params):
        """
        Perform forward pass using task-specific parameters.
        """
        original_params = [p.clone() for p in self.model.parameters()]
        
        # Update LoRA parameters temporarily
        for p, new_p in zip(self.model.parameters(), params):
            p.data = new_p.data

        output = self.model(images)

        # Restore original parameters
        for p, orig_p in zip(self.model.parameters(), original_params):
            p.data = orig_p.data

        return output

    def parse_batch_train(self, batch):
        """
        Parse batch into support and query sets.
        """
        support_set = {
            "img": batch["support"]["img"].to(self.device),
            "label": batch["support"]["label"].to(self.device),
        }
        query_set = {
            "img": batch["query"]["img"].to(self.device),
            "label": batch["query"]["label"].to(self.device),
        }
        return support_set, query_set