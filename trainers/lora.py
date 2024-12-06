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

from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers

from typing import Dict

from loralib.layers import LoRALayer, PlainMultiheadAttentionLoRA

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}

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


def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    VALIDATION = False
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
 
    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() 
    
    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0
    
    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision': 
            text_features = textual_features.t().half()
        for i, (images, target) in enumerate(tqdm(train_loader)):
            
            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()
            
            count_iters += 1
            
            if count_iters == total_iters:
                break
            
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))

        
        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
        
    
    acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return
            
class LoRACLIP(nn.module):
        def __init__(self, cfg, classnames, clip_model):
            super().__init__()
            n_cls = len(classnames)
            #n_ctx = cfg.TRAINER.COOP.N_CTX
            #ctx_init = cfg.TRAINER.COOP.CTX_INIT

            dtype = clip_model.dtype
            #ctx_dim = clip_model.ln_final.weight.shape[0]
            clip_imsize = clip_model.visual.input_resolution
            cfg_imsize = cfg.INPUT.SIZE[0]
            assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

            list_lora_layers = []
            if cfg.encoder == 'text' or cfg.encoder == 'both':
                indices = INDEX_POSITIONS_TEXT[cfg.position]
                text_encoder = clip_model.transformer
                for i, block in enumerate(text_encoder.resblocks):
                    print(f"Residual Attention Block {i}: {block}")
                    if i in indices:
                        for name, submodule in block.named_children():
                            if isinstance(submodule, nn.MultiheadAttention):
                                new_multi_head_lora = PlainMultiheadAttentionLoRA(
                                    submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                                setattr(block, name, new_multi_head_lora)
                                list_lora_layers.append(new_multi_head_lora)

            if cfg.encoder == 'vision' or cfg.encoder == 'both':
                indices = INDEX_POSITIONS_VISION[cfg.backbone][cfg.position]
                vision_encoder = clip_model.visual.transformer
                for i, block in enumerate(vision_encoder.resblocks):
                    print(f"Residual Attention Block {i}: {block}")
                    if i in indices:
                        for name, submodule in block.named_children():
                            if isinstance(submodule, nn.MultiheadAttention):
                                new_multi_head_lora = PlainMultiheadAttentionLoRA(
                                    submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                                setattr(block, name, new_multi_head_lora)
                                list_lora_layers.append(new_multi_head_lora)

            # if ctx_init:
            #     # use given words to initialize context vectors
            #     ctx_init = ctx_init.replace("_", " ")
            #     n_ctx = len(ctx_init.split(" "))
            #     prompt = clip.tokenize(ctx_init)
            #     with torch.no_grad():
            #         embedding = clip_model.token_embedding(prompt).type(dtype)
            #     ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            #     prompt_prefix = ctx_init

            # else:
            #     # random initialization
            #     if cfg.TRAINER.COOP.CSC:
            #         print("Initializing class-specific contexts")
            #         ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            #     else:
            #         print("Initializing a generic context")
            #         ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            #     nn.init.normal_(ctx_vectors, std=0.02)
            #     prompt_prefix = " ".join(["X"] * n_ctx)

            # print(f'Initial context: "{prompt_prefix}"')
            # print(f"Number of context words (tokens): {n_ctx}")

            # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]

            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

            self.n_cls = n_cls
            self.n_ctx = n_ctx
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
            self.name_lens = name_lens
            self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

        def forward(self):
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix

            if self.class_token_position == "end":
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )

            elif self.class_token_position == "middle":
                half_n_ctx = self.n_ctx // 2
                prompts = []
                for i in range(self.n_cls):
                    name_len = self.name_lens[i]
                    prefix_i = prefix[i : i + 1, :, :]
                    class_i = suffix[i : i + 1, :name_len, :]
                    suffix_i = suffix[i : i + 1, name_len:, :]
                    ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                    ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                    prompt = torch.cat(
                        [
                            prefix_i,     # (1, 1, dim)
                            ctx_i_half1,  # (1, n_ctx//2, dim)
                            class_i,      # (1, name_len, dim)
                            ctx_i_half2,  # (1, n_ctx//2, dim)
                            suffix_i,     # (1, *, dim)
                        ],
                        dim=1,
                    )
                    prompts.append(prompt)
                prompts = torch.cat(prompts, dim=0)

            elif self.class_token_position == "front":
                prompts = []
                for i in range(self.n_cls):
                    name_len = self.name_lens[i]
                    prefix_i = prefix[i : i + 1, :, :]
                    class_i = suffix[i : i + 1, :name_len, :]
                    suffix_i = suffix[i : i + 1, name_len:, :]
                    ctx_i = ctx[i : i + 1, :, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  # (1, 1, dim)
                            class_i,   # (1, name_len, dim)
                            ctx_i,     # (1, n_ctx, dim)
                            suffix_i,  # (1, *, dim)
                        ],
                        dim=1,
                    )
                    prompts.append(prompt)
                prompts = torch.cat(prompts, dim=0)

            else:
                raise ValueError

            return prompts
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class LoRA(TrainerX):
    """Low Rank Adaptation(LoRA).
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
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
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

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

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

