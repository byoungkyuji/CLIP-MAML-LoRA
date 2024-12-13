# This is for CLIP-MAML-LoRA experiment records.
## [24.12.11 13:00:00] 1. CLIP-LoRA DG trial is using Data Parallel seed 1
    * bash scripts/lora/base2new_train.sh imagenet 1
### base2new_train.sh, 1 batch takes about 1 minute.
## [24.12.11 20:25:00] 1. CLIP-LoRA DG trial is using Data Parallel seed 1
    * bash scripts/lora/base2new_train.sh imagenet 2
### base2new_train.sh, 1 batch takes about 1 minute.
## [24.12.11 22:25:00] 1. CLIP-LoRA general is using Data Parallel seed 3
## [24.12.12 16:50:00] 2. CLIP-LoRA genral using Data Parallel seed 1, modified HP
    * bash scripts/lora/all2all_train.sh imagenet 2 1
    * bash scripts/lora/all2all_train.sh caltech101 2 2
    * bash scripts/lora/all2all_train.sh dtd 2 3
    * bash scripts/lora/all2all_train.sh fgvc_aircraft 2 4
    * bash scripts/lora/all2all_train.sh food101 2 5
    * bash scripts/lora/all2all_train.sh oxford_flower 2 6
    * bash scripts/lora/all2all_train.sh oxford_pets 2 7
    * bash scripts/lora/all2all_train.sh stanford_cars 2 0 
    * bash scripts/lora/all2all_train.sh sun397 2 1
    * bash scripts/lora/all2all_train.sh ucf101 2 2 
    * bash scripts/lora/all2all_train.sh eurosat 2 3
