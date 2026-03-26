#!/bin/bash

rjob delete multiroute-nonedistill-cover-rate-grpo-uspto-50k
rjob submit \
    --name=multiroute-nonedistill-cover-rate-grpo-uspto-50k \
    --gpu=8 \
    --memory=1280000 \
    --cpu=96 \
    --namespace=ailab-mineru4sh \
    --charged-group=mineru4sh_gpu \
    --private-machine=group \
    --mount=gpfs://gpfs1/mineru4s:/mnt/shared-storage-user/mineru4s \
    --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
    --mount=gpfs://gpfs2/mineru4s-gpfs2:/mnt/shared-storage-gpfs2/mineru4s-gpfs2 \
    --image=registry.h.pjlab.org.cn/ailab-mineru4sh/dingruiyi-vllm-verl-megatron-stable:vllm0.16_verl_with_rdkit \
    --host-network=true \
    -P 1 \
    -- bash -exc /mnt/shared-storage-user/mineru4s/dingruiyi/verl_wanjuan/train_multiroute_nondistill_cover_rate_grpo.sh