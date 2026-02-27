#!/bin/bash
set -e
KUBEBRAIN_NAMESPACE="ailab-mineru4sh"
BRAIN_USERNAME="dingruiyi"

# 检查必要的环境变量
if [ -z "$KUBEBRAIN_NAMESPACE" ] || [ -z "$BRAIN_USERNAME" ]; then
    echo "⚠️  警告: KUBEBRAIN_NAMESPACE 或 BRAIN_USERNAME 未设置"
    echo "请设置环境变量:"
    echo "  export KUBEBRAIN_NAMESPACE=your_namespace"
    echo "  export BRAIN_USERNAME=your_username"
    exit 1
fi

# 源镜像（刚 pull 下来的）
SOURCE_IMAGE="verlai/verl:vllm011.2.dev3"

# 目标 registry 配置（与 build-and-push-vllm.sh 保持一致）
REGISTRY="registry.h.pjlab.org.cn"
IMAGE_NAME="vllm-verl-megatron-stable"
TARGET_TAG="vllm011.2.dev3"
FULL_TARGET="${REGISTRY}/${KUBEBRAIN_NAMESPACE}/${BRAIN_USERNAME}-${IMAGE_NAME}:${TARGET_TAG}"

echo "🏷️  源镜像:   $SOURCE_IMAGE"
echo "🏷️  目标镜像: $FULL_TARGET"

# 打标签
docker tag "$SOURCE_IMAGE" "$FULL_TARGET"

echo "🚀 开始推送镜像..."
docker push "$FULL_TARGET"

echo "✅ 推送完成!"
echo "📍 镜像地址: $FULL_TARGET"

