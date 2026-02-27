#!/bin/bash
set -ex

# 获取 Dockerfile 所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile.stable.vllm"

# 检查 Dockerfile 是否存在
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "❌ 错误: 找不到 Dockerfile: $DOCKERFILE_PATH"
    exit 1
fi

# 检查必要的环境变量
if [ -z "$KUBEBRAIN_NAMESPACE" ] || [ -z "$BRAIN_USERNAME" ]; then
    echo "⚠️  警告: KUBEBRAIN_NAMESPACE 或 BRAIN_USERNAME 未设置"
    echo "请设置环境变量:"
    echo "  export KUBEBRAIN_NAMESPACE=your_namespace"
    echo "  export BRAIN_USERNAME=your_username"
    exit 1
fi

# 生成时间戳作为镜像标签
IMAGE_TAG=$(date +%Y%m%d%H%M%S)
IMAGE_NAME="vllm-verl-megatron-stable"
REGISTRY="registry.h.pjlab.org.cn"
FULL_IMAGE_NAME="${REGISTRY}/${KUBEBRAIN_NAMESPACE}/${BRAIN_USERNAME}-${IMAGE_NAME}:${IMAGE_TAG}"

echo "⏳ 开始构建镜像..."
echo "📦 镜像名称: $FULL_IMAGE_NAME"
echo "📄 Dockerfile: $DOCKERFILE_PATH"
echo "🔧 构建配置: MAX_JOBS=4, BUILD_THREADS=1 (适合 8CPU/16GB 内存)"

# 设置系统资源限制（避免 OOM）
# 限制进程内存使用（8GB，保留 4GB 给系统）
ulimit -v 8388608 2>/dev/null || echo "⚠️  无法设置虚拟内存限制（可能需要 root）"

# 设置构建参数以限制并发和内存占用
# MAX_JOBS: 控制并行编译任务数（设为 4，不超过 CPU 核心数）
# BUILD_THREADS: 控制每个任务的线程数（设为 1，减少内存占用）
BUILD_ARGS="--build-arg MAX_JOBS=4 --build-arg BUILD_THREADS=1"

# 检测并传递代理设置（如果存在）
# 优先使用大写变量，如果没有则使用小写变量
PROXY_HTTP="${HTTP_PROXY:-${http_proxy}}"
PROXY_HTTPS="${HTTPS_PROXY:-${https_proxy:-$PROXY_HTTP}}"
PROXY_NO="${NO_PROXY:-${no_proxy}}"

if [ -n "$PROXY_HTTP" ]; then
    echo "🌐 检测到代理配置:"
    echo "   HTTP_PROXY: $PROXY_HTTP"
    echo "   HTTPS_PROXY: $PROXY_HTTPS"
    [ -n "$PROXY_NO" ] && echo "   NO_PROXY: $PROXY_NO"
    BUILD_ARGS="$BUILD_ARGS --build-arg HTTP_PROXY=$PROXY_HTTP --build-arg http_proxy=$PROXY_HTTP"
    BUILD_ARGS="$BUILD_ARGS --build-arg HTTPS_PROXY=$PROXY_HTTPS --build-arg https_proxy=$PROXY_HTTPS"
    [ -n "$PROXY_NO" ] && BUILD_ARGS="$BUILD_ARGS --build-arg NO_PROXY=$PROXY_NO --build-arg no_proxy=$PROXY_NO"
else
    echo "⚠️  未检测到代理配置"
    echo "   如果网络连接失败，请设置代理环境变量："
    echo "   export HTTP_PROXY=http://proxy.example.com:port"
    echo "   export HTTPS_PROXY=http://proxy.example.com:port"
fi

# 启用 BuildKit 以获得更好的资源控制和缓存
export DOCKER_BUILDKIT=1

# 构建 Docker 镜像
# 注意：docker build 本身不支持 --memory/--cpus，通过构建参数控制并发
docker build \
    $BUILD_ARGS \
    --progress=plain \
    -f "$DOCKERFILE_PATH" \
    -t "$FULL_IMAGE_NAME" \
    -t "${REGISTRY}/${KUBEBRAIN_NAMESPACE}/${BRAIN_USERNAME}-${IMAGE_NAME}:latest" \
    "$SCRIPT_DIR"

if [ $? -ne 0 ]; then
    echo "❌ 镜像构建失败"
    exit 1
fi

echo "✅ 镜像构建成功"
echo "🚀 开始推送镜像..."

# 推送带时间戳的镜像
docker push "$FULL_IMAGE_NAME"

# 推送 latest 标签
docker push "${REGISTRY}/${KUBEBRAIN_NAMESPACE}/${BRAIN_USERNAME}-${IMAGE_NAME}:latest"

if [ $? -ne 0 ]; then
    echo "❌ 镜像推送失败"
    exit 1
fi

echo "✅ 镜像推送完成!"
echo "📍 镜像地址: $FULL_IMAGE_NAME"
echo "📍 Latest 标签: ${REGISTRY}/${KUBEBRAIN_NAMESPACE}/${BRAIN_USERNAME}-${IMAGE_NAME}:latest"

