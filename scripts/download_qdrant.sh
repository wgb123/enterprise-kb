#!/usr/bin/env bash
# Qdrant 二进制下载脚本
# 用法: bash scripts/download_qdrant.sh
# 下载静态编译的 musl 版本（兼容 Ubuntu 20.04+）

set -euo pipefail

QDRANT_VERSION="v1.18.2"
QDRANT_DIR="qdrant_data"
QDRANT_BIN="${QDRANT_DIR}/qdrant"

# 使用代理（可选）
if [[ -n "${HTTP_PROXY:-}" ]]; then
    export http_proxy="$HTTP_PROXY"
    export https_proxy="$HTTPS_PROXY"
fi

if [[ -f "$QDRANT_BIN" ]]; then
    echo "✅ Qdrant 已存在: $QDRANT_BIN"
    "$QDRANT_BIN" --version
    exit 0
fi

echo "📥 下载 Qdrant ${QDRANT_VERSION} (musl, x86_64)..."
mkdir -p "$QDRANT_DIR"

URL="https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-musl.tar.gz"

curl -sL "$URL" -o /tmp/qdrant.tar.gz
tar xzf /tmp/qdrant.tar.gz -C "$QDRANT_DIR"
rm /tmp/qdrant.tar.gz
chmod +x "$QDRANT_BIN"

echo "✅ 下载完成"
"$QDRANT_BIN" --version
