#!/bin/bash

# 获取当前脚本所在目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR=$(realpath "$SCRIPT_DIR/piper_sdk")

echo "切换到工作空间目录: $WORKSPACE_DIR"
cd "$WORKSPACE_DIR" || { echo "❌ 目录不存在: $WORKSPACE_DIR"; exit 1; }

# 激活 CAN0
bash can_activate.sh can1 1000000 "3-5.1:1.0" || { echo "CAN0 激活失败！"; exit 1; }

echo "CAN1 已成功激活！"
