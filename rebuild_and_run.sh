#!/bin/bash
# 1. 编译内核 (增量编译非常快)
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "编译失败，请检查代码错误！"
    exit 1
fi

echo "编译成功，正在准备启动虚拟机..."

# 2. 调用你之前的 FEMU 启动脚本
# 注意：确保你的 run-blackbox.sh 里的 NEW_KERNEL 路径指向这里
cd ~/femu/build-femu/
bash run-blackbox.sh
