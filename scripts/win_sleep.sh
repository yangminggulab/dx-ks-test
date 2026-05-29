#!/bin/bash
# 让 Windows 进入休眠（可远程唤醒）
ssh -p 2222 thisislbk@192.168.1.18 \
  "/mnt/c/Windows/System32/shutdown.exe /h" && \
echo "Windows 已进入休眠，用 win_wake.sh 唤醒。"
