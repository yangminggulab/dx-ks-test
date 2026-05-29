#!/bin/bash
# 完全关机（关机后无法远程唤醒，除非插有线网线）
echo "警告：关机后无法远程唤醒（WiFi 断电）。确认请按 Enter，取消按 Ctrl+C"
read
ssh -p 2222 thisislbk@192.168.1.18 \
  "/mnt/c/Windows/System32/shutdown.exe /s /t 0" && \
echo "Windows 正在关机。"
