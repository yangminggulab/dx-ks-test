#!/bin/bash
# 发 Wake-on-LAN 魔术包唤醒 Windows（需处于睡眠/休眠状态）
python3 -c "
import socket
mac = 'E02E0B3664 42'.replace(' ','')
magic = bytes.fromhex('FF'*6 + mac*16)
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.sendto(magic, ('192.168.1.255', 9))
print('唤醒包已发送，等待 10-20 秒后尝试 SSH 连接...')
"
