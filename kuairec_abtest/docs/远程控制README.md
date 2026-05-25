# Windows SSH 配置文档

## 架构

```
Mac  ──SSH:22──▶  Windows OpenSSH Server  ──▶  WSL Ubuntu 22.04 bash
                  （连接层，Windows 服务）        （工作层，Linux 环境）
```

- **连接层**：Windows 原生 OpenSSH Server，端口 22，开机自动启动
- **工作层**：SSH 进去后直接是 WSL Ubuntu 22.04 的 bash
- **外网访问**：通过 Tailscale VPN

---

## 账户信息

| 项目 | 值 |
|------|----|
| Windows 账户名 | `thisi` |
| WSL Linux 用户名 | `thisislbk` |
| Windows SSH 端口 | `22` |
| Tailscale IP | `100.75.10.89` |
| 局域网 IP | `192.168.1.18` |

> SSH 登录用 **Windows 账户名 `thisi`**，进去后 shell 是 WSL 的 `thisislbk`。

---

## Mac 连接方式

```bash
# 从家里 / 外网（走 Tailscale）
ssh win

# 同局域网直连
ssh win-local
```

Mac `~/.ssh/config` 配置：

```
# 从家里/外网用（Tailscale）
Host win
    HostName 100.75.10.89
    Port 22
    User thisi
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3

# 同局域网直连
Host win-local
    HostName 192.168.1.18
    Port 22
    User thisi
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

---

## Windows 侧配置详情

### OpenSSH Server

- 服务名：`sshd`
- 启动类型：Automatic（开机自启）
- 监听端口：22
- 公钥文件：`C:\ProgramData\ssh\administrators_authorized_keys`
  - `thisi` 是管理员组成员，必须用这个路径，不能用 `~/.ssh/authorized_keys`
  - 文件权限：只有 SYSTEM 和 Administrators 可读（`icacls` 设置）

### DefaultShell

注册表：`HKLM:\SOFTWARE\OpenSSH\DefaultShell`

```
C:\Windows\System32\bash.exe
```

SSH 连接进来后自动启动 WSL Ubuntu bash。

### 防火墙

规则名：`OpenSSH-Server-In-TCP`
- 端口：22/TCP，入站
- 配置文件：Any（所有网络类型均生效）

### Tailscale

- 服务：`Tailscale`，Automatic 自启
- Windows 侧 IP：`100.75.10.89`
- Mac 侧需开启 Tailscale 才能用 `ssh win`

### WSL

- 发行版：Ubuntu 22.04
- `.wslconfig`（`C:\Users\thisi\.wslconfig`）：
  ```ini
  [wsl2]
  networkingMode=mirrored
  ```
- WSL 自己的 sshd **已停用**，不需要，不要再开

---

## 定时开关机

每天自动睡眠和唤醒，通过 Windows 任务计划实现。

| 任务名 | 时间 | 动作 |
|--------|------|------|
| `DailyWake_0900` | 每天 09:00 | 从睡眠唤醒 |
| `DailySleep_2130` | 每天 21:30 | 进入睡眠 |

**睡眠 vs 关机**：用的是睡眠（不是完全关机）。内存保持供电，唤醒只需几秒，任务计划可以自动唤醒。完全关机无法用软件唤醒。

**睡眠期间 SSH 是否可用**：不可用，机器睡了网络就断。唤醒后 SSH 立即可用，不需要在 Windows 上手动登录账号（sshd 是 SYSTEM 服务，锁屏不影响）。

### 查看任务状态

```powershell
Get-ScheduledTask -TaskName DailyWake_0900, DailySleep_2130 | Select-Object TaskName, State
(Get-ScheduledTaskInfo -TaskName "DailyWake_0900").NextRunTime
(Get-ScheduledTaskInfo -TaskName "DailySleep_2130").NextRunTime
```

### 临时手动睡眠

```powershell
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::SetSuspendState('Suspend', $true, $false)
```

### 修改时间（管理员 PowerShell）

```powershell
# 改唤醒时间，例如改为 08:30
$trigger = New-ScheduledTaskTrigger -Daily -At "08:30"
Set-ScheduledTask -TaskName "DailyWake_0900" -Trigger $trigger

# 改睡眠时间，例如改为 23:00
$trigger = New-ScheduledTaskTrigger -Daily -At "23:00"
Set-ScheduledTask -TaskName "DailySleep_2130" -Trigger $trigger
```

### 睡眠脚本位置

`C:\ProgramData\ssh\sleep_cmd.ps1`（任务计划调用此文件执行睡眠）

---

## 重启后是否需要手动操作

**不需要。**

- Windows sshd 是系统服务，开机自动启动
- bash.exe（WSL）在 SSH 连接时按需启动
- Tailscale 服务开机自启

重启后直接 `ssh win` 即可。

---

## 排查

### 连不上

```bash
# 1. 检查 Tailscale 是否在跑（外网连接时）
# Mac 上：
tailscale status

# 2. 局域网测试
ssh win-local

# 3. 检查 Windows sshd 服务（在 Windows 上）
Get-Service sshd
```

### Permission denied (publickey)

```powershell
# 检查 authorized_keys 内容
Get-Content "C:\ProgramData\ssh\administrators_authorized_keys"

# 检查权限（必须只有 SYSTEM/Administrators）
icacls "C:\ProgramData\ssh\administrators_authorized_keys"

# 重新设置权限
icacls "C:\ProgramData\ssh\administrators_authorized_keys" /inheritance:r /grant "SYSTEM:(F)" /grant "Administrators:(F)"
```

### 进去是 cmd 而不是 bash

```powershell
# 管理员 PowerShell 里跑
Set-ItemProperty "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Windows\System32\bash.exe" -Force
```

---

## 为什么不用 WSL 自带的 sshd

| 方案 | 稳定性 | 复杂度 | 重启后 |
|------|--------|--------|--------|
| Windows 原生 SSH ✅ | 高（系统服务） | 低 | 自动恢复 |
| WSL sshd（旧方案） | 中（依赖 WSL 进程） | 高（需 portproxy，IP 动态变） | 需手动或脚本维护 |
