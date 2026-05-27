# Windows SSH 配置文档

> 这份文档描述的是“**Windows 原生 OpenSSH 负责接入，WSL 负责工作环境**”这套方案。  
> 开机/唤醒自恢复、睡眠任务、服务兜底等内容，配套看 [win_开机配置README.md](/Users/liubike/Desktop/快手test/kuairec_abtest/docs/win_开机配置README.md:1)。

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
    IdentitiesOnly yes
    PreferredAuthentications publickey
    ServerAliveInterval 60
    ServerAliveCountMax 3

# 同局域网直连
Host win-local
    HostName 192.168.1.18
    Port 22
    User thisi
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    PreferredAuthentications publickey
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

---

## Windows 侧配置详情

### OpenSSH Server

- 服务名：`sshd`
- 启动类型：Automatic（开机自启）
- 监听端口：22
- 建议认证方式：**仅公钥登录**
- 公钥文件：`C:\ProgramData\ssh\administrators_authorized_keys`
  - `thisi` 是管理员组成员，必须用这个路径，不能用 `~/.ssh/authorized_keys`
  - 文件权限：只有 SYSTEM 和 Administrators 可读（`icacls` 设置）

建议在 `C:\ProgramData\ssh\sshd_config` 里至少确认这些项：

```text
PubkeyAuthentication yes
PasswordAuthentication no
PermitEmptyPasswords no
AllowUsers thisi
```

### DefaultShell

注册表：`HKLM:\SOFTWARE\OpenSSH\DefaultShell`

推荐值：

```
C:\Windows\System32\wsl.exe
```

说明：

- 旧写法 `bash.exe` 现在已经属于 **deprecated WSL 命令**
- 更稳的做法是把 `DefaultShell` 指向 `wsl.exe`
- 然后确保默认发行版是 `Ubuntu-22.04`，默认 Linux 用户是 `thisislbk`

检查命令：

```powershell
Get-ItemProperty "HKLM:\SOFTWARE\OpenSSH" | Select-Object DefaultShell
wsl.exe -l -v
wsl.exe --status
```

设置命令：

```powershell
Set-ItemProperty "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Windows\System32\wsl.exe" -Force
```

SSH 连接进来后自动启动默认 WSL shell。

### 防火墙

规则名：`OpenSSH-Server-In-TCP`
- 端口：22/TCP，入站
- 推荐：尽量不要长期 `Any` 全开
- 如果主要通过局域网 + Tailscale 访问，优先按 `RemoteAddress` 或网络配置文件收窄范围

### Tailscale

- 服务：`Tailscale`，Automatic 自启
- 建议：开启 **Run unattended**
- Windows 侧 IP：`100.75.10.89`
- Mac 侧需开启 Tailscale 才能用 `ssh win`

建议显式执行一次：

```powershell
tailscale up --unattended=true
```

### WSL

- 发行版：Ubuntu 22.04
- `.wslconfig`（`C:\Users\thisi\.wslconfig`）：
  ```ini
  [wsl2]
  networkingMode=mirrored
  ```
- WSL 自己的 sshd **已停用**，不需要，不要再开
- 如果使用 **Microsoft Store 版 WSL**，远程经由 Windows OpenSSH 进入 WSL 时，偶尔会碰到
  `The file cannot be accessed by the system` 这个已知问题
  - 这是 WSL 官方文档明确记录过的坑
  - 遇到时优先检查 `wsl.exe -v`
  - 必要时改用 **in-box WSL** 或 **WSL 1**

---

## 建议最小验收

Windows 侧至少确认一次：

```powershell
# 1. 服务是否正常
Get-Service sshd, Tailscale | Select-Object Name, Status, StartType

# 2. 22 端口是否真的在监听
netstat -an | findstr :22

# 3. 默认 shell 是否已切到 wsl.exe
Get-ItemProperty "HKLM:\SOFTWARE\OpenSSH" | Select-Object DefaultShell

# 4. 公钥 ACL 是否正确
icacls "C:\ProgramData\ssh\administrators_authorized_keys"

# 5. WSL 状态
wsl.exe -l -v
wsl.exe --status

# 6. Tailscale 状态
tailscale status
```

Mac 侧至少确认一次：

```bash
ssh win 'whoami && hostname'
ssh win 'wsl.exe -l -v'
tailscale ping 100.75.10.89
```

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
- wsl.exe（WSL）在 SSH 连接时按需启动
- Tailscale 服务开机自启

重启后直接 `ssh win` 即可。

---

## 排查

### 连不上

```bash
# 1. 检查 Tailscale 是否在跑（外网连接时）
# Mac 上：
tailscale status

# 2. Tailscale 链路测试（比普通 ping 更准）
tailscale ping 100.75.10.89

# 3. 局域网测试
ssh win-local

# 4. 检查 Windows sshd 服务（在 Windows 上）
Get-Service sshd

# 5. 检查 22 端口是否真的在监听（在 Windows 上）
netstat -an | findstr :22
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

### 进去是 cmd 而不是 WSL

```powershell
# 管理员 PowerShell 里跑
Set-ItemProperty "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Windows\System32\wsl.exe" -Force

# 必要时重启 sshd
Restart-Service sshd
```

### 进去时报 `The file cannot be accessed by the system`

```powershell
wsl.exe -v
wsl.exe -l -v
```

高概率是：

- 你在用 Microsoft Store 版 WSL
- 而且命中了“Windows OpenSSH 远程进入 WSL”的已知问题

优先处理方向：

- 先确认是不是 Store 版 WSL
- 再考虑切到 in-box WSL
- 或临时改用 WSL 1

### `sshd` 服务起不来（Error 1053 / 1067）

这是 2025 年后微软官方专门写过的一个高频坑。

```powershell
Get-Service sshd
Get-WinEvent -LogName System -MaxEvents 50 |
  Where-Object { $_.ProviderName -eq "Service Control Manager" -or $_.Id -in 7034,7031 } |
  Select-Object -First 10 TimeCreated, Id, ProviderName, Message
```

常见原因：

- OpenSSH Client / Server / `libcrypto.dll` 版本不匹配
- 某些 Windows 更新后的 OpenSSH 9.5.2.1 启动异常

优先处理：

- 先打到 **2025-03-11 之后** 的更新
- 不要只装 OpenSSH Server、不装 Client
- 如果还是不稳，再考虑升级 Win32-OpenSSH

### Tailscale 在跑，但 `ssh win` 还是不通

这时不要只看普通 `ping`，先看 Tailscale 自己的链路：

```powershell
tailscale status
tailscale ping 100.75.10.89
tailscale ping --tsmp 100.75.10.89
```

如果 `tailscale ping` 能通，但 SSH 不通，更像是：

- Windows 防火墙规则
- `sshd` 没监听 22
- OpenSSH 配置或 shell 启动问题

### Tailscale 突然要求重新登录 / 服务起不来

如果日志里出现 `failed to unseal state file`，要怀疑 TPM / 状态文件问题。

这是最后手段：

1. 去 Tailscale 管理后台移除旧设备
2. 删除本机状态目录

```text
C:\ProgramData\Tailscale
C:\Users\%USERNAME%\AppData\Local\Tailscale
```

3. 重新登录注册节点

---

## 为什么不用 WSL 自带的 sshd

| 方案 | 稳定性 | 复杂度 | 重启后 |
|------|--------|--------|--------|
| Windows 原生 SSH ✅ | 高（系统服务） | 低 | 自动恢复 |
| WSL sshd（旧方案） | 中（依赖 WSL 进程） | 高（需 portproxy，IP 动态变） | 需手动或脚本维护 |
