# Windows 4060 开机/唤醒自启配置

## 核心问题与解法

反复失败的根本原因有两个：

**① `AtStartup` 触发器不管用**
Windows 任务计划的"系统启动时"触发器只在**冷启动**时触发，从睡眠唤醒时完全不会触发。
必须用**事件日志触发器**监听唤醒事件（System 日志 EventID=1）。

**② sshd 唤醒后容易崩**
网络栈重新初始化时 sshd 有时会失去绑定，必须配置**服务故障自动重启**作为兜底。

---

## 一次性配置（管理员 PowerShell 全部执行）

### 第一步：创建启动脚本

```powershell
$script = @'
# sshd
if ((Get-Service sshd).Status -ne 'Running') { Start-Service sshd }

# Tailscale
$ts = Get-Service Tailscale -ErrorAction SilentlyContinue
if ($ts -and $ts.Status -ne 'Running') { Start-Service Tailscale }

# 禁止 idle 睡眠
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0

# 禁止休眠
powercfg /h off
'@
$script | Out-File "C:\ProgramData\ssh\win_boot.ps1" -Encoding UTF8

# 关键：这个脚本会被 SYSTEM 任务执行，权限必须收紧
icacls "C:\ProgramData\ssh\win_boot.ps1" /inheritance:r /grant "SYSTEM:(F)" /grant "Administrators:(F)"
```

### 第二步：注册任务计划（冷启动 + 唤醒双触发，用 XML）

```powershell
# 唤醒事件：System 日志，Power-Troubleshooter，EventID=1
$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>开机/唤醒后启动 sshd、Tailscale，禁用睡眠</Description>
  </RegistrationInfo>
  <Triggers>
    <BootTrigger>
      <Delay>PT15S</Delay>
    </BootTrigger>
    <EventTrigger>
      <Delay>PT10S</Delay>
      <Subscription>&lt;QueryList&gt;&lt;Query Id="0" Path="System"&gt;&lt;Select Path="System"&gt;*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;</Subscription>
    </EventTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>S-1-5-18</UserId>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <StartWhenAvailable>true</StartWhenAvailable>
    <ExecutionTimeLimit>PT2M</ExecutionTimeLimit>
  </Settings>
  <Actions>
    <Exec>
      <Command>powershell.exe</Command>
      <Arguments>-NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "C:\ProgramData\ssh\win_boot.ps1"</Arguments>
    </Exec>
  </Actions>
</Task>
"@
$xml | Out-File "$env:TEMP\WinBoot4060.xml" -Encoding Unicode
schtasks /Create /TN "WinBoot4060" /XML "$env:TEMP\WinBoot4060.xml" /F
```

### 第三步：配置 sshd 和 Tailscale 故障自动重启（兜底）

```powershell
# sshd：崩了 3 秒后自动重启，一天内最多 3 次
sc.exe failure sshd reset= 86400 actions= restart/3000/restart/3000/restart/5000
sc.exe failureflag sshd 1

# Tailscale
sc.exe failure Tailscale reset= 86400 actions= restart/5000/restart/5000/restart/10000
sc.exe failureflag Tailscale 1

# 确认两个服务都是自动启动
Set-Service sshd     -StartupType Automatic
Set-Service Tailscale -StartupType Automatic
```

### 第四步：建议顺手补上的安全加固

这一步不是“能不能跑”的硬条件，但如果机器要长期暴露 SSH，建议一起做。

#### 4.1 Tailscale 开 unattended mode

仅仅 `Start-Service Tailscale` 不等于“未登录用户也能稳定远程进来”。建议显式打开 unattended mode：

```powershell
tailscale up --unattended=true
```

#### 4.2 OpenSSH 尽量只开公钥，不开密码

编辑 `C:\ProgramData\ssh\sshd_config`，至少确认这些项：

```text
PubkeyAuthentication yes
PasswordAuthentication no
PermitEmptyPasswords no
AllowUsers thisi
```

改完重启 sshd：

```powershell
Restart-Service sshd
```

#### 4.3 SSH 防火墙规则不要无限放开

如果你平时只走局域网 + Tailscale，建议把 `OpenSSH-Server-In-TCP` 的暴露范围收窄，不要长期 `Any` 全开。  
如果不确定 Tailscale 接口落在哪个网络配置文件上，优先用 `RemoteAddress` 收窄，别一上来就只留 `Private`。

```powershell
# 先看当前规则
Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" | Format-List Name, Enabled, Profile, Direction, Action

# 示例 1：只允许 Private 网络
Set-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -Profile Private

# 示例 2：进一步限制 RemoteAddress（按你自己的局域网/Tailnet 实际地址改）
# Set-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -RemoteAddress 192.168.1.0/24,100.x.x.x
```

#### 4.4 顺手核对管理员公钥文件 ACL

如果登录用户是管理员组成员，`administrators_authorized_keys` 的权限也必须足够严格：

```powershell
icacls "C:\ProgramData\ssh\administrators_authorized_keys"
icacls "C:\ProgramData\ssh\administrators_authorized_keys" /inheritance:r /grant "Administrators:(F)" /grant "SYSTEM:(F)"
```

---

## 验证每一步是否成功

```powershell
# 1. 任务计划是否注册成功
Get-ScheduledTask -TaskName "WinBoot4060" | Select-Object TaskName, State
# 预期：State = Ready

# 2. 任务上次运行情况（手动触发测试）
Start-ScheduledTask -TaskName "WinBoot4060"
Start-Sleep 5
Get-ScheduledTaskInfo -TaskName "WinBoot4060" | Select-Object LastRunTime, LastTaskResult
# 预期：LastTaskResult = 0（成功）

# 3. 服务状态
Get-Service sshd, Tailscale | Select-Object Name, Status, StartType
# 预期：Status = Running，StartType = Automatic

# 4. 恢复策略是否已生效
sc.exe qfailure sshd
sc.exe qfailure Tailscale
# 预期：能看到 restart 动作

# 5. 失败标志是否已打开
sc.exe qfailureflag sshd
sc.exe qfailureflag Tailscale
# 预期：FAILURE_ACTIONS_ON_NONCRASH_FAILURES = TRUE

# 6. 启动脚本 ACL 是否安全
icacls "C:\ProgramData\ssh\win_boot.ps1"
# 预期：只剩 SYSTEM / Administrators

# 7. 电源策略
powercfg /a
# 预期："休眠"在"以下睡眠状态在此系统上不可用"里

# 8. 唤醒事件是否真的能打到日志
Get-WinEvent -LogName System -MaxEvents 20 |
  Where-Object { $_.ProviderName -eq "Microsoft-Windows-Power-Troubleshooter" -and $_.Id -eq 1 } |
  Select-Object -First 3 TimeCreated, Id, ProviderName

# 9. Tailscale unattended 是否已开启（能看到登录态/节点状态即可）
tailscale status
```

---

## 当前状态（2026-05-26）

| 步骤 | 状态 |
|------|------|
| standby-timeout = 0 | ✅ 已执行 |
| hibernate off | ✅ 已执行 |
| sshd Automatic 启动 | ✅ 已确认 |
| Tailscale Automatic 启动 | ✅ 已确认 |
| sshd 故障自动重启 | ✅ 已配置（3s/3s/5s） |
| sshd / Tailscale failureflag | ✅ 已配置 |
| win_boot.ps1 ACL 收紧 | ✅ 已配置 |
| WinBoot4060 任务计划 | ✅ 已注册（Ready） |

**连上后只需执行上面三步，约 1 分钟搞定，之后不需要再动。**

---

## 这份配置当前最值得警惕的点

### 1. `-ExecutionPolicy Bypass` 本身不是最大风险，脚本 ACL 才是

这里真正危险的不是 `Bypass` 这四个字，而是“有没有低权限用户能改 `C:\ProgramData\ssh\win_boot.ps1`”。  
如果能改，那么任务计划以 `SYSTEM` 身份执行这个脚本时，就会变成提权入口。

### 2. `OpenSSH-Server-In-TCP` 如果长期对 `Any` 网络开放，暴露面偏大

如果你只需要局域网和 Tailscale，建议至少收敛到 `Private`，更进一步再按 `RemoteAddress` 缩小。

### 3. 如果 `PasswordAuthentication` 还开着，公网/广域网口子会比你想象得大

尤其是你已经有公钥登录条件时，继续保留密码登录通常没有收益，只增加爆破面。

---

## 上网补查后，建议额外记住的常见 Bug

### A. Windows 更新后，`sshd` 直接起不来（Error 1053 / 1067 / Event ID 7034）

这是 2025 年后微软官方单独写过两篇排障文的坑，不算小概率。

常见表现：

- `OpenSSH SSH Server` 无法启动
- 报 `Error 1053`
- 或系统日志里出现 `Event ID 7034`

常见原因：

- OpenSSH Client / Server / `libcrypto.dll` 版本不匹配
- 某些 2024-10 到 2025-03 期间的 Windows 更新把 OpenSSH 9.5.2.1 搞坏了

快速处理：

```powershell
# 先看服务状态
Get-Service sshd

# 看最近系统日志
Get-WinEvent -LogName System -MaxEvents 50 |
  Where-Object { $_.ProviderName -eq "Service Control Manager" -or $_.Id -in 7034,7031 } |
  Select-Object -First 10 TimeCreated, Id, ProviderName, Message
```

如果是“装完 OpenSSH FoD 后起不来”：

- 重新安装**最新累积更新**
- 不要只装 OpenSSH Server，不装 Client

如果是“Windows 更新后突然起不来”：

- 优先装 **2025-03-11 之后** 的更新
- 如果仍不稳，再考虑升级到新版 Win32-OpenSSH，并保留 RDP/本地控制台兜底

### B. 远程 SSH 进 WSL 时，报 `The file cannot be accessed by the system`

这个是微软 WSL 文档里明确写的已知问题。

触发场景：

- Windows 上跑 `openssh-server`
- 登录后想直接进 WSL
- 使用的是 **Microsoft Store 版 WSL**

已知绕法：

- 改用 **WSL 1**
- 或改用 **Windows 内置版 WSL**

这意味着：如果你后面发现 `sshd` 明明正常、但进 shell 失败，不一定是 SSH 配置错了，也可能是 WSL 分发版来源/版本问题。

建议额外留一个检查命令：

```powershell
wsl.exe -v
wsl.exe -l -v
```

### C. Tailscale 服务在跑，但网络配置失败

Tailscale 官方现在单独有个 Windows 消息页专门讲这个。

常见表现：

- Tailscale 看起来已启动
- 但实际不能连
- 或出现 `network configuration failed`

常见原因：

- 其他 VPN / 虚拟网卡 / 网络代理冲突
- 防火墙或安全软件拦截
- Tailscale 虚拟适配器没起来
- Windows 网络组件或驱动状态异常

快速处理：

```powershell
Get-Service Tailscale
ipconfig /all
```

排查顺序建议：

1. 重启 Tailscale 客户端
2. 升级到最新版
3. 用管理员权限运行
4. 检查是否装了其他 VPN、Hyper-V/虚拟网络增强工具、代理或安全软件
5. 确认 Tailscale 虚拟网卡存在且启用

### D. Tailscale 因 TPM / 状态文件问题起不来

这是另一个很像“突然失联”的坑，尤其在固件更新、TPM 异常之后。

常见表现：

- Tailscale 服务无法正常启动
- 日志里出现 `failed to unseal state file`

官方思路：

- 先看客户端日志，确认是不是 TPM 解封失败
- 如果 TPM 状态修不好，就删节点状态，重新登录注册

重置前要先去管理后台移除旧设备，然后删除：

```text
C:\ProgramData\Tailscale
C:\Users\%USERNAME%\AppData\Local\Tailscale
```

这个动作会让节点重新注册，所以适合写进“最后手段”而不是日常步骤。

### E. `ping` 不通，不一定代表 Tailscale/SSH 真坏了

Tailscale 官方专门提醒过：Windows 对 ICMP 防火墙规则比较激进。

也就是说：

- `ping 100.x.x.x` 失败，不一定代表 Tailnet 不通
- 更应该看 `tailscale ping`

建议把排障顺序固定成：

```powershell
tailscale status
tailscale ping 100.75.10.89
tailscale ping --tsmp 100.75.10.89
```

如果 `tailscale ping` 通，但普通 `ping` 不通，更像是 Windows ICMP 防火墙问题，不是 Tailscale 主链路问题。

### F. 唤醒事件没触发时，先别怀疑 XML，先看两件事

第一，看你是不是**真的从 Sleep/Hibernate 唤醒**，而不是冷启动/重启。  
`Power-Troubleshooter Event ID 1` 主要记录的是“从低功耗状态返回”，不是普通重启。

第二，看任务历史和 TaskScheduler 运行日志，而不只看任务列表状态：

```powershell
Get-WinEvent -LogName "Microsoft-Windows-TaskScheduler/Operational" -MaxEvents 50 |
  Select-Object -First 20 TimeCreated, Id, LevelDisplayName, Message
```

如果以后你手工重建任务而不是继续用现在这份 XML，还要注意一个社区里高频踩坑点：

- 事件源要写 **`Microsoft-Windows-Power-Troubleshooter`**
- 不是只写界面里看见的 `Power-Troubleshooter`

你现在这份 XML 已经写对了。
