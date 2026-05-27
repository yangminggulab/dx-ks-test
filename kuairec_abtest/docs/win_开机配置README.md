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

# Tailscale
sc.exe failure Tailscale reset= 86400 actions= restart/5000/restart/5000/restart/10000

# 确认两个服务都是自动启动
Set-Service sshd     -StartupType Automatic
Set-Service Tailscale -StartupType Automatic
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

# 4. 电源策略
powercfg /a
# 预期："休眠"在"以下睡眠状态在此系统上不可用"里
```

---

## 当前状态（2026-05-26）

| 步骤 | 状态 |
|------|------|
| standby-timeout = 0 | ✅ 已执行 |
| hibernate off | ✅ 已执行 |
| sshd Automatic 启动 | ✅ 原有配置 |
| Tailscale Automatic 启动 | ❓ 未确认 |
| sshd 故障自动重启 | ❌ 未配置 |
| WinBoot4060 任务计划 | ❌ 未注册 |

**连上后只需执行上面三步，约 1 分钟搞定，之后不需要再动。**
