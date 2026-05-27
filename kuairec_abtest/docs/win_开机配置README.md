# Windows 4060 开机自启配置文档

每次开机 / 从睡眠唤醒后，需要以下四件事全部就绪：

| # | 项目 | 验证命令 | 预期结果 |
|---|------|---------|---------|
| 1 | sshd 服务运行中 | `Get-Service sshd` | Status = Running |
| 2 | Tailscale 服务运行中 | `Get-Service Tailscale` | Status = Running |
| 3 | idle sleep 禁用（不自动睡眠） | `powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE` | AC/DC 值均为 0x00000000 |
| 4 | 休眠禁用（hibernate off） | `powercfg /a` | 休眠 = 未启用 |

---

## 第一步：创建启动脚本

在 Windows 上（管理员 PowerShell）新建文件：

```
C:\ProgramData\ssh\win_boot.ps1
```

内容：

```powershell
# Windows 4060 开机/唤醒启动脚本
# 路径：C:\ProgramData\ssh\win_boot.ps1

# 1. 确保 sshd 在跑
if ((Get-Service sshd).Status -ne 'Running') {
    Start-Service sshd
}

# 2. 确保 Tailscale 在跑
if ((Get-Service Tailscale -ErrorAction SilentlyContinue).Status -ne 'Running') {
    Start-Service Tailscale
}

# 3. 禁用 idle 自动睡眠
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0

# 4. 禁用休眠
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0
powercfg /h off
```

---

## 第二步：注册为任务计划（开机 + 唤醒均触发）

管理员 PowerShell 里一次性运行：

```powershell
$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
               -Argument "-NonInteractive -WindowStyle Hidden -File C:\ProgramData\ssh\win_boot.ps1"

$trigger1 = New-ScheduledTaskTrigger -AtStartup
$trigger2 = New-ScheduledTaskTrigger -AtLogOn   # 兜底，Startup 有时不可靠

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
    -StartWhenAvailable `
    -WakeToRun

$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest

Register-ScheduledTask `
    -TaskName  "WinBoot4060" `
    -Action    $action `
    -Trigger   @($trigger1, $trigger2) `
    -Settings  $settings `
    -Principal $principal `
    -Force
```

---

## 第三步：验证注册成功

```powershell
Get-ScheduledTask -TaskName "WinBoot4060" | Select-Object TaskName, State
# 预期：State = Ready
```

---

## 逐项排查

### ① sshd 没起来

```powershell
Get-Service sshd
# 如果 Stopped：
Start-Service sshd
# 如果 StartType 不是 Automatic：
Set-Service sshd -StartupType Automatic
```

### ② Tailscale 没起来

```powershell
Get-Service Tailscale
Start-Service Tailscale
Set-Service Tailscale -StartupType Automatic
```

### ③ idle sleep 没禁掉（机器唤醒后自己又睡）

```powershell
# 查当前值（应该都是 0）
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE
# 重新设置
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
```

### ④ 休眠没禁掉（机器深度睡眠后 ping 也不通）

```powershell
powercfg /a
# 如果"休眠"还在可用列表里：
powercfg /h off
```

### ⑤ WinBoot4060 任务没触发

```powershell
# 查上次运行时间和结果
Get-ScheduledTaskInfo -TaskName "WinBoot4060" | Select-Object LastRunTime, LastTaskResult
# LastTaskResult = 0 表示成功，其他值表示失败
# 手动触发测试：
Start-ScheduledTask -TaskName "WinBoot4060"
```

---

## 当前状态（2026-05-26）

| 项目 | 状态 |
|------|------|
| sshd 开机自启 | ✅ Automatic（Windows 原生） |
| Tailscale 开机自启 | ❓ 未确认 |
| standby-timeout = 0 | ✅ 今日已执行 |
| hibernate off | ✅ 今日已执行 |
| WinBoot4060 任务计划 | ❌ 未注册（待执行第二步） |

**下次连上后最需要做的一件事**：执行第二步，注册 `WinBoot4060` 任务计划。
