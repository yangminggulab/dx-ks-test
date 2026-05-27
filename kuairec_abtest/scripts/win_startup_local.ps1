# ═══════════════════════════════════════════════════════════════════════
#  Windows 4060 本地启动脚本（在 Windows 上直接运行）
#  适用场景：sshd 挂了、Tailscale 断了、SSH 连不上时，去机器旁边跑这个
#
#  用法（管理员 PowerShell）：
#    .\win_startup_local.ps1
# ═══════════════════════════════════════════════════════════════════════

Write-Host "▶ 启动 sshd..." -ForegroundColor Cyan
Start-Service sshd -ErrorAction SilentlyContinue
Get-Service sshd | Select-Object Name, Status

Write-Host ""
Write-Host "▶ 启动 Tailscale..." -ForegroundColor Cyan
Start-Service Tailscale -ErrorAction SilentlyContinue
Get-Service Tailscale | Select-Object Name, Status

Write-Host ""
Write-Host "▶ 确认电源策略（idle sleep + hibernate 禁用）..." -ForegroundColor Cyan
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0
powercfg /h off
Write-Host "  电源策略已设置" -ForegroundColor Green

Write-Host ""
Write-Host "✓ 完成。现在可以从 Mac 用 ssh win-local 或 ssh win 连入。" -ForegroundColor Green
