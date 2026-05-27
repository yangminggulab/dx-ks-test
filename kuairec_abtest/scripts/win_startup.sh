#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  Windows 4060 服务器一键启动脚本
#  从 Mac 端运行，通过 SSH 远程拉起所有必要服务
#
#  用法：
#    bash win_startup.sh          # 启动所有服务 + 验证状态
#    bash win_startup.sh --status # 只查看当前服务状态
#
#  前提：sshd 必须已在运行（sshd 挂了只能去机器旁边手动 Start-Service sshd）
# ═══════════════════════════════════════════════════════════════════════

SSH="ssh win-local"

# ── 查看状态 ─────────────────────────────────────────────────────────
if [ "$1" = "--status" ]; then
    echo "=== 服务状态 ==="
    $SSH "powershell.exe -Command \"Get-Service sshd, Tailscale | Select-Object Name, Status\"" 2>/dev/null
    echo ""
    echo "=== GPU 状态 ==="
    $SSH "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null
    echo ""
    echo "=== 电源策略（睡眠超时，0=永不）==="
    $SSH "powershell.exe -Command \"
        \\\$ac = (powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String 'AC').ToString().Trim()
        \\\$dc = (powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String 'DC').ToString().Trim()
        Write-Host \\\"Standby AC: \\\$ac\\\"
        Write-Host \\\"Standby DC: \\\$dc\\\"
    \"" 2>/dev/null
    exit 0
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Windows 4060 启动序列"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 检查 SSH 是否可达 ──────────────────────────────────────────────
echo ""
echo "▶ 检查 SSH 连接..."
if ! $SSH "echo ok" &>/dev/null; then
    echo "[FAIL] SSH 不可达（sshd 未运行）"
    echo ""
    echo "  → 去机器旁边打开 PowerShell，手动运行："
    echo "      Start-Service sshd"
    echo "      Start-Service Tailscale"
    echo "  → 或者用 win_startup_local.ps1 在 Windows 本地执行（见同目录）"
    exit 1
fi
echo "  ✓ SSH 可达"

# ── 启动 Tailscale ────────────────────────────────────────────────
echo ""
echo "▶ 启动 Tailscale..."
$SSH "powershell.exe -Command \"
    \\\$svc = Get-Service Tailscale -ErrorAction SilentlyContinue
    if (\\\$svc -and \\\$svc.Status -ne 'Running') {
        Start-Service Tailscale
        Write-Host 'Tailscale 已启动'
    } elseif (\\\$svc.Status -eq 'Running') {
        Write-Host 'Tailscale 已在运行'
    } else {
        Write-Host '[WARN] Tailscale 服务不存在，请检查安装'
    }
\"" 2>/dev/null

# ── 确认电源策略（防止 idle sleep / hibernate）───────────────────
echo ""
echo "▶ 确认电源策略..."
$SSH "powershell.exe -Command \"
    powercfg /change standby-timeout-ac 0
    powercfg /change standby-timeout-dc 0
    powercfg /change hibernate-timeout-ac 0
    powercfg /change hibernate-timeout-dc 0
    powercfg /h off
    Write-Host '电源策略已确认（idle sleep + hibernate 均已禁用）'
\"" 2>/dev/null

# ── 最终状态 ──────────────────────────────────────────────────────
echo ""
echo "▶ 最终服务状态："
$SSH "powershell.exe -Command \"Get-Service sshd, Tailscale | Select-Object Name, Status\"" 2>/dev/null

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ 启动完成"
echo "  局域网连接：ssh win-local"
echo "  外网连接：  ssh win（需 Mac 侧 Tailscale 开启）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
