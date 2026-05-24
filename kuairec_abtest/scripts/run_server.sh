#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  GPU 服务器一键启动脚本（断连安全版）
#  使用方式见下方 Usage
# ═══════════════════════════════════════════════════════════════════════
#
#  Usage:
#    bash run_server.sh           # 默认：跑全部模型（不含 LLM），nohup 后台
#    bash run_server.sh --with-llm         # 含 LLMRec
#    bash run_server.sh --force            # 强制重跑（忽略 checkpoint）
#    bash run_server.sh --quick            # 快速验证（10 epoch，patience=3）
#    bash run_server.sh --status           # 查看当前训练进度（tail log）
#    bash run_server.sh --kill             # 停止后台训练进程
#
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── 路径配置 ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/output"
LOG_FILE="$OUTPUT_DIR/experiment.log"
PID_FILE="$OUTPUT_DIR/experiment.pid"

mkdir -p "$OUTPUT_DIR"

# ── 解析参数 ─────────────────────────────────────────────────────────
PY_ARGS=""
ACTION="start"

for arg in "$@"; do
    case "$arg" in
        --with-llm)    PY_ARGS="$PY_ARGS --with-llm" ;;
        --force)       PY_ARGS="$PY_ARGS --force" ;;
        --quick)       PY_ARGS="$PY_ARGS --n-epochs 10 --patience 3" ;;
        --status)      ACTION="status" ;;
        --kill)        ACTION="kill" ;;
        --help|-h)     ACTION="help" ;;
        *)             PY_ARGS="$PY_ARGS $arg" ;;
    esac
done

# ── 状态查询 ─────────────────────────────────────────────────────────
if [ "$ACTION" = "status" ]; then
    echo "=== 实验进度 ==="
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "训练进程运行中（PID=$PID）"
        else
            echo "PID=$PID 进程已退出（训练可能已完成或异常退出）"
        fi
    else
        echo "未找到 PID 文件，可能未在后台运行"
    fi
    echo ""
    echo "=== 已完成的模型 ==="
    if ls "$OUTPUT_DIR/results/"*.json 2>/dev/null | head -1 > /dev/null 2>&1; then
        for f in "$OUTPUT_DIR/results/"*.json; do
            python3 -c "
import json, sys
with open('$f') as fp: d = json.load(fp)
hr   = d.get('hit_rate')
ndcg = d.get('ndcg')
mins = d.get('train_min', 0)
hr_str   = f'{hr:.4f}' if hr is not None else 'pending'
ndcg_str = f'{ndcg:.4f}' if ndcg is not None else 'pending'
print(f\"  {d['model_key']:12s}  HR={hr_str}  NDCG={ndcg_str}  ({mins:.1f}min)\")
" 2>/dev/null || echo "  $(basename $f)"
        done
    else
        echo "  （暂无已完成结果）"
    fi
    echo ""
    echo "=== 最新日志（最后 20 行）==="
    if [ -f "$LOG_FILE" ]; then
        tail -20 "$LOG_FILE"
    else
        echo "  日志文件不存在"
    fi
    exit 0
fi

# ── 停止 ─────────────────────────────────────────────────────────────
if [ "$ACTION" = "kill" ]; then
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "已发送 SIGTERM 到进程 PID=$PID"
            echo "（checkpoint 已保存，下次启动会从中断处续训）"
        else
            echo "进程 $PID 已不存在"
        fi
        rm -f "$PID_FILE"
    else
        echo "未找到 PID 文件"
    fi
    exit 0
fi

# ── 帮助 ─────────────────────────────────────────────────────────────
if [ "$ACTION" = "help" ]; then
    head -25 "${BASH_SOURCE[0]}" | tail -18
    exit 0
fi

# ── 检查已有进程 ──────────────────────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[WARN] 已有训练进程在运行（PID=$PID）"
        echo "  查看进度：bash run_server.sh --status"
        echo "  停止训练：bash run_server.sh --kill"
        exit 1
    fi
fi

# ── 检测 Python 命令 ──────────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERROR] 找不到 python3 或 python，请先激活 conda 环境："
    echo "  conda activate <your_env>"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1)
echo "Python: $PY_VERSION"
echo "设备检测："
$PYTHON -c "
import torch
if torch.cuda.is_available():
    print(f'  CUDA 可用：{torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    print('  Apple MPS 可用')
else:
    print('  仅 CPU（训练会很慢，建议检查 CUDA 环境）')
" 2>/dev/null || echo "  torch 未安装"

echo ""

# ── 切到 scripts 目录 ─────────────────────────────────────────────────
cd "$SCRIPT_DIR"

# ── 启动（nohup 后台）────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  启动训练（nohup 后台，断 SSH 不影响）"
echo "  参数：$PY_ARGS"
echo "  日志：$LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 写入启动时间
echo "=== 训练启动：$(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_FILE"

nohup $PYTHON run_all_experiments.py $PY_ARGS >> "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "  后台进程 PID = $TRAIN_PID"
echo "  PID 已写入：$PID_FILE"
echo ""
echo "  常用命令："
echo "    查看进度：bash run_server.sh --status"
echo "    实时日志：tail -f $LOG_FILE"
echo "    停止训练：bash run_server.sh --kill"
echo ""
echo "  ✓ 现在可以安全断开 SSH，训练在后台继续运行。"
echo ""

# 等 3 秒确认进程没有立刻退出
sleep 3
if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "  [OK] 进程运行正常，最新日志："
    tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/    /'
else
    echo "  [ERROR] 进程已退出！请检查日志："
    tail -20 "$LOG_FILE" 2>/dev/null | sed 's/^/    /'
    rm -f "$PID_FILE"
    exit 1
fi
