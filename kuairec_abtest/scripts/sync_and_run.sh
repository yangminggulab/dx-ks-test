#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  Mac 端一键脚本：同步代码 → SSH 启动训练（RTX 4060 WSL2 服务器）
#
#  在 Mac 上运行：
#    bash sync_and_run.sh           # 同步 + 启动（跑全部模型，不含LLM）
#    bash sync_and_run.sh --status  # 查看训练进度
#    bash sync_and_run.sh --log     # 实时追看日志（Ctrl+C 停止查看，不影响训练）
#    bash sync_and_run.sh --kill    # 停止后台训练
#    bash sync_and_run.sh --with-llm  # 同步 + 启动含 LLM 实验
#    bash sync_and_run.sh --force      # 强制重跑（忽略已有结果）
# ═══════════════════════════════════════════════════════════════════════

# ── 服务器配置（建议通过环境变量注入，避免把个人地址写进仓库）──────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${KUAIREC_SERVER:-}"
PORT="${KUAIREC_PORT:-2222}"
REMOTE_DIR="${KUAIREC_REMOTE_DIR:-~/kuairec_abtest}"
LOCAL_SCRIPTS="${KUAIREC_LOCAL_SCRIPTS:-$SCRIPT_DIR}"
SSH="ssh -p $PORT $SERVER"
LOG_FILE="$REMOTE_DIR/output/experiment.log"
PID_FILE="$REMOTE_DIR/output/experiment.pid"

if [ -z "$SERVER" ]; then
    echo "[ERROR] 请先设置服务器地址，例如："
    echo "  export KUAIREC_SERVER='user@your-host'"
    echo "  export KUAIREC_PORT='2222'              # 可选"
    echo "  export KUAIREC_REMOTE_DIR='~/kuairec_abtest'  # 可选"
    exit 1
fi

# ── 解析参数 ─────────────────────────────────────────────────────────
PY_ARGS=""
ACTION="start"

for arg in "$@"; do
    case "$arg" in
        --status)   ACTION="status" ;;
        --log)      ACTION="log" ;;
        --kill)     ACTION="kill" ;;
        --with-llm) PY_ARGS="$PY_ARGS --with-llm" ;;
        --force)    PY_ARGS="$PY_ARGS --force" ;;
        --quick)    PY_ARGS="$PY_ARGS --n-epochs 10 --patience 3" ;;
        --help|-h)  ACTION="help" ;;
        *)          PY_ARGS="$PY_ARGS $arg" ;;
    esac
done

# ── 帮助 ─────────────────────────────────────────────────────────────
if [ "$ACTION" = "help" ]; then
    head -15 "${BASH_SOURCE[0]}" | tail -9
    exit 0
fi

# ── 查看进度 ─────────────────────────────────────────────────────────
if [ "$ACTION" = "status" ]; then
    echo "=== 服务器状态 ==="
    $SSH "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi 不可用'"
    echo ""
    echo "=== 训练进程 ==="
    $SSH "
if [ -f $PID_FILE ]; then
    PID=\$(cat $PID_FILE)
    if kill -0 \$PID 2>/dev/null; then
        echo \"训练中 PID=\$PID\"
    else
        echo \"PID=\$PID 已退出（训练完成或异常）\"
    fi
else
    echo '未找到 PID 文件（未在后台运行）'
fi
"
    echo ""
    echo "=== 已完成的模型 ==="
    $SSH "
for f in $REMOTE_DIR/output/results/*.json; do
    [ -f \"\$f\" ] || continue
    python3 -c \"
import json
with open('\$f') as fp: d = json.load(fp)
hr   = d.get('hit_rate')
ndcg = d.get('ndcg')
mins = d.get('train_min', 0)
hr_str   = f'{hr:.4f}' if hr is not None else 'pending'
ndcg_str = f'{ndcg:.4f}' if ndcg is not None else 'pending'
print(f\\\"  {d.get('model_key','?'):12s}  HR={hr_str}  NDCG={ndcg_str}  ({mins:.1f}min)\\\")
\" 2>/dev/null || echo \"  \$(basename \$f)\"
done
" 2>/dev/null || echo "  （暂无结果）"
    echo ""
    echo "=== 最新日志（最后 15 行）==="
    $SSH "tail -15 $LOG_FILE 2>/dev/null || echo '日志文件不存在'"
    exit 0
fi

# ── 实时日志 ─────────────────────────────────────────────────────────
if [ "$ACTION" = "log" ]; then
    echo "实时日志（Ctrl+C 停止查看，不影响服务器训练）："
    $SSH "tail -f $LOG_FILE"
    exit 0
fi

# ── 停止训练 ─────────────────────────────────────────────────────────
if [ "$ACTION" = "kill" ]; then
    $SSH "
if [ -f $PID_FILE ]; then
    PID=\$(cat $PID_FILE)
    if kill -0 \$PID 2>/dev/null; then
        kill \$PID
        echo \"已停止 PID=\$PID（checkpoint 已保存，下次可续训）\"
    else
        echo \"进程 \$PID 已不存在\"
    fi
    rm -f $PID_FILE
else
    echo '未找到 PID 文件'
fi
"
    exit 0
fi

# ── 同步代码 ─────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 1: 同步代码到服务器"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

rsync -avz --progress \
    -e "ssh -p $PORT" \
    "$LOCAL_SCRIPTS/" \
    "$SERVER:$REMOTE_DIR/scripts/" \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude ".DS_Store"

if [ $? -ne 0 ]; then
    echo "[ERROR] rsync 失败，请检查服务器连接"
    exit 1
fi

echo ""
echo "  代码同步完成 ✓"
echo ""

# ── 检查已有训练进程 ──────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 2: 检查服务器状态"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ALREADY_RUNNING=$($SSH "
if [ -f $PID_FILE ]; then
    PID=\$(cat $PID_FILE)
    if kill -0 \$PID 2>/dev/null; then
        echo yes
    else
        echo no
    fi
else
    echo no
fi
" 2>/dev/null)

if [ "$ALREADY_RUNNING" = "yes" ]; then
    echo ""
    echo "[WARN] 服务器上已有训练进程在运行！"
    echo "  查看进度：bash sync_and_run.sh --status"
    echo "  停止训练：bash sync_and_run.sh --kill"
    echo ""
    exit 1
fi

# GPU 信息
echo ""
$SSH "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | awk '{print \"  GPU: \" \$0}'" 2>/dev/null || true
echo ""

# ── 启动训练 ─────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 3: 启动训练（nohup 后台，断 SSH 不影响）"
echo "  参数：$PY_ARGS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TRAIN_PID=$($SSH "
cd $REMOTE_DIR/scripts
mkdir -p $REMOTE_DIR/output
echo '=== 训练启动：'\$(date '+%Y-%m-%d %H:%M:%S')'===' >> $REMOTE_DIR/output/experiment.log
nohup python3 -u run_all_experiments.py $PY_ARGS \
    >> $REMOTE_DIR/output/experiment.log 2>&1 &
echo \$!
" 2>/dev/null)

if [ -z "$TRAIN_PID" ]; then
    echo "[ERROR] 启动失败，请检查："
    $SSH "tail -20 $REMOTE_DIR/output/experiment.log" 2>/dev/null
    exit 1
fi

# 写 PID 文件
$SSH "echo $TRAIN_PID > $PID_FILE"

echo ""
echo "  ✓ 训练已在后台启动（PID=$TRAIN_PID）"
echo ""

# 等 5 秒确认进程正常
sleep 5
STILL_RUNNING=$($SSH "kill -0 $TRAIN_PID 2>/dev/null && echo yes || echo no" 2>/dev/null)

if [ "$STILL_RUNNING" = "yes" ]; then
    echo "  [OK] 进程运行正常，最新日志："
    $SSH "tail -8 $REMOTE_DIR/output/experiment.log 2>/dev/null" | sed 's/^/    /'
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ✓ 现在可以关闭终端，训练在服务器后台继续运行"
    echo ""
    echo "  后续命令（在 Mac 上运行）："
    echo "    查看进度：bash sync_and_run.sh --status"
    echo "    实时日志：bash sync_and_run.sh --log"
    echo "    停止训练：bash sync_and_run.sh --kill"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "  [ERROR] 进程已退出！查看错误日志："
    $SSH "tail -30 $REMOTE_DIR/output/experiment.log 2>/dev/null" | sed 's/^/    /'
    exit 1
fi
