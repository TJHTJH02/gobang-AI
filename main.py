"""
五子棋 AI 对战平台 —— 后端服务（FastAPI 单文件实现）

包含：
  1) 全局唯一游戏状态（内存单例，无数据库、无多会话）
  2) 增量 O(K) 胜负判定（只在最新落子周围 4 个方向延伸检查，不全盘扫描）
  3) HTTP 接口（无 WebSocket）；AI 走子支持 SSE 流式返回，前端可实时看到
     提示词原文与模型的思考过程
  4) 本地 llama.cpp server（OpenAI 兼容）模型调用 + 失败回退策略
  5) 可切换「模型思考」开关：关掉 = 快（1~2 秒出坐标）；打开 = 慢但棋力更好
  6) 实验性「传统算法辅助决策」：后端先用棋型识别 + 候选点评分 + 威胁分级做侦察，
     把结构化结论转成自然语言塞进提示词，模型只负责在建议里挑一个（默认关闭）

启动：
    python main.py
    或  uvicorn main:app --reload
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import threading
import time
import webbrowser
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
APP_NAME = "五子棋 AI 对战平台"
APP_VERSION = "1.0.0"

DEFAULT_MODEL_BASE_URL = "http://localhost:10003"
DEFAULT_MODEL_NAME = "local-model"

# 服务监听端口。本机 8000 已被其他程序占用，故默认改用空闲端口 8123。
# 需要时可用 `python main.py <端口>` 或环境变量 PORT 覆盖。
DEFAULT_PORT = 8123

ALLOWED_SIZES = (9, 13, 15, 19)
ALLOWED_WIN_LENGTHS = (3, 4, 5)
ALLOWED_MODES = ("human_vs_ai", "ai_vs_ai", "human_vs_human")
# human = 真人 · ai = 走 llama.cpp（LLM） · algorithm = 纯算法，完全不联网、不调模型
ALLOWED_PLAYER_TYPES = ("human", "ai", "algorithm")
# 玩家类型别名（前端/请求体里可能写成别的叫法）
PLAYER_TYPE_ALIAS = {
    "ai_algorithm": "algorithm", "algo": "algorithm", "search": "algorithm",
    "llm": "ai", "model": "ai",
}

# 连接要快失败，读取要给足时间（思考模式可能要跑几十秒到几分钟，但流式可见进度）
AI_TIMEOUT = httpx.Timeout(connect=8.0, read=300.0, write=30.0, pool=10.0)
MODELS_TIMEOUT = 8.0     # /v1/models 探测超时（秒）

# 关思考时：24 个 token 就够（配合 enable_thinking=false 实测 1~2 秒出坐标）
# 开思考时：实测这个模型 1024/2048 个 token 都还在推理、content 始终为空，
#           所以还要靠「取推理末尾那个坐标」兜底（见 pick_move），预算给足让它能收敛。
MAX_TOKENS_NO_THINK = 24
MAX_TOKENS_NO_THINK_RETRY = 512
MAX_TOKENS_THINK = 2048

# 「非思考模式 + 允许输出解析」时给的预算：够写一两句简短分析 + 末尾坐标
MAX_TOKENS_ANALYSIS = 220

# 思考强度档位 → token 预算（数值越大推得越久、棋力通常越好）
THINKING_BUDGETS: Dict[str, int] = {"low": 512, "medium": 1024, "high": 2048, "ultra": 4096}
DEFAULT_THINKING_BUDGET = 1024

# 棋盘在提示词里的呈现方式
#   rows   —— 经典：每行一个行号 + 15 个数字
#   cells  —— 逐格标注「第N行第M列:值」（用户点名要的格式，最啰嗦但最不容易看错行列）
#   coords —— 只列已落子的格子 + 双方清单（信息密度最高，棋子少时优势明显）
BOARD_FORMATS = ("rows", "cells", "coords")
DEFAULT_BOARD_FORMAT = "rows"

# 模型采样参数默认值（llama.cpp 的原生参数名）
MODEL_PARAM_DEFAULTS: Dict[str, float] = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1,
}
MODEL_PARAM_LIMITS = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "top_k": (0, 200),
    "repeat_penalty": (0.5, 2.0),
}

# 混合决策（实验性）：必下局面由程序直接落子，高警惕局面只给模型少量候选点。
#   · hybrid_hints —— 「高警惕局面」允许交给模型的最大候选点数（超出就退回自由决策）
#   · 上限不设太大：候选点一多就失去「收窄思考范围」的意义，等于没限制
HYBRID_HINTS_MIN = 2
HYBRID_HINTS_MAX = 5
DEFAULT_HYBRID_HINTS = 3
# 只有 S/A/B 三级（五连·活四 / 冲四 / 活三）算「高警惕」，眠三及以下不干预
HYBRID_URGENT_LEVELS = ("S", "A", "B")

# 纯算法 AI（不调用模型）—— 难度档位。
#   depth    迭代加深的目标深度上限（时间到了就停在上一层，见 time_ms）
#   time_ms  每步的墙钟预算，硬上限；超时立刻收手，保证「再难也在一两秒内落子」
#   cands    每层保留的候选点数（走法排序质量决定剪枝效率，这个值是最主要的旋钮）
#   vcf      VCF（连续冲四杀）搜索的最大层数；0 = 关闭
# 说明：Python 下纯搜索的节点开销比 C++ 高两个量级，所以这里不承诺「深度 8」，
#       而是「给多少时间就搜多深」，实际达到的深度会随盘面复杂度浮动并在日志里如实标出。
#   nodes    搜索节点预算 —— **这才是真正的主约束**。用节点数而不是秒数来卡，
#            是为了让「同一局面 → 同一落点」严格成立：秒数会随机器负载浮动，
#            节点数不会。time_ms 只是名义耗时（给人看的），实际硬上限是它的
#            ALGO_TIME_SAFETY 倍，只在极端盘面上才可能先撞上。
# 难度档位。
#   depth  = 迭代加深的目标层数（能不能达到要看节点预算够不够，见 README 的「关于深度」）
#   nodes  = 主约束：搜索节点预算。**用节点而不是墙钟时间做截止，才能保证同一局面
#            在任何机器负载下都返回同一手棋**（这是「确定性」验收标准的前提）。
#   cands  = 每层保留的候选点数
#   vcf    = VCF（连续冲四杀）搜索的最大深度
#   time_ms= 名义墙钟预算，只用于算安全网（time_ms × ALGO_TIME_SAFETY），不是主约束
# 实测（15×15 安静中局，i9-14900HX，纯 Python）：easy 约 0.03s / medium 约 0.3s /
# hard 约 2.2s / ultra 约 6.0s。前 3 档满足「3 秒内出招」，ultra 是明确标注的分析档。
ALGO_LEVELS: Dict[str, Dict[str, Any]] = {
    "easy":   {"label": "简单", "depth": 2, "nodes": 300,  "cands": 8,  "vcf": 4,
               "time_ms": 250},
    "medium": {"label": "普通", "depth": 4, "nodes": 1200, "cands": 10, "vcf": 8,
               "time_ms": 800},
    "hard":   {"label": "困难", "depth": 6, "nodes": 4000, "cands": 12, "vcf": 12,
               "time_ms": 2000},
    # 极致档会真正搜到 7 层，代价是可能超过 3 秒；界面与文档里都标了 ⚠️。
    "ultra":  {"label": "极致", "depth": 8, "nodes": 9000, "cands": 14, "vcf": 16,
               "time_ms": 4000, "over_3s": True},
}
DEFAULT_ALGO_LEVEL = "medium"
# 墙钟安全网倍数：节点预算正常情况下先到；真遇到病态盘面也不至于卡死浏览器
ALGO_TIME_SAFETY = 4.0
# VCF 单独给一份节点预算（它分支极小，正常远用不完；给上限只为防爆炸）
ALGO_VCF_NODES = 2000

# 静态评估里进攻 / 防守的权重（>1 偏保守，<1 偏激进）
ALGO_DEFENSE_WEIGHT = 0.9
# 搜索中的「胜」分数：比任何真实盘面分都大一个量级，保证优先取胜
ALGO_WIN_SCORE = 100_000_000
# 置换表容量上限（每走一步清空重建，避免跨局脏数据）
ALGO_TT_MAX = 400_000

def _find_static_dir() -> Path:
    """
    定位前端资源目录。

    · 源码运行：main.py 同级的 static/
    · PyInstaller 打包后：__file__ 不再可信，改从 sys._MEIPASS（解包目录）
      和 exe 所在目录依次找，谁里面有 index.html 就用谁。
    """
    here = Path(__file__).resolve().parent
    candidates = [here / "static"]
    if getattr(sys, "frozen", False):          # 被打包成 exe
        exe_dir = Path(sys.executable).resolve().parent
        meipass = Path(getattr(sys, "_MEIPASS", here))
        candidates += [meipass / "static", exe_dir / "static", meipass, exe_dir]
    for cand in candidates:
        try:
            if (cand / "index.html").is_file():
                return cand
        except OSError:
            continue
    return candidates[0]


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = _find_static_dir()
INDEX_FILE = STATIC_DIR / "index.html"

# 坐标解析：匹配 "7,7" / "7，7" / "7 7" / "7 , 7"
COORD_RE = re.compile(r"(\d+)\s*[,，\s]\s*(\d+)")

# 星位表（天元 + 星），坐标基于 0 起始
STAR_POINTS: Dict[int, List[Tuple[int, int]]] = {
    9: [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)],
    13: [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)],
    15: [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)],
    19: [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)],
}

# 防止并发 /api/step 相互踩踏
STEP_LOCK = asyncio.Lock()


# --------------------------------------------------------------------------- #
# 状态构造
# --------------------------------------------------------------------------- #
def new_board(size: int) -> List[List[int]]:
    """新建 size x size 的空棋盘（0=空）。"""
    return [[0] * size for _ in range(size)]


def _norm_size(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 15
    return n if 5 <= n <= 25 else 15


def _norm_win_length(value: Any, size: int) -> int:
    try:
        k = int(value)
    except (TypeError, ValueError):
        k = 5
    if k < 3 or k > 10:
        k = 5
    return min(k, size)


def _default_players(mode: str, model: str) -> Dict[str, Dict[str, Any]]:
    """按对战模式给出默认双方配置。"""
    if mode == "human_vs_human":
        return {"1": {"type": "human", "model": None, "level": None},
                "2": {"type": "human", "model": None, "level": None}}
    if mode == "ai_vs_ai":
        return {"1": {"type": "ai", "model": model, "level": None},
                "2": {"type": "ai", "model": model, "level": None}}
    # human_vs_ai：黑=人类，白=AI
    return {"1": {"type": "human", "model": None, "level": None},
            "2": {"type": "ai", "model": model, "level": None}}


def _norm_players(raw: Any, mode: str, model: str,
                  default_level: str = DEFAULT_ALGO_LEVEL) -> Dict[str, Dict[str, Any]]:
    """把前端传入的 players 归一化为 {"1": {...}, "2": {...}}，非法时回落到模式默认值。

    **难度是每位玩家自己的**（`level` 字段，只有 algorithm 才有意义）——
    所以「简单难度的算法」可以和「困难难度的算法」同场对下，不需要任何全局设置。
    某一方没写 level 时才回落到 default_level（即 state.algo_level，纯兜底用）。
    """
    players = _default_players(mode, model)
    if not isinstance(raw, dict):
        return players

    for key in ("1", "2"):
        item = raw.get(key, raw.get(int(key)))
        if item is None:
            continue
        if isinstance(item, str):                       # 允许简写 "human" / "ai"
            item = {"type": item, "model": model if item == "ai" else None}
        if not isinstance(item, dict):
            continue
        ptype = str(item.get("type", players[key]["type"])).lower()
        ptype = PLAYER_TYPE_ALIAS.get(ptype, ptype)
        if ptype not in ALLOWED_PLAYER_TYPES:
            ptype = players[key]["type"]
        pname = item.get("model")
        if ptype == "ai":
            pname = str(pname).strip() if pname else model
        else:
            # 纯算法 / 人类都不需要模型名
            pname = None
        # 难度：只对纯算法玩家有意义。**没显式给就留 None**，读取时动态跟随
        # state["algo_level"]（默认档）—— 这样 /api/settings 改默认档仍能影响它们，
        # 不会因为开局时快照一次就变成「改了没反应」。
        plvl = None
        if ptype == "algorithm":
            raw_lv = item.get("level", item.get("algo_level"))
            if raw_lv is not None:
                plvl = _norm_algo_level(raw_lv, default_level)
        players[key] = {"type": ptype, "model": pname, "level": plvl}
    return players


def _norm_algo_level(raw: Any, base: str = DEFAULT_ALGO_LEVEL) -> str:
    """纯算法难度档位归一化。"""
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in ALGO_LEVELS:
            return key
    return base if base in ALGO_LEVELS else DEFAULT_ALGO_LEVEL


def _norm_model_params(raw: Any, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """把前端传来的采样参数归一化到合法区间；缺项沿用 base（默认 MODEL_PARAM_DEFAULTS）。"""
    out = dict(base or MODEL_PARAM_DEFAULTS)
    if not isinstance(raw, dict):
        return out
    for key, (lo, hi) in MODEL_PARAM_LIMITS.items():
        if key not in raw or raw[key] is None:
            continue
        try:
            val = float(raw[key])
        except (TypeError, ValueError):
            continue
        val = max(lo, min(hi, val))
        out[key] = int(val) if key == "top_k" else round(val, 4)
    return out


def _norm_board_format(raw: Any, base: str = DEFAULT_BOARD_FORMAT) -> str:
    return raw if raw in BOARD_FORMATS else base


def _norm_thinking_budget(raw: Any, base: Any = DEFAULT_THINKING_BUDGET) -> int:
    """接受档位名（low/medium/high/ultra）或直接给 token 数。"""
    if isinstance(raw, str) and raw in THINKING_BUDGETS:
        return THINKING_BUDGETS[raw]
    try:
        val = int(raw)
    except (TypeError, ValueError):
        val = int(base) if isinstance(base, (int, float)) else DEFAULT_THINKING_BUDGET
    return max(64, min(8192, val))


def _norm_hybrid_hints(raw: Any, base: Any = DEFAULT_HYBRID_HINTS) -> int:
    """混合决策的候选点上限，夹在 [2, 5]。"""
    try:
        val = int(raw)
    except (TypeError, ValueError):
        val = int(base) if isinstance(base, (int, float)) else DEFAULT_HYBRID_HINTS
    return max(HYBRID_HINTS_MIN, min(HYBRID_HINTS_MAX, val))


def _norm_bool(raw: Any, base: bool = False) -> bool:
    """None 视为「沿用默认值」—— 用于开关类字段，避免前端漏传时把默认值抹掉。"""
    return base if raw is None else bool(raw)


def make_state(
    size: Any = 15,
    win_length: Any = 5,
    mode: str = "human_vs_ai",
    players: Any = None,
    model_base_url: Optional[str] = None,
    thinking: Any = False,
    thinking_budget: Any = None,
    model_params: Any = None,
    board_format: Any = None,
    analysis: Any = None,
    tactics: Any = None,
    assist: Any = None,
    hybrid: Any = None,
    hybrid_hints: Any = None,
    hybrid_skip_model: Any = None,
    algo_level: Any = None,
) -> Dict[str, Any]:
    """构造一份全新的完整游戏状态。"""
    n = _norm_size(size)
    k = _norm_win_length(win_length, n)
    m = mode if mode in ALLOWED_MODES else "human_vs_ai"
    base = (model_base_url or DEFAULT_MODEL_BASE_URL).strip().rstrip("/") or DEFAULT_MODEL_BASE_URL
    lvl = _norm_algo_level(algo_level)          # 默认难度：玩家没单独指定时用它兜底
    pls = _norm_players(players, m, DEFAULT_MODEL_NAME, lvl)
    return {
        "size": n,
        "win_length": k,
        "board": new_board(n),
        "current_player": 1,
        "winner": 0,
        "game_over": False,
        "history": [],
        "last_move": None,
        "mode": m,
        "players": pls,
        "model_base_url": base,
        # ---- 高级选项（都可在线修改，改这些不需要重开对局）----
        "thinking": bool(thinking),
        "thinking_budget": _norm_thinking_budget(thinking_budget),
        "model_params": _norm_model_params(model_params),
        "board_format": _norm_board_format(board_format),
        "analysis": bool(analysis),
        "tactics": bool(tactics),
        "assist": bool(assist),
        "hybrid": bool(hybrid),
        "hybrid_hints": _norm_hybrid_hints(hybrid_hints),
        "hybrid_skip_model": _norm_bool(hybrid_skip_model, True),
        "algo_level": lvl,               # 默认算法难度（每位纯算法玩家的 level 优先）
        "last_decision": None,       # 最近一次混合决策的判定结果（供前端展示）
        "last_engine": None,         # 最近一次纯算法走子的统计（深度/节点/耗时）
        "last_error": None,
        "ai_thinking": False,
    }


# 全局唯一游戏状态（内存单例）
STATE: Dict[str, Any] = make_state()


# --------------------------------------------------------------------------- #
# 核心算法
# --------------------------------------------------------------------------- #
def check_win(board: List[List[int]], row: int, col: int, player: int, K: int) -> bool:
    """增量 O(K) 胜负判定：只检查最新落子周围的 4 个方向。"""
    N = len(board)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            for i in range(1, K):
                r, c = row + dr * i * sign, col + dc * i * sign
                if 0 <= r < N and 0 <= c < N and board[r][c] == player:
                    count += 1
                else:
                    break
        if count >= K:
            return True
    return False


def is_empty_cell(state: Dict[str, Any], row: int, col: int) -> bool:
    board = state["board"]
    return 0 <= row < state["size"] and 0 <= col < state["size"] and board[row][col] == 0


def apply_move(state: Dict[str, Any], row: int, col: int, player: int) -> None:
    """落子并做增量判胜、平局判定、切换玩家。"""
    state["board"][row][col] = player
    state["history"].append({"row": row, "col": col, "player": player})
    state["last_move"] = {"row": row, "col": col}

    if check_win(state["board"], row, col, player, state["win_length"]):
        state["winner"] = player
        state["game_over"] = True
    elif len(state["history"]) >= state["size"] * state["size"]:
        state["winner"] = 3           # 平局
        state["game_over"] = True
    else:
        state["current_player"] = 2 if player == 1 else 1


def sync_last_move(state: Dict[str, Any]) -> None:
    last = state["history"][-1] if state["history"] else None
    state["last_move"] = {"row": last["row"], "col": last["col"]} if last else None


def player_type(state: Dict[str, Any], player: int) -> str:
    cfg = state["players"].get(str(player)) or {}
    return str(cfg.get("type") or "human")


def is_algorithm(state: Dict[str, Any], player: int) -> bool:
    """该玩家是否由「纯算法 AI」驱动（不调用模型）。"""
    return player_type(state, player) == "algorithm"


def is_ai(state: Dict[str, Any], player: int) -> bool:
    """是否需要「机器落子」—— LLM 与纯算法都算，人类不算。"""
    return player_type(state, player) in ("ai", "algorithm")


def player_algo_level(state: Dict[str, Any], player: int) -> str:
    """取某位玩家自己的纯算法难度档位。

    优先级：players[key]["level"]（该玩家单独设定的）→ state["algo_level"]（默认兜底）。
    这样「简单难度算法 vs 困难难度算法」可以同场对下。
    """
    cfg = state["players"].get(str(player)) or {}
    return _norm_algo_level(cfg.get("level"), state.get("algo_level"))


def players_level_map(state: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """{"1": level|None, "2": level|None} —— 只给纯算法玩家填档位，其余为 None。"""
    out: Dict[str, Optional[str]] = {}
    for key in ("1", "2"):
        out[key] = player_algo_level(state, int(key)) if is_algorithm(state, int(key)) else None
    return out


def has_human(state: Dict[str, Any]) -> bool:
    return any((v or {}).get("type") == "human" for v in state["players"].values())


def fallback_move(state: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """回退策略：全空下中心点；否则在所有空位中取「距已有棋子曼哈顿距离最近」的随机一个。"""
    N = state["size"]
    board = state["board"]
    empties = [(r, c) for r in range(N) for c in range(N) if board[r][c] == 0]
    if not empties:
        return None
    stones = [(m["row"], m["col"]) for m in state["history"]]
    if not stones:
        mid = N // 2
        return (mid, mid)
    best_d = min(
        min(abs(r - sr) + abs(c - sc) for sr, sc in stones)
        for r, c in empties
    )
    nearest = [(r, c) for r, c in empties
               if min(abs(r - sr) + abs(c - sc) for sr, sc in stones) == best_d]
    return random.choice(nearest)


def win_now_cells(board: List[List[int]], K: int, player: int) -> List[Tuple[int, int]]:
    """找出「player 落在哪就能立刻连成 K 子」的所有空位（即必须抢占/必须堵的点）。

    做法：对每个空位假设性地落一子，复用增量 check_win 判断。
    15x15 上最多 225 个空位 × 4 方向 × K 步，开销可忽略。
    """
    N = len(board)
    wins: List[Tuple[int, int]] = []
    for r in range(N):
        for c in range(N):
            if board[r][c] != 0:
                continue
            board[r][c] = player
            hit = check_win(board, r, c, player, K)
            board[r][c] = 0
            if hit:
                wins.append((r, c))
    return wins


def build_tactics_hint(state: Dict[str, Any], player: int) -> str:
    """实验性战术提示：把「谁下一手就能赢」算出来告诉模型。

    这是最能立刻改善棋力的一条 —— 实测模型自己看不出来该堵哪里，
    但把「对手下在 (4,5) 就赢了」直接写在提示词里，它基本都会去堵。
    """
    board = state["board"]
    K = state["win_length"]
    opponent = 2 if player == 1 else 1
    me_name = "黑棋(1)" if player == 1 else "白棋(2)"
    opp_name = "白棋(2)" if player == 1 else "黑棋(1)"

    fmt = lambda cells: "、".join(f"第{r + 1}行第{c + 1}列" for r, c in cells) if cells else "无"

    my_win = win_now_cells(board, K, player)
    opp_win = win_now_cells(board, K, opponent)

    lines = ["【程序算出的战术提示（实验性，请务必参考）】"]
    lines.append(f"· 你（{me_name}）下一手能立刻连成 {K} 子获胜的位置：{fmt(my_win)}")
    lines.append(f"· 对手（{opp_name}）下一手能立刻连成 {K} 子获胜的位置：{fmt(opp_win)}")
    if opp_win:
        lines.append(f"  ⚠️ 如果你不先占住对手的这些点，对手下一手就会赢 —— 优先堵它们。")
    if my_win:
        lines.append(f"  ✅ 你自己有能直接赢的点，那就直接下那里。")
    if not my_win and not opp_win:
        lines.append("  （当前双方都没有「一子即胜」的点，按常规下：靠近已有棋子、"
                     "同时兼顾进攻与防守。）")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 传统算法辅助决策（实验性）
#
# 思路：用轻量传统算法做「侦察」，把结构化结论转成自然语言塞进提示词，
#       模型仍然做最终决策。程序只回答「有哪些威胁、建议下哪儿、为什么」，
#       不再把裸棋盘直接丢给模型。
#
# 实现要点：
#   · 每个点只看 4 个方向、长度 2K+1 的局部窗口，复杂度 O(棋子数 × 4 × K)
#   · 棋型判定统一用「补子试探」实现（补一子能连成 K 子吗？补完能形成活四吗？），
#     因此对 win_length = 3 / 4 / 5 天然成立，不需要为每个 K 手写模式表
#   · 纯字符串函数全部加了 lru_cache —— 同一盘面上窗口串大量重复，命中率很高
#
# 与既有功能完全解耦：默认关闭，关掉后拼出来的提示词与本次改动前逐字节一致。
# --------------------------------------------------------------------------- #

# 棋型分值（参考常用五子棋评分体系的相对量级，只用于排序，不代表绝对棋力）
PATTERN_SCORES: Dict[str, int] = {
    "五连": 10_000_000,
    "活四": 1_000_000,
    "冲四": 100_000,
    "活三": 60_000,
    "眠三": 1_200,
    "活二": 2_000,
    "眠二": 200,
    "无": 0,
}
PATTERN_ORDER = ("五连", "活四", "冲四", "活三", "眠三", "活二", "眠二", "无")

# 威胁紧迫性分级：S 必胜/必败 · A 绝对先手/必须封堵 · B 强攻/强防 · C 弱攻/弱防 · D 布局
URGENCY_OF: Dict[str, str] = {
    "五连": "S", "活四": "S", "冲四": "A", "活三": "B",
    "眠三": "C", "活二": "D", "眠二": "D", "无": "D",
}

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
DIR_NAMES = ("横", "竖", "主对角", "副对角")

# 候选点：只看已有棋子 CANDIDATE_DISTANCE 格内的空位；评分后只取前 CANDIDATE_LIMIT 个进报告
CANDIDATE_DISTANCE = 2
CANDIDATE_LIMIT = 15
# 综合分 = 进攻分 + 防守分 × DEFENSE_WEIGHT（防守略低于进攻，鼓励主动）
DEFENSE_WEIGHT = 0.8
# 威胁报告只报「眠三」及以上；每方最多列 THREAT_LIMIT 条，免得提示词被塞爆
THREAT_MIN_SCORE = PATTERN_SCORES["眠三"]
THREAT_LIMIT = 6


def _norm_pattern(name: str) -> str:
    return name if name in PATTERN_SCORES else "无"


def _line_string(
    board: List[List[int]], row: int, col: int, dr: int, dc: int, K: int, player: int
) -> str:
    """以 (row,col) 为中心、沿 (dr,dc) 取长度 2K+1 的窗口字符串。

    字符含义：'X'=己方（中心格无论盘面如何都记作 X，调用方负责语义）、
    '.'=空、'O'=对方、'#'=出界（把边界当成对方子对待）。
    """
    N = len(board)
    opponent = 2 if player == 1 else 1
    chars: List[str] = []
    for i in range(-K, K + 1):
        r, c = row + dr * i, col + dc * i
        if not (0 <= r < N and 0 <= c < N):
            chars.append("#")
        elif i == 0:
            chars.append("X")
        else:
            value = board[r][c]
            chars.append("X" if value == player else ("." if value == 0 else "O"))
    return "".join(chars)


@lru_cache(maxsize=8192)
def _fill_win_points(s: str, K: int) -> Tuple[int, ...]:
    """返回「把某个空点补成 X 就能出现 K 连」的所有空位下标。O(len(s))。

    做法：把 s 按对手子/出界切成若干「X 与 . 组成」的段，
    在每段上滑一个长度 K 的窗口 —— 窗口内恰好只有一个空点时，那个空点就是补子点
    （因为补上它这 K 格就全是 X 了）。
    """
    win = "X" * K
    if win in s:
        return ()
    points: List[int] = []
    n = len(s)
    start = 0
    while start < n:
        if s[start] in "O#":
            start += 1
            continue
        end = start
        while end < n and s[end] not in "O#":
            end += 1
        seg = s[start:end]
        length = len(seg)
        if length >= K:
            dots = 0
            last_dot = -1
            for k, ch in enumerate(seg):
                if ch == ".":
                    dots += 1
                    last_dot = k
                if k >= K and seg[k - K] == ".":
                    dots -= 1
                if k >= K - 1 and dots == 1:
                    points.append(start + last_dot)
        start = end
    return tuple(points)


@lru_cache(maxsize=8192)
def _classify_window(s: str, K: int) -> Tuple[str, int]:
    """判断窗口里「中心那一子参与」的棋型，返回 (棋型名, 分值)。

    判定链（全部基于补子试探，因此对任意 K 成立）：
      · 已经有 K 连                        → 五连
      · 补一子就能连成 K 子，且有两个这样的点 → 活四
      · 只有一个这样的点                     → 冲四
      · 补一子能形成活四                     → 活三
      · 补一子能形成冲四                     → 眠三
      · 剩下的按「需要补几子」粗判活二 / 眠二
    """
    if "X" * K in s:
        return "五连", PATTERN_SCORES["五连"]

    direct = _fill_win_points(s, K)
    if len(direct) >= 2:                      # 两头都能成五 → 活四
        return "活四", PATTERN_SCORES["活四"]
    if len(direct) == 1:                      # 只有一处能成五 → 冲四
        return "冲四", PATTERN_SCORES["冲四"]

    open_three = closed_three = False
    for i, ch in enumerate(s):
        if ch != ".":
            continue
        after = s[:i] + "X" + s[i + 1:]
        n = len(_fill_win_points(after, K))
        if n >= 2:
            open_three = True
            break
        if n == 1:
            closed_three = True
    if open_three:
        return "活三", PATTERN_SCORES["活三"]
    if closed_three:
        return "眠三", PATTERN_SCORES["眠三"]

    run = max(1, K - 3)
    if ("." + "X" * run + ".") in s:
        return "活二", PATTERN_SCORES["活二"]
    if s.count("X") >= run:
        return "眠二", PATTERN_SCORES["眠二"]
    return "无", 0


@lru_cache(maxsize=8192)
def _upgrade_idx(s: str, K: int) -> Tuple[int, ...]:
    """返回「补上这一子能获得最大幅度升级」的空位下标（棋型的延伸点 / 封堵关键点）。

    只保留能达到的**最高档位**：活三的延伸点应该是「补一子成活四」的那两个点，
    而不是「补一子成冲四」的那些点 —— 后者分数也涨了，但并不是该去堵的地方。
    """
    current = PATTERN_SCORES[_classify_window(s, K)[0]]
    best = current
    hits: List[int] = []
    for i, ch in enumerate(s):
        if ch != ".":
            continue
        after = s[:i] + "X" + s[i + 1:]
        value = PATTERN_SCORES[_classify_window(after, K)[0]]
        if value > best:
            best = value
            hits = [i]
        elif value == best and value > current:
            hits.append(i)
    return tuple(sorted(set(hits)))


def _window_cells(
    row: int, col: int, dr: int, dc: int, K: int, N: int
) -> List[Optional[Tuple[int, int]]]:
    """把窗口下标 0..2K 映射回盘面坐标；出界的位置给 None。"""
    cells: List[Optional[Tuple[int, int]]] = []
    for i in range(-K, K + 1):
        r, c = row + dr * i, col + dc * i
        cells.append((r, c) if (0 <= r < N and 0 <= c < N) else None)
    return cells


def detect_patterns(
    board: List[List[int]],
    row: int,
    col: int,
    player: int,
    K: int,
    with_key_points: bool = False,
) -> List[Dict[str, Any]]:
    """返回 (row,col) 在 4 个方向上形成的棋型。

    调用前把 (row,col) 视为已经是 player 的子 —— 对空位来说就是「假设落子」，
    对已有棋子来说就是「它现在参与的形状」。
    """
    N = len(board)
    out: List[Dict[str, Any]] = []
    for idx, (dr, dc) in enumerate(DIRECTIONS):
        s = _line_string(board, row, col, dr, dc, K, player)
        name, score = _classify_window(s, K)
        item: Dict[str, Any] = {
            "dir": DIR_NAMES[idx],
            "pattern": name,
            "score": score,
            "urgency": URGENCY_OF[name],
        }
        if with_key_points:
            # _window_cells 与窗口字符串同序：cells[i] 就是 s[i] 对应的盘面坐标
            cells = _window_cells(row, col, dr, dc, K, N)
            # 中心格恒为 X，但只有它本来就是 player 的子时才计入 coords
            center_is_stone = board[row][col] == player
            item["coords"] = [
                cells[i] for i in range(len(s))
                if s[i] == "X" and cells[i] and (i != K or center_is_stone)
            ]
            item["key_points"] = [cells[i] for i in _upgrade_idx(s, K) if cells[i]]
        out.append(item)
    return out


def generate_candidates(
    board: List[List[int]], distance: int = CANDIDATE_DISTANCE
) -> List[Tuple[int, int]]:
    """只把已有棋子周围 distance 格内的空位当候选 —— 不遍历全盘空位。"""
    N = len(board)
    stones = [(r, c) for r in range(N) for c in range(N) if board[r][c]]
    if not stones:
        mid = N // 2
        return [(mid, mid)]
    found: set = set()
    for r, c in stones:
        for dr in range(-distance, distance + 1):
            for dc in range(-distance, distance + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N and board[rr][cc] == 0:
                    found.add((rr, cc))
    return sorted(found)


def combo_label(names: List[str], K: int) -> str:
    """阵型识别：一子落下在多个方向同时成型。

    K=5 时用习惯叫法（双四 / 四三 / 双三）；连珠数不是 5 时改用实际棋型名拼起来
    （如三子棋的「冲二＋活一」），否则换算出来的「二一」根本看不懂。
    """
    fours = [n for n in names if n in ("活四", "冲四")]
    threes = [n for n in names if n == "活三"]

    if K != 5:
        if len(fours) >= 2:
            return "＋".join(_pattern_label(n, K) for n in fours[:2])
        if fours and threes:
            return f"{_pattern_label(fours[0], K)}＋{_pattern_label(threes[0], K)}"
        if len(threes) >= 2:
            return "＋".join(_pattern_label(n, K) for n in threes[:2])
        return ""

    if len(fours) >= 2:
        return "双四"
    if fours and threes:
        return "四三"
    if len(threes) >= 2:
        return "双三"
    return ""


_CN_DIGITS = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}
# 各棋型「离成 K 连还差几子」的档位（按 K=5 命名时的偏移量）
_PATTERN_OFFSET = {"五连": 0, "活四": 1, "冲四": 1, "活三": 2, "眠三": 2, "活二": 3, "眠二": 3}
_PATTERN_PREFIX = {"活四": "活", "冲四": "冲", "活三": "活", "眠三": "眠", "活二": "活", "眠二": "眠"}


def _cn(n: int) -> str:
    return _CN_DIGITS.get(n, str(n))


def _pattern_label(name: str, K: int) -> str:
    """把内部按 5 子棋命名的棋型翻译成当前 K 下的叫法。

    内部一律用 K=5 的名字当键（五连/活四/冲四/活三/眠三/活二/眠二），
    展示时按 K 换算：五连 → {K}连、活四 → 活{K-1}、冲四 → 冲{K-1}、
    活三 → 活{K-2}、眠三 → 眠{K-2}、活二 → 活{K-3}。

    不这么做的话，9×9 三子棋下会对着一颗孤子说「形成活四」，纯属误导。
    """
    if K == 5 or name not in _PATTERN_OFFSET:
        return name
    if name == "五连":
        return f"{K}连"
    n = K - _PATTERN_OFFSET[name]
    if n < 1:
        return "无"
    return f"{_PATTERN_PREFIX[name]}{_cn(n)}"


def _best_pattern(dirs: List[Dict[str, Any]]) -> str:
    best = "无"
    for d in dirs:
        if PATTERN_ORDER.index(_norm_pattern(d["pattern"])) < PATTERN_ORDER.index(best):
            best = _norm_pattern(d["pattern"])
    return best


def score_candidate(
    board: List[List[int]], row: int, col: int, player: int, opponent: int, K: int
) -> Dict[str, Any]:
    """给一个空位打分：进攻分（我方下这儿）+ 防守分（对手下这儿）× 0.8。"""
    atk_dirs = detect_patterns(board, row, col, player, K)
    dfd_dirs = detect_patterns(board, row, col, opponent, K)
    atk = sum(d["score"] for d in atk_dirs)
    dfd = sum(d["score"] for d in dfd_dirs)
    atk_best = _best_pattern(atk_dirs)
    dfd_best = _best_pattern(dfd_dirs)
    combo = combo_label([d["pattern"] for d in atk_dirs], K)
    atk_label = _pattern_label(atk_best, K)
    dfd_label = _pattern_label(dfd_best, K)

    if combo:
        label = "必抢"
    elif PATTERN_SCORES[atk_best] >= PATTERN_SCORES["冲四"]:
        label = "强攻"
    elif PATTERN_SCORES[dfd_best] >= PATTERN_SCORES["活三"]:
        label = "强防"
    else:
        label = "布局"

    bits: List[str] = []
    if combo:
        bits.append(f"一子形成{combo}")
    if PATTERN_SCORES[atk_best] >= PATTERN_SCORES["眠三"]:
        bits.append(f"己方在此形成{atk_label}")
    # 防守分的含义是「对手落在这里会形成什么」——按这个语义说清楚，
    # 否则「封堵对手五连」会被误读成对手已经连成五子了。
    if PATTERN_SCORES[dfd_best] >= PATTERN_SCORES["冲四"]:
        bits.append(f"抢占：对手下这里会形成{dfd_label}")
    elif PATTERN_SCORES[dfd_best] >= PATTERN_SCORES["眠三"]:
        bits.append(f"压制对手的{dfd_label}")
    if not bits:
        bits.append("靠近已有棋子的常规落点")

    return {
        "coord": [row, col],
        "score": int(atk + dfd * DEFENSE_WEIGHT),
        "attack": int(atk),
        "defense": int(dfd),
        "label": label,
        "reason": "；".join(bits),
        "my_pattern": atk_label,
        "opponent_pattern": dfd_label,
        "combo": combo,
        "urgency": URGENCY_OF[atk_best if PATTERN_SCORES[atk_best] >= PATTERN_SCORES[dfd_best]
                              else dfd_best],
    }


def _threat_candidates(
    state: Dict[str, Any], player: int
) -> List[Dict[str, Any]]:
    """扫描 player 已有棋子形成的「眠三及以上」棋型，去重后按分值降序返回。

    只评估「邻域内至少有 2 颗己方棋子」的点 —— 彻底孤立的单子不可能构成任何威胁，
    先剪掉能省掉早期盘面上的大量无谓计算（K=3 时 `X.X` 这类两子棋型仍会被保留）。
    """
    board, K, N = state["board"], state["win_length"], state["size"]
    found: Dict[Tuple[str, Tuple[Tuple[int, int], ...]], Dict[str, Any]] = {}

    for r in range(N):
        for c in range(N):
            if board[r][c] != player:
                continue
            # 便宜的密度预筛：5×5 邻域内己方棋子不足 2 颗就跳过
            near = 0
            for rr in range(max(0, r - 2), min(N, r + 3)):
                for cc in range(max(0, c - 2), min(N, c + 3)):
                    if board[rr][cc] == player:
                        near += 1
            if near < 2:
                continue

            for d in detect_patterns(board, r, c, player, K, with_key_points=True):
                if d["score"] < THREAT_MIN_SCORE:
                    continue
                points = tuple(sorted(d.get("key_points") or []))
                # 同一个棋型会被它自己的每一颗子各检出一次 —— 关键点相同即视为同一条，
                # 这样 15×15 上一条活三只会出现一次，而不是三到四次
                key = points if points else (d["pattern"], tuple(sorted(d.get("coords") or [])))
                if key in found:
                    continue
                found[key] = {
                    "type": _pattern_label(d["pattern"], K),
                    "pattern": d["pattern"],
                    "dir": d["dir"],
                    "coords": [list(x) for x in (d.get("coords") or [])],
                    "key_points": [list(x) for x in (d.get("key_points") or [])],
                    "urgency": d["urgency"],
                    "score": d["score"],
                }

    items = sorted(found.values(), key=lambda x: -x["score"])
    return items[:THREAT_LIMIT]


def analyze_board(state: Dict[str, Any], player: int) -> Dict[str, Any]:
    """产出完整局势分析报告（结构化，可直接 JSON 返回）。"""
    board, K, N = state["board"], state["win_length"], state["size"]
    opponent = 2 if player == 1 else 1

    candidates: List[Dict[str, Any]] = []
    for r, c in generate_candidates(board):
        candidates.append(score_candidate(board, r, c, player, opponent, K))
    candidates.sort(key=lambda x: -x["score"])
    top = candidates[:CANDIDATE_LIMIT]

    my_win = win_now_cells(board, K, player)
    opp_win = win_now_cells(board, K, opponent)

    return {
        "game_info": {
            "board_size": N,
            "win_length": K,
            "current_player": player,
            "total_moves": len(state["history"]),
            "candidate_count": len(candidates),
        },
        "my_threats": _threat_candidates(state, player),
        "opponent_threats": _threat_candidates(state, opponent),
        "critical_points": {
            "winning_move": [list(x) for x in my_win] or None,
            "must_block": [list(x) for x in opp_win] or None,
            "best_attack": list(top[0]["coord"]) if top else None,
        },
        "candidates": top,
    }


# --------------------------------------------------------------------------- #
# 混合决策（实验性）：程序「侦察」完之后，还要不要「动手」
#
# 上一版里程序只做侦察 —— 把所有结论摊给模型，由模型拍板。实测棋力提升明显，
# 但有两类局面仍然会翻车：
#   · 只有一个正解的局面（对手冲四、自己四连）—— 模型偶尔会在正解旁边落子，
#     一步走错直接输；这种局面其实不需要模型判断，程序算得比它准。
#   · 高警惕局面（对手活三）—— 候选点其实只有 2 个，把 15 个候选点摊给模型
#     反而是在稀释信息，它容易跑去选一个「看起来分数高」的无关点。
#
# 所以这一层做的是「分层放权」：越确定的局面程序越强势。
#   R1 自己下一手就能赢        → forced      程序直接落子取胜
#   R2 对手只有一个必胜点      → forced      程序直接占住
#   R3 对手有多个必胜点        → restricted  挡不住全部，只把这几个点交给模型
#   R4 对手活三一类强威胁      → restricted  封堵点通常 2 个，只给这几个
#   其余                       → normal      不干预，回到「报告 + 模型自由决策」
#
# 一个刻意的取舍：restricted 只由**对手的威胁**触发，己方有活三时不收窄范围。
# 因为防守点是唯一的（堵 A 或堵 B），而进攻点常常不是 —— 自己活三时程序建议的
# 两个延伸点未必比模型看到的「双三」更好，此时限制它反而会丢掉棋力。
# --------------------------------------------------------------------------- #

def _pick_center_first(cells: List[Tuple[int, int]], N: int) -> List[Tuple[int, int]]:
    """同为「必下点」时优先靠棋盘中心的那个 —— 后续变化更多，观感也更自然。"""
    mid = (N - 1) / 2.0
    return sorted(cells, key=lambda rc: (abs(rc[0] - mid) + abs(rc[1] - mid), rc[0], rc[1]))


def _xy_text(cell: Any) -> str:
    """给人看的中文坐标写法。**只用于日志和接口返回，绝不进提示词** ——
    提示词里一旦出现中文坐标，模型就会照抄，而现有解析正则认不出这种写法。"""
    r, c = cell
    return f"第{r + 1}行第{c + 1}列"


def decide_hybrid(state: Dict[str, Any], player: int, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """混合决策：返回本步该怎么放权。

    返回字段：
      mode        off / normal / restricted / forced
      rule        命中的规则代号（win_now / block_unique / block_multi / block_*）
      reason      给日志看的人话解释
      points      受限候选点（0 起算），restricted 时为「只允许这几个」
      move        forced 时的最终落子（0 起算）
      threat_level 触发时的紧迫度 S/A/B
      skip_model  **本步**是否跳过模型调用 —— 只有 forced 才可能为 True

    注意 skip_model 返回的是「本步的实际行为」而不是用户配置：非 forced 时一律 False，
    调用方只需要 `if decision["skip_model"]` 就能判断，不必再去看 mode。
    """
    blank: Dict[str, Any] = {
        "mode": "off", "rule": "", "reason": "", "points": [],
        "move": None, "threat_level": "-", "skip_model": False,
    }
    if not state.get("hybrid"):
        return blank

    N, K = state["size"], state["win_length"]
    max_hints = _norm_hybrid_hints(state.get("hybrid_hints"))
    skip = _norm_bool(state.get("hybrid_skip_model"), True)

    # 兜底：决策模块是实验性的，绝不能因为它抛异常就把整局弄挂
    try:
        rep = report if report is not None else analyze_board(state, player)
    except Exception as exc:                        # noqa: BLE001
        return {**blank, "mode": "normal", "rule": "error",
                "reason": f"局势分析失败，本步不干预（{type(exc).__name__}: {exc}）"}

    my_win = [tuple(x) for x in (rep["critical_points"].get("winning_move") or [])]
    opp_win = [tuple(x) for x in (rep["critical_points"].get("must_block") or [])]

    # ---- R1：自己下一手就能连成 K 子 ----
    if my_win:
        ordered = _pick_center_first(my_win, N)
        extra = f"（另有 {len(ordered) - 1} 个同样能赢的点）" if len(ordered) > 1 else ""
        return {
            "mode": "forced", "rule": "win_now", "move": ordered[0],
            "points": ordered, "threat_level": "S", "skip_model": skip,
            "reason": f"你下在 {_xy_text(ordered[0])} 就能连成 {K} 子直接获胜{extra}，"
                      f"本步由程序直接执行",
        }

    # ---- R2 / R3：对手下一手就能连成 K 子 ----
    if opp_win:
        ordered = _pick_center_first(opp_win, N)
        if len(ordered) == 1:
            only = ordered[0]
            return {
                "mode": "forced", "rule": "block_unique", "move": only,
                "points": ordered, "threat_level": "S", "skip_model": skip,
                "reason": f"对手只有 {_xy_text(only)} 这一个「一子即胜」点，"
                          f"不占住就直接输，本步由程序直接执行",
            }
        return {
            "mode": "restricted", "rule": "block_multi", "move": None,
            "points": ordered[:max_hints], "threat_level": "S", "skip_model": False,
            "reason": f"对手有 {len(ordered)} 个「一子即胜」点，已经挡不住全部了；"
                      f"只能占住其中一个拖延，把其中 {min(len(ordered), max_hints)} 个点交给模型判断",
        }

    # ---- R4：对手活三一类的强威胁 —— 封堵点通常只有 1~3 个 ----
    urgent = [t for t in (rep.get("opponent_threats") or [])
              if t.get("urgency") in HYBRID_URGENT_LEVELS]
    keys: List[Tuple[int, int]] = []
    for t in urgent:
        for x in (t.get("key_points") or []):
            xy = tuple(x)
            if xy not in keys:
                keys.append(xy)

    if not keys:
        return {**blank, "mode": "normal",
                "reason": "双方都没有必须抢占的点，交给模型自由决策"}

    if len(keys) == 1:
        only = keys[0]
        return {
            "mode": "forced", "rule": "block_single_key", "move": only,
            "points": keys, "threat_level": urgent[0].get("urgency", "A"), "skip_model": skip,
            "reason": f"对手的威胁只有一个封堵点 {_xy_text(only)}，不占住就陷入劣势，"
                      f"本步由程序直接执行",
        }

    if len(keys) <= max_hints:
        types = "、".join(dict.fromkeys(t["type"] for t in urgent))
        return {
            "mode": "restricted", "rule": "block_open_three", "move": None,
            "points": keys, "threat_level": urgent[0].get("urgency", "B"), "skip_model": False,
            "reason": f"对手有{types}一类的强威胁，封堵点只有 {len(keys)} 个，"
                      f"把这 {len(keys)} 个点交给模型判断",
        }

    return {**blank, "mode": "normal",
            "reason": f"对手的封堵点有 {len(keys)} 个（超过上限 {max_hints}），候选收窄意义不大，"
                      f"不限制模型"}


def _fmt_cell(cell: Any) -> str:
    """0 起算 → 提示词里统一的 1 起算「(行,列)」写法。"""
    r, c = cell
    return f"({r + 1},{c + 1})"


def _fmt_cells(cells: Any) -> str:
    if not cells:
        return "无"
    return "、".join(_fmt_cell(x) for x in cells)


def _point_reason(rep: Dict[str, Any], xy: Tuple[int, int]) -> str:
    """某个受限点「凭什么」被程序挑出来 —— 不解释清楚，模型会当成普通建议随手忽略。"""
    K = rep["game_info"]["win_length"]
    crit = rep["critical_points"]
    if xy in {tuple(x) for x in (crit.get("winning_move") or [])}:
        return f"你下这里就能直接连成 {K} 子获胜"
    if xy in {tuple(x) for x in (crit.get("must_block") or [])}:
        return "对手下这里就直接获胜，必须占住"
    for t in (rep.get("opponent_threats") or []):
        if xy in {tuple(x) for x in (t.get("key_points") or [])}:
            return f"封堵对手的{t['type']}（{t['dir']}方向）"
    for t in (rep.get("my_threats") or []):
        if xy in {tuple(x) for x in (t.get("key_points") or [])}:
            return f"你自己的{t['type']}可以下在这里升级"
    return "程序评分最高的落点之一"


def render_restrict_hint(rep: Dict[str, Any], decision: Dict[str, Any]) -> str:
    """只输出「本轮硬性约束」那一段 —— 让它在没有完整报告的场合也能单独使用。"""
    if decision.get("mode") != "restricted":
        return ""
    allowed = [tuple(x) for x in (decision.get("points") or [])]
    if not allowed:
        return ""
    lines = ["【本轮硬性约束（程序判定，必须遵守）】"]
    lines.append(f"程序用传统算法判定：本局面下只有以下 {len(allowed)} 个落点是合理的，"
                 "落在其他任何位置都会立刻陷入劣势：")
    for i, xy in enumerate(allowed, 1):
        lines.append(f"{i}. {_fmt_cell(xy)} —— {_point_reason(rep, xy)}")
    lines.append(f"你必须从这 {len(allowed)} 个点里选一个，不得选择其他位置。")
    return "\n".join(lines)


def render_assist(
    state: Dict[str, Any],
    player: int,
    report: Optional[Dict[str, Any]] = None,
    decision: Optional[Dict[str, Any]] = None,
) -> str:
    """把分析报告转成自然语言，嵌入提示词。

    坐标刻意全部用「(行,列)」的纯数字写法（1 起算），与提示词末尾要求的输出格式一致，
    避免模型照抄中文写法后解析不出来。

    decision 的 mode 为 restricted 时，候选点列表会被**裁剪到那几个点**并追加一段硬性
    约束 —— 否则报告里列着 15 个候选、约束里说只能选 2 个，模型会无所适从。
    """
    rep = report if report is not None else analyze_board(state, player)
    dec = decision or {"mode": "off", "points": [], "reason": ""}
    allowed = [tuple(x) for x in (dec.get("points") or [])] if dec.get("mode") == "restricted" else None
    info = rep["game_info"]
    K = info["win_length"]
    threshold = _pattern_label("眠三", K)
    me_name = "黑棋(1)" if player == 1 else "白棋(2)"
    opp_name = "白棋(2)" if player == 1 else "黑棋(1)"

    lines: List[str] = [
        "【程序用传统算法算出的局势分析报告（供参考，最终由你决定）】",
        f"棋盘 {info['board_size']}×{info['board_size']}，连成 {info['win_length']} 子获胜；"
        f"已下 {info['total_moves']} 手，你执{me_name}。",
        "以下坐标一律写成「行,列」且行列都从 1 开始（与棋盘图一致）。",
        "",
        f"■ 你（{me_name}）已有的棋型：",
    ]
    if rep["my_threats"]:
        for t in rep["my_threats"]:
            lines.append(f"· {t['type']}（{t['dir']}）：{_fmt_cells(t['coords'])}，"
                         f"延伸点 {_fmt_cells(t['key_points'])}"
                         f"（下在这些点可以升级棋型）【紧迫度 {t['urgency']}】")
    else:
        lines.append(f"· 暂无明显棋型（{threshold}及以上）。")

    lines += ["", f"■ 对手（{opp_name}）已有的棋型："]
    if rep["opponent_threats"]:
        for t in rep["opponent_threats"]:
            lines.append(f"· {t['type']}（{t['dir']}）：{_fmt_cells(t['coords'])}，"
                         f"你必须堵 {_fmt_cells(t['key_points'])}"
                         f"【紧迫度 {t['urgency']}】")
    else:
        lines.append(f"· 暂无明显棋型（{threshold}及以上）。")

    crit = rep["critical_points"]
    lines += ["", "■ 关键点："]
    lines.append(f"· 你下一手能立刻获胜的点：{_fmt_cells(crit['winning_move'])}")
    if crit["must_block"]:
        lines.append(f"· ⚠️ 对手下一手就能获胜的点：{_fmt_cells(crit['must_block'])}"
                     f" —— 不先占住，你下一手就输了")
    else:
        lines.append("· 对手没有「一子即胜」的点。")

    if allowed:
        lines += ["", f"■ 候选落子点（程序判定：本局面只有以下 {len(allowed)} 个点合理）："]
        for i, xy in enumerate(allowed, 1):
            hit = next((c for c in rep["candidates"] if tuple(c["coord"]) == xy), None)
            score = f"评分 {hit['score']}" if hit else "程序判定必抢"
            lines.append(f"{i}. {_fmt_cell(xy)} {score} —— {_point_reason(rep, xy)}")
    else:
        lines += ["", f"■ 候选落子点（程序评分降序，共 {len(rep['candidates'])} 个）："]
        for i, cand in enumerate(rep["candidates"], 1):
            lines.append(f"{i}. {_fmt_cell(cand['coord'])} 评分 {cand['score']}"
                         f"（进攻 {cand['attack']} / 防守 {cand['defense']}）"
                         f" · {cand['label']} —— {cand['reason']}")

    if allowed:
        lines += ["", render_restrict_hint(rep, dec)]
    else:
        lines += [
            "",
            "■ 提示",
            "· 优先级：先封堵对手的必胜点 > 执行自己的必胜点 > "
            "抢占组合威胁（上面标为「必抢」的点）> 常规布局。",
            "· 上面的候选点是程序算出来的建议，不是命令；如果你有更好的判断，可以另选一个空位。",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 纯算法 AI（不调用模型）
#
# 完全自包含的博弈树搜索引擎，作为「玩家类型」之一参与对局：
#   · 难度档位选出「目标深度 + 时间预算 + 候选点上限 + VCF 层数」四元组
#   · 迭代加深 + NegaScout(PVS) + Zobrist 置换表 + 杀手走法
#   · 静态评估用「点分」：每颗子只看自己 4 个方向上的棋型，落子/悔棋时
#     只重算受影响的那些子（增量维护），避免每个叶子节点全盘重扫
#   · VCF（连续冲四）独立搜索，分支因子极小，是「会赢」的关键
#
# 与 LLM 完全无关：不发任何 HTTP 请求，不读 state 里的模型配置。
# --------------------------------------------------------------------------- #

ALGO_INF = float("inf")

# Zobrist 随机数表：按棋盘边长缓存，用固定种子保证同一局面哈希稳定
_ZOBRIST_CACHE: Dict[int, List[List[List[int]]]] = {}


def _zobrist_table(N: int) -> List[List[List[int]]]:
    table = _ZOBRIST_CACHE.get(N)
    if table is None:
        rnd = random.Random(0x5A17 + N)      # 固定种子：不同进程/不同次调用结果一致
        table = [[[rnd.getrandbits(64) for _ in range(3)] for _ in range(N)]
                 for _ in range(N)]
        _ZOBRIST_CACHE[N] = table
    return table


def _point_score(board: List[List[int]], row: int, col: int, player: int, K: int) -> int:
    """一颗子（把 (row,col) 视为 player 的子）在 4 个方向上形成的棋型分之和。"""
    total = 0
    for dr, dc in DIRECTIONS:
        s = _line_string(board, row, col, dr, dc, K, player)
        total += PATTERN_SCORES[_classify_window(s, K)[0]]
    return total


class AlgoEngine:
    """一次走子用的搜索上下文。棋盘是**副本**，全程只在副本上落子/悔棋。"""

    def __init__(self, board: List[List[int]], K: int, me: int, cfg: Dict[str, Any]):
        self.board = board
        self.K = int(K)
        self.N = len(board)
        self.me = me
        self.opp = 3 - me
        self.cfg = cfg
        self.deadline = 0.0                 # 墙钟安全网（不是主约束）
        self.stop = False
        self.nodes = 0                      # 常规搜索节点数
        self.vcf_nodes = 0                  # VCF 节点数（单独计）
        self.node_budget = int(cfg.get("nodes", 1200))
        self.vcf_node_budget = ALGO_VCF_NODES
        self.vcf_stop = False
        self.tt: Dict[int, Tuple[int, int, int, Optional[Tuple[int, int]]]] = {}
        self.killers: List[List[Optional[Tuple[int, int]]]] = [[None, None] for _ in range(64)]
        self.best_move: Optional[Tuple[int, int]] = None

        # Zobrist
        self.zob = _zobrist_table(self.N)
        self.hash = 0

        # 增量静态评估：side[c] = 颜色 c 所有子的点分之和
        self.side: Dict[int, int] = {1: 0, 2: 0}
        self.pts: Dict[Tuple[int, int], int] = {}
        # near[cell] = 该空位 2 格内的棋子数（用于快速列出候选点）
        self.near: Dict[Tuple[int, int], int] = {}
        self.stack: List[Dict[str, Any]] = []

        for r in range(self.N):
            for c in range(self.N):
                v = board[r][c]
                if v:
                    self.hash ^= self.zob[r][c][v]
                    self.pts[(r, c)] = _point_score(board, r, c, v, self.K)
                    self.side[v] += self.pts[(r, c)]
                    self._near_shift(r, c, +1)

    # ----------------------------- 候选点维护 ----------------------------- #
    def _near_shift(self, r: int, c: int, delta: int) -> List[Tuple[int, int]]:
        """把 (r,c) 这颗子对周围 2 格内空位的计数 +delta，返回被动过的格子列表。"""
        touched: List[Tuple[int, int]] = []
        for dr in range(-2, 3):
            rr = r + dr
            if not (0 <= rr < self.N):
                continue
            for dc in range(-2, 3):
                cc = c + dc
                if 0 <= cc < self.N and self.board[rr][cc] == 0:
                    self.near[(rr, cc)] = self.near.get((rr, cc), 0) + delta
                    touched.append((rr, cc))
        return touched

    # ------------------------------- 落子 --------------------------------- #
    def make(self, r: int, c: int, color: int) -> None:
        board = self.board
        board[r][c] = color
        self.hash ^= self.zob[r][c][color]

        # 自己这个格子不再是候选点，记下旧值待恢复
        near_self = self.near.pop((r, c), 0)
        touched = self._near_shift(r, c, +1)

        # 受影响的所有棋子：自身 + 4 个方向上 K 格内的已有棋子
        cells: List[Tuple[int, int]] = [(r, c)]
        for dr, dc in DIRECTIONS:
            for i in range(-self.K, self.K + 1):
                if i == 0:
                    continue
                rr, cc = r + dr * i, c + dc * i
                if 0 <= rr < self.N and 0 <= cc < self.N and board[rr][cc]:
                    cells.append((rr, cc))

        for rr, cc in cells:
            v = board[rr][cc]
            old = self.pts.get((rr, cc), 0)
            new = _point_score(board, rr, cc, v, self.K)
            if new != old:
                self.pts[(rr, cc)] = new
                self.side[v] += new - old

        self.stack.append({"move": (r, c, color), "near_self": near_self, "touched": touched})

    def undo(self) -> None:
        rec = self.stack.pop()
        r, c, color = rec["move"]
        board = self.board

        board[r][c] = 0
        self.hash ^= self.zob[r][c][color]
        self.side[color] -= self.pts.pop((r, c), 0)

        for rr, cc in rec["touched"]:
            self.near[(rr, cc)] = self.near.get((rr, cc), 0) - 1
        if rec["near_self"]:
            self.near[(r, c)] = rec["near_self"]

        # 重算受影响棋子的点分（被清空那颗的周围所有子）
        for rr, cc in self._affected(r, c):
            v = board[rr][cc]
            if not v:
                continue
            old = self.pts.get((rr, cc), 0)
            new = _point_score(board, rr, cc, v, self.K)
            if new != old:
                self.pts[(rr, cc)] = new
                self.side[v] += new - old

    def _affected(self, r: int, c: int) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for dr, dc in DIRECTIONS:
            for i in range(-self.K, self.K + 1):
                if i == 0:
                    continue
                rr, cc = r + dr * i, c + dc * i
                if 0 <= rr < self.N and 0 <= cc < self.N and self.board[rr][cc]:
                    out.append((rr, cc))
        return out

    # ------------------------------- 评估 --------------------------------- #
    def eval_for(self, color: int) -> int:
        other = 3 - color
        return int(self.side[color] - self.side[other] * ALGO_DEFENSE_WEIGHT)

    # --------------------- 候选点扫描（含必胜/必堵判定）--------------------- #
    def _scan(self, r: int, c: int, color: int) -> Tuple[str, int, str, int]:
        """一个空位对双方的「价值」：(己方最佳棋型, 己方点分, 对方最佳棋型, 对方点分)。

        点分语义与全局评估一致：把该点假设成某一方的子，它自己在 4 个方向上
        形成的棋型分之和。「对方点分」就是防守价值 —— 对手下这儿会有多强。
        """
        board, K = self.board, self.K
        other = 3 - color
        best_i = PATTERN_ORDER.index("无")
        atk_best, dfd_best = "无", "无"
        atk_sum = dfd_sum = 0
        for dr, dc in DIRECTIONS:
            name = _classify_window(_line_string(board, r, c, dr, dc, K, color), K)[0]
            atk_sum += PATTERN_SCORES[name]
            i = PATTERN_ORDER.index(name)
            if i < best_i:
                best_i, atk_best = i, name
            name = _classify_window(_line_string(board, r, c, dr, dc, K, other), K)[0]
            dfd_sum += PATTERN_SCORES[name]
            i = PATTERN_ORDER.index(name)
            if i < PATTERN_ORDER.index(dfd_best):
                dfd_best = name
        return atk_best, atk_sum, dfd_best, dfd_sum

    def _limit(self, ply: int) -> int:
        """越深留的候选越少 —— 这是控制节点数最有效的旋钮。"""
        base = int(self.cfg.get("cands", 10))
        return max(4, base - 2 * ply)

    def _move_list(self, color: int, ply: int) -> List[Tuple[int, int]]:
        """生成并排序走法。

        两处刻意的「降分支」处理（既提速又更准）：
          · 自己有一子即胜的点 → 只返回它（立刻赢，不用再看别的）
          · 对手有一子即胜的点 → 只返回这些点（不堵就直接输，其余走法没意义）
        """
        board = self.board
        scored: List[Tuple[float, int, int, str, str]] = []
        my_win: List[Tuple[int, int]] = []
        opp_win: List[Tuple[int, int]] = []
        for (r, c), cnt in self.near.items():
            if cnt <= 0 or board[r][c] != 0:
                continue
            ab, asum, ob, osum = self._scan(r, c, color)
            if ab == "五连":
                my_win.append((r, c))
                continue
            if ob == "五连":
                opp_win.append((r, c))
            scored.append((asum + osum * ALGO_DEFENSE_WEIGHT, r, c, ab, ob))

        if my_win:
            return my_win[:1]
        if opp_win:
            return opp_win[:self._limit(ply)]
        scored.sort(key=lambda x: -x[0])
        return [(r, c) for _s, r, c, _a, _o in scored[:self._limit(ply)]]

    # ------------------------------- 搜索 --------------------------------- #
    def _negascout(self, depth: int, alpha: float, beta: float,
                   color: int, ply: int) -> float:
        self.nodes += 1
        # 主约束是节点预算（可复现），墙钟只是安全网
        if self.stop or self.nodes > self.node_budget or time.monotonic() > self.deadline:
            self.stop = True
            return 0.0
        if depth <= 0:
            return float(self.eval_for(color))

        key = (self.hash << 1) | (1 if color == self.me else 0)
        tte = self.tt.get(key)
        tt_move: Optional[Tuple[int, int]] = None
        if tte is not None:
            t_depth, t_score, t_flag, tt_move = tte
            if t_depth >= depth:
                if t_flag == 0:
                    return float(t_score)
                if t_flag == 1 and t_score > alpha:
                    alpha = float(t_score)
                elif t_flag == 2 and t_score < beta:
                    beta = float(t_score)
                if alpha >= beta:
                    return float(t_score)

        moves = self._move_list(color, ply)
        if not moves:
            return float(self.eval_for(color))
        # 置换表 / 杀手走法优先 —— 走法排序质量直接决定 PVS 的剪枝效率
        killers = self.killers[ply] if ply < len(self.killers) else [None, None]
        prio = [tt_move] + [k for k in killers if k]
        if prio:
            head = [m for m in moves if m in prio]
            if head:
                head.sort(key=lambda m: 0 if m == tt_move else 1)
                moves = head + [m for m in moves if m not in head]

        orig_alpha = alpha
        best = -ALGO_INF
        best_move = moves[0]
        for idx, (r, c) in enumerate(moves):
            self.make(r, c, color)
            if check_win(self.board, r, c, color, self.K):
                score = float(ALGO_WIN_SCORE - ply)          # 越早赢越好
            elif idx == 0:
                score = -self._negascout(depth - 1, -beta, -alpha, 3 - color, ply + 1)
            else:
                # 零窗口快速验证，失败再全窗口重搜（NegaScout / PVS 的核心）
                score = -self._negascout(depth - 1, -alpha - 1, -alpha, 3 - color, ply + 1)
                if alpha < score < beta:
                    score = -self._negascout(depth - 1, -beta, -alpha, 3 - color, ply + 1)
            self.undo()

            if self.stop:
                return 0.0
            if score > best:
                best, best_move = score, (r, c)
            if best > alpha:
                alpha = best
            if alpha >= beta:
                if ply < len(self.killers) and self.killers[ply][0] != (r, c):
                    self.killers[ply][1] = self.killers[ply][0]
                    self.killers[ply][0] = (r, c)
                break

        if not self.stop and len(self.tt) < ALGO_TT_MAX:
            flag = 0 if orig_alpha < best < beta else (1 if best >= beta else 2)
            self.tt[key] = (depth, int(best), flag, best_move)
        if ply == 0 and not self.stop:
            # 根节点：把这一层迭代加深的结果记下来。超时被中断时不写 ——
            # 上层循环靠「best_move 还在不在」判断本层有没有跑完。
            self.best_move = best_move
        return best

    # --------------------------- VCF（连续冲四）--------------------------- #
    def _four_moves(self, color: int) -> List[Tuple[int, int]]:
        """只列「下这里能形成冲四及以上」的点 —— VCF 的分支因子就靠这个压住。"""
        board = self.board
        hits: List[Tuple[int, int, int]] = []
        for (r, c), cnt in self.near.items():
            if cnt <= 0 or board[r][c] != 0:
                continue
            top = 0
            for dr, dc in DIRECTIONS:
                name = _classify_window(_line_string(board, r, c, dr, dc, self.K, color),
                                        self.K)[0]
                sc = PATTERN_SCORES[name]
                if sc > top:
                    top = sc
            if top >= PATTERN_SCORES["冲四"]:
                hits.append((top, r, c))
        hits.sort(key=lambda x: -x[0])
        return [(r, c) for _s, r, c in hits]

    def _vcf(self, color: int, depth: int) -> Optional[Tuple[int, int]]:
        """连续冲四取胜搜索：我们只走冲四，对手只能堵，看能不能一路逼死。"""
        if depth <= 0 or self.stop or self.vcf_stop or time.monotonic() > self.deadline:
            return None
        self.vcf_nodes += 1
        if self.vcf_nodes > self.vcf_node_budget:
            self.vcf_stop = True
            return None
        other = 3 - color
        for r, c in self._four_moves(color)[:10]:
            self.make(r, c, color)
            if check_win(self.board, r, c, color, self.K):
                self.undo()
                return (r, c)
            if win_now_cells(self.board, self.K, other):
                self.undo()                       # 对手自己有杀，我们的冲四挡不住
                continue
            blocks = win_now_cells(self.board, self.K, color)
            if len(blocks) >= 2:
                self.undo()                       # 双四：对手堵不过来
                return (r, c)
            found = None
            if len(blocks) == 1:
                br, bc = blocks[0]
                self.make(br, bc, other)
                found = self._vcf(color, depth - 1)
                self.undo()
            self.undo()
            if found is not None:
                return (r, c)
        return None

    # ----------------------------- 对外主入口 ----------------------------- #
    def get_move(self, level: str) -> Dict[str, Any]:
        """算一步棋，返回 (落点 + 统计信息)。保证一定给出合法落点。"""
        t0 = time.monotonic()
        nominal = float(self.cfg.get("time_ms", 800)) / 1000.0
        # 只是安全网：正常情况先撞到的是节点预算，而不是这个
        self.deadline = t0 + nominal * ALGO_TIME_SAFETY
        info: Dict[str, Any] = {
            "level": level, "depth": 0, "nodes": 0, "ms": 0,
            "budget": self.node_budget,
            "vcf": "", "reason": "", "move": None, "score": 0,
        }

        # ① 空盘：直接天元（也顺手把搜索的第一个候选点定好）
        if not self.pts:
            mid = self.N // 2
            info.update(move=(mid, mid), reason="空盘开局，落在天元")
            return info

        # ② 自己一子即胜 → 立刻赢
        my_wins = win_now_cells(self.board, self.K, self.me)
        if my_wins:
            mv = _pick_center_first(my_wins, self.N)[0]
            info.update(move=mv, reason="己方一子即可连成 K 子获胜，直接取胜")
            return info

        # ③ 对手只剩唯一必胜点 → 必堵（不堵下一手就输，没有别的选择）
        opp_wins = win_now_cells(self.board, self.K, self.opp)
        if len(opp_wins) == 1:
            info.update(move=opp_wins[0], reason="对手唯一的一子即胜点，必须占住")
            return info

        # ④ VCF：连续冲四能不能直接算死 —— 这是「会赢」的关键，先于常规搜索
        vcf_depth = int(self.cfg.get("vcf", 0) or 0)
        if vcf_depth > 0 and len(self.pts) >= 2 * self.K - 2:
            hit = self._vcf(self.me, vcf_depth)
            self.vcf_stop = False                # VCF 没算出来不算失败，继续常规搜索
            if hit is not None:
                info.update(move=hit, vcf=f"深度 {vcf_depth} 内找到连续冲四杀",
                            reason="VCF：连续冲四，对手无法同时化解")
                info["ms"] = int((time.monotonic() - t0) * 1000)
                info["nodes"] = self.nodes + self.vcf_nodes
                return info

        # ⑤ 迭代加深 NegaScout —— 每层完整跑完才更新答案，超时就停在上一层
        target = int(self.cfg.get("depth", 4))
        best: Optional[Tuple[int, int]] = None
        for d in range(1, target + 1):
            before = self.nodes
            self.stop = False
            self.best_move = None
            score = self._negascout(d, -ALGO_INF, ALGO_INF, self.me, 0)
            if self.stop or self.best_move is None:
                break
            best = self.best_move
            info["depth"] = d
            info["score"] = int(score)
            if abs(score) >= ALGO_WIN_SCORE - 64:       # 已经算到必胜 / 必败
                break
            # 用「上一层用了多少节点」预测下一层（PVS 通常按 3 倍左右增长），
            # 预测跑不完就不开这一层 —— 开了也是白开，只是白等。
            # 用节点数而不是秒数做这个判断，落点才严格可复现。
            used = self.nodes - before
            if self.nodes + used * 3 > self.node_budget:
                break

        if best is None:                                # 一层都没跑完（预算极小）
            lst = self._move_list(self.me, 0)
            best = lst[0] if lst else None

        info["nodes"] = self.nodes + self.vcf_nodes
        info["ms"] = int((time.monotonic() - t0) * 1000)
        info["move"] = best
        if best is not None:
            if info["score"] >= ALGO_WIN_SCORE - 64:
                info["reason"] = f"搜索判定己方必胜（{info['depth']} 层内算到）"
            elif info["score"] <= -(ALGO_WIN_SCORE - 64):
                info["reason"] = f"搜索判定己方劣势，选择最顽强的一手（{info['depth']} 层）"
            else:
                info["reason"] = f"常规搜索 {info['depth']} 层选出评分最高的点"
        return info


def algorithm_move(state: Dict[str, Any], player: int,
                   level: Optional[str] = None) -> Dict[str, Any]:
    """给「纯算法」玩家算一步棋。全程不发任何 HTTP 请求。

    棋盘用副本：搜索过程会大量落子/悔棋，绝不能碰到正在展示的真实棋盘。
    `level` 可显式指定（「帮我想一步」用）；不给就取该玩家自己的难度。
    """
    lv = _norm_algo_level(level, player_algo_level(state, player))
    cfg = ALGO_LEVELS[lv]
    board = [row[:] for row in state["board"]]
    engine = AlgoEngine(board, state["win_length"], player, cfg)
    info = engine.get_move(lv)
    # 补上「这一手是谁、用的哪档」—— 前端状态面板与日志要靠它区分黑白双方各自的难度
    info["player"] = player
    info["label"] = cfg["label"]
    if info.get("move") is None:                        # 兜底：绝不允许算不出落点
        info["move"] = fallback_move(state)
        info["reason"] = info.get("reason") or "搜索未产出落点，回退为启发式选点"
    return info


# --------------------------------------------------------------------------- #
# 模型调用
# --------------------------------------------------------------------------- #
def render_board(state: Dict[str, Any]) -> str:
    """按 board_format 把棋盘渲染成文本。

    rows   —— 经典「行号 + 数字」，紧凑
    cells  —— 逐格标注「第N行第M列:值」，行列绝不会看错，但很啰嗦（15x15 约 2200 字）
    coords —— 只列已落子的格子 + 双方清单，信息密度最高；棋子少时优势极大
              （实测能缓解模型「照抄棋盘上唯一那个非零坐标」的毛病）
    """
    N = state["size"]
    board = state["board"]
    fmt = state.get("board_format") or DEFAULT_BOARD_FORMAT

    if fmt == "cells":
        out = []
        for r in range(N):
            cells = " ".join(f"第{r + 1}行第{c + 1}列:{board[r][c]}" for c in range(N))
            out.append(cells)
        return ("棋盘逐格列出（格式为 第N行第M列:值）：\n"
                + "\n".join(out) + "\n（行、列编号都从 1 开始）")

    if fmt == "coords":
        black = [(m["row"], m["col"]) for m in state["history"] if m["player"] == 1]
        white = [(m["row"], m["col"]) for m in state["history"] if m["player"] == 2]
        show = lambda cells: "、".join(f"({r + 1},{c + 1})" for r, c in cells) if cells else "（无）"
        return (
            f"棋盘大小 {N}×{N}，当前所有已落子的位置（行列都从 1 开始编号）：\n"
            f"黑棋(1)：{show(black)}\n"
            f"白棋(2)：{show(white)}\n"
            f"除上述位置外，其余格子都是空的（0）。"
        )

    lines = [f"行{r + 1}: " + " ".join(str(board[r][c]) for c in range(N)) for r in range(N)]
    return ("当前棋盘（每行左侧是行号，行列都从 1 开始编号）：\n" + "\n".join(lines))


def build_prompt(
    state: Dict[str, Any],
    player: int,
    report: Optional[Dict[str, Any]] = None,
    decision: Optional[Dict[str, Any]] = None,
) -> str:
    """拼装提示词。

    相对最初模板的调整（都是为了减少「模型乱下」）：
      · 「你在下{N}子棋」→「你在下{N}×{N}的连珠棋（Gomoku）」，前者会写出「15子棋」这种不存在的棋种
      · 棋盘渲染支持三种格式，见 render_board
      · 可选插入程序算好的战术提示（tactics）
      · 可选插入传统算法算出的完整局势分析报告（assist，实验性）
      · 可选按混合决策收窄候选点（hybrid，实验性）
      · 非思考模式可选「先写一小段解析再给坐标」（analysis）

    report / decision 允许调用方传入 —— 同一轮里分析报告要算好几处，传进来省掉重复计算。
    """
    N = state["size"]
    K = state["win_length"]
    color = "黑棋(1)" if player == 1 else "白棋(2)"
    analysis = bool(state.get("analysis"))
    coord_rule = "坐标格式统一为：行,列（例如：8,8）"

    # 先确定本步的放权层级。hybrid 关着时它就是个 mode=off 的空壳，不产生任何计算。
    if decision is None:
        decision = decide_hybrid(state, player, report)
    dec = decision
    restricted = dec.get("mode") == "restricted"

    parts = [
        f"你在下{N}×{N}的连珠棋（Gomoku），连成{K}子获胜。",
        "棋盘状态用数字表示：0=空 1=黑棋 2=白棋。",
        "",
        render_board(state),
    ]

    # 传统算法辅助：把程序算好的局势报告整段塞进提示词。
    # 放在战术提示之前 —— 让短促紧迫的「必须堵这点」离任务指令更近，模型更当回事。
    # 整块用 try 包住：辅助模块是实验性的，绝不能因为它抛异常就把整局弄挂。
    # 注意 restricted 的判断独立于 assist：只开了混合决策、没开完整报告时，
    # 硬性约束那一段照样要进提示词。
    if state.get("assist") or restricted:
        try:
            if report is None:
                report = analyze_board(state, player)
            if state.get("assist"):
                parts += ["", render_assist(state, player, report, dec)]
            else:
                parts += ["", render_restrict_hint(report, dec)]
        except Exception as exc:                     # noqa: BLE001 - 兜底，避免影响对局
            parts += ["", f"（程序局势分析生成失败：{type(exc).__name__}: {exc}）"]

    if state.get("tactics"):
        parts += ["", build_tactics_hint(state, player)]

    parts += [
        "",
        f"你是{color}。",
        "注意：只能落在空位上，不能落在已有棋子的位置。",
    ]

    if analysis:
        # 注意措辞：实测「说明你为什么选这里」这种开放式问法会让模型「说的和下的不一致」
        # （嘴上说"第八行四连要堵"，坐标却给第6行）—— 在一个必堵局面上只有 0/3 堵对。
        # 改成「先定好落子再写理由 + 理由必须与坐标一致 + 强制从战术提示里选」后提升到 2/3。
        parts += [
            "请先在心里定好落子，然后输出两行：",
            "第 1 行：一句不超过 30 字的理由，必须和你最终输出的坐标一致；",
            f"第 2 行：只输出坐标，{coord_rule}，不要有任何其他文字。",
        ]
        if state.get("assist"):
            parts.append("如果上面的局势分析报告给出了候选点，请优先从候选点里选一个。")
        if state.get("tactics"):
            parts.append("如果上面的战术提示列出了「必须抢占的点」，请直接从这些点里选。")
    else:
        parts += [
            f"请只输出一个落子坐标，{coord_rule}",
            "不要输出任何解释、标点或其他文字。",
        ]

    if restricted:
        # 放在最后一行 —— 离模型开始生成的位置最近，约束力最强
        n_pts = len(dec.get("points") or [])
        parts.append(f"⚠️ 最后再强调一次：本轮你只能从上面列出的 {n_pts} 个坐标里选一个，"
                     "选其他任何位置都算违反约束。")

    return "\n".join(parts)


def parse_coord(text: str, state: Dict[str, Any], prefer_last: bool = False) -> Optional[Tuple[int, int]]:
    """正则容错解析文本里的落子坐标，返回第一个（或最后一个）合法且为空的坐标。

    prefer_last=True 用于解析「思考过程」：模型的结论总是在推理的**末尾**，
    取第一个会抓到它早期列举的候选点（实测会挑出 (0,0) 这种废点，
    而末尾那句才是「(4,4) is good」）。

    坐标系：提示词里行列都是 **1 开始**（第1行第1列），而内部 `board` 是 0 开始。
    模型偶尔会忘记、按 0 起算输出 —— 所以这里对每个候选**先按 1-based 解释，
    不合法再按 0-based 试一次**，两边都不行才跳到下一个候选。
    """
    if not text:
        return None
    matches = list(COORD_RE.finditer(text))
    if prefer_last:
        matches.reverse()
    for m in matches:
        a, b = int(m.group(1)), int(m.group(2))
        for base in (1, 0):
            row, col = a - base, b - base
            if is_empty_cell(state, row, col):
                return row, col
    return None


def pick_move(
    content: str,
    reasoning: str,
    state: Dict[str, Any],
) -> Tuple[Optional[Tuple[int, int]], str]:
    """从模型的输出里选出落子，并说明来源。

    返回 ((row, col) | None, source)，source ∈ {"model", "reasoning", ""}。
    · content 非空 → 取坐标。开了「输出解析」时取**最后一个**（结论在末尾）。
    · content 为空（思考型模型被 token 预算截断的常态）→ 退而从推理文本里取末尾坐标。
    """
    prefer_last = bool(state.get("analysis"))
    coord = parse_coord(content, state, prefer_last=prefer_last)
    if coord is not None:
        return coord, "model"
    coord = parse_coord(reasoning, state, prefer_last=True)
    if coord is not None:
        return coord, "reasoning"
    return None, ""


def extract_text(data: Any) -> Tuple[str, str]:
    """从 OpenAI 兼容响应里取出 (content, reasoning_content)。

    Qwen3 系「思考模型」会把 token 全部花在 reasoning_content 上，content 留成空串，
    所以两段都要取出来备用。
    """
    try:
        choice = (data.get("choices") or [{}])[0]
    except (AttributeError, IndexError, TypeError):
        return "", ""
    if not isinstance(choice, dict):
        return "", ""
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        message = {}
    content = message.get("content") or choice.get("text") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return str(content), str(reasoning)


# 模型调用重试阶梯 (max_tokens, 是否带 chat_template_kwargs, enable_thinking)
#   背景：Qwen3 系「思考模型」默认 enable_thinking=True，会把 token 全部花在
#   reasoning_content 上、content 留成空串，坐标永远解析不出来（实测 24 与 512 都一样）。
#   · 关思考：24 个 token 就够，实测 1~2 秒返回坐标 —— 默认走这条
#   · 关思考 + 输出解析：给 220，让它写一句分析 + 末尾坐标
#   · 开思考：按「思考强度」给 512/1024/2048/4096，让它推够再落子（慢，但棋力明显更好）
#   另：仅当服务端「拒收」chat_template_kwargs（HTTP 4xx）时，才补一次不带该字段、
#   改用提示词 /no_think 的尝试。不要无条件重试 —— 模型只是答不出坐标时，
#   再问一次既慢又不会变好，直接交给启发式回退更划算。
NO_THINK_SUFFIX = "\n/no_think"


def attempt_plan(state: Dict[str, Any]) -> List[Tuple[int, bool, bool]]:
    """按当前高级选项算出要依次尝试的 (max_tokens, use_kwargs, enable_thinking) 序列。"""
    if state.get("thinking"):
        return [(_norm_thinking_budget(state.get("thinking_budget")), True, True)]
    if state.get("analysis"):
        return [(MAX_TOKENS_ANALYSIS, True, False),
                (MAX_TOKENS_NO_THINK_RETRY, True, False)]
    return [(MAX_TOKENS_NO_THINK, True, False),
            (MAX_TOKENS_NO_THINK_RETRY, True, False)]


def build_payload(
    model_name: str,
    prompt: str,
    max_tokens: int,
    use_kwargs: bool,
    enable_thinking: bool,
    stream: bool,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    params = _norm_model_params(model_params)
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user",
                      "content": prompt if use_kwargs else prompt + NO_THINK_SUFFIX}],
        "max_tokens": max_tokens,
        # llama.cpp 原生采样参数（实测该服务端全部接受）
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "repeat_penalty": params["repeat_penalty"],
    }
    if use_kwargs:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if stream:
        payload["stream"] = True
    return payload


def model_target(state: Dict[str, Any], player: int) -> Tuple[str, str]:
    """返回 (模型名, chat/completions 地址)。"""
    cfg = state["players"].get(str(player)) or {}
    model_name = cfg.get("model") or DEFAULT_MODEL_NAME
    base = (state["model_base_url"] or DEFAULT_MODEL_BASE_URL).rstrip("/")
    return model_name, f"{base}/v1/chat/completions"


async def iter_sse_deltas(resp: httpx.Response) -> AsyncIterator[Tuple[str, str]]:
    """异步解析 OpenAI 兼容的流式响应体，产出 (kind, text)。kind ∈ reasoning/content。"""
    async for raw in resp.aiter_lines():
        line = raw.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if not isinstance(delta, dict):
            continue
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        content = delta.get("content") or ""
        if reasoning:
            yield "reasoning", str(reasoning)
        if content:
            yield "content", str(content)


async def call_ai(state: Dict[str, Any], player: int) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """非流式调用模型（供 /api/step 与自测使用）。

    成功返回 (row, col, None)，失败返回 (None, None, err)。
    """
    # 混合决策先跑一遍：判定为 forced 时直接返回坐标，这一步连模型都不问。
    rep: Optional[Dict[str, Any]] = None
    if state.get("assist") or state.get("hybrid"):
        try:
            rep = analyze_board(state, player)
        except Exception:                            # noqa: BLE001 - 实验性模块，失败就退回自由决策
            rep = None
    decision = decide_hybrid(state, player, rep)
    state["last_decision"] = decision
    if decision.get("mode") == "forced" and decision.get("skip_model") and decision.get("move"):
        mr, mc = decision["move"]
        if is_empty_cell(state, mr, mc):
            return mr, mc, None

    model_name, url = model_target(state, player)
    prompt = build_prompt(state, player, rep, decision)
    thinking = bool(state.get("thinking"))
    problems: List[str] = []
    state["last_engine"] = None          # 本步走模型，清掉上一次纯算法搜索的统计

    attempts = attempt_plan(state)
    async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
        i = 0
        while i < len(attempts):
            max_tokens, use_kwargs, enable_thinking = attempts[i]
            i += 1
            payload = build_payload(model_name, prompt, max_tokens,
                                    use_kwargs, enable_thinking, stream=False,
                                    model_params=state.get("model_params"))
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as exc:
                problems.append(f"HTTP {exc.response.status_code}")
                if use_kwargs:
                    # 服务端可能不认 chat_template_kwargs → 补一次不带该字段的尝试
                    attempts.append((max_tokens, False, enable_thinking))
                continue
            except Exception as exc:
                # 连接失败 / 超时 → 换 payload 也连不上，直接返回，避免无谓重试
                return None, None, f"模型服务调用失败（{url}）：{type(exc).__name__}: {exc}"

            content, reasoning = extract_text(data)
            coord, _src = pick_move(content, reasoning, state)
            if coord is not None:
                return coord[0], coord[1], None

            snippet = (content or reasoning or "").strip().replace("\n", " ")[:70]
            problems.append(f"max_tokens={max_tokens} 输出无法解析 {snippet!r}")

    return None, None, f"模型调用未取得合法坐标（{url}）：" + " | ".join(problems[-2:])


def sse(event: str, data: Any) -> str:
    """拼一条 SSE 消息。data 用 JSON 编码，换行会被转义，因此始终是单行。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _finish_move(
    state: Dict[str, Any],
    coord: Tuple[int, int],
    player: int,
    source: str,
    error: Optional[str],
) -> List[str]:
    """落子并给出收尾的 move / state 两条事件 —— 让「算法代打」和「模型走子」共用同一条尾巴。"""
    apply_move(state, coord[0], coord[1], player)
    return [
        sse("move", {
            "row": coord[0], "col": coord[1], "player": player,
            # 提示词里的行列是 1 起算的，这里一并给出，方便对着日志核对
            "row1": coord[0] + 1, "col1": coord[1] + 1,
            "source": source, "fallback": source == "fallback",
            "error": error,
        }),
        sse("state", {"state": state}),
    ]


async def stream_algorithm_step(state: Dict[str, Any], player: int) -> AsyncIterator[str]:
    """SSE 流式走一步 —— 纯算法版。

    事件流：engine(start) → engine(done) → move → state
    没有 prompt / reasoning / content：这一路完全不碰模型。
    """
    level = player_algo_level(state, player)
    cfg = ALGO_LEVELS[level]
    state["ai_thinking"] = True
    state["last_error"] = None
    state["last_decision"] = None        # 纯算法不做混合决策，别让上一局的判定残留
    yield sse("engine", {
        "phase": "start", "level": level, "label": cfg["label"],
        "player": player, "depth": cfg["depth"], "ms": cfg["time_ms"],
        "nodes": cfg["nodes"], "cands": cfg["cands"], "vcf": cfg["vcf"],
    })

    info: Dict[str, Any] = {}
    try:
        # 搜索是纯 CPU 的同步计算，扔到线程里跑，别把事件循环堵死
        async with STEP_LOCK:
            info = await asyncio.to_thread(algorithm_move, state, player)
    except Exception as exc:                         # noqa: BLE001 - 绝不让算法异常中断对局
        info = {"move": None, "reason": f"算法异常（{type(exc).__name__}: {exc}）",
                "depth": 0, "nodes": 0, "ms": 0, "vcf": "", "level": level, "score": 0,
                "player": player, "label": cfg["label"]}
    finally:
        state["ai_thinking"] = False

    if STATE is not state:                           # 期间被 /api/new_game 重置
        yield sse("state", {"state": STATE})
        return

    coord = info.get("move")
    source = "engine"
    if coord is None or not is_empty_cell(state, coord[0], coord[1]):
        fb = fallback_move(state)
        if fb is None:
            state["last_error"] = "棋盘已满，无法落子。"
            yield sse("state", {"state": state})
            return
        if coord is not None:
            state["last_error"] = "算法给出的落点已被占用，已回退为启发式落子。"
        coord = fb
        source = "fallback"

    info["move"] = [coord[0], coord[1]]
    state["last_engine"] = info
    yield sse("engine", dict(info, phase="done"))

    try:
        for ev in _finish_move(state, coord, player, source, state["last_error"]):
            yield ev
    except Exception as exc:                         # noqa: BLE001 - 收尾失败也要把状态推回去
        state["last_error"] = f"落子收尾失败（{type(exc).__name__}: {exc}）"
        yield sse("notice", {"text": state["last_error"]})
        yield sse("state", {"state": state})


async def stream_step(state: Dict[str, Any]) -> AsyncIterator[str]:
    """SSE 流式走一步：把提示词、模型的思考过程、最终状态实时推给前端。

    事件类型：
      prompt     —— {prompt, model, url, thinking, player}  本步用的提示词原文
      notice     —— {text}                                  重试 / 提示
      reasoning  —— {delta}                                 模型的思考过程（流式）
      content    —— {delta}                                 模型的正稿输出（流式）
      move       —— {row, col, player, fallback, error}     最终落子
      state      —— {state}                                 完整对局状态（最后一个事件）
    """
    player = state["current_player"]

    # 纯算法玩家：整条链路与模型完全无关 —— 不发请求、不拼提示词、不看采样参数。
    # 单独走一条分支，保证「关掉模型服务也能正常对局」。
    if is_algorithm(state, player):
        async for ev in stream_algorithm_step(state, player):
            yield ev
        return

    thinking = bool(state.get("thinking"))
    model_name, url = model_target(state, player)
    state["last_engine"] = None          # 本步走模型，清掉上一次纯算法搜索的统计

    # 混合决策：决定本步是「程序代打」「只给少量候选」还是「不干预」
    rep: Optional[Dict[str, Any]] = None
    if state.get("assist") or state.get("hybrid"):
        try:
            rep = analyze_board(state, player)
        except Exception:                            # noqa: BLE001 - 实验性模块，失败就退回自由决策
            rep = None
    decision = decide_hybrid(state, player, rep)
    state["last_decision"] = decision
    prompt = build_prompt(state, player, rep, decision)

    state["ai_thinking"] = True
    state["last_error"] = None

    yield sse("prompt", {
        "prompt": prompt,
        "model": model_name,
        "url": url,
        "thinking": thinking,
        "player": player,
        "board_format": state.get("board_format"),
        "analysis": bool(state.get("analysis")),
        "tactics": bool(state.get("tactics")),
        "assist": bool(state.get("assist")),
        "hybrid": decision,
        "max_tokens": attempt_plan(state)[0][0],
        "model_params": _norm_model_params(state.get("model_params")),
    })

    # ---- 算法强制落子：这一步的答案程序已经算死了，不必再问模型 ----
    if decision.get("mode") == "forced" and decision.get("skip_model") and decision.get("move"):
        state["ai_thinking"] = False

        if STATE is not state:                       # 期间被 /api/new_game 重置
            yield sse("state", {"state": STATE})
            return

        coord = tuple(decision["move"])
        source = "algorithm"
        if not is_empty_cell(state, coord[0], coord[1]):
            # 理论上不会发生（forced 点是当步现算的），保险起见仍走启发式
            fb = fallback_move(state)
            if fb is None:
                state["last_error"] = "棋盘已满，无法落子。"
                yield sse("state", {"state": state})
                return
            coord = fb
            source = "fallback"
            state["last_error"] = "算法强制落点已被占用，已回退为启发式落子。"

        yield sse("notice", {"text": f"🧮 算法强制落子：{decision.get('reason') or ''}"
                                     f"（本步未请求模型）"})
        for ev in _finish_move(state, coord, player, source, state["last_error"]):
            yield ev
        return

    coord: Optional[Tuple[int, int]] = None
    source = ""                                  # model / reasoning / fallback
    problems: List[str] = []
    fatal: Optional[str] = None

    async with STEP_LOCK:
        attempts = attempt_plan(state)
        try:
            async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
                i = 0
                while i < len(attempts):
                    max_tokens, use_kwargs, enable_thinking = attempts[i]
                    i += 1
                    if i > 1:
                        yield sse("notice", {"text": f"第 {i} 次尝试（max_tokens={max_tokens}）…"})
                    if not use_kwargs:
                        yield sse("notice", {"text": "服务端不认 chat_template_kwargs，"
                                                     "改用提示词追加 /no_think 重试"})

                    payload = build_payload(model_name, prompt, max_tokens,
                                            use_kwargs, enable_thinking, stream=True,
                                            model_params=state.get("model_params"))
                    content_parts: List[str] = []
                    reasoning_parts: List[str] = []
                    try:
                        async with client.stream("POST", url, json=payload) as resp:
                            resp.raise_for_status()
                            async for kind, text in iter_sse_deltas(resp):
                                if kind == "reasoning":
                                    reasoning_parts.append(text)
                                    yield sse("reasoning", {"delta": text})
                                else:
                                    content_parts.append(text)
                                    yield sse("content", {"delta": text})
                    except httpx.HTTPStatusError as exc:
                        problems.append(f"HTTP {exc.response.status_code}")
                        if use_kwargs:
                            attempts.append((max_tokens, False, enable_thinking))
                        continue
                    except Exception as exc:
                        fatal = f"模型服务调用失败（{url}）：{type(exc).__name__}: {exc}"
                        problems.append(fatal)
                        break

                    content = "".join(content_parts)
                    reasoning = "".join(reasoning_parts)
                    coord, source = pick_move(content, reasoning, state)
                    if coord is not None:
                        break
                    snippet = (content or reasoning).strip().replace("\n", " ")[:70]
                    problems.append(f"max_tokens={max_tokens} 输出无法解析 {snippet!r}")
        finally:
            state["ai_thinking"] = False

    if STATE is not state:                      # 等待期间被 /api/new_game 重置
        yield sse("state", {"state": STATE})
        return

    if coord is None:
        fb = fallback_move(state)
        if fb is None:
            state["last_error"] = "棋盘已满，无法落子。"
            yield sse("state", {"state": state})
            return
        coord = fb
        source = "fallback"
        state["last_error"] = (f"{problems[-1] if problems else '模型未返回结果'}"
                               f" 已回退为启发式落子 第{coord[0] + 1}行第{coord[1] + 1}列。")
    elif source == "reasoning":
        # 思考被截断、回答没写完 —— 落子是从推理末尾推断出来的，明确告知用户
        state["last_error"] = ("模型思考超出 token 预算，答案被截断；"
                               f"已按其推理末尾的结论落在 第{coord[0] + 1}行第{coord[1] + 1}列。"
                               "可在设置里调高思考强度或改用关思考模式。")

    # forced 局面上用户选择「仍然问一次模型」：这一步答案以程序为准，
    # 模型的落点只作为对照记录下来 —— 否则这个选项就变成了「只加一句提示」，
    # 与它「既能对比观察、又不牺牲正确率」的定位不符。
    if decision.get("mode") == "forced" and decision.get("move"):
        algo = tuple(decision["move"])
        if is_empty_cell(state, algo[0], algo[1]):
            if (coord[0], coord[1]) == algo:
                yield sse("notice", {"text": f"🧮 算法与模型判断一致：{decision.get('reason') or ''}"})
            else:
                yield sse("notice", {
                    "text": f"🧮 算法强制落子（模型本步落在 第{coord[0] + 1}行第{coord[1] + 1}列，未被采用）："
                            f"{decision.get('reason') or ''}",
                })
            coord = algo
            source = "algorithm"
            state["last_error"] = None
        else:
            yield sse("notice", {"text": "算法给出的强制落点已被占用，改用模型给出的落点。"})

    # restricted 只是「提示词里限定范围」，并不强制纠正落点 —— 但模型听不听话正是
    # 这个实验项要观察的东西，所以越界时如实记一条，别让它悄悄溜过去。
    if decision.get("mode") == "restricted":
        allowed = {tuple(x) for x in (decision.get("points") or [])}
        if allowed and (coord[0], coord[1]) not in allowed:
            yield sse("notice", {
                "text": f"⚠️ 模型没有遵守「只能从这 {len(allowed)} 个点里选」的约束，"
                        f"落在 第{coord[0] + 1}行第{coord[1] + 1}列；本次仍然采用它的落点。",
            })

    for ev in _finish_move(state, coord, player, source, state["last_error"]):
        yield ev


# --------------------------------------------------------------------------- #
# FastAPI 应用
# --------------------------------------------------------------------------- #
app = FastAPI(title="五子棋 AI 对战平台", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ------------------------------- 请求体模型 -------------------------------- #
class NewGameRequest(BaseModel):
    size: int = 15
    win_length: int = 5
    mode: str = "human_vs_ai"
    players: Optional[Dict[str, Any]] = None
    model_base_url: Optional[str] = None
    # 以下高级选项为 None 时沿用上一局的设置
    thinking: Optional[bool] = None
    thinking_budget: Optional[Any] = None       # 档位名 low/medium/high/ultra 或 token 数
    model_params: Optional[Dict[str, Any]] = None
    board_format: Optional[str] = None          # rows / cells / coords
    analysis: Optional[bool] = None
    tactics: Optional[bool] = None
    assist: Optional[bool] = None               # 传统算法辅助决策（实验性）
    # 混合决策（实验性）：必下局面由程序直接落子，高警惕局面只给模型少量候选点
    hybrid: Optional[bool] = None
    hybrid_hints: Optional[Any] = None
    hybrid_skip_model: Optional[bool] = None
    # 默认算法难度（兜底）。真正的难度写在 players 里：{"1": {"type":"algorithm","level":"hard"}}
    algo_level: Optional[str] = None


class SettingsRequest(BaseModel):
    """在线修改高级选项 —— 不重置对局，下一次 AI 走子立即生效。"""
    thinking: Optional[bool] = None
    thinking_budget: Optional[Any] = None
    model_params: Optional[Dict[str, Any]] = None
    board_format: Optional[str] = None
    analysis: Optional[bool] = None
    tactics: Optional[bool] = None
    assist: Optional[bool] = None
    hybrid: Optional[bool] = None
    hybrid_hints: Optional[Any] = None
    hybrid_skip_model: Optional[bool] = None
    algo_level: Optional[str] = None
    # 边下边调：{"1": "easy", "2": "hard"} —— 各改各的，不需要重开对局
    players_level: Optional[Dict[str, Any]] = None
    # 允许连带改模型与地址（同样不重置对局）
    model_base_url: Optional[str] = None
    model: Optional[str] = None


class HintRequest(BaseModel):
    """「帮我下一步」—— 让算法替你算一个落点，**只给建议不落子**。"""
    level: Optional[str] = None     # 本次计算用的难度；不给就用「当前方」自己的档位
    player: Optional[int] = None    # 替谁算；不给就取当前轮到的一方


class MoveRequest(BaseModel):
    row: int
    col: int


class ThinkingRequest(BaseModel):
    enabled: bool


# --------------------------------- 路由 ----------------------------------- #
@app.get("/")
async def index():
    """返回前端页面。"""
    if INDEX_FILE.is_file():
        return FileResponse(str(INDEX_FILE), media_type="text/html")
    return {"error": "static/index.html 不存在", "expected": str(INDEX_FILE)}


@app.get("/api/state")
async def api_state():
    """返回当前完整状态。"""
    return STATE


@app.get("/api/models")
async def api_models(base_url: Optional[str] = None):
    """代理请求 {model_base_url}/v1/models，返回可用模型名列表。"""
    base = (base_url or STATE["model_base_url"] or DEFAULT_MODEL_BASE_URL).strip().rstrip("/")
    url = f"{base}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=MODELS_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        models: List[str] = []
        for item in (data.get("data") or []):
            mid = item.get("id") if isinstance(item, dict) else None
            if mid:
                models.append(str(mid))
        return {"models": models}
    except Exception as exc:
        return {"models": [], "error": f"{type(exc).__name__}: {exc}"}


def _pick(req_value: Any, current: Any) -> Any:
    """请求里没传（None）就用当前值。"""
    return current if req_value is None else req_value


@app.post("/api/new_game")
async def api_new_game(req: NewGameRequest):
    """重置全局状态，开始新的一局。"""
    global STATE
    prev = STATE
    STATE = make_state(
        size=req.size,
        win_length=req.win_length,
        mode=req.mode,
        players=req.players,
        model_base_url=req.model_base_url or prev["model_base_url"],
        thinking=_pick(req.thinking, prev.get("thinking", False)),
        thinking_budget=_pick(req.thinking_budget, prev.get("thinking_budget")),
        model_params=_pick(req.model_params, prev.get("model_params")),
        board_format=_pick(req.board_format, prev.get("board_format")),
        analysis=_pick(req.analysis, prev.get("analysis", False)),
        tactics=_pick(req.tactics, prev.get("tactics", False)),
        assist=_pick(req.assist, prev.get("assist", False)),
        hybrid=_pick(req.hybrid, prev.get("hybrid", False)),
        hybrid_hints=_norm_hybrid_hints(
            _pick(req.hybrid_hints, prev.get("hybrid_hints")), DEFAULT_HYBRID_HINTS),
        hybrid_skip_model=_pick(req.hybrid_skip_model, prev.get("hybrid_skip_model", True)),
        algo_level=_pick(req.algo_level, prev.get("algo_level")),
    )
    return STATE


@app.post("/api/thinking")
async def api_thinking(req: ThinkingRequest):
    """开关模型的「思考」模式（不重置对局，立即生效）。"""
    STATE["thinking"] = bool(req.enabled)
    STATE["last_error"] = None
    return STATE


@app.post("/api/settings")
async def api_settings(req: SettingsRequest):
    """在线修改高级选项：采样参数 / 棋盘格式 / 解析 / 战术提示 / 思考强度。

    **不重置对局** —— 下一次 AI 走子就按新设置走，方便边下边调。
    """
    state = STATE
    if req.thinking is not None:
        state["thinking"] = bool(req.thinking)
    if req.thinking_budget is not None:
        state["thinking_budget"] = _norm_thinking_budget(req.thinking_budget)
    if req.model_params is not None:
        state["model_params"] = _norm_model_params(req.model_params, state.get("model_params"))
    if req.board_format is not None:
        state["board_format"] = _norm_board_format(req.board_format, state.get("board_format"))
    if req.analysis is not None:
        state["analysis"] = bool(req.analysis)
    if req.tactics is not None:
        state["tactics"] = bool(req.tactics)
    if req.assist is not None:
        state["assist"] = bool(req.assist)
    if req.hybrid is not None:
        state["hybrid"] = bool(req.hybrid)
    if req.hybrid_hints is not None:
        state["hybrid_hints"] = _norm_hybrid_hints(req.hybrid_hints, state.get("hybrid_hints"))
    if req.hybrid_skip_model is not None:
        state["hybrid_skip_model"] = bool(req.hybrid_skip_model)
    if req.algo_level is not None:
        state["algo_level"] = _norm_algo_level(req.algo_level, state.get("algo_level"))
    # 每位纯算法玩家自己的难度 —— 边下边改，立即对下一步生效
    if isinstance(req.players_level, dict):
        for key in ("1", "2"):
            lv = req.players_level.get(key, req.players_level.get(int(key)))
            if lv is None:
                continue
            cfg = state["players"].get(key)
            if not isinstance(cfg, dict) or cfg.get("type") != "algorithm":
                continue                 # 这一方不是纯算法，难度无意义，忽略
            # 空串 / "default" = 清掉单独设置，改为跟随默认档
            if isinstance(lv, str) and lv.strip().lower() in ("", "default"):
                cfg["level"] = None
            else:
                cfg["level"] = _norm_algo_level(lv, state.get("algo_level"))
    if req.model_base_url:
        state["model_base_url"] = req.model_base_url.strip().rstrip("/")
    if req.model:
        base_name = req.model.strip()
        for key in ("1", "2"):
            cfg = state["players"].get(key) or {}
            if cfg.get("type") == "ai":
                state["players"][key] = {"type": "ai", "model": base_name,
                                         "level": cfg.get("level")}
    state["last_error"] = None
    return state


@app.get("/api/prompt")
async def api_prompt():
    """返回「当前轮到的一方」即将发给模型的提示词原文，便于前端展示与排查。"""
    state = STATE
    player = state["current_player"]
    # 轮到「纯算法」玩家时不存在提示词 —— 如实返回引擎参数与上次搜索统计，
    # 而不是硬拼一段模型提示词骗人。
    if is_algorithm(state, player):
        level = player_algo_level(state, player)
        cfg = ALGO_LEVELS[level]
        return {
            "engine": "algorithm",
            "prompt": (f"【纯算法模式】本步由搜索算法直接落子，不调用任何模型。\n"
                       f"难度：{cfg['label']}（{level}）—— 这是「{player} 号玩家」自己的档位\n"
                       f"目标深度：{cfg['depth']} 层　时间预算：{cfg['time_ms']} ms\n"
                       f"候选点上限：{cfg['cands']} 个/层　VCF 搜索层数：{cfg['vcf']}\n"
                       f"上次搜索：{json.dumps(state.get('last_engine') or {}, ensure_ascii=False)}"),
            "player": player,
            "level": level,
            "level_label": cfg["label"],
            "algo_cfg": cfg,
            "players_level": players_level_map(state),
            "last_engine": state.get("last_engine"),
        }
    model_name, url = model_target(state, player)
    plan = attempt_plan(state)
    return {
        "engine": "llm",
        "prompt": build_prompt(state, player),
        "player": player,
        "algo_level": _norm_algo_level(state.get("algo_level")),
        "players_level": players_level_map(state),
        "thinking": bool(state.get("thinking")),
        "thinking_budget": state.get("thinking_budget"),
        "model": model_name,
        "url": url,
        "board_format": state.get("board_format"),
        "analysis": bool(state.get("analysis")),
        "tactics": bool(state.get("tactics")),
        "assist": bool(state.get("assist")),
        "hybrid": decide_hybrid(state, player),
        "hybrid_hints": _norm_hybrid_hints(state.get("hybrid_hints")),
        "hybrid_skip_model": _norm_bool(state.get("hybrid_skip_model"), True),
        "model_params": _norm_model_params(state.get("model_params")),
        "max_tokens": plan[0][0],
    }


@app.get("/api/analysis")
async def api_analysis(player: Optional[int] = None):
    """返回当前局面的传统算法分析报告（结构化 JSON）。

    默认分析「当前轮到的一方」；可用 `?player=1|2` 指定看谁的视角。
    这份 JSON 就是塞进提示词那段自然语言的数据源，便于对照排查。
    """
    state = STATE
    who = player if player in (1, 2) else state["current_player"]
    return analyze_board(state, who)


@app.post("/api/hint")
async def api_hint(req: HintRequest):
    """「帮我想一步」—— 用纯算法替人类玩家算一个落点，**只给建议、不落子、不改状态**。

    与前两个实验功能的区别：这里算法不是在「辅助模型」，而是**直接替人做决策**，
    用户看过建议后可以自己点「就下这里」落子，也可以忽略它自己下。
    因为全程只在棋盘副本上搜索，本接口对 STATE 完全无副作用（也不写 last_engine，
    免得面板里把「一次建议」误显示成「刚走了一手算法」）。
    """
    state = STATE
    if state["game_over"]:
        return {"ok": False, "error": "对局已结束，无需再算。"}

    who = req.player if req.player in (1, 2) else state["current_player"]
    level = _norm_algo_level(req.level, player_algo_level(state, who))
    cfg = ALGO_LEVELS[level]

    async with STEP_LOCK:                      # 与 /api/step 串行，避免两边同时抢占 CPU
        snap = (len(state["history"]), state["current_player"], state["last_move"])
        try:
            info = await asyncio.to_thread(algorithm_move, state, who, level)
        except Exception as exc:                # noqa: BLE001 - 建议失败不该影响对局
            return {"ok": False, "error": f"算法异常（{type(exc).__name__}: {exc}）"}
        untouched = snap == (len(state["history"]), state["current_player"],
                             state["last_move"])

    mv = info.get("move")
    if not mv:
        return {"ok": False, "error": info.get("reason") or "算法未产出落点"}
    row, col = int(mv[0]), int(mv[1])

    return {
        "ok": True,
        "player": who,
        "row": row, "col": col,
        "row1": row + 1, "col1": col + 1,        # 1 起算，与界面口径一致
        "level": level,
        "level_label": cfg["label"],
        "label": f"第{row + 1}行第{col + 1}列",
        "depth": info.get("depth"),
        "nodes": info.get("nodes"),
        "ms": info.get("ms"),
        "score": info.get("score"),
        "vcf": info.get("vcf"),
        "reason": info.get("reason"),
        "changed": not untouched,   # 恒为 False，用于自证「本接口不碰对局状态」
    }


@app.post("/api/move")
async def api_move(req: MoveRequest):
    """人类落子。"""
    state = STATE
    row, col = req.row, req.col

    if state["game_over"]:
        state["last_error"] = "对局已结束，请点击「重开」开始新的一局。"
        return state
    if is_ai(state, state["current_player"]):
        # 当前玩家不是人类
        state["last_error"] = "当前轮到 AI 落子，请稍候。"
        return state
    if not (0 <= row < state["size"] and 0 <= col < state["size"]):
        state["last_error"] = f"坐标越界：({row}, {col})，合法范围 0 ~ {state['size'] - 1}。"
        return state
    if state["board"][row][col] != 0:
        state["last_error"] = f"位置 ({row}, {col}) 已有棋子。"
        return state

    state["last_error"] = None
    apply_move(state, row, col, state["current_player"])
    return state


@app.post("/api/step")
async def api_step():
    """让当前玩家（必须是 AI）走一步；失败时回退到启发式落子。"""
    state = STATE

    if state["game_over"]:
        state["last_error"] = "对局已结束，请点击「重开」开始新的一局。"
        return state
    if not is_ai(state, state["current_player"]):
        state["last_error"] = "当前玩家不是 AI，无法执行 AI 落子。"
        return state

    async with STEP_LOCK:
        player = state["current_player"]
        state["ai_thinking"] = True
        state["last_error"] = None
        try:
            if is_algorithm(state, player):
                # 纯算法玩家：直接跑搜索引擎，一次模型请求都不发
                info = await asyncio.to_thread(algorithm_move, state, player)
                state["last_engine"] = info
                mv = info.get("move")
                row, col = (mv[0], mv[1]) if mv else (None, None)
                err = None if mv else (info.get("reason") or "算法未产出落点")
            else:
                row, col, err = await call_ai(state, player)
            # forced 局面上若仍问了模型（hybrid_skip_model 关闭），答案同样以程序为准
            dec = state.get("last_decision") or {}
            if not is_algorithm(state, player) and dec.get("mode") == "forced" and dec.get("move"):
                ar, ac = dec["move"]
                if is_empty_cell(state, ar, ac):
                    if row is not None and (row, col) != (ar, ac):
                        state["last_error"] = (f"🧮 算法强制落子（模型原本落在 ({row}, {col})，"
                                               f"未被采用）。")
                    row, col, err = ar, ac, None
            if row is None:
                fb = fallback_move(state)
                if fb is None:
                    state["last_error"] = "棋盘已满，无法落子。"
                    return state
                row, col = fb
                state["last_error"] = f"{err} 已回退为启发式落子 ({row}, {col})。"
            # 双保险：模型返回的点在等待期间被占用
            elif not is_empty_cell(state, row, col):
                fb = fallback_move(state)
                if fb is None:
                    state["last_error"] = "棋盘已满，无法落子。"
                    return state
                row, col = fb
                state["last_error"] = "给出的坐标已被占用，已回退为启发式落子。"
        finally:
            state["ai_thinking"] = False

    if STATE is not state:      # 等待期间被 /api/new_game 重置，丢弃这次结果
        return STATE
    apply_move(state, row, col, player)
    return state


@app.post("/api/step_stream")
async def api_step_stream():
    """SSE 版 /api/step：把提示词、模型思考过程、最终状态实时推给前端。

    事件流：prompt → (notice)* → (reasoning|content)* → move → state
    所有响应都以 `state` 事件收尾，前端只要应用它即可。
    """
    state = STATE

    async def guard(message: str) -> AsyncIterator[str]:
        state["last_error"] = message
        yield sse("notice", {"text": message})
        yield sse("state", {"state": state})

    if state["game_over"]:
        return StreamingResponse(
            guard("对局已结束，请点击「重开」开始新的一局。"),
            media_type="text/event-stream")

    if not is_ai(state, state["current_player"]):
        return StreamingResponse(
            guard("当前玩家不是 AI，无法执行 AI 落子。"),
            media_type="text/event-stream")

    return StreamingResponse(
        stream_step(state),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.post("/api/undo")
async def api_undo():
    """悔棋：移除最后一步；机机模式（或退后仍轮到 AI）时多退一步，人机模式退回到人类回合。"""
    state = STATE
    if not state["history"]:
        state["last_error"] = "尚无可悔的棋步。"
        return state

    def pop_one() -> None:
        m = state["history"].pop()
        state["board"][m["row"]][m["col"]] = 0
        state["current_player"] = m["player"]

    pop_one()
    # 最后两步都是 AI → 再退一步；人机模式下若退后轮到 AI，也再退一步回到人类回合
    if state["history"] and is_ai(state, state["current_player"]) and (
        has_human(state) or is_ai(state, state["history"][-1]["player"])
    ):
        pop_one()

    state["winner"] = 0
    state["game_over"] = False
    state["last_error"] = None
    state["ai_thinking"] = False
    sync_last_move(state)
    return state


def _port_in_use(host: str, port: int) -> bool:
    """端口是否已被占用（能连上 = 有人占着）。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def _pick_port(host: str, want: int, tries: int = 20) -> int:
    """从 want 起往上找一个空闲端口；全部占用才放弃。"""
    for port in range(want, want + tries):
        if not _port_in_use(host, port):
            return port
    return want


def _can_encode(text: str) -> bool:
    """当前控制台能不能显示这些字符（非中文 Windows 的默认代码页装不下汉字）。"""
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _print_banner(url: str, want: int, port: int) -> None:
    """打印启动横幅。控制台吃不消中文时自动退回纯英文，保证任何机器上都不会崩。"""
    zh = _can_encode(APP_NAME)
    line = "  " + "=" * 60
    dash = "  " + "-" * 60
    print()
    print(line)
    print(f"    {APP_NAME}  v{APP_VERSION}" if zh else f"    WuZiQi - AI Gomoku  v{APP_VERSION}")
    print(line)
    print(f"    URL      : {url}")
    if port != want:
        if zh:
            print(f"    （端口 {want} 已被占用，自动改用 {port}）")
        else:
            print(f"    (port {want} was busy, using {port} instead)")
    print(f"    前端资源 : {STATIC_DIR}" if zh else f"    Frontend : {STATIC_DIR}")
    print(dash)
    if zh:
        print("    模型服务没启动也能玩：把玩家类型选成「人类」或「传统算法」即可。")
        print("    想让 LLM 下棋，先在设置里填好本地 llama.cpp 的 OpenAI 兼容地址。")
        print(dash)
        print("    关闭：在本窗口按 Ctrl+C")
    else:
        print("    No model server needed: pick 'Human' or 'Algorithm' as the player type.")
        print("    To play against an LLM, set your local llama.cpp endpoint in Settings.")
        print(dash)
        print("    Press Ctrl+C in this window to stop the server.")
    print()


def main() -> int:
    import argparse

    # 控制台代码页不确定：无法表示的字符替换成 '?'，而不是让 print 抛异常崩掉进程
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass

    zh_help = _can_encode(APP_NAME)
    parser = argparse.ArgumentParser(
        prog="WuZiQi",
        description=(f"{APP_NAME} v{APP_VERSION} —— 启动本地网页服务并打开浏览器"
                     if zh_help else
                     f"WuZiQi - AI Gomoku v{APP_VERSION}: start the local web server"),
    )
    parser.add_argument("port", nargs="?", type=int, default=None,
                        help=(f"监听端口，默认 {DEFAULT_PORT}（被占用则自动顺延）" if zh_help
                              else f"listen port, default {DEFAULT_PORT} (auto-shifts if busy)"))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                        help="listen host, default 127.0.0.1")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open the browser automatically")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} v{APP_VERSION}")
    args = parser.parse_args()

    # 端口优先级：命令行 > 环境变量 PORT > 默认值；被占用则自动顺延到下一个空闲端口
    want = args.port or int(os.environ.get("PORT") or DEFAULT_PORT)
    host = args.host
    port = _pick_port(host, want)
    url = f"http://{host}:{port}/"

    _print_banner(url, want, port)

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        print("\n  stopped.")
    return 0


if __name__ == "__main__":
    import uvicorn

    sys.exit(main())
