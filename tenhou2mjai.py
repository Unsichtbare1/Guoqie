#!/usr/bin/env python3
r"""
tenhou.net/6 JSON 牌谱 -> MJAI 格式 转换器

将一个文件夹里所有 tenhou.net/6 格式的 JSON 牌谱转换为 MJAI（JSON Lines）格式，
输出到指定文件夹。每个输入文件对应一个输出文件（同名 .json，内容为 MJAI 事件流，
每行一个 JSON 对象，符合 Mortal 等麻将模型的训练数据格式）。

用法：
    python tenhou2mjai.py D:\trae_project\download_paipu\paipu_json dataset/mjai

tenhou.net/6 JSON 每局结构（log 数组的一个元素）：
    [
        [round, honba, kyotaku],    # 0
        scores,                     # 1
        dora_indicators,            # 2
        ura_indicators,             # 3
        haipai0, draws0, discards0, # 4, 5, 6
        haipai1, draws1, discards1, # 7, 8, 9
        haipai2, draws2, discards2, # 10, 11, 12
        haipai3, draws3, discards3, # 13, 14, 15
        result                      # 16  ("和了" / "流局" 等)
    ]

牌编码（天凤）：
    11-19 万 1-9, 21-29 筒 1-9, 31-39 索 1-9,
    41-47 东南西北白發中, 51/52/53 赤五万/赤五筒/赤五索, 60 摸切标记
副露串标记：
    draws 中：c=吃, p=碰, m=大明杠
    discards 中：a=暗杠, k=加杠, r=立直(前缀于打牌)
"""

import argparse
import json
import os
import sys
from typing import Any


# --------------------------------------------------------------------------- #
# 牌编码转换
# --------------------------------------------------------------------------- #
_TENHOU_TO_MJAI: dict[int, str] = {}
for _i in range(1, 10):
    _TENHOU_TO_MJAI[10 + _i] = f"{_i}m"
    _TENHOU_TO_MJAI[20 + _i] = f"{_i}p"
    _TENHOU_TO_MJAI[30 + _i] = f"{_i}s"
for _k, _v in [(41, "E"), (42, "S"), (43, "W"), (44, "N"),
               (45, "P"), (46, "F"), (47, "C")]:
    _TENHOU_TO_MJAI[_k] = _v
_TENHOU_TO_MJAI[51] = "5mr"
_TENHOU_TO_MJAI[52] = "5pr"
_TENHOU_TO_MJAI[53] = "5sr"

TSUMOGIRI_MARK = 60

RYUUKYOKU_TYPES = {
    "流局", "流し満貫", "四開槓", "三家和",
    "九種九牌", "四風連打", "四家立直",
}


def tile_to_mjai(t: int) -> str:
    """天凤牌编号 -> MJAI 牌字符串。"""
    return _TENHOU_TO_MJAI.get(t, "?")


# --------------------------------------------------------------------------- #
# 副露串解析
# --------------------------------------------------------------------------- #
def tokenize_call(s: str) -> list[Any]:
    """把副露串拆成 [整数牌, '标记', 整数牌, ...]。"""
    tokens: list[Any] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isalpha():
            tokens.append(c)
            i += 1
        else:
            tokens.append(int(s[i:i + 2]))
            i += 2
    return tokens


def parse_call(s: str, actor: int) -> dict:
    """
    解析 draws / discards 中的副露串，返回 MJAI 事件字典（不含 target）。
    chi/pon/daiminkan 的 target 需要由调用方填入（被吃碰杠的出牌者）。
    """
    tokens = tokenize_call(s)
    tiles = [t for t in tokens if isinstance(t, int)]

    if "c" in tokens:
        # chi: c + 被吃牌 + 两张手牌
        called = tiles[0]
        consumed = tiles[1:]
        return {
            "type": "chi", "actor": actor,
            "pai": tile_to_mjai(called),
            "consumed": [tile_to_mjai(x) for x in consumed],
        }
    if "p" in tokens:
        idx = tokens.index("p")
        called = tokens[idx + 1]
        consumed = [t for i, t in enumerate(tokens)
                    if isinstance(t, int) and i != idx + 1]
        return {
            "type": "pon", "actor": actor,
            "pai": tile_to_mjai(called),
            "consumed": [tile_to_mjai(x) for x in consumed],
        }
    if "m" in tokens:
        # daiminkan: m 标记后的牌是被杠牌，其余三张是手牌
        # （m 的位置随被杠者相对座位变化，不固定在开头）
        idx = tokens.index("m")
        called = tokens[idx + 1]
        consumed = [t for i, t in enumerate(tokens)
                    if isinstance(t, int) and i != idx + 1]
        return {
            "type": "daiminkan", "actor": actor,
            "pai": tile_to_mjai(called),
            "consumed": [tile_to_mjai(x) for x in consumed],
        }
    if "a" in tokens:
        # ankan: 四张相同
        return {
            "type": "ankan", "actor": actor,
            "consumed": [tile_to_mjai(x) for x in tiles],
        }
    if "k" in tokens:
        idx = tokens.index("k")
        called = tokens[idx + 1]
        consumed = [t for i, t in enumerate(tokens)
                    if isinstance(t, int) and i != idx + 1]
        return {
            "type": "kakan", "actor": actor,
            "pai": tile_to_mjai(called),
            "consumed": [tile_to_mjai(x) for x in consumed],
        }
    raise ValueError(f"无法解析的副露串: {s!r}")


# --------------------------------------------------------------------------- #
# 单局转换
# --------------------------------------------------------------------------- #
def convert_round(round_entry: list) -> list[dict]:
    """把一局（log 中的一个数组）转换为 MJAI 事件列表。"""
    meta = round_entry[0]                  # [round, honba, kyotaku]
    scores = round_entry[1]
    doras = round_entry[2] or []
    uras = round_entry[3] or []

    haipais = [round_entry[4 + 3 * p] for p in range(4)]
    draws = [round_entry[5 + 3 * p] for p in range(4)]
    discards = [round_entry[6 + 3 * p] for p in range(4)]
    result = round_entry[16] if len(round_entry) > 16 else None

    round_num, honba, sticks = meta[0], meta[1], meta[2]
    bakaze = ["E", "S", "W"][round_num // 4]
    kyoku = round_num % 4 + 1
    oya = round_num % 4

    events: list[dict] = []
    tehais = [[tile_to_mjai(t) for t in haipais[p]] for p in range(4)]
    events.append({
        "type": "start_kyoku",
        "bakaze": bakaze,
        "dora_marker": tile_to_mjai(doras[0]) if doras else "?",
        "kyoku": kyoku,
        "honba": honba,
        "kyotaku": sticks,
        "oya": oya,
        "scores": list(scores),
        "tehais": tehais,
    })

    draw_i = [0, 0, 0, 0]
    discard_i = [0, 0, 0, 0]
    # 用单一可变 dict 管理跨闭包共享的状态，避免 nonlocal + 列表双重不同步
    state = {"kan_count": 0, "pending_dora": False}

    # 庄家的第 14 张牌在 draws[oya][0]，作为初次摸牌发出
    if draws[oya]:
        events.append({"type": "tsumo", "actor": oya,
                       "pai": tile_to_mjai(draws[oya][0])})
        draw_i[oya] = 1
    current = oya

    def last_drawn_tile(p: int) -> str:
        if draw_i[p] > 0 and draws[p]:
            return tile_to_mjai(draws[p][draw_i[p] - 1])
        return "?"

    def last_drawn_code(p: int):
        """最近一次摸到牌的原始 tenhou 编码（用于摸切还原）。"""
        if draw_i[p] > 0 and draws[p]:
            v = draws[p][draw_i[p] - 1]
            if isinstance(v, int):
                return v
        return None

    def base_tile(code: int) -> int:
        """赤五(51/52/53)归一化到普通五(15/25/35)，便于比较鸣牌/打牌。"""
        return code - 36 if code in (51, 52, 53) else code

    def claim_matches_discard(call_str: str, caller: int,
                              discarder: int, discard_code: int) -> bool:
        """
        判断 draws 里的下一个副露串是否就是「针对当前这张打牌」的鸣牌。
        必须同时满足：
        1) 串内编码的被鸣者座位与当前打牌者一致；
        2) 被鸣的牌与实际打出的牌牌面一致（忽略赤五差异）。

        仅凭座位会误判：同一打牌者连续两张牌可能分别被不同玩家鸣，
        若只看座位会把针对后一张牌的鸣牌提前触发，吞掉中间的正常轮次。

        tensoul: relative_seating = (caller - feeder + 3) % 4
          rel=0 被鸣者是 caller 的上家 (feeder = caller-1)
          rel=1 对家 (feeder = caller+2)
          rel=2 下家 (feeder = caller+1)
        吃(c)永远只能吃上家；碰(p)标记位置即 rel；
        大明杠(m)标记位置 0/1 即 rel，位置 3 是 rel=2 的改写。
        """
        if discard_code is None:
            return False
        tokens = tokenize_call(call_str)
        for pos, tok in enumerate(tokens):
            if isinstance(tok, str) and tok in ("c", "p", "m") \
                    and pos + 1 < len(tokens):
                kind = tok
                rel_pos = pos
                break
        else:
            return False
        claimed = tokens[rel_pos + 1]
        if not isinstance(claimed, int):
            return False
        if base_tile(claimed) != base_tile(discard_code):
            return False
        if kind == "c":
            expected = (caller - 1) % 4
        elif kind == "p":
            expected = (caller + 3 - rel_pos) % 4
        else:  # m
            rel = 2 if rel_pos == 3 else rel_pos
            expected = (caller + 3 - rel) % 4
        return expected == discarder

    def find_caller(discarder: int, discard_code: int):
        """
        在 discarder 之后寻找吃/碰/大明杠的玩家。
        除了下一个 draws 槽位必须是副露串外，还要同时校验串内编码的
        被鸣者座位与被鸣牌牌面都匹配本次打牌，防止副露被提前触发。
        """
        for offset in range(1, 4):
            p = (discarder + offset) % 4
            if draw_i[p] < len(draws[p]):
                item = draws[p][draw_i[p]]
                if isinstance(item, str) and \
                        claim_matches_discard(item, p, discarder, discard_code):
                    return p
        return None

    def emit_dora_if_pending():
        """杠后下一次打牌出现时翻开新的宝牌指示牌。"""
        if state["pending_dora"]:
            state["kan_count"] += 1
            if state["kan_count"] < len(doras):
                events.append({"type": "dora",
                               "dora_marker": tile_to_mjai(doras[state["kan_count"]])})
            state["pending_dora"] = False

    def step_after_discard(discard_code):
        """处理完一次打牌后的流程：翻宝牌 -> 判断被叫 -> 推进。"""
        nonlocal current
        emit_dora_if_pending()
        caller = find_caller(current, discard_code)
        if caller is not None:
            call_str = draws[caller][draw_i[caller]]
            draw_i[caller] += 1
            call_ev = parse_call(call_str, caller)
            call_ev["target"] = current
            events.append(call_ev)
            current = caller
            if call_ev["type"] == "daiminkan":
                # 岭上摸牌
                if draw_i[current] < len(draws[current]):
                    events.append({"type": "tsumo", "actor": current,
                                   "pai": tile_to_mjai(draws[current][draw_i[current]])})
                    draw_i[current] += 1
                state["pending_dora"] = True
            # chi / pon / daiminkan 后由 caller 打牌，回到主循环
        else:
            nxt = (current + 1) % 4
            if draw_i[nxt] >= len(draws[nxt]):
                # 下家无牌可摸 -> 流局或荣和
                return False
            nxt_tile = draws[nxt][draw_i[nxt]]
            if not isinstance(nxt_tile, int):
                # 该摸牌了，槽位里却是未消费的副露串：说明鸣牌匹配逻辑
                # 与原始数据不一致，属于转换 bug，直接报错而非发出 pai="?"
                raise ValueError(
                    f"turn desync at seat {nxt} round {round_num}: "
                    f"expected draw, got {nxt_tile!r}")
            events.append({"type": "tsumo", "actor": nxt,
                           "pai": tile_to_mjai(nxt_tile)})
            draw_i[nxt] += 1
            current = nxt
        return True

    guard = 20000
    while guard > 0:
        guard -= 1

        if discard_i[current] >= len(discards[current]):
            # 当前玩家已摸但未打 -> 自摸和了 / 九种九牌等流局
            break

        d = discards[current][discard_i[current]]

        if d == 0:
            # tensoul ZeroSymbol：大明杠后 discards 里的占位符。
            # 杠事件与岭上摸牌已在 step_after_discard 中发出，这里直接跳过。
            discard_i[current] += 1
            continue

        if isinstance(d, int):
            if d == TSUMOGIRI_MARK:
                pai = last_drawn_tile(current)
                discard_code = last_drawn_code(current)
                events.append({"type": "dahai", "actor": current,
                               "pai": pai, "tsumogiri": True})
            else:
                discard_code = d
                events.append({"type": "dahai", "actor": current,
                               "pai": tile_to_mjai(d), "tsumogiri": False})
            discard_i[current] += 1
            if not step_after_discard(discard_code):
                break

        elif isinstance(d, str):
            if d.startswith("r"):
                rest = d[1:]
                if rest == "60" or rest == "":
                    pai = last_drawn_tile(current)
                    discard_code = last_drawn_code(current)
                    tsumogiri = True
                else:
                    discard_code = int(rest)
                    pai = tile_to_mjai(discard_code)
                    tsumogiri = False
                events.append({"type": "reach", "actor": current})
                events.append({"type": "dahai", "actor": current,
                               "pai": pai, "tsumogiri": tsumogiri})
                events.append({"type": "reach_accepted", "actor": current})
                discard_i[current] += 1
                if not step_after_discard(discard_code):
                    break
            elif "a" in d:
                ev = parse_call(d, current)
                events.append(ev)
                discard_i[current] += 1
                # 暗杠后岭上摸牌
                if draw_i[current] < len(draws[current]):
                    events.append({"type": "tsumo", "actor": current,
                                   "pai": tile_to_mjai(draws[current][draw_i[current]])})
                    draw_i[current] += 1
                state["pending_dora"] = True
            elif "k" in d:
                ev = parse_call(d, current)
                events.append(ev)
                discard_i[current] += 1
                # 加杠后岭上摸牌
                if draw_i[current] < len(draws[current]):
                    events.append({"type": "tsumo", "actor": current,
                                   "pai": tile_to_mjai(draws[current][draw_i[current]])})
                    draw_i[current] += 1
                state["pending_dora"] = True
            else:
                # 无法识别，跳过
                discard_i[current] += 1
        else:
            discard_i[current] += 1

    # ---------------- 结算 ---------------- #
    if result and isinstance(result, list) and result:
        tag = result[0]
        if tag == "和了":
            # tensoul 布局（支持双响/三响）：
            # ["和了", deltas1, agari1, deltas2, agari2, ...]
            # agari = [和牌者座位, 放铳/自摸座位, 包牌座位, 符飜, 役种...]
            k = 1
            while k + 1 < len(result):
                deltas_raw = result[k]
                ag = result[k + 1]
                k += 2
                if not isinstance(ag, list) or len(ag) < 2:
                    continue
                deltas = list(deltas_raw) if isinstance(deltas_raw, list) \
                    else [0, 0, 0, 0]
                deltas = (deltas + [0, 0, 0, 0])[:4]
                who = ag[0]
                from_who = ag[1]
                if not isinstance(who, int) or not isinstance(from_who, int):
                    continue
                if who == from_who:
                    pai = last_drawn_tile(who)
                else:
                    ld = discards[from_who][discard_i[from_who] - 1] \
                        if discard_i[from_who] > 0 else None
                    pai = _pai_from_discard(ld, from_who, draws, draw_i)
                hora: dict[str, Any] = {
                    "type": "hora", "actor": who, "target": from_who,
                    "pai": pai, "deltas": deltas,
                }
                if uras:
                    hora["ura_markers"] = [tile_to_mjai(u) for u in uras]
                events.append(hora)
        elif tag in RYUUKYOKU_TYPES:
            deltas = list(result[1]) if len(result) > 1 else [0, 0, 0, 0]
            tehais_r: list[list[str]] = [[] for _ in range(4)]
            if len(result) > 2 and isinstance(result[2], list):
                for hand in result[2]:
                    if isinstance(hand, list) and hand:
                        p = hand[0]
                        tiles = hand[1:]
                        if isinstance(p, int) and 0 <= p < 4:
                            tehais_r[p] = [tile_to_mjai(t) for t in tiles]
            events.append({"type": "ryukyoku", "tehais": tehais_r,
                           "deltas": deltas})

    events.append({"type": "end_kyoku"})
    return events


def _pai_from_discard(ld, player: int, draws, draw_i) -> str:
    """从打牌记录还原被荣和的牌。"""
    if ld is None:
        return "?"
    if isinstance(ld, int):
        if ld == TSUMOGIRI_MARK:
            if draw_i[player] > 0:
                return tile_to_mjai(draws[player][draw_i[player] - 1])
            return "?"
        return tile_to_mjai(ld)
    if isinstance(ld, str):
        if ld.startswith("r"):
            rest = ld[1:]
            if rest == "60" or rest == "":
                if draw_i[player] > 0:
                    return tile_to_mjai(draws[player][draw_i[player] - 1])
                return "?"
            return tile_to_mjai(int(rest))
        # 杠等，取最后一张
        toks = tokenize_call(ld)
        ints = [t for t in toks if isinstance(t, int)]
        return tile_to_mjai(ints[-1]) if ints else "?"
    return "?"


# --------------------------------------------------------------------------- #
# 整局牌谱转换
# --------------------------------------------------------------------------- #
def convert_game(data: dict) -> list[dict]:
    """把一整场 tenhou JSON 转换为 MJAI 事件列表。"""
    names = data.get("name") or ["P0", "P1", "P2", "P3"]
    if len(names) != 4:
        raise ValueError(
            f"仅支持四人麻将牌谱，检测到 {len(names)} 人（三麻不支持）: "
            f"{names}")
    events: list[dict] = [{"type": "start_game", "names": list(names)}]

    log = data.get("log") or []
    for rnd in log:
        if not isinstance(rnd, list) or not rnd:
            continue
        events.extend(convert_round(rnd))

    events.append({"type": "end_game"})
    return events


# --------------------------------------------------------------------------- #
# 文件夹批量处理
# --------------------------------------------------------------------------- #
def convert_file(in_path: str, out_path: str) -> int:
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = convert_game(data)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False))
            f.write("\n")
    return len(events)


def main():
    parser = argparse.ArgumentParser(
        description="将 tenhou.net/6 格式 JSON 牌谱批量转换为 MJAI 格式（JSON Lines）。")
    parser.add_argument("input_dir", help="包含 tenhou JSON 牌谱的文件夹")
    parser.add_argument("output_dir", help="输出 MJAI 文件的文件夹")
    parser.add_argument("--ext", default=".json",
                        help="输入文件扩展名（默认 .json）")
    args = parser.parse_args()

    in_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(in_dir):
        print(f"错误：输入文件夹不存在: {in_dir}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir)
             if f.lower().endswith(args.ext.lower())]
    if not files:
        print(f"在 {in_dir} 中未找到 {args.ext} 文件。", file=sys.stderr)
        sys.exit(1)

    ok = 0
    for fname in sorted(files):
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)
        try:
            n = convert_file(in_path, out_path)
            print(f"[OK]   {fname} -> {n} 个事件")
            ok += 1
        except Exception:  # noqa: BLE001
            if os.path.exists(out_path):
                os.remove(out_path)
            continue

    print(f"\n完成：成功转换 {ok} 个文件。")


if __name__ == "__main__":
    main()
