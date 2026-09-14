#!/usr/bin/env python3
"""
把 download_browser.mjs 从浏览器 WS 帧里截获的原始牌谱，
转换成 tenhou.net/6 JSON（与 ninklang/tensoul 输出格式一致）。

输入目录中每个牌谱是一对文件：
    <uuid>.head.bin   ResGameRecord.head 字段的原始 protobuf 字节（lq.RecordGame）
    <uuid>.data.bin   牌谱体原始字节（lq.Wrapper{name=".lq.GameDetailRecords", ...}，
                      若来自 data_url 则是 gunzip 之后的同样结构）

输出：
    <输出目录>/paipu-<uuid>.json

用法：
    python convert_paipu.py paipu_raw paipu_json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor"
sys.path.insert(0, str(VENDOR))

import ms.protocol_pb2 as pb  # noqa: E402
from tensoul.cfg import cfg  # noqa: E402
from tensoul.constants import RUNES, JPNAME  # noqa: E402
from tensoul.parser import MajsoulPaipuParser  # noqa: E402


def record_to_tenhou(head: "pb.RecordGame", data: bytes) -> dict:
    """tensoul.downloader.MajsoulPaipuDownloader._handle_game_record 的离线等价实现。"""
    res = {}
    ruledisp = ""
    lobby = ""
    nplayers = len(head.result.players)
    nakas = nplayers - 1
    tsumoloss_off = False

    res["ver"] = "2.3"
    res["ref"] = head.uuid
    res["ratingc"] = f"PF{nplayers}"

    try:
        if nplayers == 3:
            ruledisp += RUNES["sanma"][JPNAME]
        if head.config.meta.mode_id:
            ruledisp += cfg["desktop"]["matchmode"]["map_"][str(head.config.meta.mode_id)]["room_name_jp"]
        elif head.config.meta.room_id:
            lobby = f": {head.config.meta.room_id}"
            ruledisp += RUNES["friendly"][JPNAME]
            nakas = head.config.mode.detail_rule.dora_count
            tsumoloss_off = nplayers == 3 and not head.config.mode.detail_rule.have_zimosun
        elif head.config.meta.contest_uid:
            lobby = f": {head.config.meta.contest_uid}"
            ruledisp += RUNES["tournament"][JPNAME]
            nakas = head.config.mode.detail_rule.dora_count
            tsumoloss_off = nplayers == 3 and not head.config.mode.detail_rule.have_zimosun

        if head.config.mode.mode == 1:
            ruledisp += RUNES["tonpuu"][JPNAME]
        elif head.config.mode.mode == 2:
            ruledisp += RUNES["hanchan"][JPNAME]

        if head.config.meta.mode_id == 0 and head.config.mode.detail_rule.dora_count == 0:
            res["rule"] = {"disp": ruledisp, "aka53": 0, "aka52": 0, "aka51": 0}
        else:
            res["rule"] = {"disp": ruledisp, "aka53": 1, "aka52": 2 if nakas == 4 else 1,
                           "aka51": 1 if nplayers == 4 else 0}
    except (KeyError, AttributeError):
        # 新版本引入了 cfg.json 里还没有的模式 id，保证转换不崩，仅缺模式文案
        res["rule"] = {"disp": ruledisp, "aka53": 1, "aka52": 2 if nakas == 4 else 1,
                       "aka51": 1 if nplayers == 4 else 0}

    res["lobby"] = 0

    res["dan"] = [""] * nplayers
    for e in head.accounts:
        try:
            res["dan"][e.seat] = cfg["level_definition"]["level_definition"]["map_"][str(e.level.id)]["full_name_jp"]
        except (KeyError, IndexError):
            res["dan"][e.seat] = ""

    res["rate"] = [0] * nplayers
    for e in head.accounts:
        res["rate"][e.seat] = e.level.score

    res["sx"] = ["C"] * nplayers

    res["name"] = ["AI"] * nplayers
    for e in head.accounts:
        res["name"][e.seat] = e.nickname

    scores = [[e.seat, e.part_point_1, e.total_point / 1000] for e in head.result.players]
    res["sc"] = [0] * nplayers * 2
    for seat, part, total in scores:
        res["sc"][2 * seat] = part
        res["sc"][2 * seat + 1] = total

    res["title"] = [ruledisp + lobby, datetime.fromtimestamp(head.end_time).strftime("%Y-%m-%d %H:%M:%S")]

    wrapper = pb.Wrapper()
    wrapper.ParseFromString(data)

    details = pb.GameDetailRecords()
    details.ParseFromString(wrapper.data)

    converter = MajsoulPaipuParser(tsumoloss_off=tsumoloss_off)
    res["log"] = []
    if details.version < 210715 and len(details.records) > 0:
        for rec in details.records:
            round_record_wrapper = pb.Wrapper()
            round_record_wrapper.ParseFromString(rec)
            log = getattr(pb, round_record_wrapper.name[len(".lq."):])()
            log.ParseFromString(round_record_wrapper.data)
            converter.feed(log)
            res["log"] = [e.dump() for e in converter.getvalue()]
    else:
        for act in details.actions:
            if len(act.result) != 0:
                round_record_wrapper = pb.Wrapper()
                round_record_wrapper.ParseFromString(act.result)
                log = getattr(pb, round_record_wrapper.name[len(".lq."):])()
                log.ParseFromString(round_record_wrapper.data)
                converter.feed(log)
                res["log"] = [e.dump() for e in converter.getvalue()]

    return res


def convert_pair(head_path: Path, data_path: Path, out_path: Path) -> int:
    head = pb.RecordGame()
    head.ParseFromString(head_path.read_bytes())
    tenhou = record_to_tenhou(head, data_path.read_bytes())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(tenhou, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, out_path)
    return len(tenhou.get("log") or [])


def main():
    parser = argparse.ArgumentParser(description="原始雀魂牌谱字节 -> tenhou.net/6 JSON")
    parser.add_argument("raw_dir", help="download_browser.mjs 输出的 .head.bin/.data.bin 目录")
    parser.add_argument("out_dir", help="tenhou JSON 输出目录")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    if not raw_dir.is_dir():
        print(f"输入目录不存在: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    heads = sorted(raw_dir.glob("*.head.bin"))
    if not heads:
        print(f"{raw_dir} 下没有 *.head.bin 文件", file=sys.stderr)
        sys.exit(1)

    ok = fail = 0
    for head_path in heads:
        uuid = head_path.name[: -len(".head.bin")]
        data_path = raw_dir / f"{uuid}.data.bin"
        if not data_path.exists():
            print(f"[跳过] {uuid}：缺少 {data_path.name}")
            fail += 1
            continue
        try:
            n = convert_pair(head_path, data_path, out_dir / f"paipu-{uuid}.json")
            print(f"[OK]   {uuid} -> {n} 局")
            ok += 1
        except Exception as exc:  # 单局失败不影响整批
            print(f"[失败] {uuid}: {type(exc).__name__}: {exc}", file=sys.stderr)
            fail += 1

    print(f"\n完成：成功 {ok}，失败 {fail}，共 {len(heads)} 个文件。")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
