from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

import prelude
from libriichi.dataset import GameplayLoader
#cd Mortal-main\mortal
#python export_decision_points.py "D:\Project-list\guoqie\dataset\mjai" -o "D:\Project-list\guoqie\output\huolongguo" --version 4 --player-names "むほうむてん"


ACTION_NAMES = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
    "5mr", "5pr", "5sr",
    "riichi",
    "chi_low", "chi_mid", "chi_high",
    "pon",
    "kan",
    "hora",
    "ryukyoku",
    "pass",
]

assert len(ACTION_NAMES) == 46


def _is_gz(path: Path) -> bool:
    return path.name.lower().endswith(".json.gz")


def _kana_to_hira(text: str) -> str:
    """Fold katakana (U+30A1..U+30F6) to hiragana for name comparison."""
    return "".join(
        chr(ord(ch) - 0x60) if 0x30A1 <= ord(ch) <= 0x30F6 else ch
        for ch in text
    )


def _norm_name(name: str) -> str:
    """Loose name comparison key.

    NFKC folds full/half-width variants, katakana is folded to hiragana,
    surrounding whitespace is ignored and ASCII case is folded, so minor
    spelling differences still match.
    """
    return _kana_to_hira(unicodedata.normalize("NFKC", name)).strip().casefold()


def _read_start_game_names(path: Path) -> list[str]:
    """Read the names array from the leading start_game event of an mjai log."""
    opener = gzip.open if _is_gz(path) else open
    with opener(path, "rt", encoding="utf-8") as f:
        for _ in range(16):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "start_game":
                return [str(n) for n in event.get("names", [])]
    return []


def _expand_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        candidates: list[Path]
        if path.exists():
            if path.is_dir():
                candidates = sorted(path.rglob("*.json")) + sorted(path.rglob("*.json.gz"))
            else:
                candidates = [path]
        else:
            candidates = [Path(p) for p in glob.glob(raw, recursive=True)] #适配通配符

        for candidate in candidates:
            if candidate in seen:
                continue
            if candidate.name.lower().endswith((".json", ".json.gz")):
                files.append(candidate)
                seen.add(candidate)
    return files


def _load_games(loader: GameplayLoader, path: Path):
    if _is_gz(path):
        return loader.load_gz_log_files([str(path)])[0]
    raw = path.read_text(encoding="utf-8")
    return loader.load_log(raw)


def _stack_or_empty(items: list[np.ndarray], dtype: np.dtype[Any] | type, shape: tuple[int, ...]):
    if not items:
        return np.empty((0, *shape), dtype=dtype)
    return np.stack(items, axis=0).astype(dtype, copy=False)


def export_decision_points(args: argparse.Namespace) -> Path:
    inputs = _expand_inputs(args.inputs)
    if not inputs:
        raise SystemExit("no input files matched")

    # Resolve requested player names against the names actually stored in each
    # log. GameplayLoader filters by exact spelling, so we collect the exact
    # in-log spellings here; loose matching (NFKC / whitespace / case) is
    # applied only for comparison. Files without a match are skipped with a
    # warning instead of silently contributing zero samples.
    selected: list[tuple[Path, list[str] | None]] = []
    seen_names: set[str] = set()
    if args.player_names:
        wanted = {_norm_name(n) for n in args.player_names}
        for path in inputs:
            try:
                names = _read_start_game_names(path)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] cannot read start_game names from {path}: {exc}",
                      file=sys.stderr)
                continue
            seen_names.update(names)
            hits = [n for n in names if _norm_name(n) in wanted]
            if hits:
                selected.append((path, hits))
            else:
                print(
                    f"[SKIP] no player in {path.name} matches "
                    f"{args.player_names}; names in log: {names or '(none)'}",
                    file=sys.stderr,
                )
        if not selected:
            raise SystemExit(
                "no log contains any of the requested player names "
                f"{args.player_names}. Names seen across inputs: "
                f"{sorted(seen_names) or '(none)'}. For anonymous logs "
                "(e.g. tenhou logs with placeholder names), use --player-id "
                "instead."
            )
        # Exact spellings as they appear inside the logs, for Rust-side filtering.
        exact_names = sorted({name for _, hits in selected for name in hits})
    else:
        selected = [(path, None) for path in inputs]
        exact_names = None

    oracle = args.oracle or args.include_invisible_obs
    loader = GameplayLoader(
        version=args.version,
        oracle=oracle,
        player_names=exact_names,
        excludes=args.excludes,
        trust_seed=args.trust_seed,
        always_include_kan_select=args.always_include_kan_select,
        augmented=args.augmented,
    )

    obs_list: list[np.ndarray] = []
    invisible_obs_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    action_list: list[int] = []
    player_id_list: list[int] = []
    file_index_list: list[int] = []
    game_index_list: list[int] = []
    sample_index_list: list[int] = []
    at_kyoku_list: list[int] = []
    at_turn_list: list[int] = []
    shanten_list: list[int] = []
    done_list: list[bool] = []
    apply_gamma_list: list[bool] = []
    sample_records: list[dict[str, Any]] = []

    total_games = 0
    total_samples = 0

    for file_idx, (path, _matched_names) in enumerate(selected):
        try:
            games = _load_games(loader, path)
        except BaseException as exc:  # noqa: BLE001
            print(f"[SKIP] failed to parse {path.name}: {exc}", file=sys.stderr)
            continue
        for game_idx, game in enumerate(games):
            player_id = int(game.take_player_id())
            if args.player_id is not None and player_id != args.player_id:
                continue

            obs = game.take_obs()
            masks = game.take_masks()
            actions = game.take_actions()
            at_kyoku = game.take_at_kyoku()
            dones = game.take_dones()
            apply_gamma = game.take_apply_gamma()
            at_turns = game.take_at_turns()
            shantens = game.take_shantens()
            invisible_obs = game.take_invisible_obs() if oracle else []

            total_games += 1
            for sample_idx, action in enumerate(actions):
                mask_arr = np.asarray(masks[sample_idx], dtype=np.bool_)
                legal_action_ids = np.flatnonzero(mask_arr).tolist()
                action_id = int(action)
                obs_list.append(np.asarray(obs[sample_idx], dtype=np.float32))
                mask_list.append(mask_arr)
                action_list.append(action_id)
                player_id_list.append(player_id)
                file_index_list.append(file_idx)
                game_index_list.append(game_idx)
                sample_index_list.append(sample_idx)
                at_kyoku_list.append(int(at_kyoku[sample_idx]))
                at_turn_list.append(int(at_turns[sample_idx]))
                shanten_list.append(int(shantens[sample_idx]))
                done_list.append(bool(dones[sample_idx]))
                apply_gamma_list.append(bool(apply_gamma[sample_idx]))
                if oracle:
                    invisible_obs_list.append(np.asarray(invisible_obs[sample_idx], dtype=np.float32))
                sample_records.append({
                    "file_index": file_idx,
                    "game_index": game_idx,
                    "sample_index": sample_idx,
                    "player_id": player_id,
                    "at_kyoku": int(at_kyoku[sample_idx]),
                    "at_turn": int(at_turns[sample_idx]),
                    "shanten": int(shantens[sample_idx]),
                    "done": bool(dones[sample_idx]),
                    "apply_gamma": bool(apply_gamma[sample_idx]),
                    "action": action_id,
                    "action_name": ACTION_NAMES[action_id],
                    "legal_action_ids": legal_action_ids,
                    "legal_action_names": [ACTION_NAMES[i] for i in legal_action_ids],
                })
                total_samples += 1

    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if total_samples == 0:
        print(
            "[WARN] 0 decision points exported (check --player-id / --player-names filters); "
            "an empty dataset will still be written.",
            file=sys.stderr,
        )

    obs = _stack_or_empty(obs_list, np.float32, obs_list[0].shape if obs_list else (0, 0))
    masks = _stack_or_empty(mask_list, np.bool_, mask_list[0].shape if mask_list else (0,))
    action = np.asarray(action_list, dtype=np.int64)
    player_id = np.asarray(player_id_list, dtype=np.int64)
    file_index = np.asarray(file_index_list, dtype=np.int64)
    game_index = np.asarray(game_index_list, dtype=np.int64)
    sample_index = np.asarray(sample_index_list, dtype=np.int64)
    at_kyoku = np.asarray(at_kyoku_list, dtype=np.int64)
    at_turn = np.asarray(at_turn_list, dtype=np.int64)
    shanten = np.asarray(shanten_list, dtype=np.int64)
    done = np.asarray(done_list, dtype=np.bool_)
    apply_gamma = np.asarray(apply_gamma_list, dtype=np.bool_)

    payload = {
        "obs": obs,
        "masks": masks,
        "action": action,
        "player_id": player_id,
        "file_index": file_index,
        "game_index": game_index,
        "sample_index": sample_index,
        "at_kyoku": at_kyoku,
        "at_turn": at_turn,
        "shanten": shanten,
        "done": done,
        "apply_gamma": apply_gamma,
    }
    if oracle:
        payload["invisible_obs"] = _stack_or_empty(
            invisible_obs_list,
            np.float32,
            invisible_obs_list[0].shape if invisible_obs_list else (0, 0),
        )

    np.savez_compressed(out_prefix.with_suffix(".npz"), **payload)

    out_prefix.with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in sample_records)
        + ("\n" if sample_records else ""),
        encoding="utf-8",
    )

    manifest = {
        "version": args.version,
        "oracle": oracle,
        "player_id": args.player_id,
        "player_names": args.player_names,
        "resolved_player_names": exact_names,
        "excludes": args.excludes,
        "augmented": args.augmented,
        "trust_seed": args.trust_seed,
        "always_include_kan_select": args.always_include_kan_select,
        "sample_count": total_samples,
        "game_count": total_games,
        "action_names": ACTION_NAMES,
        "source_files": [str(p) for p, _ in selected],
        "skipped_files": [str(p) for p in inputs if p not in {p for p, _ in selected}],
    }
    out_prefix.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"exported {total_samples} decision points / {total_games} games "
        f"from {len(selected)} file(s) -> {out_prefix.with_suffix('.npz')}"
    )
    if exact_names is not None:
        print(f"matched in-log player name(s): {exact_names}")
    return out_prefix


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export per-decision mjai samples for BC / imitation learning.",
    )
    parser.add_argument("inputs", nargs="+", help="Input .json/.json.gz files, directories, or glob patterns.")
    parser.add_argument("-o", "--output", required=True, help="Output prefix, e.g. data/player0_decisions")
    parser.add_argument("--version", type=int, required=True, help="Observation version passed to GameplayLoader.")
    parser.add_argument("--player-id", type=int, default=None, choices=range(4), help="Keep only one player id.")
    parser.add_argument("--player-names", nargs="*", default=None, help="Whitelist of player names; matched loosely (NFKC, surrounding whitespace, ASCII case ignored).")
    parser.add_argument("--excludes", nargs="*", default=None, help="Optional blacklist of player names.")
    parser.add_argument("--oracle", action="store_true", help="Also export oracle invisible observations.")
    parser.add_argument("--include-invisible-obs", action="store_true", help="Alias of --oracle for convenience.")
    parser.add_argument("--trust-seed", action="store_true", help="Trust seed in StartGame when building invisible states.")
    parser.add_argument("--always-include-kan-select", dest="always_include_kan_select", action="store_true", help="Keep kan-select samples when a kan choice exists.")
    parser.add_argument("--no-always-include-kan-select", dest="always_include_kan_select", action="store_false")
    parser.set_defaults(always_include_kan_select=True)
    parser.add_argument("--augmented", action="store_true", help="Augment tiles with red/normal equivalents.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    export_decision_points(args)


if __name__ == "__main__":
    main()
