"""第1回 宿題（発展）: 応答時間とトークン数を計測し、モデル・質問ごとに比較する。

使い方:
    # モデル×質問を順番に計測して CSV に追記（無料枠の RPM を超えないよう間隔を空ける）
    uv run python scripts/bench_ask.py --models gemini-3.5-flash-lite --repeat 2 --interval 5

    # 同じ質問を同時に N 件送り、レート制限（429）がいつ・どの枠で起きるかを観察
    uv run python scripts/bench_ask.py --models gemini-3.5-flash-lite --burst 20

    # これまでの CSV を集計して Markdown の表を出す（API は呼ばない）
    uv run python scripts/bench_ask.py --report

計測方法:
    generate_content_stream で受け取り、最初の本文が届くまで（TTFT）と最後まで（応答時間）を測る。
    トークン数は最後のチャンクの usage_metadata（入力 / 出力 / 思考 / 合計）。
"""

import argparse
import csv
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import errors

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "lectures" / "01" / "bench_results.csv"

# 出力の長さが違う質問を用意する（短い / 中くらい / 講座固有 / 長い）
QUESTIONS = {
    "short": "日本の首都はどこですか？一言で答えてください。",
    "lora": "LoRAとは何ですか？",
    "course": "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
    "long": "Transformerの仕組みを、初学者向けに見出しを付けて詳しく説明してください。",
}

FIELDS = [
    "timestamp", "mode", "model", "question", "status",
    "latency_s", "ttft_s", "input_tokens", "output_tokens", "thought_tokens",
    "total_tokens", "output_chars", "error",
]


def get_client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("環境変数 GEMINI_API_KEY が見つかりません（第1回 演習の手順3を参照）")
    return genai.Client(api_key=key)


def quota_info(e: errors.APIError) -> str:
    """429 のレスポンスから、どの枠（RPM / TPM / RPD）に当たったかと再試行までの待ち時間を取り出す。"""
    found = []
    details = e.details.get("error", {}).get("details", []) if isinstance(e.details, dict) else []
    for d in details:
        kind = d.get("@type", "")
        if kind.endswith("QuotaFailure"):
            for v in d.get("violations", []):
                found.append(f"{v.get('quotaId')} limit={v.get('quotaValue')}")
        elif kind.endswith("RetryInfo"):
            found.append(f"retry={d.get('retryDelay')}")
    return "; ".join(found) or (e.message or "")[:120]


def measure(client: genai.Client, model: str, qid: str, question: str, mode: str) -> dict:
    row = {f: "" for f in FIELDS}
    row.update(timestamp=datetime.now().isoformat(timespec="seconds"), mode=mode,
               model=model, question=qid, status="ok")
    usage, ttft, chars = None, None, 0
    t0 = time.perf_counter()
    try:
        for chunk in client.models.generate_content_stream(model=model, contents=question):
            for cand in chunk.candidates or []:
                for part in (cand.content.parts if cand.content else None) or []:
                    if part.text and not part.thought:  # 思考（thought）は本文に数えない
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        chars += len(part.text)
            if chunk.usage_metadata:
                usage = chunk.usage_metadata
    except errors.APIError as e:
        row.update(status=str(e.code), error=quota_info(e))
    row["latency_s"] = round(time.perf_counter() - t0, 2)
    row["ttft_s"] = round(ttft, 2) if ttft is not None else ""
    row["output_chars"] = chars
    if usage:
        row.update(input_tokens=usage.prompt_token_count or 0,
                   output_tokens=usage.candidates_token_count or 0,
                   thought_tokens=usage.thoughts_token_count or 0,
                   total_tokens=usage.total_token_count or 0)
    return row


def save(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def show(row: dict) -> None:
    print(f"{row['model']:<24} {row['question']:<7} {row['status']:>4} "
          f"{row['latency_s']:>6}s ttft={row['ttft_s']:<5} "
          f"in={row['input_tokens'] or '-'} out={row['output_tokens'] or '-'} "
          f"think={row['thought_tokens'] or '-'} {row['error']}", flush=True)


def run_sequential(client, models, qids, repeat, interval, out) -> None:
    for model in models:
        for qid in qids:
            for _ in range(repeat):
                row = measure(client, model, qid, QUESTIONS[qid], "seq")
                show(row)
                save([row], out)
                time.sleep(interval)


def run_burst(client, models, n, out) -> None:
    for model in models:
        with ThreadPoolExecutor(max_workers=n) as ex:
            rows = list(ex.map(lambda _: measure(client, model, "short", QUESTIONS["short"], "burst"), range(n)))
        for row in rows:
            show(row)
        save(rows, out)
        ok = sum(r["status"] == "ok" for r in rows)
        print(f"[burst] {model}: {n}件同時 → 成功 {ok} / 失敗 {n - ok}\n")


def mean(values, digits: int = 2) -> str:
    nums = [float(v) for v in values if v not in ("", None)]
    return f"{statistics.mean(nums):.{digits}f}" if nums else "-"


def report(out: Path) -> None:
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    seq = [r for r in rows if r["mode"] == "seq" and r["status"] == "ok"]
    print("| モデル | 質問 | n | 応答時間[s] | TTFT[s] | 入力tok | 出力tok | 思考tok | 合計tok | 出力tok/s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    keys = sorted({(r["model"], r["question"]) for r in seq},
                  key=lambda k: (k[0], list(QUESTIONS).index(k[1]) if k[1] in QUESTIONS else 99))
    for model, qid in keys:
        g = [r for r in seq if r["model"] == model and r["question"] == qid]
        speed = [(int(r["output_tokens"] or 0) + int(r["thought_tokens"] or 0)) / float(r["latency_s"])
                 for r in g if float(r["latency_s"]) > 0]
        print(f"| {model} | {qid} | {len(g)} | {mean(r['latency_s'] for r in g)} | "
              f"{mean(r['ttft_s'] for r in g)} | {mean((r['input_tokens'] for r in g), 0)} | "
              f"{mean((r['output_tokens'] for r in g), 0)} | {mean((r['thought_tokens'] for r in g), 0)} | "
              f"{mean((r['total_tokens'] for r in g), 0)} | {mean(speed, 1)} |")
    errs = [r for r in rows if r["status"] != "ok"]
    if errs:
        print("\n| モード | モデル | ステータス | 件数 | 内容 |\n|---|---|---|---|---|")
        for key in sorted({(r["mode"], r["model"], r["status"], r["error"]) for r in errs}):
            n = sum((r["mode"], r["model"], r["status"], r["error"]) == key for r in errs)
            print(f"| {key[0]} | {key[1]} | {key[2]} | {n} | {key[3]} |")


def main() -> None:
    p = argparse.ArgumentParser(description="応答時間とトークン数を計測する")
    p.add_argument("--models", nargs="+", default=[os.getenv("AIE_MODEL", "gemini-3.5-flash-lite")])
    p.add_argument("--questions", nargs="+", default=list(QUESTIONS), choices=list(QUESTIONS))
    p.add_argument("--repeat", type=int, default=2, help="同じ質問を何回ずつ送るか")
    p.add_argument("--interval", type=float, default=13.0,
                   help="リクエスト間の待ち秒数（既定13秒 = 5 RPM のモデルでも超えない間隔）")
    p.add_argument("--burst", type=int, default=0, help="同時に送るリクエスト数（レート制限の観察用）")
    p.add_argument("--report", action="store_true", help="CSV を集計して表を出す（API は呼ばない）")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if args.report:
        report(args.out)
        return
    client = get_client()
    if args.burst:
        run_burst(client, args.models, args.burst, args.out)
    else:
        run_sequential(client, args.models, args.questions, args.repeat, args.interval, args.out)


if __name__ == "__main__":
    main()
