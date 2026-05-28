"""
CSV 데이터셋 전체 이미지 유효성 검사 스크립트 (고속 버전).

- 파일 magic bytes만 확인 (파일 전체 로드 없음) → 수십 배 빠름
  * npy : b'\x93NUMPY' (6 bytes)
  * jpg : b'\xff\xd8' (2 bytes)
  * png : b'\x89PNG\r\n\x1a\n' (8 bytes)
- 손상/열 수 없는 파일 목록을 <csv>.bad.txt 로 저장
- 멀티프로세스 병렬 처리 (기본 32 workers)
- 중간에 죽어도 bad_files.txt에 지금까지 결과 유지 (line-buffered write)
- 검사 완료 후 CSV에서 bad 파일 제거한 clean CSV 자동 생성
"""

import argparse
import csv
import multiprocessing as mp
import os

_MAGIC = {
    "npy":  b'\x93NUMPY',
    "jpg":  b'\xff\xd8',
    "jpeg": b'\xff\xd8',
    "png":  b'\x89PNG\r\n\x1a\n',
}
_MAGIC_LEN = {k: len(v) for k, v in _MAGIC.items()}
_MIN_SIZE = 256


def check_one(args):
    path, fmt = args
    fmt = fmt.lower()
    try:
        size = os.path.getsize(path)
        if size < _MIN_SIZE:
            return (path, False, f"too_small:{size}")
        magic = _MAGIC.get(fmt)
        if magic is not None:
            mlen = _MAGIC_LEN[fmt]
            with open(path, 'rb') as f:
                header = f.read(mlen)
            if header != magic:
                return (path, False, f"bad_magic:{header[:8].hex()}")
        return (path, True, "")
    except FileNotFoundError:
        return (path, False, "not_found")
    except Exception as e:
        return (path, False, type(e).__name__ + ":" + str(e)[:60])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--bad-out", default=None)
    parser.add_argument("--clean-csv", default=None)
    args = parser.parse_args()

    csv_path = args.csv
    bad_out = args.bad_out or (csv_path + ".bad.txt")
    clean_csv = args.clean_csv or (csv_path.replace(".csv", "_clean.csv"))

    print(f"[*] CSV 읽는 중: {csv_path}", flush=True)
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    total = len(rows)
    print(f"[*] 총 {total:,}개 파일 검사 시작 (workers={args.workers})", flush=True)

    already_bad = set()
    if os.path.exists(bad_out):
        with open(bad_out) as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    already_bad.add(parts[0])
        print(f"[*] 이전 bad 목록 {len(already_bad):,}개 로드", flush=True)

    tasks = [(row["path"], row["fmt"]) for row in rows
             if row["path"] not in already_bad]
    print(f"[*] 새로 검사할 파일: {len(tasks):,}개", flush=True)

    bad_new = []

    if tasks:
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        bad_file_handle = open(bad_out, "a", buffering=1)

        with mp.Pool(processes=args.workers) as pool:
            if use_tqdm:
                it = tqdm(pool.imap_unordered(check_one, tasks, chunksize=512),
                          total=len(tasks), ncols=100, unit="img")
            else:
                it = pool.imap_unordered(check_one, tasks, chunksize=512)

            done = 0
            for path, ok, reason in it:
                done += 1
                if not ok:
                    bad_new.append((path, reason))
                    bad_file_handle.write(f"{path}\t{reason}\n")
                if not use_tqdm and done % 20000 == 0:
                    pct = done / len(tasks) * 100
                    print(f"  {done:,}/{len(tasks):,} ({pct:.1f}%)  bad={len(bad_new)}", flush=True)

        bad_file_handle.close()
        total_bad = len(already_bad) + len(bad_new)
        print(f"\n[결과] 전체 bad: {total_bad:,}개", flush=True)

    all_bad_paths = already_bad | {p for p, _ in bad_new}
    clean_rows = [row for row in rows if row["path"] not in all_bad_paths]
    with open(clean_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)
    removed = total - len(clean_rows)
    print(f"[*] clean CSV 저장: {clean_csv}", flush=True)
    print(f"    원본 {total:,}개 → clean {len(clean_rows):,}개 (제거 {removed:,}개)", flush=True)


if __name__ == "__main__":
    main()
