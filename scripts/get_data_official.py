"""Download PhysioNet/CinC 2019 training data from PhysioNet's official open-data bucket on AWS
(s3://physionet-open, public HTTPS), verify every file's MD5 against its S3 ETag, and write a
SHA-256 manifest so the exact data used can be cited and re-verified.

Output: data/raw_official/training_set{A,B}/*.psv and data/raw_official/MANIFEST.csv
"""
import csv
import hashlib
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

BUCKET = "https://physionet-open.s3.amazonaws.com"
PREFIX = "challenge-2019/1.0.0/training/"
OUT = "data/raw_official"


def _get(url):
    return subprocess.run(["curl", "-sSf", "--retry", "8", "--retry-all-errors", "--retry-delay", "2", url],
                          check=True, capture_output=True).stdout


def list_objects():
    objs, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={PREFIX}&max-keys=1000"
        if token:
            url += "&continuation-token=" + subprocess.run(
                ["python3", "-c", "import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=''))", token],
                capture_output=True, text=True).stdout.strip()
        xml = _get(url).decode()
        for block in re.findall(r"<Contents>(.*?)</Contents>", xml, re.S):
            key = re.search(r"<Key>(.*?)</Key>", block).group(1)
            if key.endswith(".psv"):
                etag = re.search(r"<ETag>&quot;(.*?)&quot;</ETag>|<ETag>\"(.*?)\"</ETag>", block)
                objs.append((key, (etag.group(1) or etag.group(2)), int(re.search(r"<Size>(\d+)</Size>", block).group(1))))
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m:
            return objs
        token = m.group(1)


def fetch(obj):
    key, etag, size = obj
    path = os.path.join(OUT, key[len(PREFIX):])
    if not (os.path.exists(path) and os.path.getsize(path) == size):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = _get(f"{BUCKET}/{key}")
        with open(path, "wb") as f:
            f.write(data)
    data = open(path, "rb").read()
    md5 = hashlib.md5(data).hexdigest()
    if "-" not in etag and md5 != etag:
        raise RuntimeError(f"MD5 mismatch for {key}: {md5} != {etag}")
    return key[len(PREFIX):], size, md5, hashlib.sha256(data).hexdigest()


def main():
    objs = list_objects()
    print(f"listed {len(objs)} .psv objects", flush=True)
    with ThreadPoolExecutor(16) as ex:
        rows = sorted(ex.map(fetch, objs))
    with open(os.path.join(OUT, "MANIFEST.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "bytes", "md5", "sha256"])
        w.writerows(rows)
    digest = hashlib.sha256("".join(r[3] for r in rows).encode()).hexdigest()
    print(f"verified {len(rows)} files against S3 ETags; dataset digest (sha256 of ordered file sha256s): {digest}")
    return 0 if len(rows) == 40336 else 1


if __name__ == "__main__":
    sys.exit(main())
