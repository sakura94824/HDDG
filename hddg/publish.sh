#!/usr/bin/env bash
# Publish HDDG to https://github.com/sakura94824/HDDG
set -euo pipefail

cd "$(dirname "$0")"

if ! gh auth status >/dev/null 2>&1; then
  echo "请先登录 GitHub："
  echo "  gh auth login"
  exit 1
fi

# 若远程仓库尚不存在则创建；已存在则跳过
if ! gh repo view sakura94824/HDDG >/dev/null 2>&1; then
  gh repo create sakura94824/HDDG \
    --public \
    --source=. \
    --remote=origin \
    --description "HDDG: Hilbert-guided multi-graph network for EEG emotion recognition"
else
  echo "仓库 sakura94824/HDDG 已存在，直接推送..."
  git remote set-url origin https://github.com/sakura94824/HDDG.git
fi

git push -u origin main

echo ""
echo "Done: https://github.com/sakura94824/HDDG"
