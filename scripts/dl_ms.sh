#!/bin/bash
cd /data/yaokang/evishare
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
~/.local/bin/uv pip install --python .venv/bin/python modelscope
for m in Qwen/Qwen2.5-7B-Instruct-AWQ BAAI/bge-base-en-v1.5 BAAI/bge-small-en-v1.5 Qwen/Qwen2.5-3B-Instruct-AWQ Qwen/Qwen2.5-14B-Instruct-AWQ; do
  n=$(basename $m)
  for t in 1 2 3; do .venv/bin/modelscope download --model $m --local_dir models/$n && echo "OK $m" && break; done
done
echo ALLDONE
