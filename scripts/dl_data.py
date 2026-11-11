import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
D = '/data/yaokang/evishare/data/hf'
for rid, pat in [('hotpotqa/hotpot_qa', ['distractor/*']), ('dgslibisey/MuSiQue', None)]:
    for t in range(5):
        try:
            p = snapshot_download(repo_id=rid, repo_type='dataset', local_dir=os.path.join(D, rid.split('/')[-1]), allow_patterns=pat, max_workers=8)
            print('OK', rid, p, flush=True); break
        except Exception as e:
            print('ERR', rid, t, e, flush=True)
