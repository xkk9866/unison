import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
for rid in ['BAAI/bge-base-en-v1.5', 'BAAI/bge-small-en-v1.5']:
    for t in range(5):
        try:
            p = snapshot_download(repo_id=rid, local_dir='/data/yaokang/evishare/models/' + rid.split('/')[-1] + '-hf',
                                  allow_patterns=['*.json', '*.txt', 'model.safetensors', '1_Pooling/*', '*.model'], max_workers=4)
            print('OK', rid, flush=True); break
        except Exception as e:
            print('ERR', rid, e, flush=True)
