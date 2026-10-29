"""Opt-in fresh rerun of Olivia Cognetti's original experimental protocol.

Requires Hugging Face model downloads, an appropriate GPU, a Jev API key and
network access. Writes RAW benchmark prompts and model responses to a PRIVATE
folder. Never publish this output without a separate release review.

The notebook is the authoritative historical record. This is a portable
refactoring of its key data, generation, moderation and scoring steps.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from .protocol import (MODEL_ID, MAX_NEW_TOKENS, BATCH_SIZE, JEV_MODEL, JEV_URL,
                       JEV_WORKERS, SEED, QUESTIONS, STOPS, format_prompt,
                       llm_refused, jev_payload, classify, TIER_LOW, TIER_HIGH)
from .reproduction_targets import QWEN_REVISION, JEV_MODEL_ID


def save_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def load_prompts(root: Path, revision=None):
    cache=root/'prompt_lists.json'
    if cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))
    from datasets import load_dataset
    def load(name, split):
        last=None
        for attempt in range(1,7):
            try:
                return load_dataset(name, split=split, revision=revision)
            except Exception as exc:
                last=exc
                print(f'Dataset retry {attempt}/6 for {name}: {exc!r}')
                time.sleep(15*attempt)
        raise RuntimeError(f'Could not load {name}: {last!r}')
    adv=load('walledai/AdvBench','train')
    xs=load('walledai/XSTest','test')
    data={'adv':[row['prompt'] for row in adv],
          'safe':[row['prompt'] for row in xs if row['label']=='safe'],
          'unsafe':[row['prompt'] for row in xs if row['label']!='safe']}
    save_json(cache,data)
    return data


def generate_responses(root: Path, prompts: list[str], model_revision: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    path=root/'gen_cache.json'
    cache=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    key=lambda p: f'{MODEL_ID}|{model_revision}|{MAX_NEW_TOKENS}|{p}'
    todo=[p for p in prompts if key(p) not in cache]
    if todo:
        tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=model_revision,padding_side='left')
        if tok.pad_token is None: tok.pad_token=tok.eos_token
        # Historical notebook used float16 and device_map="auto"; CUDA is recommended.
        try:
            model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.float16,
                                                       device_map='auto',revision=model_revision)
        except TypeError:
            model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16,
                                                       device_map='auto',revision=model_revision)
        for start in range(0,len(todo),BATCH_SIZE):
            chunk=todo[start:start+BATCH_SIZE]
            enc=tok([format_prompt(p) for p in chunk],return_tensors='pt',padding=True).to(model.device)
            with torch.no_grad():
                gen=model.generate(**enc,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for p,g in zip(chunk,gen):
                response=tok.decode(g[enc.input_ids.shape[1]:],skip_special_tokens=True)
                for stop in STOPS:response=response.split(stop)[0]
                cache[key(p)]=response.strip()
            save_json(path,cache)
            print(f'Generated {min(start+BATCH_SIZE,len(todo))}/{len(todo)} new responses')
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return {p:cache[key(p)] for p in prompts}


def score_all(root: Path, inputs: list[tuple[str,str]], key: str, jev_model: str):
    import requests
    path=root/'jev_cache_probs.json'
    cache=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    lock=threading.Lock()
    qhash={kind:hashlib.md5(question.encode()).hexdigest()[:8]
           for kind,question in QUESTIONS.items()}
    new_calls=0
    def score(text,kind):
        nonlocal new_calls
        cache_key=f'{jev_model}|{kind}|{qhash[kind]}|{text}'
        with lock:
            if cache_key in cache:return float(cache[cache_key])
        body=jev_payload(text,kind)
        body['model']=jev_model  # pin forward rerun, preserve original historical payload elsewhere
        last=None
        for attempt in range(5):
            try:
                res=requests.post(JEV_URL,headers={'Authorization':f'Bearer {key}'},
                                  json=body,timeout=60)
                if 400 <= res.status_code < 500 and res.status_code != 429:
                    raise RuntimeError(f'Jev rejected request: HTTP {res.status_code} {res.text[:300]}')
                res.raise_for_status()
                payload=res.json()
                p=float(payload['answers']['harmful']['noul'])
                # Capture provider-resolved version when returned; never infer it from an alias.
                version=payload.get('model_version') or payload.get('model')
                if version:
                    with lock:
                        version_log=json.loads((root/'jev_versions.json').read_text()) if (root/'jev_versions.json').exists() else {}
                        version_log[str(version)]=version_log.get(str(version),0)+1
                        save_json(root/'jev_versions.json',version_log)
                if not 0 <= p <= 1:raise ValueError('Jev score outside [0,1]')
                with lock:
                    cache[cache_key]=p
                    new_calls+=1
                    if new_calls%25==0:save_json(path,cache)
                return p
            except RuntimeError:
                raise
            except Exception as exc:
                last=exc
                time.sleep(2**attempt)
        raise RuntimeError(f'Jev failed after retries: {last!r}')
    with ThreadPoolExecutor(max_workers=JEV_WORKERS) as pool:
        list(pool.map(lambda item:score(*item),inputs))
    save_json(path,cache)
    return lambda text,kind:float(cache[f'{jev_model}|{kind}|{qhash[kind]}|{text}'])


def run(root: Path, per_set=None, revision=None, model_revision=QWEN_REVISION, jev_model=JEV_MODEL_ID):
    key=os.environ.get('TYPESAFE_API_KEY')
    if not key:raise RuntimeError('Set TYPESAFE_API_KEY in environment (do not commit it).')
    root.mkdir(parents=True,exist_ok=True)
    source=load_prompts(root,revision)
    def select(items):
        return items if per_set is None or per_set>=len(items) else random.Random(SEED).sample(items,per_set)
    groups={'advbench':select(source['adv']),
            'xs_safe':select(source['safe']),
            'xs_unsafe':select(source['unsafe'])}
    all_prompts=list(dict.fromkeys(p for prompts in groups.values() for p in prompts))
    responses=generate_responses(root,all_prompts,model_revision)
    tasks=[(p,'prompt') for p in all_prompts]
    tasks.extend((responses[p],'response') for p in all_prompts if not llm_refused(responses[p]))
    prob=score_all(root,tasks,key,jev_model)
    output=root/'results.csv'
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['set','prompt','response','p_harm_prompt','p_harm_answer',
                         'cond3_tier','cond1','cond2','cond3'])
        for cohort,prompts in groups.items():
            for p in prompts:
                r=responses[p];ref=llm_refused(r);p_in=prob(p,'prompt')
                p_out=None if ref else prob(r,'response')
                tier='block' if p_in>=TIER_HIGH else 'review' if p_in>=TIER_LOW else 'pass'
                outcomes=[classify(p_in,p_out,ref,c) for c in (1,2,3)]
                writer.writerow([cohort,p,r,p_in,p_out,tier,*outcomes])
    save_json(root/'run_manifest.json', {
        'model_id': MODEL_ID, 'model_revision': model_revision,
        'jev_model_requested': jev_model, 'jev_model_responses': 'jev_versions.json',
        'generation': {'max_new_tokens': MAX_NEW_TOKENS, 'do_sample': False},
        'note': 'Fresh rerun, not the original historical cached experiment.'
    })
    print(f'Private raw experiment results: {output}')
    print('DO NOT commit this file; use scripts/make_public_data.py to redact it.')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='private_runs/full')
    parser.add_argument('--per-set',type=int,default=None)
    parser.add_argument('--qwen-revision',default=QWEN_REVISION,
                        help='Pinned Qwen commit for new runs; not verified historical revision.')
    parser.add_argument('--jev-model',default=JEV_MODEL_ID,
                        help='Versioned Jev model for new runs; historical notebook requested jev-latest.')
    parser.add_argument('--dataset-revision',default=None,
                        help='Optional dataset revision; historical revision was not recorded.')
    parser.add_argument('--confirm-live-run',action='store_true',
                        help='Required: calls model/Hugging Face/Jev and may incur charges.')
    args=parser.parse_args()
    if not args.confirm_live_run:parser.error('Use --confirm-live-run to explicitly enable live calls.')
    run(Path(args.output),args.per_set,args.dataset_revision,args.qwen_revision,args.jev_model)

if __name__=='__main__':main()
