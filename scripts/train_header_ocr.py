"""Isolated mixed-v2 fine-tuning on verified real headers plus synthetic replay."""
import argparse
import csv
import hashlib
import json
import random
import sys
import unicodedata
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.utils import PROJECT_ROOT
from scripts.train_ocr import (OCRDataset, OCRRecord, collate, load_datasets, run_epoch,
    save_ocr_recovery, load_ocr_recovery, move_optimizer_state)


MIN_REAL_SOURCES=100


def verified_records(root):
    info=json.loads((root/'dataset_info.json').read_text(encoding='utf-8'))
    with (root/'labels.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    records=[]; groups={}; digest=hashlib.sha256()
    for row in rows:
        relative=row['image']; meta=info['samples'].get(relative,{})
        path=(root/relative).resolve()
        if not path.is_relative_to((root/'images').resolve()) or not meta.get('verified') or meta.get('label_source')!='manual_cli':
            raise ValueError(f'Unverified or invalid record: {relative}')
        text=unicodedata.normalize('NFC',row['text'])
        if not text.strip() or text!=meta.get('text'):
            raise ValueError('Label differs from manually verified provenance')
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha!=meta['crop_sha256']:
            raise ValueError('Image changed after labeling')
        digest.update((sha+text+meta['source_sha256']).encode())
        records.append(OCRRecord(path,text))
        groups[str(path)]=meta['source_sha256']
    return records,groups,digest.hexdigest()


def split_real(records,groups,smoke=False):
    sources=sorted(set(groups.values()))
    if len(sources)<MIN_REAL_SOURCES and not smoke:
        raise ValueError(f'Dataset preparation complete. More verified real samples are recommended before production fine-tuning. '
                         f'Need {MIN_REAL_SOURCES} independent source images; found {len(sources)}.')
    if not records:
        raise ValueError('No manually verified records; unlabeled images are never used, including smoke tests')
    random.Random(42).shuffle(sources)
    validation=set(sources[:max(1,round(len(sources)*.2))])
    train=[r for r in records if groups[str(r.path)] not in validation]
    val=[r for r in records if groups[str(r.path)] in validation]
    if not train and smoke:
        train=val # Deliberate plumbing-only overlap, prominently marked smoke.
    return train,val


def initialize_model(checkpoint,vocab):
    import torch
    from doctr.models import parseq
    old=checkpoint['vocab']; length=checkpoint['config']['max_length']
    model=parseq(pretrained=False,pretrained_backbone=False,vocab=vocab,max_length=length)
    if old==vocab:
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        return model
    # Append vocabulary without reinterpreting any existing token or special token.
    source=checkpoint['model_state_dict']; target=model.state_dict()
    with torch.no_grad():
        for key,value in source.items():
            if key not in ('head.weight','head.bias','embed.embedding.weight'):
                target[key].copy_(value)
        for key in ('head.weight','head.bias','embed.embedding.weight'):
            for index,char in enumerate(old):
                target[key][vocab.index(char)].copy_(source[key][index])
            for offset in range(3 if key=='embed.embedding.weight' else 1):
                target[key][len(vocab)+offset].copy_(source[key][len(old)+offset])
    model.load_state_dict(target,strict=True)
    return model


class HeaderData(OCRDataset):
    def __getitem__(self,index):
        import torch
        from scripts.train_ocr import DOCTR_MEAN, DOCTR_STD
        tensor,text,path=super().__getitem__(index)
        mean,std=torch.from_numpy(DOCTR_MEAN),torch.from_numpy(DOCTR_STD)
        rgb=tensor*std+mean
        if random.random()<.3:
            blurred=torch.nn.functional.avg_pool2d(rgb.unsqueeze(0),3,1,1)[0]
            rgb=.95*rgb+.05*blurred
        rgb=(rgb*random.uniform(.97,1.03)+torch.randn_like(rgb)*.002).clamp(0,1)
        tensor=(rgb-mean)/std
        return tensor,text,path


def main():
    import torch
    from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=PROJECT_ROOT/'data/real_header_ocr_dataset')
    p.add_argument('--synthetic-dir',type=Path,default=PROJECT_ROOT/'data/mixed_ocr_dataset_v2')
    p.add_argument('--source',type=Path,default=PROJECT_ROOT/'models/ocr_mixed_v2/best.pt')
    p.add_argument('--checkpoint-dir',type=Path,default=PROJECT_ROOT/'models/ocr_header_real_v1')
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--device',default='cpu')
    p.add_argument('--synthetic-samples',type=int,default=1000)
    p.add_argument('--extend-vocab',action='store_true',help='Explicitly append verified missing characters; preserve existing token weights')
    p.add_argument('--augment-headers',action='store_true',help='Very mild brightness/noise/blur on real training rows only')
    p.add_argument('--smoke-test',action='store_true',help='One tiny epoch; output cannot be used by production header inference')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args()
    if args.source.resolve() != (PROJECT_ROOT/'models/ocr_mixed_v2/best.pt').resolve():
        p.error('This mode starts only from the protected mixed-v2 checkpoint; --resume loads header recovery separately')
    if not 0<args.lr<=5e-5 or min(args.batch,args.epochs,args.synthetic_samples)<1:
        p.error('Use positive sizes and a learning rate no greater than 5e-5')
    root=args.checkpoint_dir.resolve()
    if root.parent!= (PROJECT_ROOT/'models').resolve() or not root.name.startswith('ocr_header_') or root.is_symlink():
        p.error('Output must be a separate models/ocr_header_* directory')
    if args.smoke_test and root.name=='ocr_header_real_v1':
        p.error('Use a separate --checkpoint-dir models/ocr_header_smoke for smoke tests')
    records,groups,fingerprint=verified_records(args.data_dir)
    real_train,real_val=split_real(records,groups,args.smoke_test)
    source=torch.load(args.source,map_location='cpu',weights_only=False)
    if source.get('config',{}).get('input_normalization')!='doctr_parseq':
        p.error('Source must be a normalized mixed PARSeq checkpoint')
    missing=sorted(set(''.join(r.label for r in records))-set(source['vocab']))
    if missing and not args.extend_vocab:
        p.error(f'Verified labels contain characters absent from the source: {missing}. Explicit --extend-vocab is required.')
    vocab=source['vocab']+''.join(missing)
    if any(len(r.label)>source['config']['max_length']-2 for r in records):
        p.error('Verified header exceeds checkpoint sequence capacity; do not silently truncate labels')
    synthetic_train,synthetic_val,_,_,_=load_datasets(args.synthetic_dir,.1,42)
    rng=random.Random(42)
    synthetic_train=rng.sample(synthetic_train,min(args.synthetic_samples,len(synthetic_train)))
    if args.smoke_test:
        real_train,real_val,synthetic_train,synthetic_val=real_train[:2],real_val[:2],synthetic_train[:2],synthetic_val[:10]
        args.epochs=1
        print('SMOKE TEST ONLY: metrics are not evidence of production accuracy; splits may overlap.')
    config={'task':'header_finetune','target_epochs':args.epochs,'batch':args.batch,'learning_rate':args.lr,
        'vocab':vocab,'max_length':source['config']['max_length'],'input_normalization':'doctr_parseq',
        'image_size':[32,128],'smoke_test':args.smoke_test,'real_fingerprint':fingerprint,
        'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'synthetic_manifest_sha256':hashlib.sha256((args.synthetic_dir/'labels.csv').read_bytes()).hexdigest(),
        'synthetic_samples':args.synthetic_samples,'real_sampling_fraction':.75,
        'augmentation':'mild_header_v1' if args.augment_headers else 'none_conservative_v1'}
    if root.exists() and any(root.glob('*.pt')) and not (args.resume or args.evaluate_only):
        p.error('Checkpoint directory already contains weights; use explicit --resume or choose a new directory')
    root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'.training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        recovery=load_ocr_recovery(root,config,args.resume) if not args.evaluate_only else None
        checkpoint=recovery[1] if recovery else source
        if args.evaluate_only:
            checkpoint=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
            from scripts.train_ocr import validate_ocr_checkpoint
            validate_ocr_checkpoint(checkpoint,config)
        torch.manual_seed(42)
        model=initialize_model(checkpoint,vocab).to(args.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
        start,best=1,-1.
        if recovery:
            if recovery[1]['config']['target_epochs']!=args.epochs:
                p.error('Resume must use the original epoch target')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict']);move_optimizer_state(optimizer,args.device)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start,best=checkpoint['epoch']+1,checkpoint['best_exact_accuracy']
            random.setstate(checkpoint['python_random_state']);np.random.set_state(checkpoint['numpy_random_state'])
            torch.set_rng_state(checkpoint['torch_random_state'])
        else:
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
        def loader(records):
            return DataLoader(OCRDataset(records,False,True),batch_size=args.batch,collate_fn=collate)
        with (args.synthetic_dir/'labels.csv').open(encoding='utf-8',newline='') as stream:
            categories={str((args.synthetic_dir/r['image']).resolve()):r['category'] for r in csv.DictReader(stream)}
        subsets={'real_header_validation':real_val,'synthetic_mixed_validation':synthetic_val}
        for category in ['latin_only','devanagari_only','mixed_script','latin_letters_devanagari_digits','devanagari_letters_ascii_digits']:
            subsets[category]=[r for r in synthetic_val if categories[str(r.path.resolve())]==category]
        def evaluate():
            report={key:(run_epoch(model,loader(rows),args.device,vocab) | {'samples':len(rows)}) if rows else {'samples':0,'unavailable':True}
                    for key,rows in subsets.items()}
            report['smoke_test']=args.smoke_test
            return report
        if args.evaluate_only:
            print(json.dumps(evaluate(),ensure_ascii=False,indent=2));return
        baseline_path=root/'baseline_evaluation.json'
        if recovery:
            baseline=json.loads(baseline_path.read_text())
        else:
            baseline=evaluate()
            # Registration retention is measured against the original source,
            # not against random newly appended token logits.
            original_model=initialize_model(source,source['vocab']).to(args.device)
            for key,rows in subsets.items():
                if key!='real_header_validation' and rows:
                    baseline[key]=run_epoch(original_model,loader(rows),args.device,source['vocab']) | {'samples':len(rows)}
            del original_model
            baseline_path.write_text(json.dumps(baseline,ensure_ascii=False,indent=2),encoding='utf-8')
        datasets=[(HeaderData if args.augment_headers else OCRDataset)(real_train,False,True),OCRDataset(synthetic_train,False,True)]
        weights=[.75/len(real_train)]*len(real_train)+[.25/len(synthetic_train)]*len(synthetic_train)
        train=DataLoader(ConcatDataset(datasets),batch_size=args.batch,collate_fn=collate,
            sampler=WeightedRandomSampler(weights,4 if args.smoke_test else max(args.batch,4*len(real_train)),replacement=True))
        for epoch in range(start,args.epochs+1):
            run_epoch(model,train,args.device,vocab,optimizer,epoch,args.epochs)
            report=evaluate(); scheduler.step()
            metric=report['real_header_validation']['exact']; improved=metric>best;best=max(best,metric)
            config['production_ready']=bool(not args.smoke_test and metric>=.80 and all(
                report[key]['exact']>=baseline[key]['exact']-.01 and report[key]['cer']<=baseline[key]['cer']+.005
                for key in subsets if key!='real_header_validation' and not report[key].get('unavailable')))
            report['production_ready']=config['production_ready']
            checkpoint={'epoch':epoch,'config':config,'vocab':vocab,'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
                'best_exact_accuracy':best,'best_cer':report['real_header_validation']['cer'],
                'val_metrics':report['real_header_validation'],'python_random_state':random.getstate(),
                'numpy_random_state':np.random.get_state(),'torch_random_state':torch.get_rng_state()}
            save_ocr_recovery(checkpoint,root,improved)
            (root/f'evaluation_epoch_{epoch:03d}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(str(error))
