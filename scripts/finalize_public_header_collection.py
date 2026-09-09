"""Publish visually audited, UNVERIFIED collection manifests; never training labels."""
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
ROOT=Path(__file__).resolve().parents[1]/'data/real_header_ocr_dataset'
PROJECT=ROOT.parents[1]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    labels_hash=digest(ROOT/'labels.csv')
    decisions=json.loads((ROOT/'collection_visual_decisions.json').read_text())
    records=json.loads((ROOT/'collection_download_audit.json').read_text())
    audit={str(x['id']):x for x in records}
    extraction=json.loads((ROOT/'collection_extraction_audit.json').read_text())
    accepted=decisions['accepted']
    source_fields=['image','source_url','source_page_url','search_query','vehicle_type','plate_visible','header_visible','header_language','candidate_header_text','verification_status','notes']
    review_fields=['header_crop','source_image','source_url','candidate_header_text','verification_status','notes']
    manifest=[];queue=[];crops=[]
    for key,attrs in accepted.items():
        item=audit[key];entry=extraction[key]
        source=ROOT/entry['source_image']
        assert digest(source)==item['sha256'],f'Source changed: {source}'
        notes=attrs[2]+'; Publicly accessible does not imply a training/republication license; rights review required.'
        good=[]
        for header in entry['headers']:
            src=ROOT/header['crop']
            if src.name not in decisions['approved_crops']:continue
            dest=ROOT/'unlabeled'/src.name
            if dest.exists():
                assert digest(dest)==digest(src),f'Refusing overwrite: {dest}'
            else:shutil.copyfile(src,dest)
            good.append(dest.relative_to(ROOT).as_posix())
        if not good:
            notes+='; HEADER_EXTRACTION_FAILED'
            if key in decisions.get('rejected_extractions',{}):notes+='; '+decisions['rejected_extractions'][key]
        manifest.append(dict(zip(source_fields,[entry['source_image'],item['source_url'],item['source_page_url'],item['search_query'],attrs[0],'yes','yes',attrs[1],'','NEEDS_REVIEW',notes])))
        for crop in good or ['']:
            queue.append(dict(zip(review_fields,[crop,entry['source_image'],item['source_url'],'','NEEDS_REVIEW',notes])))
            if crop:crops.append(crop)
    # A source rejected on closer inspection is moved out of the usable set,
    # not deleted. Only this run's exact, hash-verified extra copy is handled.
    rejected=ROOT/'source_images/source_000061.jpg'
    if rejected.exists():
        assert digest(rejected)==audit['61']['sha256']
        target=ROOT/'collection_staging/rejected_source_000061.jpg'
        if target.exists():raise ValueError(f'Refusing overwrite: {target}')
        rejected.rename(target)
    for name,fields,rows in [('source_manifest.csv',source_fields,manifest),('review_queue.csv',review_fields,queue)]:
        with (ROOT/name).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    assert digest(ROOT/'labels.csv')==labels_hash
    duplicates=set(decisions['duplicates'])|set(decisions['preexisting_duplicates'])
    final=[]
    for x in records:
        x=dict(x);key=str(x['id'])
        if key in accepted:x['collection_status']='RETAINED_NEEDS_REVIEW'
        elif key in duplicates or x['status'].startswith('DUPLICATE'):x['collection_status']='DUPLICATE_EXCLUDED'
        elif x['status']=='DOWNLOAD_UNAVAILABLE':x['collection_status']='UNAVAILABLE'
        else:x['collection_status']='REJECTED_UNSUITABLE'
        if x.get('download') and not (ROOT/x['download']).exists():
            x['artifact_note']='Downloaded artifact no longer present; excluded, never referenced by manifest'
        final.append(x)
    (ROOT/'collection_final_audit.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    counts=Counter(x['collection_status'] for x in final)
    inventory=json.loads((ROOT/'collection_inventory.json').read_text())
    additional=json.loads((ROOT/'collection_additional.json').read_text())
    queries=inventory['earlier_queries_without_raw_archive']+inventory['normal_web_queries']+[q for s in inventory['searches']+additional['searches'] for q in s['queries']]
    counts.update({'search_queries':len(queries),'distinct_recorded_image_urls':len(records),
        'recorded_search_image_appearances':len(inventory['candidates'])+len(additional['candidates']),
        'downloaded_images':sum(bool(x.get('download')) or x['status']=='DUPLICATE_FILE_HASH' for x in records),
        'header_crops':len(crops),'extraction_failures':sum('HEADER_EXTRACTION_FAILED' in x['notes'] for x in manifest),
        'devanagari_source_candidates':len(manifest),'verified_labels_added':0})
    safety={'labels_sha256':labels_hash,'training_started':False,'checkpoints_sha256':{str(p.relative_to(PROJECT)):digest(p) for p in [PROJECT/'models'/d/'best.pt' for d in ['plate_detector','row_detector','ocr','ocr_devanagari','ocr_mixed_v2']]}}
    (ROOT/'collection_summary.json').write_text(json.dumps({'counts':counts,'safety':safety,'target':100,'target_met':len(manifest)>=100,
        'caveats':['Counts describe archived candidate URLs, not every image on result pages.','All retained data is unverified; candidate transcriptions deliberately blank.','RTO-prefix and adjacent official placard cases are explicitly identified in notes.','Rejected downloads remain in collection_staging and must not be used for training.']},indent=2),encoding='utf-8')
    # Ten review panels: nine unique sources plus one crop from an existing panel.
    panels=[(x['image'],Path(x['image']).stem) for x in manifest]
    panels.extend((p,Path(p).stem+' (crop of source above)') for p in crops)
    panels=panels[:10]
    sheet=Image.new('RGB',(1200,340*((len(panels)+1)//2)),'#f5f5f5');draw=ImageDraw.Draw(sheet)
    for k,(path,label) in enumerate(panels):
        im=ImageOps.exif_transpose(Image.open(ROOT/path)).convert('RGB');im.thumbnail((585,295))
        x=(k%2)*600;y=(k//2)*340
        draw.text((x+10,y+8),label+' | NEEDS_REVIEW',fill='black')
        sheet.paste(im,(x+(600-im.width)//2,y+35+(295-im.height)//2))
    sheet.save(ROOT/'representative_samples.jpg',quality=95)
    print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
