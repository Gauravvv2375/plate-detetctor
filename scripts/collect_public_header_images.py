"""Download an audited search inventory; never generate labels or run training."""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from urllib.parse import urlsplit, unquote
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / 'data/real_header_ocr_dataset'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', required=True)
    args = parser.parse_args()
    inventory = json.loads((ROOT/'collection_inventory.json').read_text())
    extra=ROOT/'collection_additional.json'
    if extra.exists():
        inventory['candidates'].extend(json.loads(extra.read_text())['candidates'])
    audit_path = ROOT/'collection_download_audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else []
    seen = {x['source_url'] for x in audit}
    hashes = {x['sha256'] for x in audit if 'sha256' in x}
    canonical = {x['canonical_url'] for x in audit if 'canonical_url' in x}
    blocked = {urlsplit(x['source_page_url']).netloc for x in audit if x.get('page_http_status') in (401,403,429)}
    pages = {}
    stage = ROOT/'collection_staging'
    stage.mkdir(exist_ok=True)
    for item in inventory['candidates']:
        if item['source_url'] in seen:
            continue
        item = dict(item)
        seen.add(item['source_url'])
        item['id'] = len(audit)+1
        url = item['source_url']
        parsed = urlsplit(url)
        # Query variations often resize a single source photograph.
        key = unquote(parsed.netloc + parsed.path)
        item['canonical_url'] = key
        title = (item['title'] + ' ' + item['source_page_url']).lower()
        if any(s in title for s in ['trysticker', 'tradeindia', 'nepal', 'नेपाली', 'kamalauto', 'dharma', 'imdb', 'pinterest', 'rusticplates']):
            item['status'] = 'REJECTED_SEARCH_IRRELEVANT'
        elif key in canonical:
            item['status'] = 'DUPLICATE_URL_VARIANT'
        else:
            canonical.add(key)
            try:
                page = item['source_page_url']
                host = urlsplit(page).netloc
                if host in blocked or parsed.netloc in blocked:
                    raise ValueError('host previously returned an access restriction')
                if page not in pages:
                    time.sleep(0.6)
                    response = requests.get(page, timeout=15, stream=True)
                    pages[page] = response.status_code
                    if response.status_code in (401,403,429):
                        blocked.add(host)
                    response.close()
                item['page_http_status'] = pages[page]
                if pages[page] != 200:
                    raise ValueError(f'source page HTTP {pages[page]}; no bypass attempted')
                time.sleep(0.6)
                response = requests.get(url, timeout=20, stream=True)
                item['image_http_status'] = response.status_code
                if response.status_code in (401,403,429):
                    blocked.add(parsed.netloc)
                response.raise_for_status()
                data = bytearray()
                for chunk in response.iter_content(65536):
                    data.extend(chunk)
                    if len(data)>20_000_000:
                        raise ValueError('image exceeds 20 MB safety limit')
                response.close()
                digest = hashlib.sha256(data).hexdigest()
                item['sha256'] = digest
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    item['size'] = list(image.size)
                    ext = {'JPEG':'.jpg','PNG':'.png','WEBP':'.webp'}.get(image.format)
                    if ext is None:
                        raise ValueError('unsupported image format')
                if digest in hashes:
                    item['status'] = 'DUPLICATE_FILE_HASH'
                else:
                    hashes.add(digest)
                    path = stage/f"candidate_{item['id']:06d}{ext}"
                    with path.open('xb') as stream:
                        stream.write(data)
                    item['download'] = path.relative_to(ROOT).as_posix()
                    item['status'] = 'DOWNLOADED_PENDING_VISUAL_REVIEW'
            except Exception as exc:
                item['status'] = 'DOWNLOAD_UNAVAILABLE'
                item['error'] = str(exc)
        audit.append(item)
        audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
        print(item['id'],item['status'],item.get('download',item.get('error','')),flush=True)


if __name__ == '__main__':
    main()
