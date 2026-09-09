"""Create visual contact sheets without altering downloaded image pixels."""
import json
import argparse
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
ROOT=Path(__file__).resolve().parents[1]/'data/real_header_ocr_dataset'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',type=int,default=0)
    args=parser.parse_args()
    records=[x for x in json.loads((ROOT/'collection_download_audit.json').read_text()) if x.get('download') and (ROOT/x['download']).exists()]
    records=records[args.start:args.start+20]
    sheet=Image.new('RGB',(1200,300*((len(records)+3)//4)),'#eeeeee')
    draw=ImageDraw.Draw(sheet)
    for k,item in enumerate(records):
        im=ImageOps.exif_transpose(Image.open(ROOT/item['download'])).convert('RGB')
        im.thumbnail((294,266))
        x=(k%4)*300;y=(k//4)*300
        sheet.paste(im,(x+(300-im.width)//2,y+24))
        draw.text((x+8,y+5),f"ID {item['id']} | {item['size']}",fill='black')
    path=ROOT/f'collection_review_{args.start:03d}.jpg'
    sheet.save(path,quality=94)
    print(path)

if __name__=='__main__':main()
