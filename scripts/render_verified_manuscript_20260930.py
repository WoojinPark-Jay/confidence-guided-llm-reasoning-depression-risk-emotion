from pathlib import Path
import json
import pymupdf as fitz
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[2] / 'Paper_20260930_verified'
doc = fitz.open(root / 'main.pdf')
qa = root / 'qa'
images = []
for i, page in enumerate(doc):
    pix = page.get_pixmap(matrix=fitz.Matrix(1.4,1.4), alpha=False)
    path = qa / f'page-{i+1:02d}.png'
    pix.save(path)
    thumb = Image.open(path).convert('RGB')
    thumb.thumbnail((400,565))
    images.append(thumb)
for start in range(0,len(images),6):
    sheet=Image.new('RGB',(1230,1200),'#dddddd')
    draw=ImageDraw.Draw(sheet)
    for k,im in enumerate(images[start:start+6]):
        x,y=(k%3)*410,(k//3)*600
        sheet.paste(im,(x,y+25))
        draw.text((x+8,y+5),f'Page {start+k+1}',fill='black')
    sheet.save(qa/f'contact-{start+1:02d}.png')
print('Rendered',len(doc),'pages:',qa)
