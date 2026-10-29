"""Lossless XWD screen-capture decoding, preserving original pixels and size."""
from pathlib import Path
import struct
import sys
from PIL import Image

data=Path(sys.argv[1]).read_bytes()
h=struct.unpack('>25I',data[:100])
width,height,order,bpp,stride=h[4],h[5],h[7],h[11],h[12]
if tuple(h[14:17])!=(0xff0000,0xff00,0xff):raise ValueError('unsupported XWD color masks')
raw=({(32,0):'BGRX',(32,1):'XRGB',(24,0):'BGR',(24,1):'RGB'})[(bpp,order)]
offset=h[0]+12*h[19]
Image.frombytes('RGB',(width,height),data[offset:offset+stride*height],'raw',raw,stride,1).save(sys.argv[2])
