from sndfileio import *
from argparse import ArgumentParser
import os
parser = ArgumentParser()
parser.add_argument('-o', '--output', default='out.flac')
args = parser.parse_args()

samples, sr = sndread("snd/bourre-fragment.wav")
samples *= 0.5
if not os.path.exists('sndout'):
    os.mkdir("sndout")
outfile = os.path.join('sndout', args.output)
sndwrite(outfile, samples, sr)

base, ext = os.path.splitext(outfile)
outfile2 = f'{base}-fragment{ext}'

f = sndwrite_chunked(outfile2, sr)

for samples in sndread_chunked("snd/finneganswake.mp3", start=0.5, stop=1.5):
    f.write(samples)
f.close()


