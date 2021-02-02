from sndfileio import *
samples, sr = sndread("snd/bourre-fragment.wav")
samples *= 0.5
sndwrite(samples, sr, "sndout/bourre-fragment.flac")

f = sndwrite_chunked(44100, "sndout/finnegan-fragment.flac")
for samples in sndread_chunked("snd/finneganswake.mp3", start=0.5, stop=1.5):
    f.write(samples)
f.close()

    