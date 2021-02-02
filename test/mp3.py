from sndfileio import *
samples, sr = sndread("snd/finneganswake.mp3")
sndwrite(samples, sr, "sndout/finneganswake.wav")

f = sndwrite_chunked(44100, "sndout/outchunked.wav", encoding="pcm16")
for samples in sndread_chunked("snd/finneganswake.mp3"):
    print(samples.shape)
    f.write(samples)
f.close()

    