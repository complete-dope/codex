---
date : 2026-03-09
title : Audio models for generation, ASR, Trigger word etc    
tags : ['audio', 'generation' ,'ASR']
---

# Audio Modality 

* Codec : piece of hardware/software that compresses / decompresses digital data to reduce file size 
* FFT : fast fourier transform converts from time-amplitude domain to frequency-amplitude domain
* Resampling : an audio was recorded at 44100 Hz, but we want to resample it to 16000 Hz, so that is called resampling 
* Spectrogram : converting from time-amplitude domain to frequency-amplitude domain
* Mels : mel scale, approximates how humans percieve pitch and in this freq axis is converted to mel scale 
* Channel : no. of seperate audio  how many microphones were used to record the audio 
* Mono channel : single sound , more like one headphone sound 
* Stereo channel : surround sounds , more like two headphones sound (tv , songs , youtube vids )
* Sampling Rate : no. of sound point extracted from 1 sec of audio
* Waveform : so this is audio plot , on x-axis we have time , on y-axis we have decibels, pitch . This is what we hear and what music players shows   
* Spectogram : so we convert from time domain to freq domain using FFT , 
* Mel-spectogram : Inspired from how humans listen to sound, and we listen on a logscale so therefore mel-spectogram is made for humans to listen 

```bash
Example :

SAMPLE_RATE = 16000
HOP_LENGTH = 256 # number of audio samples between spectrogram frames between 2 short time frame windows 
N_FFT = 1024 
MAX_MEL_FRAMES = 512 # no. of timestamps in a mel spectrogram 
```
So in this audio is sampled at 16Khz 
`N_FFT` : tells in a 1 fft how many samples to analyse,

`16000 samples = 1 second`
then, `1024 samples / 16000 ≈ 0.064 sec` , so each fft is analysing 64ms of audio

`1 FFT computation = 1 spectrogram frame`
Hop length tells how many samples to skip before we start analysing 2nd fft
`256 / 16000 = 0.016 sec , 16ms`

```bash
Frame 1
[0 ms -------- 64 ms]

Frame 2
     [16 ms -------- 80 ms]

Frame 3
          [32 ms -------- 96 ms]
```

No. of spectogram frames : 
`frames ≈ (num_samples - N_FFT) / HOP_LENGTH + 1`

# ASR models
Input : waveform  
Output : text   
Loss : [CTC loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)   
Dataset : (Audio, text) pairs   

In this we have input as waveform and we try to get features out from this waveform to a latent space (using an CNN model) then we pass that features to transformer encoder for training to get samples out.    

# Generation model 

Input : text   
Output : spectrogram  
Loss : original spectrogram with predicted spectrogram   
Vocoder : converts from spectrogram to waveform     


so this is more like a seq to seq problem , where in input we have text and in output we spectrogram values for that input

Architecture includes an full transformer model (encoder , decoder model) . 
This can be done using diffusion also 

## Basics

### Converting to tokens aka making a codec (tokenization) 
Codec refers to a speech tokenizer (nothing else)
so the very first step is to convert these speech signal to some representable that models can then learn from . In short we need to represent this to an float array format 

Some of the methods of doing this is :

1. Residual vector quantization
2. dMel (speech tokenization)

#### Residual vector quantization
best : https://kyutai.org/codec-explainer

"Make sampling 100x and the model also becomes 100x more coherent"

So we use an autoencoder, a simple autoencoder misses many parts while decoding (if its a small model with limited data) so we need to make it efficient one way is to add embedding quantization and the algo that does this is : 

```python 
x = get_batch()
z = encoder(x)

residual = z - to_nearest_cluster(z)
# .detach() means "forget that this needs a gradient"
z_quantized = z - residual.detach()
x_reconstructed = decoder(z_quantized)

loss = reconstruction_loss(x, x_reconstructed)
```
the encoder’s weights will be updated to improve the reconstruction loss, but they’re updated as if the quantization didn’t happen, so they won’t move in the optimal direction.

This is called as VQ-VAE (vector quantized , variational autoencoder) 

Residual vector quantization : 
To improve reconstruction fidelity, we can just increase the number of cluster centers. Get this residual and quantize that as well and do it till you reach your desired level 

```python
def rvq_quantize(z):
    residual = z
    codes = []

    for level in range(levels):
        quantized, cluster_i = to_nearest_cluster(level, residual)
        residual -= quantized
        codes.append(cluster_i)

    return codes
```

### dMel
