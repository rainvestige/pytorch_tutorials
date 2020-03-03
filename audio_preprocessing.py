# coding=utf-8
''' Torch audio tutorial
    In this tutorial, we will see how to load and preprecess audio data from
    a simple dataset.
'''
import torch
import torchaudio
import matplotlib.pyplot as plt


''' Opening a file
    `torchaudio` also supports loading sound files in the wav and mp3 format.
    We call waveform the resulting raw audio signal.
'''
#import requests
#
#url = 'https://pytorch.org/tutorials//_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav'
#r = requests.get(url)
#with open('steam-train-whistle-daniel_simon-converted-from-mp3.wav', 'wb') as f:
#    f.write(r.content)


filename = 'steam-train-whistle-daniel_simon-converted-from-mp3.wav'
waveform, sample_rate = torchaudio.load(filename)

print('Shape of waveform: {}'.format(waveform.size()))
print('Sample rate of waveform: {}'.format(sample_rate))

plt.figure()
plt.title('Original Waveform')
plt.plot(waveform.t().numpy())


''' Transformations
    `torchaudio` support a growing list of Transformations.
    - Resample: Resample waveform to a different sample rate.
    - Spectrogram: Create a spectrogram from a waveform.
    - GriffinLim: Compute waveform from a linear scale magnitude spectrogram 
        using the Griffin-Lim transformation.
    - ComputeDeltas: Compute delta coefficients of a tensor, usually a spectrogram.
    - ComplexNorm: Compute the norm of a complex tensor.
    - MelScale: This turns a normal STFT into a Mel-frequency STFT, using a 
        conversion matrix.
    - AmplitudeToDB: This turns a spectrogram from the power/amplitude scale 
        to the decibel scale.
    - MFCC: Create the Mel-frequency cepstrum coefficients from a waveform.
    - MelSpectrogram: Create MEL Spectrograms from a waveform using the STFT
        function in PyTorch.
    - MuLawEncoding: Encode waveform based on mu-law companding.
    - MuLawDecoding: Decode mu-law encoded waveform.
    - TimeStretch: Stretch a spectrogram in time without modifying pitch for a
        given rate.
    - FrequencyMasking: Apply masking to a spectrogram in the frequency domain.
    - TimeMasking: Apply masking to a spectrogram in the time domain.

    Each transform supports batching: you can perform a transform on a single 
    raw audio singal or spectrogram, or many of the same shape.

    Since all transforms are `nn.Modules` or `jit.ScriptModules`, they can be
    used as part of a neural network at any point.
'''

# To start, we can look at the log of the spectrogram on a log scale.
specgram = torchaudio.transforms.Spectrogram()(waveform)
print('Shape of spectrogram: {}'.format(specgram.size()))
plt.figure(figsize=[6.4, 4.6])
plt.subplot(2, 1, 1)
plt.title('Spectrogram')
plt.imshow(specgram.log2()[0, :, :].numpy(), cmap='gray')

# or we can look at the Mel Spectrogram on a log scale
specgram = torchaudio.transforms.MelSpectrogram()(waveform)
print('Shape of spectrogram: {}'.format(specgram.size()))
plt.subplot(2, 1, 2)
plt.title('MelSpectrogram')
plt.imshow(specgram.log2()[0, :, :].detach().numpy(), cmap='gray')

# we can resample the waveform, one channel at a time.
new_sample_rate = sample_rate / 10

# Since Resample applies to a single channel, we resample first channel here
channel = 0
transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(
        waveform[channel, :].view(1, -1))
print('Shape of transformed waveform: {}'.format(transformed.size()))
plt.figure()
plt.title('Resampled')
plt.plot(transformed[0, :].numpy())

# As another example of transformations, we can encode the singal based on 
# Mu-Law encoding. But to do so, we need the signal to be between -1 and 1.
# Since the tensor is just a regular Pytorch Tensor, we can apply standard
# operators on it.

# Let's check if the tensor is in the interval[-1, 1]
print('Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}'.format(
    waveform.min(), waveform.max(), waveform.mean()))

def normalize(tensor):
    # Subtract the mean, and scale to the interval [-1, 1]
    tensor_minus_mean = tensor - tensor.mean()
    return tensor_minus_mean / tensor_minus_mean.abs().max()

# let's normalize to the full interval [-1, 1]
# however, since waveform is already between -1 and 1, we do not need to 
# normalize it.

#waveform = normalize(waveform)

# Let's apply encode the waveform
transformed = torchaudio.transforms.MuLawEncoding()(waveform)
print('Shape of transformed waveform: {}'.format(transformed.size()))
plt.figure(figsize=[12.8, 4.8])
plt.subplot(1, 2, 1)
plt.title('MuLawEncoding')
plt.plot(transformed[0, :].numpy())

# and now decode.
reconstructed = torchaudio.transforms.MuLawDecoding()(transformed)
print('Shape of recovered waveform: {}'.format(reconstructed.size()))
plt.subplot(1, 2, 2)
plt.title('MuLawDecoding')
plt.plot(reconstructed[0, :].numpy())

# We can finally compare the orginal waveform with its reconstructed version.
# Compute median relative difference
err  = ((waveform - reconstructed).abs() / waveform.abs()).median()

print('Median relative difference between original and MuLaw reconstructed signals: {:.2f}'.format(err))


''' Functional
    The transformations seen above rely on lower level stateless functions for 
    their computations. These functions are available under `torchaudio.functional`
    The complete list is available here(https://pytorch.org/audio/functional.html)
    and includes:

    - istft: Inverse short time Fourier Transform.
    - gain: Applies amplification or attenuation to the whole waveform.
    - dither: Increases the perceived dynamic range of audio stored at a particular bit-depth.
    - compute_deltas: Compute delta coefficients of a tensor.
    - equalizer_biquad: Design biquad peaking equalizer filter and perform filtering.
    - lowpass_biquad: Design biquad lowpass filter and perform filtering.
    - highpass_biquad:Design biquad highpass filter and perform filtering.
'''

# For example, let's try the mu_law_encoding functional:
mu_law_encoding_waveform = torchaudio.functional.mu_law_encoding(waveform,
        quantization_channels=256)
print('Shape of transformed waveform: {}'.format(mu_law_encoding_waveform.size()))
plt.figure()
plt.plot(mu_law_encoding_waveform[0, :].numpy())
# the output is the same as the output from `torchaudio.transforms.MuLawEncoding`

# Now let's experiment with a few of the other functionals and visualize their
# output. Taking our spectrogram, we can compute it's deltas:
computed = torchaudio.functional.compute_deltas(specgram, win_length=3)
print('Shape of computed deltas: {}'.format(computed.size()))
plt.figure()
plt.imshow(computed.log2()[0, :, :].detach().numpy(), cmap='gray')

# We can take the original waveform and apply different effects to it
gain_waveform = torchaudio.functional.gain(waveform, gain_db=5.0)
print('Min of gain_waveform: {}\nMax of gain_waveform: {}\nMean of gain_waveform: {}'.format(
    gain_waveform.min(), gain_waveform.max(), gain_waveform.mean()))

dither_waveform = torchaudio.functional.dither(waveform)
print('min of dither_waveform: {}\nmax of dither_waveform: {}\nmean of dither_waveform: {}'.format(dither_waveform.min(), dither_waveform.max(), dither_waveform.mean()))

# Another example of the capabilities in `torchaudio.functional` are applying 
# filters to our waveform. Applying the lowpass biquad filter to our waveform 
# will output a new waveform with the signal of the frequency modified.
lowpass_waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, 
        cutoff_freq=3000)
print('min of lowpass_waveform: {}\nmax of lowpass_waveform: {}\nmean of lowpass_waveform: {}'.format(lowpass_waveform.min(), lowpass_waveform.max(), lowpass_waveform.mean()))

plt.figure(figsize=[10, 6])
plt.subplot(1, 2, 1)
plt.title('Lowpass Biquad')
plt.plot(lowpass_waveform.t().numpy())

# We can also visualize a waveform with the highpass biquad filter.
highpass_waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, 
        cutoff_freq=2000)
print('min of highpass_waveform: {}\nmax of highpass_waveform: {}\nmean of highpass_waveform: {}'.format(highpass_waveform.min(), highpass_waveform.max(), highpass_waveform.mean()))

plt.subplot(1, 2, 2)
plt.title('Highpass Biquad')
plt.plot(highpass_waveform.t().numpy())


''' Migrating to torchaudio from Kaldi
    Users may be familiar with kaldi(xD), a toolkit for speech recognition. 
    `torchaudio` offers compatiblity with it in `torchaudio.kaldi_io`. It
    can indeed read from kaldi scp, or ark file or streams with:
    - read_vec_int_ark
    - read_vec_flt_scp
    - read_vec_flt_arkfile/stream
    - read_mat_scp
    - read_mat_ark

    `torchaudio` provides Kaldi-compatibel transforms for `Spectrogram, fbank,
    mfcc` and `resample_waveform` with be benefit of GPU support, see for more
    information.
'''
n_fft = 400.0
frame_length = n_fft / sample_rate * 1000.0
frame_shift = frame_length / 2.0
params = {
    'channel': 0,
    'dither': 0.0,
    'window_type': 'hanning',
    'frame_length': frame_length,
    'frame_shift': frame_shift,
    'remove_dc_offset': False,
    'round_to_power_of_two': False,
    'sample_frequency': sample_rate,
}

specgram = torchaudio.compliance.kaldi.spectrogram(waveform, **params)
print('Shape of spectrogram: {}'.format(specgram.size()))

plt.figure(figsize=[8, 12])
plt.subplot(3, 1, 1)
plt.title('Kaldi Compatible')
plt.imshow(specgram.t().numpy(), cmap='gray')

# We also support computing the filterbank features from waveforms, matching
# Kaldi's implementation
fbank = torchaudio.compliance.kaldi.fbank(waveform, **params)
print('Shape of fbank: {}'.format(fbank.size()))

plt.subplot(3, 1, 2)
plt.title('Filterbank')
plt.imshow(fbank.t().numpy(), cmap='gray')

# You can create mel frequency cepstral coefficients from a raw audio signal.
# This matches the input/output of kaldi's compute-mfcc-feats
mfcc = torchaudio.compliance.kaldi.mfcc(waveform, **params)
print('Shape of mfcc: {}'.format(mfcc.size()))

plt.subplot(3, 1, 3)
plt.title('mfcc')
plt.imshow(mfcc.t().numpy(), cmap='gray')


''' Available Datasets
    If you do not want to create your own dataset to train your model, 
    `torchaudio` offers a unified dataset interface. This interface supports
    lazy-loading of files to memory, download and extract functions, and 
    datasets to build models.

    The datasets `torchaudio` currently supports are:
    * VCTK: Speech data uttered by 109 native speakers of English with various
    accents(https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)
    * Yesno: Sixty recordings of one individual saying yes or no in Hebrew; each
    recording is eight words long(https://www.openslr.org/1/)
    * Common Voice: An open source, multi-language dataset of voices that anyone
    can use to train speech-enabled applications(https://voice.mozilla.org/en/datasets)
    * LibriSpeech: Large-scale(1000 hours) corpus of read of English speech(
    http://www.openslr.org/12)
'''
yesno_data = torchaudio.datasets.YESNO('./data', download=True)

# A data point in Yesno is a tuple(wave, sample_rate, labels) where labels is a
# list of integers with 1 for yes and 0 for no.

# Pick data point number 3 to see an example of the the yesno_data:
n = 3
waveform, sample_rate, labels = yesno_data[n]
print('Waveform: {}\nSample rate: {}\nLabel: {}'.format(waveform, sample_rate, 
    labels))

plt.figure()
plt.plot(waveform.t().numpy())
plt.show()

# Now, whenever you ask for a sound file from the dataset, it is loaded in 
# memory only when you ask for it. Meaning, the dataset only loads and keeps 
# in memory the items that you want and use, saving on memory.


''' Conclusion
    We used an example raw audio signal, or waveform, to illustrate how to 
    open an audio file using `torchaudio`, and how to pre-process, transform,
    and apply functions to such wave form. We also demonstrated how to use 
    familiar kaldi functions, as well as utilize built-in datasets to construct 
    our models. Given that `torchaudio` is built on Pytorch, these techniques 
    can be used as building blocks for more advanced audio applications, such
    as speech recognition, while leveraging Gpus.
'''


