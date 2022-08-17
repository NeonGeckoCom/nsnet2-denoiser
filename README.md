# Noise Suppression Net 2 (NSNet2) baseline inference script

* As a baseline for ICASSP 2021 Deep Noise Suppression challenge, we will use the recently developed SE method based on Recurrent Neural Network (RNN). For ease of reference, we will call this method as Noise Suppression Net 2 (NSNet 2). More details about this method can be found in [here](https://arxiv.org/abs/2008.06412).


## Installation
`pip install nsnet2-denoiser`

## Usage:
From the NSNet2-baseline directory, run run_nsnet2.py with the following required arguments:
- -i "Specify the path to noisy speech files that you want to enhance"
- -o "Specify the path to a directory where you want to store the enhanced clips"
- -fs "Sampling rate of the input audio. (48000/16000)"

`python -m nsnet2_denoiser.denoise -i audio/`

Use default values for the rest. Run to enhance the clips.

### Python
```python
from nsnet2_denoiser import NSnet2Enhancer
enhancer = NSnet2Enhancer(fs=48000)

# numpy
import soundfile as sf
sigIn, fs = sf.read("audio.wav")
outSig = enhancer(sigIn, fs)

# pcm_16le
from pydub import AudioSegment
audioIn = AudioSegment.from_wav("audio.wav")
audioOut = enhancer.pcm_16le(audioIn.raw_data)
```

## Attribution:
Pretrained model [NSNet2](https://github.com/microsoft/DNS-Challenge/tree/master/NSNet2-baseline) by [Microsoft](https://github.com/microsoft) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)


## Citation:
The baseline NSNet noise suppression:<br />  
```BibTex
@misc{braun2020data,
    title={Data augmentation and loss normalization for deep noise suppression},
    author={Sebastian Braun and Ivan Tashev},
    year={2020},
    eprint={2008.06412},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```

