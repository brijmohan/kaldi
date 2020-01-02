# Recipe for voice privacy challenge 2020

To successfully run the recipe, you must configure some variables in the scripts, particularly in the main script: `run.sh`. VPC uses several datasets and modules to evaluate generalized anonymization techniques. Visit the [challenge website](https://www.voiceprivacychallenge.org/) for detailed information.

Some of the datasets we use are:
* [LibriSpeech](http://www.openslr.org/12/)
* [LibriTTS](http://www.openslr.org/60/)
* [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)
* [VoxCeleb 1 & 2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

The architecture of VPC is composed of several independent modules:
* Phonetic posteriorgram (PPG) extractor
* x-vector extractor
* Voice conversion using acoustic and neural source filter models
* Anonymization using PLDA distance

Some of these modules are pretrained and must be downloaded and put in appropriate directories for the recipe to work successfully.

## Dataset

- `librispeech_corpus`: change this variable to point at your extracted LibriSpeech corpus.
- `anoni_pool`: change this variable to the data directory in `data/` folder which will be used as anonymization pool of speakers. Please note that this directiry must be in Kaldi data format.

## Modules

### PPG extractor

This is a chain ASR model trained using 600 hours (train-clean-100 and train-other-500) of LibriSpeech. It produces 346 dimentional PPGs. This must include:

- `ivec_extractor`: i-vector extractor trained during training the chain model.
- `tree_dir`: Tree directory created during traininig the chain model.
- `lang_dir`: Lang directory for chain model
- `model_dir`: Directory where pretrained chain model is stored


### x-vector extractor

- `xvec_nnet_dir`: Directory where trained xvector network is stored
- `pseudo_xvec_rand_level`: anonymized x-vectors will be produced at this level, e.g. `spk` or `utt`
- `cross_gender`: should anonymization be done within same gender or across gender.



