'''
Usage: python local/make_librispeech_eval.py <protocol_files> <data_path> <exp_tag>
'''
import sys
import os
import shutil
from os.path import join, exists

args = sys.argv
proto_path = args[1]
data_path = args[2]
exp_tag = args[3]

enroll_files = ["Librispeech.asv_test_female.trn",
                "Librispeech.asv_test_male.trn"]

trial_files = ["Librispeech.asv_test_male.trl",
               "Librispeech.asv_test_female.trl"]

# Prepare enroll data
enroll_dir = 'data/test_clean_enroll'+exp_tag
if exists(enroll_dir):
    shutil.rmtree(enroll_dir)
os.makedirs(enroll_dir)
enroll_wav_scp = []
enroll_utt2spk = []
for ef in enroll_files:
    ef_path = join(proto_path, ef)
    with open(ef_path) as f:
        for line in f.read().splitlines():
            line = line.strip().split()
            spkid = line[0]
            uttarr = line[1].split(',')
            for utt in uttarr:
                uttid = spkid+'-'+utt.split('/')[-1]
                uttpath = join(data_path, utt)+'.flac'
                enroll_wav_scp.append(uttid + '\t flac -c -d -s ' + uttpath + ' | ')
                enroll_utt2spk.append(uttid + '\t' + spkid)

enroll_wav_scp = sorted(enroll_wav_scp)
enroll_utt2spk = sorted(enroll_utt2spk)
with open(join(enroll_dir, 'wav.scp'), 'w') as f:
    f.write('\n'.join(enroll_wav_scp) + '\n')
with open(join(enroll_dir, 'utt2spk'), 'w') as f:
    f.write('\n'.join(enroll_utt2spk) + '\n')


# Prepare trial data
trial_dir = 'data/test_clean_trial'+exp_tag
if exists(trial_dir):
    shutil.rmtree(trial_dir)
os.makedirs(trial_dir)
trial_wav_scp = []
trial_utt2spk = []
trial_trials_male = []
trial_trials_female = []
for i, tf in enumerate(trial_files):
    tf_path = join(proto_path, tf)
    with open(tf_path) as f:
        for line in f.read().splitlines():
            line = line.strip().split()
            spkid = line[0]
            utt = line[1]
            target_type = line[3]
            uttid = spkid+'-'+utt.split('/')[-1]
            uttpath = join(data_path, utt)+'.flac'
            trial_wav_scp.append(uttid + '\t flac -c -d -s ' + uttpath + ' | ')
            trial_utt2spk.append(uttid + '\t' + spkid)
            if i == 0:
                trial_trials_male.append(spkid + '\t' + uttid + '\t' + target_type)
            else:
                trial_trials_female.append(spkid + '\t' + uttid + '\t' + target_type)

trial_wav_scp = sorted(trial_wav_scp)
trial_utt2spk = sorted(trial_utt2spk)
trial_trials = sorted(trial_trials_male + trial_trials_female)
with open(join(trial_dir, 'wav.scp'), 'w') as f:
    f.write('\n'.join(trial_wav_scp) + '\n')
with open(join(trial_dir, 'utt2spk'), 'w') as f:
    f.write('\n'.join(trial_utt2spk) + '\n')
with open(join(trial_dir, 'trials'), 'w') as f:
    f.write('\n'.join(trial_trials) + '\n')

tt_male = sorted(trial_trials_male)
tt_female = sorted(trial_trials_female)
with open(join(trial_dir, 'trials_male'), 'w') as f:
    f.write('\n'.join(tt_male) + '\n')
with open(join(trial_dir, 'trials_female'), 'w') as f:
    f.write('\n'.join(tt_female) + '\n')

