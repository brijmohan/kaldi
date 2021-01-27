import sys
sys.path.append('/home/bsrivast/ssw_paper/project-CURRENNT-public/pyTools')

from os.path import join, basename

from ioTools import readwrite
import kaldi_io
from kaldiio import ReadHelper

args = sys.argv
ppg_file = args[1]
mspec_file = args[2]
out_dir = args[3]

ppg_out_dir = join(out_dir, "ppg")
mspec_out_dir = join(out_dir, "mel")

'''
print("Writing PPG feats.....")
# Write ppg features
with ReadHelper('scp:'+ppg_file) as reader:
    for key, mat in reader:
        readwrite.write_raw_mat(mat, join(ppg_out_dir, key+'.ppg'))
print("Finished writing PPG feats.")
'''

print("Writing MEL feats.....")
# Write mspec features
for key, mat in kaldi_io.read_mat_scp(mspec_file):
    #print key, mat.shape
    readwrite.write_raw_mat(mat, join(mspec_out_dir, key+'.mel'))
print("Finished writing MEL feats.")


