import sys
sys.path.append('/home/bsrivast/ssw_paper/project-CURRENNT-public/pyTools')

from os.path import join, basename

from ioTools import readwrite
import kaldi_io
import numpy as np

args = sys.argv
data_dir = args[1]
out_dir = args[2]

dataname = basename(data_dir)
xvector_file = "exp/0007_voxceleb_v2_1a/exp/xvector_nnet_1a/am_nsf/xvectors_"+dataname+"/xvector.scp"
xvec_out_dir = join(out_dir, "xvector")
pitch_out_dir = join(out_dir, "f0")

# Write pitch features
pitch_file = join(data_dir, 'pitch.scp')
pitch2shape = {}
for key, mat in kaldi_io.read_mat_scp(pitch_file):
    #pitch2shape[key] = mat.shape
    readwrite.write_raw_mat(mat[:, 1], join(pitch_out_dir, key+'.f0'))


'''
# Write xvector features
with open(xvector_file) as f:
    for key, mat in kaldi_io.read_vec_flt_scp(f):
        #print key, mat.shape
        plen = pitch2shape[key][0]
        mat = mat[np.newaxis]
        xvec = np.repeat(mat, plen, axis=0)
        readwrite.write_raw_mat(xvec, join(xvec_out_dir, key+'.xvector'))
'''
