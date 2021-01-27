export KALDI_ROOT=`pwd`/../../..
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C


export NETCDF_BIN=/home/bsrivast/ssw_paper/netcdf-c-4.3.3.1/build/bin
export CURRENNT_PUBLIC=/home/bsrivast/vc_tools/eurecom_nii_paper/project-CURRENNT-public
export CURRENNT_SCRIPTS=/home/bsrivast/vc_tools/eurecom_nii_paper/project-CURRENNT-scripts

PYTOOLS_PATH=${CURRENNT_PUBLIC}/pyTools
export PYTHONPATH=$PYTHONPATH:$PYTOOLS_PATH
export PATH=$PATH:$NETCDF_BIN

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color


