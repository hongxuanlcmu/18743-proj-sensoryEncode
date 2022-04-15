###################################### SYNOPSYS ################################################
### Synopsys License
SYNOPSYS=/afs/ece/support/synopsys
export LM_LICENSE_FILE="$LM_LICENSE_FILE:${SYNOPSYS}/license.dat"

#### Synopsys Library Compiler
export SYNOPSYS_SYN_ROOT="/afs/ece/support/synopsys/synopsys.release/P-Foundation/2019.03/syn"
export PATH="/afs/ece/support/synopsys/synopsys.release/L-Foundation/2016.06/lib_compiler_vL-2016.06/bin:$PATH"

#### VCS 2015
## DVE doesn't work in 2019 b/c not in gui folder
##export VCS_HOME="$SYNOPSYS/synopsys.release/P-Foundation/vcs/P-2019.06-SP2-1"
##export VCS_HOME="$SYNOPSYS/synopsys.release/L-Foundation/2016.06/vcs_vL-2016.06"
## As of 6/01/2020, this is the newest version that I could get working
export VCS_HOME=$SYNOPSYS/synopsys.release/vcs-mx_vK-2015.09
export PATH=$PATH:$VCS_HOME/bin
export VCS_ARCH_OVERRIDE=linux
export VCS_MODE_FLAG=64
# export VCS_TARGET_ARCH=amd64
export MANPATH=$MANPATH:$VCS_HOME/doc/man
export VCS_LIC_EXPIRE_WARNING=0


#### Rest of Synopsys 2019
export PATH="/afs/ece/support/synopsys/synopsys.release/P-Foundation/2019.03/syn/P-2019.03/bin:$PATH"
################################################################################################