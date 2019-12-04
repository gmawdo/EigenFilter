"""Path hacks to make tests work."""

import os
import sys
path = os.getcwd()
sys.path.insert(0, path)

sys.path.insert(0, path+'/src')

pathname = os.path.abspath(sys.argv[0])
sys.path.insert(0, pathname.split('/src/')[0]+'/src'
                )
bp = os.path.dirname(os.path.realpath('..')).split(os.sep)
mod_path = os.sep.join(bp + ['.'])


sys.path.insert(0, mod_path)
