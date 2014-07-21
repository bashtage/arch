import arch
t=arch.test()
import sys
sys.exit((len(t.failures)+len(t.errors))>0)

