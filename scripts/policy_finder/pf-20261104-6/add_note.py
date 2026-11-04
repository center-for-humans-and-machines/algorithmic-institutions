"""Append an Explorations entry (stdin) just before '## Key findings'."""
import sys
p = 'notes/policy_finder/pf-20261104-6.md'
s = open(p).read(); e = sys.stdin.read().strip('\n')
i = s.index('## Key findings')
s = s[:i].rstrip('\n') + '\n' + e + '\n\n' + s[i:]
open(p, 'w').write(s)
