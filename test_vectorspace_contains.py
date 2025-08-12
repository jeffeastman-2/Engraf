#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/jeff/Python/Engraf')

from engraf.lexer.vector_space import VectorSpace

# Test VectorSpace __contains__ behavior
vs = VectorSpace("test")
vs['scaleX'] = 0.0
print(f"After setting scaleX=0.0:")
print(f"'scaleX' in vs: {'scaleX' in vs}")
print(f"vs['scaleX']: {vs['scaleX']}")

vs['scaleX'] = 2.4
print(f"\nAfter setting scaleX=2.4:")
print(f"'scaleX' in vs: {'scaleX' in vs}")
print(f"vs['scaleX']: {vs['scaleX']}")

# Test the specific condition
scale_value = vs['scaleX'] if 'scaleX' in vs else 1.0
print(f"\nScale value from condition: {scale_value}")
