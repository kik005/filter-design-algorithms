# filter-design-algorithms
Contains python programs that are used in the general filter design process.


transmission_poly_optimization.py:
- Optimizes placement of reflection zeros based on the given transmission zero frequencies.
- Once reflection zeros are optimized, the normalized transmission polynomial is used to calculate the Hurwitz polynomial E(S)
- Outputs: optimized roots of transmission polynomial, coefficients of F(S), P(S), and E(S), plots displaying passband equiripple, lossless check, and TZ placement
