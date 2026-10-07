import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ==================== 1. Specifications ====================
N = 4
tz_phys = np.array([6.0, 17.0, 47.0, 54.0])  # GHz, from paper
f0 = np.sqrt(22.0 * 44.0)          # 31.11 GHz (geometric mean)
FBW = (44.0 - 22.0) / f0           # 0.707
ripple_dB = 0.1
epsilon = np.sqrt(10**(ripple_dB / 10) - 1)   # ~0.1526
S11_target = epsilon / np.sqrt(1 + epsilon**2)  # ~0.151 (-16.4 dB)

# ==================== 2. P(s) from TZs (P(0)=1 normalized) ====================
omega_tz = (1.0 / FBW) * (tz_phys / f0 - f0 / tz_phys)
s_tz = 1j * omega_tz
P_monic = np.poly(s_tz)
P = P_monic / np.polyval(P_monic, 0.0)   # normalize: P(0) = 1
print("Normalized TZ omegas:", omega_tz)
print("P(0) check:", np.polyval(P, 0.0), "(should be 1.0)")

# ==================== 3. Helpers ====================
def paraconj(coeffs):
    """Paraconjugate: F~(s) = conj(F(-conj(s))). Coeffs highest-first."""
    n = len(coeffs) - 1
    signs = np.array([(-1)**(n - k) for k in range(n + 1)])
    return np.conj(coeffs) * signs

def spectral_factor(A, N):
    """
    Given A(s) = E(s)*E~(s), return Hurwitz E(s).
    Scales by enforcing Feldtkeller at s=0 (robust for complex coeffs).
    """
    r = np.roots(A)
    # N roots with most negative real part -> LHP (Hurwitz)
    idx = np.argsort(np.real(r))[:N]
    E_monic = np.poly(r[idx])
    # Scale: |E(0)|^2 must equal A(0)
    E0 = np.polyval(E_monic, 0.0)
    A0 = np.polyval(A, 0.0)
    if np.real(A0) <= 0:
        print(f"WARNING: A(0) = {A0}, expected positive real. Check Feldtkeller.")
    scale = np.sqrt(A0 / (np.abs(E0)**2 + 1e-300))
    E = E_monic * scale
    if np.real(E[0]) < 0:
        E = -E
    return E

def build_polys(omega_roots):
    """
    omega_roots: 4 real values -> reflection zeros constrained to j*Omega axis.
    Returns F(s), E(s).
    """
    roots_F = 1j * np.asarray(omega_roots, dtype=float)
    F = np.poly(roots_F)
    FFt = np.polymul(F, paraconj(F))
    PPt = np.polymul(P, paraconj(P)) / epsilon**2
    L = max(len(FFt), len(PPt))
    FFt = np.pad(FFt, (L - len(FFt), 0))
    PPt = np.pad(PPt, (L - len(PPt), 0))
    A = FFt + PPt
    E = spectral_factor(A, N)
    return F, E

def cost(omega_roots):
    F, E = build_polys(omega_roots)
    Om = np.linspace(-1.0, 1.0, 2001)
    s = 1j * Om
    S11 = np.abs(np.polyval(F, s) / np.polyval(E, s))
    peaks = [S11[i] for i in range(1, len(S11)-1)
             if S11[i] >= S11[i-1] and S11[i] >= S11[i+1] and S11[i] > 0.01]
    peaks = np.array(peaks)
    if len(peaks) == 0:
        return 1e6
    err_target = np.sum((peaks - S11_target)**2)
    err_ripple = (np.max(peaks) - np.min(peaks))**2
    err_over = max(0.0, np.max(S11) - S11_target * 1.05)**2 * 10.0
    return err_target + 5.0 * err_ripple + err_over

# ==================== 4. Optimize F(s) ====================
x0 = np.cos((2*np.arange(1, N+1) - 1) * np.pi / (2*N))  # Chebyshev omegas
print("\nInitial guess (Chebyshev omegas):", x0)

res = minimize(cost, x0, method='Nelder-Mead',
               options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
print("Success:", res.success, "| Final cost: %.3e" % res.fun,
      "| Iterations:", res.nit)

omega_opt = res.x
F_opt, E_opt = build_polys(omega_opt)

print("\nOptimized reflection-zero omegas:")
for om in omega_opt:
    print(f"  {om:+.6f}")
print("\nF(s) coeffs:\n", F_opt)
print("E(s) coeffs:\n", E_opt)
print("P(s) coeffs:\n", P)

# ==================== 5. Evaluate & plot ====================
def eval_s_params(F, E, P, epsilon, Om):
    s = 1j * np.asarray(Om)
    S11 = np.polyval(F, s) / np.polyval(E, s)
    S21 = np.polyval(P, s) / (epsilon * np.polyval(E, s))
    return S11, S21

def to_db(z):
    return 20 * np.log10(np.maximum(np.abs(z), 1e-12))

# --- Plot 1: Passband equiripple ---
Om_pb = np.linspace(-2, 2, 2001)
S11_pb, S21_pb = eval_s_params(F_opt, E_opt, P, epsilon, Om_pb)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(Om_pb, to_db(S11_pb), label=r'$|S_{11}|$ (return loss)')
ax.plot(Om_pb, to_db(S21_pb), label=r'$|S_{21}|$ (insertion loss)')
ax.axhline(-ripple_dB, color='k', ls='--', lw=1,
           label=f'{ripple_dB} dB ripple target')
ax.axvline(-1, color='gray', ls=':', lw=1); ax.axvline(1, color='gray', ls=':', lw=1)
ax.set_xlim(-2, 2); ax.set_xlabel(r'Normalized frequency $\Omega$')
ax.set_ylabel('Magnitude (dB)'); ax.set_title('Passband: equiripple check')
ax.legend(loc='lower center'); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# --- Plot 2: Wideband TZ verification ---
Om_wb = np.linspace(-8.0, 8.0, 4001)
_, S21_wb = eval_s_params(F_opt, E_opt, P, epsilon, Om_wb)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(Om_wb, to_db(S21_wb), label=r'$|S_{21}|$')
for tz in omega_tz:
    ax.axvline(tz, color='r', ls='--', lw=1)
ax.plot([], [], color='r', ls='--', lw=1, label='Prescribed TZ locations')
ax.axvline(-1, color='gray', ls=':', lw=1); ax.axvline(1, color='gray', ls=':', lw=1)
ax.set_ylim(-80, 5); ax.set_xlabel(r'Normalized frequency $\Omega$')
ax.set_ylabel(r'$|S_{21}|$ (dB)'); ax.set_title('Wideband: TZ verification')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# --- Plot 3: Lossless sanity ---
Om_ck = np.linspace(-2.0, 2.0, 2001)
S11_ck, S21_ck = eval_s_params(F_opt, E_opt, P, epsilon, Om_ck)
unity = np.abs(S11_ck)**2 + np.abs(S21_ck)**2
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(Om_ck, unity); ax.axhline(1.0, color='k', ls='--', lw=1)
ax.set_xlabel(r'Normalized frequency $\Omega$')
ax.set_ylabel(r'$|S_{11}|^2 + |S_{21}|^2$')
ax.set_title('Lossless check (should be 1.0)')
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
print(f"Max deviation from unity: {np.max(np.abs(unity - 1)):.2e}")
