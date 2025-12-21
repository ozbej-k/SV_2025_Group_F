import numpy as np
from scipy.special import iv
from scipy.integrate import cumulative_trapezoid
import config

THETA_GRID = np.linspace(-np.pi, np.pi, 720) # all possible orientations for fish

def von_mises(theta, dispersion_koef):
    return np.exp(dispersion_koef * np.cos(theta)) / (2 * np.pi * iv(0, dispersion_koef))

def f_component(theta, A, AT, mu, dispersion_koef): # general von mises mixing function
    if AT == 0: return np.zeros_like(theta)
    weights = A / AT
    diff = theta[None, :] - mu[:, None]
    pdf = von_mises(diff, dispersion_koef)
    return np.dot(weights, pdf)

def f0_forward(theta, _): # fish going forward
    return von_mises(theta, config.PDF_K0)

def f0_wall(theta, mu_w): # fish following walls
    # deviation from source: perffer the direction along the wall which is closer to forward
    weights = np.exp(config.PDF_KWB * np.cos(mu_w))
    return f_component(theta, weights, np.sum(weights), mu_w, config.PDF_KW)

def total_f(theta, near_wall, mu_w, A_f, A_s, mu_f, mu_s): # full PDF
    f0 = f0_forward
    alpha = config.PDF_ALPHA_0
    beta = config.PDF_BETA_0
    if near_wall:
        f0 = f0_wall
        alpha = config.PDF_ALPHA_W
        beta = config.PDF_BETA_W
    
    AT_f = np.sum(A_f)
    AT_s = np.sum(A_s)

    f0_pdf = f0(theta, mu_w)
    fF_pdf = f_component(theta, A_f, AT_f, mu_f, config.PDF_KF)
    fS_pdf = f_component(theta, A_s, AT_s, mu_s, config.PDF_KS)
    
    top = f0_pdf + alpha * AT_f * fF_pdf + beta * AT_s * fS_pdf
    bottom = 1 + alpha * AT_f + beta * AT_s
    full_pdf = top / bottom

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(6,6))
    # ax = plt.subplot(111)
    # # ax = plt.subplot(111, polar=True)
    # ax.set_ylim(-0.5, np.max([f0_pdf, fF_pdf, fS_pdf, full_pdf]) * 1.2)
    # ax.plot(THETA_GRID, f0_pdf, color='blue', linewidth=1, label=r"$f_0(\theta)$")
    # ax.plot(THETA_GRID, fF_pdf, color='red', linewidth=1, label=r"$f_f(\theta)$")
    # ax.plot(THETA_GRID, fS_pdf, color='black', linewidth=1, label=r"$f_s(\theta)$")
    # ax.plot(THETA_GRID, full_pdf, color='green', linewidth=2, label=r"$f(\theta)$")
    # # ax.fill_between(THETA_GRID, 0, full_pdf, color='blue', alpha=0.3)
    # ax.set_yticklabels([])
    # plt.legend(loc='best')
    # # ax.set_yticks([])
    # ax.set_title('Mixture von Mises PDF on a Circle', va='bottom')
    # plt.show()

    # plt.figure(figsize=(6,6))
    # ax = plt.subplot(111, polar=True)
    # ax.set_ylim(-0.5, np.max([f0_pdf, fF_pdf, fS_pdf, full_pdf]) * 1.2)
    # ax.plot(THETA_GRID, f0_pdf, color='blue', linewidth=1, label=r"$f_0(\theta)$")
    # ax.plot(THETA_GRID, fF_pdf, color='red', linewidth=1, label=r"$f_f(\theta)$")
    # ax.plot(THETA_GRID, fS_pdf, color='black', linewidth=1, label=r"$f_s(\theta)$")
    # ax.plot(THETA_GRID, full_pdf, color='green', linewidth=2, label=r"$f(\theta)$")
    # # ax.fill_between(THETA_GRID, 0, full_pdf, color='blue', alpha=0.3)
    # ax.set_yticklabels([])
    # plt.legend(loc='best')
    # # ax.set_yticks([])
    # ax.set_title('Mixture von Mises PDF on a Circle', va='bottom')
    # plt.show()

    return full_pdf

def sample_from_pdf(theta_grid, pdf_values):
    cdf_values = cumulative_trapezoid(pdf_values, theta_grid, initial=0)  # numerical CDF using cumulative trapezoid
    cdf_values /= cdf_values[-1]  # normalize to 1
    return np.interp(np.random.rand(1), cdf_values, theta_grid)[0]
