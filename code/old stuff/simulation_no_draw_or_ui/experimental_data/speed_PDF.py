import json
import numpy as np
import os

"""
convert histograms of fish speeds under different conditions into probability density functions (PDFs)
"""

class SpeedPDF:
    def __init__(self, bins, pdf=None, pdf_inside=None, pdf_outside=None):
        self.bins = bins
        self.pdf = pdf
        self.pdf_inside = pdf_inside
        self.pdf_outside = pdf_outside

def gaussian_pdf(x, mean, std):
    coeff = 1 / (std * np.sqrt(2 * np.pi))
    exponent = -((x - mean) ** 2) / (2 * std**2)
    return coeff * np.exp(exponent)

speed_pdfs = {}
speed_hists = json.load(open(os.path.join(os.path.dirname(__file__), "speed_histograms.json"), "r"))
for k in speed_hists: # convert to pdf
    speed_pdfs[k] = {}

    bins = np.array(speed_hists[k]["bins"])
    s_bins = 0.5 * (bins[:-1] + bins[1:])
    speed_pdfs[k]["bins"] = s_bins
    for k_ in speed_hists[k]:
        if k_ == "bins":
            continue
        hist = np.array(speed_hists[k][k_])
        pdf_values = hist / np.sum(hist)
    
        mean = np.sum(s_bins * pdf_values)
        variance = np.sum(((s_bins - mean) ** 2) * pdf_values)
        std_dev = np.sqrt(variance)
        g_pdf = gaussian_pdf(s_bins, mean, std_dev) / 0.01
        g_pdf = g_pdf / np.sum(g_pdf)
        speed_pdfs[k][k_] = g_pdf
        # print(mean, std_dev, mean)
        # import matplotlib.pyplot as plt
        # plt.plot(s_bins, pdf_values)
        # plt.show()
        # plt.plot(s_bins, g_pdf)
        # plt.show()

HOMOGENEOUS_1AB_PDF = SpeedPDF(
    bins=speed_pdfs["homogeneous_1AB"]["bins"], 
    pdf=speed_pdfs["homogeneous_1AB"]["counts"]
)

HOMOGENEOUS_10AB_PDF = SpeedPDF(
    bins=speed_pdfs["homogeneous_10AB"]["bins"], 
    pdf=speed_pdfs["homogeneous_10AB"]["counts"]
)

HETEROGENEOUS_1AB_PDF = SpeedPDF(
    bins=speed_pdfs["heterogeneous_1AB"]["bins"], 
    pdf_inside=speed_pdfs["heterogeneous_1AB"]["counts_inside"], 
    pdf_outside=speed_pdfs["heterogeneous_1AB"]["counts_outside"]
)

HETEROGENEOUS_10AB_PDF = SpeedPDF(
    bins=speed_pdfs["heterogeneous_10AB"]["bins"], 
    pdf_inside=speed_pdfs["heterogeneous_10AB"]["counts_inside"], 
    pdf_outside=speed_pdfs["heterogeneous_10AB"]["counts_outside"]
)

# for 10 fish use 1 fish pdfs for now as there is probably an 
# issue with how we measure speed when theres many fish, 
# as it says that sometimes the fish move 1 meter in 1 second, 
# so there is for sure something wrong, we get the same shape of
# distribution, but it's stretched too much and makes fish too fast
SPEED_PDF_MAP = {
    # (sees_fish, sees_spots, under_spot)
    (False, False, False): (HOMOGENEOUS_1AB_PDF.bins, HOMOGENEOUS_1AB_PDF.pdf),
    (False, False, True): (HOMOGENEOUS_1AB_PDF.bins, HOMOGENEOUS_1AB_PDF.pdf),
    (True,  False, True): (HOMOGENEOUS_1AB_PDF.bins, HOMOGENEOUS_1AB_PDF.pdf),
    (True,  False, False): (HOMOGENEOUS_1AB_PDF.bins, HOMOGENEOUS_1AB_PDF.pdf),

    (False, True, False): (HETEROGENEOUS_1AB_PDF.bins, HETEROGENEOUS_1AB_PDF.pdf_outside),
    (True,  True, False): (HETEROGENEOUS_1AB_PDF.bins, HETEROGENEOUS_1AB_PDF.pdf_outside),
    (False, True, True): (HETEROGENEOUS_1AB_PDF.bins, HETEROGENEOUS_1AB_PDF.pdf_inside),
    (True,  True, True): (HETEROGENEOUS_1AB_PDF.bins, HETEROGENEOUS_1AB_PDF.pdf_inside),
}

# for k in SPEED_PDF_MAP:
#     midpoints, pdf = SPEED_PDF_MAP[k] 
#     p = pdf * 0.01
#     mean = np.sum(midpoints * p)
#     variance = np.sum(((midpoints - mean) ** 2) * p)
#     std_dev = np.sqrt(variance)
#     print(mean,std_dev)