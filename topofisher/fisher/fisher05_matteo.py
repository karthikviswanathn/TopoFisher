import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Ellipse
import sys

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

fontsize = 20
ticksize = 10
matplotlib.rcParams.update({'font.size': fontsize})

def gaussian(x, mu, sig):
    # return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / sig / np.sqrt(2 * np.pi)

names = [r"$\alpha$", r"$\sigma_{\rm{log}}$$_M$", r"$\rm{log}$$M_0$", r"$\rm{log}$$M_1$", r"$\rm{log}$$M_{\rm{min}}$", r"$M_{\nu}$"+" "+r"$\rm{(eV)}$", r"$n_{\rm{s}}$", r"$\Omega_{\rm{b}}$", r"$\Omega_{\rm{m}}$", r"$\sigma_8$", r"$h$"]
means = [1.1, 0.2, 14, 14, 13.65, 0, 0.9624, 0.049, 0.3175, 0.834, 0.6711]

whichcov = int(sys.argv[1])
if whichcov == 3:  # THIS
    cov = [np.genfromtxt("cov_densities_meanD_10_7500_10000_7500_1_5_15_30_60_100_ran49.txt"), np.genfromtxt("cov_densities_meanD_7500_10000_7500_P0kP2kB0k_ran49.txt"), np.genfromtxt("cov_densities_meanD_10_7500_10000_7500_1_5_15_30_60_100_P0kP2kB0k_ran49.txt")]
    for haha in range(len(cov)):
        cov[haha] = np.linalg.inv(np.linalg.inv(cov[haha])[:11,:11])
        for haha2 in range(5,11):
            print(names[haha2], np.sqrt(cov[haha][haha2, haha2]), np.sqrt(cov[haha][haha2, haha2])/np.sqrt(2.4))
    # covlabel = [r"$PH$", r"$P^{\rm{g}}+B^{\rm{g}}$", r"$PH+P^{\rm{g}}+B^{\rm{g}}$"]
    covlabel = [r"$PH$", r"$P^{(0)}+P^{(2)}+B^{(0)}$", r"$PH+P^{(0)}+P^{(2)}+B^{(0)}$"]
    # covlabel = ["Histogram (PH)", "Power Spectrum + Bispectrum", "Combined"]  # ; $k_{max}=0.4$ $Mpc/h$"
# elif whichcov == 2:
#     cov = [np.genfromtxt("cov_densities_meanD_50_1_7500_9000_7500_16.txt"), np.genfromtxt("cov_Pk_P0kP2k_7500_9000.txt")]
#     covlabel = [r"$PH=PH_{birth}+PH_{death}$", r"$PS=PS_0+PS_2$"]
elif whichcov == 4:  # THIS
    cov = []
    covthis = np.genfromtxt("cov_densities_meanD_10_7500_10000_7500_1_5_15_30_60_100_ran49.txt")
    cov.append(np.linalg.inv(np.linalg.inv(covthis)[:11,:11]))
    covthis_HODfixed = np.linalg.inv(np.linalg.inv(covthis)[5:11,5:11])
    filled = np.zeros((11,11))
    for i in range(filled.shape[0]):
        for j in range(filled.shape[1]):
            if i >= 5 and j >= 5:
                filled[i,j] = covthis_HODfixed[i-5, j-5]
    cov.append(filled)   
    # covlabel = [r"$PH_{\rm{marg}}$", r"$PH_{\rm{fixed}}$"]
    covlabel = [r"$PH$", r"$PH^{\rm{HOD\mbox{-}fixed}}$"]
elif whichcov == 5:  # THIS
    cov = []
    covthis = np.genfromtxt("cov_densities_meanD_7500_10000_7500_P0kP2kB0k_ran49.txt")
    cov.append(np.linalg.inv(np.linalg.inv(covthis)[:11,:11]))
    covthis_HODfixed = np.linalg.inv(np.linalg.inv(covthis)[5:11,5:11])
    filled = np.zeros((11,11))
    for i in range(filled.shape[0]):
        for j in range(filled.shape[1]):
            if i >= 5 and j >= 5:
                filled[i,j] = covthis_HODfixed[i-5, j-5]
    cov.append(filled)   
    # covlabel = [r"$(P^{\rm{g}}+B^{\rm{g}})_{\rm{marg}}$", r"$(P^{\rm{g}}+B^{\rm{g}})_{\rm{fixed}}$"]
    covlabel = [r"$P^{(0)}+P^{(2)}+B^{(0)}$", r"$(P^{(0)}+P^{(2)}+B^{(0)})^{\rm{HOD\mbox{-}fixed}}$"]
# elif whichcov == 1:
#     cov = [np.genfromtxt("cov_densities_meanD_50_1_7500_9000_7500_16.txt")]
#     covlabel = [r"$PH=PH_{birth}+PH_{death}$"]
# elif whichcov == 6:
#     cov = [np.genfromtxt("cov_densities_meanD_50_3_22500_27000_22500_16.txt"), np.genfromtxt("cov_densities_meanD_50_3_22500_27000_22500_16_fixHOD_filled.txt")]
#     covlabel = [r"$PH^{\rm{sub}}_{\rm{HODmarg}}$", r"$PH^{\rm{sub}}_{\rm{HODfixed}}$"]

if len(cov) == 3 and whichcov == 3:
    # colorr = ["tab:blue", "tab:green", "tab:orange"]
    # gaussian_colorr = ["tab:blue", "tab:green", "tab:orange"]
    # # ellipse_alphaa = [[0, 0.6, 0.4], [0, 0.4, 0.3], [0, 0.6, 0.3]]
    # ellipse_alphaa = [[0, 0.35, 0.2], [0, 0.55, 0.25], [0, 0.75, 0.45]]
    # ellipse_lw = [1, 1, 1]
    colorr = ["tab:blue", "green", "tab:orange"]
    gaussian_colorr = ["tab:blue", "tab:green", "tab:orange"]
    ellipse_alphaa = [[0, 0.3, 0.2],[0, 0.5, 0.25], [0, 0.65, 0.45]]
    ellipse_lw = [1, 1, 1]
elif len(cov) == 2 and whichcov == 2:
    colorr = ["tab:blue", "green"]
    gaussian_colorr = ["tab:blue", "#15B01A"]
    ellipse_alphaa = [[0, 0.3, 0.3], [0, 1, 0.35]]
    ellipse_lw = [0.8, 1.1]
elif len(cov) == 2 and whichcov == 4:
    colorr = ["tab:green", "tab:red"]
    gaussian_colorr = ["tab:green", "tab:red"]
    ellipse_alphaa = [[0, 0.5, 0.35], [0, 0.95, 0.3]]
    ellipse_lw = [0.8, 1.1]
elif len(cov) == 2 and whichcov == 5:
    colorr = ["tab:blue", "fuchsia"]
    gaussian_colorr = ["tab:blue", "orchid"]
    ellipse_alphaa = [[0, 0.6, 0.35], [0, 0.9, 0.3]]
    ellipse_lw = [0.8, 1.1]
elif len(cov) == 1:
    colorr = ["tab:green"]
    gaussian_colorr = ["tab:green"]
    ellipse_alphaa = [[0, 1, 0.6]]
    ellipse_lw = [1.1]
elif len(cov) == 2 and whichcov == 6:
    colorr = ["tab:green", "tab:red"]
    gaussian_colorr = ["tab:green", "tab:red"]
    ellipse_alphaa = [[0, 0.6, 0.5], [0, 0.95, 0.3]]
    ellipse_lw = [0.8, 1.1]

names = [r"$\alpha$", r"$\sigma_{\rm{log}}$$_M$", r"$\rm{log}$$M_0$", r"$\rm{log}$$M_1$", r"$\rm{log}$$M_{\rm{min}}$", r"$\sum$$m_\nu$"+" "+r"$\rm{(eV)}$", r"$n_{\rm{s}}$", r"$\Omega_{\rm{b}}$", r"$\Omega_{\rm{m}}$", r"$\sigma_8$", r"$h$"]
means = [1.1, 0.2, 14, 14, 13.65, 0, 0.9624, 0.049, 0.3175, 0.834, 0.6711]

m = int(sys.argv[2])  # 5=cosmo only; 0=all
sigma_axis = 3
# sigma_plot = 2

alpha = [0, 1.52, 2.48, 3.44]

names = names[m:]
means = means[m:]

order_cosmo = [3, 2, 5, 1, 4, 0]
order_hod = [2, 3, 4, 1, 0]
if m == 5:
    order = order_cosmo
elif m == 0:
    order = list(np.array(order_cosmo)+5)
    order.extend(np.array(order_hod))
print(order)


# onesig = 0
onesigt = []
for i in range(len(cov)):
    # dad = -1
    onesigt.append(cov[i][0,0])
    # if np.sqrt(cov[i][0,0]) > onesig:
    #     onesig = np.sqrt(cov[i][0,0])
    #     dad = i
cov = [x for _, x in sorted(zip(onesigt, cov), reverse=True)]
covlabel = [x for _, x in sorted(zip(onesigt, covlabel), reverse=True)]

fig, axs = plt.subplots(11-m, 11-m, figsize=(10, 10))
for h in range(11-m):
    hid = order[h]
    
    if whichcov == 3:
        xrlim = means[hid]+alpha[sigma_axis]*np.sqrt(cov[1][hid+m,hid+m])
        xllim = means[hid]-alpha[sigma_axis]*np.sqrt(cov[1][hid+m,hid+m])
    else:
        xrlim = means[hid]+alpha[sigma_axis]*np.sqrt(cov[0][hid+m,hid+m])
        xllim = means[hid]-alpha[sigma_axis]*np.sqrt(cov[0][hid+m,hid+m])
    # print(xrlim)
    xx = np.linspace(xllim, xrlim, 100)
    
    for v in range(11-m):
        axs[v, h].tick_params(direction="in", labelsize=ticksize, labelrotation=45)
        if h == 1 and whichcov in [3, 4]:
            axs[v, h].set_xticks([0.04, 0.06])
            axs[v, h].set_xticklabels(["0.04", "0.06"])
        if v == 1 and whichcov in [3, 4]:
            axs[v, h].set_yticks([0.04, 0.06])
            axs[v, h].set_yticklabels(["0.04", "0.06"])
        axs[v, h].locator_params(tight=True, nbins=4)

        vid = order[v]

        axs[v, h].set_xlim(left=xllim, right=xrlim)
        if h > v:
            fig.delaxes(axs[v][h])
        elif v == h:
            if whichcov == 1:
                fig.delaxes(axs[v][h])
            else:
                for i in range(len(cov)):
                    axs[v, h].plot(xx, gaussian(xx, means[hid], np.sqrt(cov[i][hid+m,hid+m])), lw=1, color=gaussian_colorr[i], label=covlabel[i])
                axs[v, h].set_ylim(bottom=0)
                axs[v, h].tick_params(left=False)
                axs[v, h].set_yticklabels([])
        else:
            if whichcov == 3:
                ytlim = means[vid]+alpha[sigma_axis]*np.sqrt(cov[1][vid+m,vid+m])
                yblim = means[vid]-alpha[sigma_axis]*np.sqrt(cov[1][vid+m,vid+m])            
            else:
                ytlim = means[vid]+alpha[sigma_axis]*np.sqrt(cov[0][vid+m,vid+m])
                yblim = means[vid]-alpha[sigma_axis]*np.sqrt(cov[0][vid+m,vid+m])
            axs[v, h].set_ylim(top=ytlim, bottom=yblim)
            
            for i in range(len(cov)):
            # for i in [0]:
                a2 = (cov[i][hid+m,hid+m] + cov[i][vid+m,vid+m]) / 2 + np.sqrt((cov[i][hid+m,hid+m] - cov[i][vid+m,vid+m])**2 / 4 + cov[i][hid+m, vid+m]**2)
                b2 = (cov[i][hid+m,hid+m] + cov[i][vid+m,vid+m]) / 2 - np.sqrt((cov[i][hid+m,hid+m] - cov[i][vid+m,vid+m])**2 / 4 + cov[i][hid+m, vid+m]**2)
                tan2th = 2 * cov[i][hid+m, vid+m] / (cov[i][hid+m,hid+m] - cov[i][vid+m,vid+m])

                # print(names[vid], names[hid])
                # print(a2, b2, tan2th)
                # print("\n")

                a = np.sqrt(a2)
                b = np.sqrt(b2)
                th = np.degrees(np.arctan(tan2th) / 2)
                th = np.degrees(np.arctan2(2 * cov[i][hid+m, vid+m], cov[i][hid+m,hid+m] - cov[i][vid+m,vid+m]) / 2)

                for sigma_plot in [2, 1]:
                    # if sigma_plot == 1:
                    #     ellipse_alpha = 0.9
                    # elif sigma_plot == 2:
                    #     ellipse_alpha = 0.5
                    ellipse1 = Ellipse(xy=(means[hid], means[vid]), width=alpha[sigma_plot]*a*2, height=alpha[sigma_plot]*b*2, angle=th, lw=0, alpha=ellipse_alphaa[i][sigma_plot], color=colorr[i])
                    ellipse2 = Ellipse(xy=(means[hid], means[vid]), width=alpha[sigma_plot]*a*2, height=alpha[sigma_plot]*b*2, angle=th, lw=1, alpha=1, color=colorr[i], fill=False)
                    axs[v, h].add_patch(ellipse1)
                    axs[v, h].add_patch(ellipse2)

        if h != 0:
            axs[v, h].set_yticklabels([])
        elif v != 0:
            axs[v, h].set_ylabel(names[vid])
        
        if v != range(11-m)[-1]:
            axs[v, h].set_xticklabels([])
        else:
            axs[v, h].set_xlabel(names[hid])

lines = []
labels = []
axLine, axLabel = axs[0, 0].get_legend_handles_labels()
lines.extend(axLine)
labels.extend(axLabel)
if whichcov != 4 and whichcov != 1 and whichcov != 5 and whichcov != 6:
    lines[0], lines[1] = lines[1], lines[0]
    labels[0], labels[1] = labels[1], labels[0]
leg = fig.legend(lines, labels, bbox_to_anchor=(0.9,0.8), frameon=False, labelcolor='linecolor')
for legobj in leg.legendHandles:
    legobj.set_linewidth(10.0)

if len(cov) == 1:
    for i in range(11-m):
        fig.delaxes(axs[i][i])

fig.align_ylabels(axs[:, 0])
fig.align_xlabels(axs[-1, :])

# fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()
# fig.savefig("fisher_"+str(whichcov)+"_"+str(m)+".pdf", dpi=600)
fig.savefig("fisher_cosmo_"+str(whichcov)+"_"+str(m)+".pdf", dpi=600)