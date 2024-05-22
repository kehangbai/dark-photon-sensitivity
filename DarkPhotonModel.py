import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import darkcast


class DarkPhotonModel:
    
    def __init__(self, BR_hApAp=1e-4, Lmin=0.001, Lmax=0.3):
        # dark photon mass
        self.m_Ap = np.array([mass*1e-1 for mass in range(2, 200)])[1:]

        # kinetic mixing
        self.eps_list = np.logspace(-7, -3, 30)

        # BR of h->2A'
        self.BR_hApAp = BR_hApAp
        
        # ggF production xsection for mH = 125 GeV (fb)
        self.sigma_ggF = 48580
        
        # Full run2 luminosity (fb^-1)
        self.Lrun2 = 139

        # Load dark photon model in darkcast/models and DARKCAST_MODEL_PATH.
        models = darkcast.Models()
        channel = "mu_mu"
        
        # Loop over all the models.
        for name, model in models.items():
            # Br(A' -> mu mu)
            self.BR_Ap = np.array([model.bfrac(channel, mass) for mass in self.m_Ap])
    
            # Total decay width assuming epsilon=1
            self.total_width = np.array([model.width("total", mass) for mass in self.m_Ap])

        # Min and max decay length for acceptance in m
        self.Lmin = Lmin
        self.Lmax = Lmax
    
    def get_boost(self):
        m_h = 125
        E_Ap = m_h/2
        p_Ap = np.sqrt(E_Ap**2 - self.m_Ap**2)
        return p_Ap/self.m_Ap

    def get_tau(self):
        decay_width = self.total_width*(self.eps_list[:, np.newaxis]**2)
        return (1/decay_width)*6.58*1e-25

    def get_ctau(self, m, eps):
        decay_width = self.total_width * eps * eps
        tau = (1/decay_width)*6.58*1e-25
        c = 3*10**8
        return c*tau

    def get_decay_prob(self, boost, ctau):
        return np.exp(-self.Lmin/(boost*ctau))-np.exp(-self.Lmax/(boost*ctau))
    
    def get_acceptance_per_eps(self):
        c = 3*10**8
        tau = self.get_tau()
        boost = self.get_boost()
        acceptance = self.get_decay_prob(boost, c*tau)
        return np.array(acceptance)
    
    def num_2Ap(self):
        # 2 A' -> mu mu
        xs, L, Br = self.sigma_ggF, self.Lrun2, self.BR_Ap
        return xs*L*Br*Br
    
    def num_1Ap(self):
        # 1 A' -> mu mu
        xs, L, Br = self.sigma_ggF, self.Lrun2, self.BR_Ap
        return 2*xs*L*Br*(1-Br)
    
    def num_0Ap(self):
        # 0 A' -> mu mu
        xs, L, Br = self.sigma_ggF, self.Lrun2, self.BR_Ap
        return xs*L*(1-Br)*(1-Br)

    def num_2Ap_2accept(self, acceptance):
        n_Ap = self.num_2Ap()
        n_sig = self.BR_hApAp * n_Ap
        return n_sig*(acceptance**2)
    
    def num_2Ap_1accept(self, acceptance):
        n_Ap = self.num_2Ap()
        n_sig = self.BR_hApAp * n_Ap
        return n_sig*2*acceptance*(1-acceptance)
    
    def num_2Ap_0accept(self, acceptance):
        n_Ap = self.num_2Ap()
        n_sig = self.BR_hApAp * n_Ap
        return n_sig*(1-acceptance)*(1-acceptance)
    
    def num_1Ap_1accept(self, acceptance):
        n_Ap = self.num_1Ap()
        n_sig = self.BR_hApAp * n_Ap
        return n_sig*2*acceptance
    
    def num_1Ap_0accept(self, acceptance):
        n_Ap = self.num_1Ap(self.BR_Ap)
        n_sig = self.BR_hApAp * n_Ap
        return n_sig*2*(1-acceptance)

    def get_n_sig_list(self):
        n_sig_list = []
        acceptance = self.get_acceptance_per_eps()
        n_sig_list.append(self.num_2Ap_2accept(acceptance))
        n_sig_list.append(self.num_2Ap_1accept(acceptance)+self.num_1Ap_1accept(acceptance))
        return np.array(n_sig_list)
    
    def plot_num_sig(self, title_list="Num. events"):
    
        x = self.m_Ap
        y = self.eps_list
        X, Y = np.meshgrid(x, y)

        # Epsilon for ctau=1 mm
        y1 = np.sqrt(self.get_ctau(x, 1)/0.0001)
        y2 = np.sqrt(self.get_ctau(x, 1)/0.001)
        y3 = np.sqrt(self.get_ctau(x, 1)/0.005)

        n_sig_list = self.get_n_sig_list()
    
        N = len(n_sig_list)
        fig, ax = plt.subplots(1, N, figsize=(10*N,8))
        for i in range(N):
            z = n_sig_list[i]
            cs = ax[i].contourf(X, Y, z, cmap=cm.Blues)
            ax[i].plot(x, y1, color="dimgray", ls="--")
            ax[i].annotate(r"$c\tau = 0.1$ mm", (x[-80], y1[-80]), textcoords="offset points", xytext=(0,15), ha='center') 
            ax[i].plot(x, y2, color="dimgray", ls="--")
            ax[i].annotate(r"$c\tau = 1$ mm", (x[-80], y2[-80]), textcoords="offset points", xytext=(0,15), ha='center') 
            ax[i].plot(x, y3, color="dimgray", ls="--")
            ax[i].annotate(r"$c\tau = 5$ mm", (x[-80], y3[-80]), textcoords="offset points", xytext=(0,15), ha='center') 
            ax[i].set_yscale('log')
            ax[i].set_xscale('log')
            cbar = fig.colorbar(cs)
            ax[i].text(0.5, 0.8, f"$Br(h\\to A'A')=${self.BR_hApAp:.0e}", transform=ax[i].transAxes, fontsize=16, color='black')
            ax[i].text(0.5, 0.75, f"$L_{{dis}} \in [{self.Lmin*1e3:.0f},{self.Lmax*1e3:.0f}]$ mm", transform=ax[i].transAxes, fontsize=16, color='black')
            ax[i].set_ylabel(r"$\epsilon$", fontsize=20)
            ax[i].set_xlabel(r"$m_{A'}$ (GeV)", fontsize=20)
            ax[i].set_title(title_list[i], fontsize=24)
        plt.show
        plt.savefig(f"./HAMA_sensitivity_{self.Lmin*1e3:.0f}mm_{self.Lmax*1e3:.0f}mm.png")