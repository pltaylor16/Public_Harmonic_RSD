import numpy as np
import camb
from camb import model
import os.path as op
from math import pi as pi
import scipy
from scipy.special import sinc
from scipy.constants import c # speed of light in m/s
print('Using CAMB %s installed at %s'%(camb.__version__,op.dirname(camb.__file__)))



class ProjectedSpectra():
    
    def __init__(self, param_dict, z, chi_fid, nz_survey, n_gals, weight_func_list, m_list, rsd_model, alt_rsd_model, sigma8_rescale, n_int_sub_divs, harmonic_weight, sigma_z, scale_dict, load_kernel_dic_name):
        
        self.param_dict = param_dict
        self.z = z
        '''this variable only used if doing harmonic weighting'''
        self.chi_fid = chi_fid
        self.nz_survey = nz_survey
        self.n_gals = n_gals
        '''This will only have 1 element in the harmonic case'''
        self.weight_func_list = weight_func_list
        '''this only used if doing harmonic weighting'''
        self.m_list = m_list
        self.rsd_model = rsd_model
        self.alt_rsd_model = alt_rsd_model
        self.sigma8_rescale = sigma8_rescale
        self.n_int_sub_divs = n_int_sub_divs
        self.harmonic_weight = harmonic_weight
        self.sigma_z = sigma_z

        '''a useful quantity'''
        self.n_tomo_bins = len(self.weight_func_list)
        self.grid_res = np.shape(self.z)[0]
        if self.harmonic_weight == False:
            self.n_weights = len(self.weight_func_list)
        elif self.harmonic_weight == True:
            self.n_weights = len(self.m_list)

        self.load_kernel_dic_name = load_kernel_dic_name

        '''get the scales'''
        self.scale_dict = scale_dict
        self.kmin = self.scale_dict['kmin']
        self.kmax = self.scale_dict['kmax']
        self.nk = self.scale_dict['nk']
        self.n_ell = self.scale_dict['n_ell']
        self.lmin = self.scale_dict['lmin']
        self.lmax = self.scale_dict['lmax']
        self.ell = np.logspace(np.log10(self.lmin), np.log10(self.lmax), self.n_ell)




        '''First normalise the weight functions and get gals per bin'''
        print ('Normalising the Survey Window')
        self.nz_survey = self.nz_survey / np.trapz(self.nz_survey, self.z)
        for i in range(self.n_tomo_bins):
            self.weight_func_list[i] = self.weight_func_list[i] / (np.trapz(self.weight_func_list[i], self.z)) 

        weight_sum = np.zeros(self.grid_res)
        for t in range(self.n_tomo_bins):
            weight_sum += weight_func_list[t]
        self.n_gals_list = []
        for i in range(self.n_tomo_bins):
            gal_frac = np.trapz(weight_func_list[i],z) / np.trapz(weight_sum,z)
            self.n_gals_list += [gal_frac * self.n_gals]
       



    '''fix p(k) over the entire spectrum with no z evolution'''
    def do_cosmo_calc(self):

        '''first compute the average z over the survey'''
        z_avg = np.trapz(self.z * self.nz_survey, self.z) / np.trapz(self.nz_survey, self.z)


        '''initiate CAMB'''
        # Set up a new set of parameters for CAMB
        pars = camb.CAMBparams()
        # This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
        pars.set_cosmology(H0 = self.param_dict['h0'] * 100., ombh2=self.param_dict['omb'] * self.param_dict['h0'] ** 2., omch2=(self.param_dict['omm'] - self.param_dict['omb']) * self.param_dict['h0'] ** 2., mnu=self.param_dict['mnu'], omk=self.param_dict['omk'], tau=self.param_dict['tau'])
        pars.set_dark_energy(w=self.param_dict['w0'], wa=self.param_dict['wa'])
        pars.InitPower.set_params(As=self.param_dict['As'], ns=self.param_dict['ns'], r=0.)
        # Calculate results for these parameters
        results = camb.get_results(pars)
        # Note non-linear corrections couples to smaller scales than you want
        pars.set_matter_power(redshifts=[z_avg], kmax=self.kmax)
        
        if self.sigma8_rescale == True:
            print ('Rescaling A_s to get correct sigma_8')

            # Linear spectra
            pars.NonLinear = model.NonLinear_none
            results = camb.get_results(pars)
            kh, zs, pk = results.get_matter_power_spectrum(minkh=self.kmin, maxkh=self.kmax, npoints=self.nk)
            sigma8 = np.array(results.get_sigma8())
            #print(sigma8)
            new_As = (self.param_dict['sigma8'] / sigma8 ) ** 2. * self.param_dict['As']
            self.param_dict['As'] = new_As


        print ('Computing nonlinear power spectrum')
        '''now compute the nonlinear power spectrum using the rescaled As'''
        if self.sigma8_rescale == True:
            pars.InitPower.set_params(As=new_As, ns=self.param_dict['ns'], r=0.)
        results = camb.get_results(pars)
        # Note non-linear corrections couples to smaller scales than you want
        pars.set_matter_power(redshifts=[z_avg], kmax=self.kmax)
        # Non-Linear spectra (Halofit)
        pars.NonLinear = model.NonLinear_both
        results.calc_power_spectra(pars)
        kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=self.kmin, maxkh=self.kmax, npoints=self.nk)

        #dbugging
        '''rescale everything to remove factors of h'''
        k_nonlin = self.param_dict['h0'] * kh_nonlin
        pk_nonlin *= 1. / self.param_dict['h0'] ** 3.


        '''we want the quantities for the rest of the class'''
        self.k_nonlin, self.pk_nonlin = k_nonlin, pk_nonlin
        self.chi = results.comoving_radial_distance(self.z)
        self.k_par = self.k_nonlin.copy()
        

        '''get the average chi for each of the weights'''
        self.chi_list = []
        self.H_list = []
        self.z_list = []
        if self.harmonic_weight == False:
            for i in range(len(self.weight_func_list)):
                nz = self.weight_func_list[i] * self.nz_survey 
                z_avg_i = np.trapz(nz * self.z, self.z) / np.trapz(nz, self.z)
                self.chi_list += [results.comoving_radial_distance(z_avg_i)]
                self.H_list += [results.hubble_parameter(z_avg_i)]
                self.z_list += [z_avg_i]
        elif self.harmonic_weight == True:
            for i in range(self.n_weights):
                z_avg = np.trapz(self.z * self.nz_survey, self.z) / np.trapz(self.nz_survey, self.z)
                self.chi_list += [results.comoving_radial_distance(z_avg)]
                self.H_list += [results.hubble_parameter(z_avg)]
                self.z_list += [z_avg]


        return 0.


    
    def get_rsd_Pk(self, i, j, alt_spectra = False):

        self.param_dict['beta'] = self.param_dict['f'] / self.param_dict['b']

        #print ('Applying RSD Correction P(k)-->P(k,mu)')

        '''compute useful quantities for all models'''
        P = np.zeros((self.nk, self.n_ell))
        r0 = 0.5 * (self.chi_list[i] + self.chi_list[j])
        k = (self.k_par[:,None] ** 2. + ((self.ell[None,:] + 0.5) / r0) ** 2.) ** 0.5
        mu = self.k_par[:,None] / k

        '''interpolate onto the new grid'''
        pk_new_grid = np.zeros((self.nk, self.n_ell))
        for l in range(self.n_ell):
            pk_new_grid[:,l] = np.interp(k[:,l], self.k_nonlin, self.pk_nonlin[0,:])

        kaiser_factor = (1. + self.param_dict['beta'] * mu ** 2.)

        '''conpute the spectra for the baseline or the alternative model'''
        if alt_spectra == False:
            rsd_model = self.rsd_model
        elif alt_spectra == True:
            rsd_model = self.alt_rsd_model

        '''do the computation for different models'''   
        if rsd_model == 'None':
            P = self.param_dict['b'] ** 2. * np.ones((self.nk, self.n_ell)) * pk_new_grid
        elif rsd_model == 'Kaiser':
            P = self.param_dict['b'] ** 2. * kaiser_factor ** 2. * pk_new_grid
        elif rsd_model == 'Kaiser Gauss FOG':
            gauss = np.exp(-1. * k ** 2. * mu ** 2. * self.param_dict['f'] ** 2. * self.param_dict['sigma_v'] ** 2.)
            P = self.param_dict['b'] ** 2. * gauss * kaiser_factor ** 2. * pk_new_grid
        elif rsd_model == 'Kaiser Lorentz FOG':
            lorentzian = 1. / (1. + k ** 2. * mu ** 2. * self.param_dict['f'] ** 2. * self.param_dict['sigma_v'] ** 2. )
            P = self.param_dict['b'] ** 2. * lorentzian * kaiser_factor ** 2. * pk_new_grid


        try:
            P *= self.param_dict['p_amp']
        except:
            pass


        return P
        


    '''compute the harmonic radial efficiency'''
    def get_rad_eff_harmonic(self, i, j):

        #print ('Computing radial efficiency Kernels')

        '''redfine variable names  because I copied / pasted code below'''
        n_divs = self.n_int_sub_divs
        nr = self.weight_func_list[0]
        chi = self.chi
        nk = self.nk
        chi_ref = self.chi_fid
        k = self.k_par
        m1 = self.m_list[i]
        m2 = self.m_list[j]
        z = self.z

        '''normalise the tomobin'''
        nr = nr / np.trapz(nr, chi)
        
        '''initalise csome variables'''
        keep_idx = np.where(nr!= 0)
        delta_r = chi[int(np.max(keep_idx))] - chi[int(np.min(keep_idx))]
        bin_idx_width = np.shape(keep_idx[0])[0] / n_divs
        start_idx = int(np.min(keep_idx))
        start_centre = int(start_idx + bin_idx_width/2)
        alphas = np.zeros(n_divs)
        rns = np.zeros(n_divs)
        rns_ref = np.zeros(n_divs)
        delta_r_n = np.zeros(n_divs)
        r_plus = np.zeros(n_divs)
        r_minus = np.zeros(n_divs)
        delta_z = np.zeros(n_divs)
        nu = np.zeros(n_divs)
        nu_ref = np.zeros(n_divs)
        a = np.zeros(n_divs)
        b = np.zeros(n_divs)
        c = np.zeros(n_divs)
        d = np.zeros(n_divs)


        '''compute cnm and dnm to go between chi and chi_fid'''
        for a in range(n_divs):
            alphas[a]  = nr[int(start_centre + bin_idx_width * a)]
            rns[a] = chi[int(start_centre + bin_idx_width * a)]
            rns_ref[a] = chi_ref[int(start_centre + bin_idx_width * a)]
            delta_r_n[a] = chi[int(start_idx + bin_idx_width * (a+1)-1)] -  chi[int(start_idx + bin_idx_width * a)]
            r_plus[a] = chi[int(start_idx + bin_idx_width * (a+1)-1)] 
            r_minus[a] = chi[int(start_idx + bin_idx_width * (a))] 
            delta_z[a] = z[int(start_idx + bin_idx_width * (a+1)-1)] -  z[int(start_idx + bin_idx_width * a)]
            nu[a] = (chi[int(start_idx + bin_idx_width * (a+1)-1)] - chi[int(start_idx + bin_idx_width * (a))]) / delta_z[a]
            nu_ref[a] = (chi_ref[int(start_idx + bin_idx_width * (a+1)-1)] - chi_ref[int(start_idx + bin_idx_width * (a))]) / delta_z[a]
        a = nu_ref / nu
        b = rns_ref - rns * nu_ref / nu
               
        cm1 = 2. * pi * m1 / delta_r
        dm1 = 2. * pi * m1 * b / a / delta_r
        cm2 = 2. * pi * m2 / delta_r
        dm2 = 2. * pi * m2 * b / a / delta_r

        '''inialise some arrays'''
        prefactor = np.zeros((nk, n_divs, n_divs))
        trig = np.zeros((nk, n_divs, n_divs))
        R1, R2 = np.zeros((nk, n_divs, n_divs)), np.zeros((nk, n_divs, n_divs))
        R1_1, R1_2, R1_3, R1_4 = R1.copy(), R1.copy(), R1.copy(), R1.copy()
        R2_1, R2_2, R2_3, R2_4 = R2.copy(), R2.copy(), R2.copy(), R2.copy()
        I1, I2 = R1.copy(), R2.copy()
        I1_1, I1_2, I1_3, I1_4 = I1.copy(), I1.copy(), I1.copy(), I1.copy()
        I2_1, I2_2, I2_3, I2_4 = I2.copy(), I2.copy(), I2.copy(), I2.copy()
        
        
        '''compute quantities in the final sum'''
        prefactor[None,:,:] = alphas[None,:,None] * alphas[None,None,:] 
        denominator1 = cm1**2. - k[:,None,None]**2. 
        denominator2 = cm2**2. - k[:,None,None]**2. 
        R1_1 = - cm1 * np.cos(r_minus[None,:,None]*k[:,None,None])* np.sin(r_minus[None,:,None]*cm1+dm1[None,:,None]) / denominator1
        R1_2 = k[:,None,None] * np.sin(r_minus[None,:,None]*k[:,None,None])* np.cos(r_minus[None,:,None]*cm1+dm1[None,:,None])  / denominator1
        R1_3 = cm1 * np.cos(r_plus[None,:,None]*k[:,None,None]) * np.sin(r_plus[None,:,None]*cm1+dm1[None,:,None])  / denominator1
        R1_4 = -k[:,None,None] * np.sin(r_plus[None,:,None]*k[:,None,None]) * np.cos(r_plus[None,:,None]*cm1+dm1[None,:,None])  / denominator1
        R1 = R1_1 + R1_2 + R1_3 + R1_4
        R2_1 = - cm2 * np.cos(r_minus[None,None,:]*k[:,None,None])* np.sin(r_minus[None,None,:]*cm2+dm2[None,None,:]) / denominator2
        R2_2 = k[:,None,None] * np.sin(r_minus[None,None,:]*k[:,None,None]) * np.cos(r_minus[None,None,:]*cm2+dm2[None,None,:]) / denominator2
        R2_3 = cm2 * np.cos(r_plus[None,None,:]*k[:,None,None]) * np.sin(r_plus[None,None,:]*cm2+dm2[None,None,:]) / denominator2
        R2_4 = -k[:,None,None] * np.sin(r_plus[None,None,:]*k[:,None,None]) * np.cos(r_plus[None,None,:]*cm2+dm2[None,None,:]) / denominator2
        R2 = R2_1 + R2_2 + R2_3 + R2_4
        I1_1 = cm1 * np.sin(r_minus[None,:,None]*k[:,None,None])* np.sin(r_minus[None,:,None]*cm1+dm1[None,:,None])  / denominator1
        I1_2 = k[:,None,None] * np.cos(r_minus[None,:,None]*k[:,None,None]) * np.cos(r_minus[None,:,None]*cm1+dm1[None,:,None])  / denominator1
        I1_3 = -cm1 * np.sin(r_plus[None,:,None]*k[:,None,None]) * np.sin(r_plus[None,:,None]*cm1+dm1[None,:,None])  / denominator1
        I1_4 = -k[:,None,None] * np.cos(r_plus[None,:,None]*k[:,None,None]) * np.cos(r_plus[None,:,None]*cm1+dm1[None,:,None])  / denominator1
        I1 = I1_1 + I1_2 + I1_3 + I1_4
        I2_1 = cm2 * np.sin(r_minus[None,None,:]*k[:,None,None])* np.sin(r_minus[None,None,:]*cm2+dm2[None,None,:]) / denominator2
        I2_2 = k[:,None,None] * np.cos(r_minus[None,None,:]*k[:,None,None]) * np.cos(r_minus[None,None,:]*cm2+dm2[None,None,:]) / denominator2
        I2_3 = -cm2 * np.sin(r_plus[None,None,:]*k[:,None,None]) * np.sin(r_plus[None,None,:]*cm2+dm2[None,None,:]) / denominator2
        I2_4 = -k[:,None,None] * np.cos(r_plus[None,None,:]*k[:,None,None]) * np.cos(r_plus[None,None,:]*cm2+dm2[None,None,:]) / denominator2
        I2 = I2_1 + I2_2 + I2_3 + I2_4

        '''do the final sum'''
        summand = prefactor * (R1 * R2 + I1 * I2) 
        Kij = np.sum(np.sum(summand, axis = 1), axis =1)

        return Kij




    '''compute the radial efficeny breaking up into piecewise functions'''
    def get_radial_efficiency(self, i, j):

        if self.harmonic_weight == False:
            return self.get_rad_eff_tomo(i, j)
        elif self.harmonic_weight == True:
            return self.get_rad_eff_harmonic(i,j)
        else:
            print('Error: harmonic_weight must be True or False')



    '''compute all the kernel at once and store in a dictionary'''
    def compute_rad_kernel_dict(self, save_dict = False, save_name = 'example_dict.npy'):

        if self.load_kernel_dic_name == None:
            print ('Computing the radial efficency kernels')
            self.kernel_dict = {}
            for i in range(self.n_weights):
                for j in range(self.n_weights):
                    if i >= j:
                        self.kernel_dict['%s%s' %(i,j)] = self.get_radial_efficiency(i,j)
                        self.kernel_dict['%s%s' %(j,i)] = self.kernel_dict['%s%s' %(i,j)] 
            if save_dict == True:
                np.save(save_name, self.kernel_dict)


        elif self.load_kernel_dic_name != None:
            print ('Loading the radial efficiency kernels')
            self.kernel_dict = np.load(self.load_kernel_dic_name, allow_pickle='TRUE').item()

        return 0.


    
    def get_spectra(self, i, j, alt_spectra = False):
        prefactor = 1. / pi / self.chi_list[i] / self.chi_list[j]
        power = self.get_rsd_Pk(i, j, alt_spectra)
        eff_kernel = self.kernel_dict['%s%s' %(i,j)]
        return (prefactor * np.trapz(eff_kernel[:,None] * power, self.k_par, axis = 0))
    
    


    def get_shot_noise(self, i, j):
        if self.harmonic_weight == False:
            if i == j:
                shot_noise = 1. / self.n_gals_list[i]
            else:
                shot_noise = 0.
        elif self.harmonic_weight == True:
            keep_idx = np.where(self.weight_func_list[0] != 0)
            delta_r = self.chi_fid[int(np.max(keep_idx))] - self.chi_fid[int(np.min(keep_idx))]
            w1 = np.cos(2. * pi * self.m_list[i] * self.chi_fid / delta_r)
            w2 = np.cos(2. * pi * self.m_list[j] * self.chi_fid / delta_r)
            shot_noise = 1. / self.n_gals * np.trapz(self.weight_func_list[0] * w1 * w2, self.z)
        else:
            pass
        return shot_noise
        

    def get_noisy_spectra(self, i, j, alt_spectra = False):
        return (self.get_spectra(i,j, alt_spectra) + self.get_shot_noise(i,j))












