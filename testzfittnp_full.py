import os
ncpu = 1
os.environ["OMP_NUM_THREADS"] = str(ncpu)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncpu)
os.environ["MKL_NUM_THREADS"] = str(ncpu)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncpu)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncpu)
os.environ["TF_NUM_INTEROP_THREADS"] = str(ncpu)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(ncpu)

import zfit
import narf
import hist
import ROOT
import scipy
import numpy as np
import mplhep
import matplotlib.pyplot as plt
import pickle as pkl
import tensorflow as tf
from multiprocessing import Pool,Process
#import os
import time
#import array

start = time.time()
start_cpu = time.process_time()

#os.environ["OMP_NUM_THREADS"] = "2"
#os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
#os.environ["TF_NUM_INTEROP_THREADS"] = "1"

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

#zfit.run.set_n_cpu(n_cpu=1,strict=True)

fnamedata = "/home/users/rajarshi/Steve_Erc_Cipriani/Steve_Marc_Raj/Histos_9_11_2022/tnp_isominus_data_vertexWeights1_oscharge1.root"
fnamemc = "/home/users/rajarshi/Steve_Erc_Cipriani/Steve_Marc_Raj/Histos_9_11_2022/tnp_isominus_mc_vertexWeights1_oscharge1.root"

fdata = ROOT.TFile.Open(fnamedata)
fmc = ROOT.TFile.Open(fnamemc)

th1_pass_data = fdata.Get("pass_mu_RunGtoH")
th1_fail_data = fdata.Get("fail_mu_RunGtoH")

th1_pass_mc = fmc.Get("pass_mu_DY_postVFP")
th1_fail_mc = fmc.Get("fail_mu_DY_postVFP")

axis_names = ["mass", "pt", "eta"]

hist_pass_data = narf.root_to_hist(th1_pass_data, axis_names = axis_names)
hist_fail_data = narf.root_to_hist(th1_fail_data, axis_names = axis_names)
hist_pass_mc = narf.root_to_hist(th1_pass_mc, axis_names = axis_names)
hist_fail_mc = narf.root_to_hist(th1_fail_mc, axis_names = axis_names)

def fit_bin(ptsel, etasel):
#def fit_bin(fit_bin):
#    ptsel = fit_bin[0]
#    etasel = fit_bin[1]

    hist_pass_data_bin = hist_pass_data[:,ptsel, etasel]
    hist_fail_data_bin = hist_fail_data[:,ptsel, etasel]
    hist_pass_mc_bin = hist_pass_mc[:,ptsel, etasel]
    hist_fail_mc_bin = hist_fail_mc[:,ptsel, etasel]

     
    data_pass = zfit.data.BinnedData.from_hist(hist_pass_data_bin)
    data_fail = zfit.data.BinnedData.from_hist(hist_fail_data_bin)
    mc_pass = zfit.data.BinnedData.from_hist(hist_pass_mc_bin)
    mc_fail = zfit.data.BinnedData.from_hist(hist_fail_mc_bin)
 
    # Fail fit
   
    ntotal_fail = hist_fail_data_bin.sum().value

    mass_binned = data_fail.space
    mass = mass_binned

    mu_fail = zfit.Parameter("mu_fail", 0., -20., 20.)
    sigma_fail = zfit.Parameter("sigma_fail", 0.1, 0., 2.)
    nsig_fail = zfit.Parameter("nsig_fail", 0.9*ntotal_fail, 0., 2.*ntotal_fail)
    nbkg_fail = zfit.Parameter("nbkg_fail", 0.1*ntotal_fail, 0., 2.*ntotal_fail)
    lamda_fail = zfit.Parameter("lamda_fail", -0.01,-5,5)


    template_fail = zfit.pdf.HistogramPDF(mc_fail)
    smooth_template_fail = zfit.pdf.SplinePDF(template_fail, obs = mass_binned)

    mass = smooth_template_fail.space

    #bkg = zfit.pdf.Chebyshev(mass, [lam]).create_extended(nbkg)
    bkg_fail = zfit.pdf.Exponential(lamda_fail, obs=mass).create_extended(nbkg_fail)
    # func and kernel are swapped wrt to intuitive order to work around limitations on the ranges.  limits_func (for the gaussian) must be explicitly specified since otherwise the gaussian is out of range of the mass observable
    conv_kernel_fail = zfit.pdf.Gauss(mu_fail, sigma_fail, obs = mass)
    conv_template_fail = zfit.pdf.FFTConvPDFV1(conv_kernel_fail, smooth_template_fail, limits_func=[-80., 80.], n=500)

    # workaround bug in FFTConvPDFV1 which prevents extending it directly
    sig_fail = zfit.pdf.SumPDF([conv_template_fail, bkg_fail], [1., 0.]).create_extended(nsig_fail)


    totalpdf_fail = zfit.pdf.SumPDF([sig_fail, bkg_fail])

    data_unbinned_fail = data_fail.to_unbinned()
    loss_fail = zfit.loss.ExtendedUnbinnedNLL(totalpdf_fail, data_unbinned_fail, options = { "numhess" : False })

    minimizer_fail = zfit.minimize.ScipyTrustConstrV1(hessian = "zfit")
    result_fail = minimizer_fail.minimize(loss_fail)
    status_fail = result_fail.valid

    try:
        hessval_fail = result_fail.loss.hessian(list(result_fail.params)).numpy()
        cov_fail = np.linalg.inv(hessval_fail)
        eigvals_fail = np.linalg.eigvalsh(hessval_fail)
        covstatus_fail = eigvals_fail[0] > 0.
        print("eigvals", eigvals_fail)
    except:
        cov_fail = None
        covstatus_fail = False
    

    print(result_fail)
    print("Fail ptbin:", ptsel, "etabin:", etasel, "status:", status_fail, "covstatus:", covstatus_fail)
    errors_fail = np.sqrt(np.diag(cov_fail))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
  
    x_plot_fail = np.linspace(60., 120., num=1000)
    y_plot_bkg_fail = zfit.run(bkg_fail.ext_pdf(x_plot_fail))
    y_plot_fail = zfit.run(totalpdf_fail.ext_pdf(x_plot_fail))

    hist_fail_data_bin.plot(ax=ax3,label='Data')
    ax3.plot(x_plot_fail, y_plot_fail,label='Total pdf')
    ax3.plot(x_plot_fail, y_plot_bkg_fail,label='Background')

    ax3.legend(loc='upper left')
    ax3.set_ylabel('Events')
    ax3.set_title('Failing')

    #end of fail fit

    # Start of pass fit

    ntotal_pass = hist_pass_data_bin.sum().value

    mass_binned = data_pass.space
    mass = mass_binned

    mu_pass = zfit.Parameter("mu_pass", 0., -20., 20.)
    sigma_pass = zfit.Parameter("sigma_pass", 0.1, 0., 2.)
    nsig_pass = zfit.Parameter("nsig_pass", 0.9*ntotal_pass, 0., 2.*ntotal_pass)
    nbkg_pass = zfit.Parameter("nbkg_pass", 0.1*ntotal_pass, 0., 2.*ntotal_pass)
    lamda_pass = zfit.Parameter("lamda_pass", -0.01,-5,5)


    template_pass = zfit.pdf.HistogramPDF(mc_pass)
    smooth_template_pass = zfit.pdf.SplinePDF(template_pass, obs = mass_binned)

    mass = smooth_template_pass.space

    #bkg = zfit.pdf.Chebyshev(mass, [lam]).create_extended(nbkg)
    bkg_pass = zfit.pdf.Exponential(lamda_pass, obs=mass).create_extended(nbkg_pass)
    # func and kernel are swapped wrt to intuitive order to work around limitations on the ranges.  limits_func (for the gaussian) must be explicitly specified since otherwise the gaussian is out of range of the mass observable
    conv_kernel_pass = zfit.pdf.Gauss(mu_pass, sigma_pass, obs = mass)
    conv_template_pass = zfit.pdf.FFTConvPDFV1(conv_kernel_pass, smooth_template_pass, limits_func=[-80., 80.], n=500)

    # workaround bug in FFTConvPDFV1 which prevents extending it directly
    sig_pass = zfit.pdf.SumPDF([conv_template_pass, bkg_pass], [1., 0.]).create_extended(nsig_pass)


    totalpdf_pass = zfit.pdf.SumPDF([sig_pass, bkg_pass])

    data_unbinned_pass = data_pass.to_unbinned()
    loss_pass = zfit.loss.ExtendedUnbinnedNLL(totalpdf_pass, data_unbinned_pass, options = { "numhess" : False })

    minimizer_pass = zfit.minimize.ScipyTrustConstrV1(hessian = "zfit")
    result_pass = minimizer_pass.minimize(loss_pass)

    status_pass = result_pass.valid

    try:
        hessval_pass = result_pass.loss.hessian(list(result_pass.params)).numpy()
        cov_pass = np.linalg.inv(hessval_pass)
        eigvals_pass = np.linalg.eigvalsh(hessval_pass)
        covstatus_pass = eigvals_pass[0] > 0.
        print("eigvals", eigvals_pass)
    except:
        cov_pass = None
        covstatus_pass = False


    print(result_pass)
    print("Pass ptbin:", ptsel, "etabin:", etasel, "status:", status_pass, "covstatus:", covstatus_pass)
    errors_pass = np.sqrt(np.diag(cov_pass))

    x_plot_pass = np.linspace(60., 120., num=1000)
    y_plot_bkg_pass = zfit.run(bkg_pass.ext_pdf(x_plot_pass))
    y_plot_pass = zfit.run(totalpdf_pass.ext_pdf(x_plot_pass))

    hist_pass_data_bin.plot(ax=ax2,label='Data')
    ax2.plot(x_plot_pass, y_plot_pass,label='Total pdf')
    ax2.plot(x_plot_pass, y_plot_bkg_pass,label='Background')

    ax2.legend(loc='upper left')
    ax2.set_ylabel('Events')
    ax2.set_title('Passing')


    ax1.axis([0, 10, 0, 15])
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.axis('off')
    ax1.text(0, 15, f"Ptbin: {np.imag(ptsel)} Etabin: {np.imag(etasel)}")
    ax1.text(0, 14, f"Fit Status: Passing: {status_pass} Failing: {status_fail}")
    ax1.text(0,13,f"Covaraince Status: Passing: {covstatus_pass} Failing: {covstatus_fail}")
    ax1.text(0,12,f"Converged: Passing: {result_pass.converged} Failing: {result_fail.converged}")
    ax1.text(0,11,f"Params at limit: Passing: {result_pass.params_at_limit} Failing: {result_fail.params_at_limit}")
    ax1.text(0,10,f"nsig_pass: {result_pass.params[nsig_pass]['value']:.2f} $\pm$ {errors_pass[0]:.2f}")
    ax1.text(0,9,f"nbkg_pass: {result_pass.params[nbkg_pass]['value']:.2f} $\pm$ {errors_pass[1]:.2f}")
    ax1.text(0,8,f"mu_pass: {result_pass.params[mu_pass]['value']:.2f} $\pm$ {errors_pass[2]:.2f}")
    ax1.text(0,7,f"sigma_pass: {result_pass.params[sigma_pass]['value']:.2f} $\pm$ {errors_pass[3]:.2f}")
    ax1.text(0,6,f"lamda_pass: {result_pass.params[lamda_pass]['value']:.2f} $\pm$ {errors_pass[4]:.2f}")
    ax1.text(0,5,f"nsig_fail: {result_fail.params[nsig_fail]['value']:.2f} $\pm$ {errors_fail[0]:.2f}")
    ax1.text(0,4,f"nbkg_fail: {result_fail.params[nbkg_fail]['value']:.2f} $\pm$ {errors_fail[1]:.2f}")
    ax1.text(0,3,f"mu_fail: {result_fail.params[mu_fail]['value']:.2f} $\pm$ {errors_fail[2]:.2f}")
    ax1.text(0,2,f"sigma_fail: {result_fail.params[sigma_fail]['value']:.2f} $\pm$ {errors_fail[3]:.2f}")
    ax1.text(0,1,f"lamda_fail: {result_fail.params[lamda_fail]['value']:.2f} $\pm$ {errors_fail[4]:.2f}")
   
    plt.savefig(f"bin_{np.imag(ptsel)}pT_{np.imag(etasel)}Eta.pdf") 
    plt.savefig(f"bin_{np.imag(ptsel)}pT_{np.imag(etasel)}Eta.png")
    pkl.dump(fig, open(f"bin_{np.imag(ptsel)}pT_{np.imag(etasel)}Eta.pickle", 'wb')) 

    #return {"nsig_pass": result_pass.params[nsig_pass]['value'], "nbkg_pass" : result_pass.params[nbkg_pass]['value'],
   #         "mu_pass" : result_pass.params[mu_pass]['value'], "sigma_pass": result_pass.params[sigma_pass]['value'],
   #         "lamda_pass": result_pass.params[lamda_pass]['value'], #"errors_pass": errors_pass,
            #"cov_pass": cov_pass, "Status_pass": status_pass, "covstatus_pass": covstatus_pass,
   #         "converged_pass": result_pass.converged, "Params_at_limit_pass": result_pass.params_at_limit,
   #         "edm_pass": result_pass.edm, "fmin_pass": result_pass.fmin,
   #         "nsig_fail": result_fail.params[nsig_fail]['value'], "nbkg_fail" : result_fail.params[nbkg_fail]['value'],
   #         "mu_fail" : result_fail.params[mu_fail]['value'], "sigma_fail": result_fail.params[sigma_fail]['value'],
   #         "lamda_fail": result_fail.params[lamda_fail]['value'], #"errors_fail": errors_fail, 
            #"cov_fail": cov_fail, "Status_fail": status_fail, "covstatus_fail": covstatus_fail, 
   #         "converged_fail": result_fail.converged, "Params_at_limit_fail": result_fail.params_at_limit, 
   #         "edm_fail": result_fail.edm, "fmin_fail": result_fail.fmin,
   #         "Ptbin": np.imag(ptsel), "Etabin": np.imag(etasel) }



 

    #return result, cov, status, covstatus
pt_bins = [26.j, 28.j, 30.j, 32.j, 34.j]
eta_bins = [ -0.4j, -0.3j, -0.2j, -0.1j, 0.1j, 0.2j, 0.3j, 0.4j, 0.6j, 0.7j, 0.9j, 1.0j, 1.2j, 1.3j, 1.5j]
#binning_pt = [24., 26., 28., 30., 32., 34., 36., 38., 40., 42., 44., 47., 50., 55., 60., 65.]
#binning_eta = [round(-2.4 + i*0.1,2) for i in range(49)]
bins = []
for ptsel in pt_bins:
    for etasel in eta_bins:
        bins.append((ptsel,etasel))


#pool = Pool(256)

#pool.map(fit_bin,bins)

#fit_bin(28.j, -0.3j)

procs = []
for ptsel in pt_bins:
    for etasel in eta_bins:
       proc = Process(target=fit_bin, args=(ptsel,etasel))
       procs.append(proc)
       proc.start()


for proc in procs:
    proc.join() 

#Mc Efficiency

#for ptsel in binning_pt:
#    for etasel in binning_eta:
#         hist_pass_mc_bin = hist_pass_mc[:,ptsel, etasel]
#         hist_fail_mc_bin = hist_fail_mc[:,ptsel, etasel]
         
#         nsig_pass = hist_pass_data_bin.sum().value
#         error_pass = np.sqrt(hist_pass_data_bin.sum().variance)

#         nsig_fail = hist_fail_data_bin.sum().value
#         error_fail = np.sqrt(hist_fail_data_bin.sum().variance)

#         nsig_total = nsig_pass + nsig_fail
         
#         efficiency = nsig_pass / (nsig_pass+nsig_fail)
#         errors = np.sqrt(error_pass*error_pass*nsig_fail*nsig_fail + error_fail*error_fail*nsig_pass*nsig_pass) / (nsig_total*nsig_total)

elapsed = time.time() - start
elapsed_cpu = time.process_time() - start_cpu
print('Execution time:', elapsed, 'seconds')
print('CPU Execution time:', elapsed_cpu , 'seconds')
