import tool as tl
import numpy as np
import argparse
import sys
import learn_nl
import learn_sir

#------------------------#
#     main function   
#------------------------#

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    #--- arguments ---#
    parser = argparse.ArgumentParser()
    #-----------------------------------#
    parser.add_argument("-s",  "--seqfn", type=str, help="input seqs filename")
    parser.add_argument("-o",  "--outdir", type=str, help="output dir")
    parser.add_argument("-t",  "--runtype", type=int, help="run type")
    parser.add_argument("-d",  "--depth", type=int, help="depth (top-down)")
    parser.add_argument("-e",  "--errRAT", type=float, help="error ratio")
    parser.add_argument("-k",  "--k", type=int, help="latent dimension")
    parser.add_argument("-lf", "--linearFit", type=str2bool, help="linear fit")

    #-----------------------------------#
    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
    #-----------------------------------#
    
    #--- check sequencefn ---#
    if(args.seqfn!=None):
        seqfn = args.seqfn
    else: parser.print_help()
    #--- check output dir ---#
    if(args.outdir!=None): 
        outdir = args.outdir
    else: parser.print_help()
    #--- check run type ---#
    if(args.runtype!=None):
        runtype = args.runtype
    else: parser.print_help()
    #--- check error ratio ---#
    if(args.errRAT!=None):
        errRAT = args.errRAT
    #--- check linear fit ---#
    if(args.linearFit!=None):
        linearFit = args.linearFit
    else: parser.print_help()
    #--- check linear fit ---#
    if(args.k!=None):
        k = args.k
    else: parser.print_help()
        
    #-----------------------------------#
    tl.comment("COVID TF")
    #-----------------------------------#
    # create directory
    try:
        tl.mkdir("%s_mdb"%outdir)
    except:
        tl.error("cannot find: %s"%outdir)
    
    # load data (event stream)
    data = tl.loadsq(seqfn).T
    (n,d) = np.shape(data)
    data += 0.0001 * np.random.rand(n,d)
    data, mean, std = tl.normalizeZ(data)

    #-----------------------------------#
    # start COVID TF 
    #-----------------------------------#
    if runtype == 1:
        learn_nl.fit_data(data, outdir, linearFit, mean, std, k)
    elif runtype == 2:
        learn_nl.fit_data_single(data, outdir, linearFit, mean, std, k)
    elif runtype == 3:
        learn_nl.fit_data_inc(data, outdir, errRAT, linearFit, mean, std, k)
    elif runtype == 4:
        learn_nl.forecast_data(data, outdir, linearFit, mean, std, k)
    elif runtype == 5:
        learn_nl.forecast_data_single(data, outdir, linearFit, mean, std, k)
    elif runtype == 6:
        learn_sir.fit_sir(data, outdir)
    elif runtype == 7:
        learn_sir.fit_sir_single(data, outdir)
    elif runtype == 8:
        learn_sir.fit_sir_inc(data, outdir, errRAT)
    elif runtype == 9:
        learn_sir.forecast_sir(data, outdir)
    elif runtype == 10:
        learn_sir.forecast_sir_single(data, outdir)
    #-----------------------------------#

    sys.exit(0);
