import soapcw as soap
import numpy as np
import lalpulsar
import lal

class SFT:
    """Container for SFT data and related metadata."""

    def __init__(self,nsft=None,tsft=None,delta_f=None,fmin=None,fmax=None,det_name=None,tstart=None):
        self.sft = None
        self.norm_sft_power = None
        self.tstart = tstart
        self.nsft = nsft
        self.tsft = tsft
        self.epochs = None
        self.delta_f = delta_f
        self.det_name = det_name
        self.fmin = fmin
        self.fmax = fmax
        
        if self.delta_f is None and self.tsft is not None:
            self.delta_f = 1./self.tsft

class LoadSFT:
  def __init__(self,sftpath,fmin=None,fmax=None,norm=False,summed=False,filled=False,remove_sft=True,save_rngmed=False,tmin=None,tmax=None,vetolist = None,norm_timebin_power=False,eps=1e-12):
        """
        Load an SFT from multiple detectors
        args
        -----------------
        sftpath: str or np.ndarray
            path to the sft files, for multiple files separate by semicolon, 'filename1;filename2;.....' can input from multiple detectors
            or input an np.ndarray of shape (Ndetectors, Ntimesteps, Nfreqbins)
        fmin: float
            minimum frequency to load from sft
        fmax: float
            maximum frequency to load from sft
        norm: bool or int
            normalise sft to running median, if integer running median of that many bins
        summed: bool or int
            sum normalised spectrograms over number of bins (default 48 (1day of 1800s)) or set to value
        filled: bool
            fill the gaps in the sfts with the expected value (2 for normalised sfts, nan for sfts)
        remove_sft: bool
            remove original sft after normalising to running median
        save_rndmed: bool
            save the running median as an estimate of noise floor
        tmin: float
            start time for sfts
        tmax: float
            end time for sfts
        vetolist: list
            list of frequency bins to set to expected value (2 or nan) (and +/- 3 bins of selected bin)
        det_names: list
            list of detector names (only necessary when using np.ndarray as input to sftpath)
        """


        self.norm_timebin_power = norm_timebin_power
        self.eps = eps

        if type(sftpath) is str:
            self.get_sft(sftpath,fmin=fmin,fmax=fmax,tmin=tmin,tmax=tmax)
        elif type(sftpath) is dict:
            self.get_sft_from_array(sftpath, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
        else:
            raise Exception(f"Please use a type str or np.ndarray for the input of sftpath, you used type {type(sftpath)}")

  
  def get_sft(self,sftpath,fmin=None,fmax=None,tmin=None,tmax=None,vetolist = None):
        '''
        load an sft to a numpy array, 
        args
        -------
        sftpath : string
            path to the sft file
        detector: string
            which detector i.e 'H1'
        fmin    : float
            minimum frequency
        fmax    : float
            maximum frequency
        tmin    : float
            min time
        tmax    : float
            max time
        '''
        constraints = lalpulsar.SFTConstraints()

        if fmin is None:
            fmin = -1
        if fmax is None:
            fmax = -1

        if tmin is not None:
            self.tmin_gps = lal.LIGOTimeGPS(int(tmin),0)
            constraints.minStartTime=self.tmin_gps
        if tmax is not None:
            self.tmax_gps = lal.LIGOTimeGPS(int(tmax),0)
            constraints.maxStartTime=self.tmax_gps

        catalogue = lalpulsar.SFTdataFind(sftpath,constraints)
        sfts = lalpulsar.LoadMultiSFTs(catalogue,fmin,fmax)

        self.det_names = []
        tsft = []
        nbins = [] 
        for det in sfts.data:
            tsft.append(1.0/det.data[0].deltaF)
            nbins.append(det.data[0].data.length)
        
        if len(set(tsft)) > 1:
            print("Warning: tsft not the same between detectors.")
            
        if len(set(nbins)) > 1:
            print("Warning: different detectors do not have the same number of frequency bins")
        
        for det in sfts.data:
            detname = det.data[0].name
            self.det_names.append(detname)

            data = SFT()
            data.nsft = det.length
            data.nbins = det.data[0].data.length
            data.delta_f = det.data[0].deltaF
            data.f0 = det.data[0].f0
            data.tsft = 1.0/det.data[0].deltaF

            data.frequencies = np.arange(data.nbins)*data.delta_f + data.f0
            data.sft = np.zeros((data.nsft,data.nbins)).astype(np.complex128)
            data.epochs = np.zeros(data.nsft)
            data.fmin = det.data[0].f0
            data.fmax = data.fmin + data.nbins/data.tsft

            for i,sft in enumerate(det.data):
                data.sft[i,:] = sft.data.data
                data.epochs[i] = sft.epoch

            if self.norm_timebin_power:
                power = np.abs(data.sft)**2
                mean_t = power.mean(axis=1, keepdims=True)
                data.norm_sft_power = power / (mean_t + self.eps)
            setattr(self,detname,data)

def find_rms_n(pul_track,vit_track,ref_CSh):
    diff = []
    for elem in range(len(pul_track)):
        weight = ref_CSh[elem]/np.sum(ref_CSh[elem])
        pathsqs = weight*((np.array(pul_track[elem])-np.array(vit_track[elem]))**2)
        #diff.append(np.sum(np.median(np.array(pathsqs))))
        diff.append(1./len(pathsqs)*np.sum(np.array(pathsqs)))

    return np.sqrt(1./(len(diff))*np.sum(diff))
