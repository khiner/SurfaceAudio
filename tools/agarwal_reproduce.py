#!/usr/bin/env python3
"""Author-input baselines and explicitly calibrated 2021-model reconstructions."""
import argparse, hashlib, json, pathlib, subprocess, urllib.request
import numpy as np
from scipy import signal, optimize
from scipy.io import wavfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMIT = '465c379fd0833296a5372edecc51c7aba1999649'
RAW = f'https://raw.githubusercontent.com/vinayakagarwal4/tdw/{COMMIT}/Python/tdw/'
SPACING = 1394.068e-9
CASES = {
 'scrape': ('bass_wood', 'wood_medium_2_mm', 'expt1/Wood_on_Basswood_bfslow/2_1.wav'),
 'roll': ('poplar_wood', 'wood_hard_2_mm', 'expt3/Wood_on_Poplar_down/11_1.wav'),
 'roll-glass': ('ceramic', 'ceramic_2_mm', 'expt3/Glass_on_Ceramic_up/13_1.wav'),
}

def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url) as response: path.write_bytes(response.read())
    return dict(path=str(path.relative_to(ROOT)), url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def wave(path):
    fs, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer): x = x.astype(float) / 2**(np.iinfo(x.dtype).bits-1)
    if x.ndim == 2: x = x.mean(axis=1)
    return fs, x.astype(float)

def parameters(directory, p):
    (directory/'parameters.txt').write_text(' '.join(str(x) for x in p)+'\n')

def run(binary, *args):
    if args[0]=='prepare':
        domain=pathlib.Path(args[1])/'smoothing.json'
        if domain.exists() and json.loads(domain.read_text())['domain']=='temporal':args=('prepare-temporal',*args[1:])
    subprocess.run([str(binary), *map(str,args)], check=True)

def envelope(x, fs):
    width = int(.04*fs)
    return np.sqrt(np.mean(x[:len(x)//width*width].reshape(-1,width)**2,axis=1))

def stats(reference, synthesis, fs, start, end):
    if end<=start:return {'available':False}
    a, b = reference[start:end], synthesis[start:end]
    ea, eb = envelope(a,fs), envelope(b,fs)
    audible_filter=signal.butter(3,80,fs=fs,btype='highpass',output='sos')
    aa=signal.sosfilt(audible_filter,reference)[start:end];bb=signal.sosfilt(audible_filter,synthesis)[start:end]
    aea,aeb=envelope(aa,fs),envelope(bb,fs)
    f, pa = signal.welch(a-np.mean(a),fs,nperseg=min(4096,len(a)))
    _, pb = signal.welch(b-np.mean(b),fs,nperseg=min(4096,len(b)))
    mask=(f>=80)&(f<=12000)
    pa,pb=pa[mask],pb[mask]; pa/=pa.sum();pb/=pb.sum()
    return dict(rms_reference=float(np.sqrt(np.mean(a*a))),rms_synthesis=float(np.sqrt(np.mean(b*b))),
      envelope_correlation=float(np.corrcoef(ea,eb)[0,1]),
      audible_envelope_correlation=float(np.corrcoef(aea,aeb)[0,1]),
      audible_rms_reference=float(np.sqrt(np.mean(aa*aa))),audible_rms_synthesis=float(np.sqrt(np.mean(bb*bb))),
      normalized_log_psd_mae_db=float(np.mean(np.abs(10*np.log10(np.maximum(pa,1e-12)/np.maximum(pb,1e-12))))),
      power_weighted_log_psd_rmse_db=float(np.sqrt(np.sum(pa*(10*np.log10(np.maximum(pa,1e-12)/np.maximum(pb,1e-12)))**2))),
      centroid_reference_hz=float(np.sum(f[mask]*pa)),centroid_synthesis_hz=float(np.sum(f[mask]*pb)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=pathlib.Path,default=ROOT/'build/agarwalReproduce')
    parser.add_argument('--upstream', action='store_true', help='Also execute the pinned author scraping implementation and check GPU convolution')
    parser.add_argument('--retained', action='store_true', help='Render the four retained rolling reconstructions from committed inputs')
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--whole-record',action='store_true')
    parser.add_argument('--temporal',action='store_true')
    parser.add_argument('--coordinate-system',choices=['metres','micrometres'],default='metres',help='Hypothesized coordinate units of the paper alpha constants; measured profiles remain in metres.')
    parser.add_argument('--full-profile',action='store_true',help='Use the complete measured rolling profile instead of the assumed 4 cm patch.')
    parser.add_argument('--case',choices=[*CASES,'all'],default='all')
    args=parser.parse_args()
    if args.retained:
        render_retained(args.binary.resolve())
        return
    out=ROOT/'outputs/reproduction/agarwal';out.mkdir(parents=True,exist_ok=True)
    refs=ROOT/'references/agarwal';manifest=[];listening=[];results={}
    if (out/'cases.json').exists(): listening=json.loads((out/'cases.json').read_text())['cases']
    suffix=('-temporal' if args.temporal else '')+('-micrometres' if args.coordinate_system=='micrometres' else '')+('-full-profile' if args.full_profile else '')
    alpha_scale=1e-6 if args.coordinate_system=='micrometres' else 1.
    metrics_path=out/('metrics'+suffix+('-whole' if args.whole_record else '')+'.json')
    if metrics_path.exists():results=json.loads(metrics_path.read_text())
    if (refs/'manifest.json').exists():manifest=json.loads((refs/'manifest.json').read_text())['files']
    for name,(surface,material,audio) in CASES.items():
        if args.case!='all' and args.case!=name:continue
        profile_path=refs/(surface+'.npy'); material_path=refs/(material+'.json'); reference_path=refs/(name+'-author.wav')
        manifest.append(fetch(RAW+'add_ons/py_impact/scrape_surfaces/'+surface+'.npy',profile_path))
        manifest.append(fetch(RAW+'add_ons/py_impact/material_data/'+material+'.json',material_path))
        manifest.append(fetch('https://mcdermottlab.mit.edu/scraping_rolling/'+audio,reference_path))
        profile=np.load(profile_path).astype('<f8').ravel(); fs,reference=wave(reference_path)
        # One measured 4 cm patch limits the otherwise unspecified integration domain. No resampling.
        if name!='scrape' and not args.full_profile: profile=profile[:int(.04/SPACING)+1]
        n=len(reference); t=np.arange(n)/fs; duration=n/fs; train=n if args.whole_record else int(.75*n)
        tag=name+suffix+('-whole' if args.whole_record else '')
        listening=[case for case in listening if pathlib.Path(case['synthesis']).stem not in [tag+'-baseline',tag+'-calibrated']]
        directory=out/(tag+'-case');directory.mkdir(exist_ok=True)
        (directory/'calibration.json').write_text(json.dumps({'reference':str(reference_path.relative_to(ROOT)),'training_frames':train,'total_frames':n,'whole_record':args.whole_record},indent=2)+'\n')
        (directory/'smoothing.json').write_text(json.dumps({'domain':'temporal' if args.temporal else 'spatial','half_width_at_44100':5.,'sigma_ratio':.4})+'\n')
        coordinate_metadata={'hypothesized_paper_coordinate_system':args.coordinate_system,'alpha_scale_to_SI':alpha_scale,'author_coordinate_units_verified':False,'full_measured_profile':args.full_profile,'profile_samples':len(profile),'profile_spacing_m':SPACING,'profile_length_m':(len(profile)-1)*SPACING}
        (directory/'coordinates.json').write_text(json.dumps(coordinate_metadata,indent=2)+'\n')
        profile.tofile(directory/'profile.f64')
        length=(len(profile)-1)*SPACING
        # These are reconstruction choices, not recovered publication motion records.
        if name=='scrape':
            frequency=.9;phase=0.;amplitude=min(.025,.35*length)
            x=length/2-amplitude*np.cos(2*np.pi*frequency*t+phase)
            velocity=amplitude*2*np.pi*frequency*np.sin(2*np.pi*frequency*t+phase)
            normal=3.5+2.5*np.cos(2*np.pi*frequency*t+phase)
            mass,beta,k,lam,radius,ecc=.1,.05,0.,0.,.01,.0001
        else:
            distance=.8*length; x=.1*length+distance*(t/duration)**2
            velocity=2*distance*t/duration**2;normal=3.5-2.5*np.sin(x/.01)
            mass,beta,k,lam,radius,ecc=.1,.05,1.,1.,.01,.0001
            if name=='roll-glass':
                x=.1*length+distance*(2*t/duration-(t/duration)**2)
                velocity=2*distance/duration*(1-t/duration)
                normal=3.5-2.5*np.sin(x/radius)
            density=2500. if name=='roll-glass' else 650.
            mass=4*np.pi/3*radius**3*density
        morph=np.clip(x/length,0,1)
        np.column_stack([x,velocity,normal,morph]).astype('<f8').tofile(directory/'motion.f64')
        p=[fs,n,129,int(.25*fs),SPACING,5.,mass,beta,1.,radius,ecc,k,lam,1.,alpha_scale]
        parameters(directory,p);run(args.binary,'prepare',directory)
        baseline_force=np.fromfile(directory/'scrape.f32',dtype='<f4').astype(float)+k*np.fromfile(directory/'elastic.f32',dtype='<f4')+lam*np.fromfile(directory/'damping.f32',dtype='<f4')
        baseline_force.astype('<f4').tofile(directory/'baseline-force.f32')
        baseline_force_features={'mean_N':float(np.mean(baseline_force)),'rms_N':float(np.sqrt(np.mean(baseline_force**2))),'ac_rms_N':float(np.std(baseline_force)),'peak_abs_N':float(np.max(abs(baseline_force)))}
        md=json.loads(material_path.read_text())
        modes=np.column_stack([md['cf'],np.array(md['rt'])/np.log(1000),10**(np.array(md['op'])/20),10**(np.array(md['op'])/20)])
        np.savetxt(directory/'modes.txt',modes)
        baseline=out/(tag+'-baseline');run(args.binary,'render',directory,baseline)
        # A single scalar sets listening level from the training interval; timbre/motion remain unfitted.
        _,y=wave(baseline.with_suffix('.wav'));p[13]=np.sqrt(np.mean(reference[:train]**2)/np.mean(y[:train]**2))
        parameters(directory,p);run(args.binary,'render',directory,baseline)
        _,y=wave(baseline.with_suffix('.wav'))
        record={'coordinates':coordinate_metadata,'baseline_force':baseline_force_features,'assumed_motion':{'mass_kg':mass,'radius_m':radius if name!='scrape' else None,'density_kg_m3':density if name!='scrape' else None,'travel_m':float(np.ptp(x)),'maximum_speed_m_s':float(np.max(abs(velocity)))},'baseline':{'training':stats(reference,y,fs,0,train),'held_out':stats(reference,y,fs,train,n)}}
        listening.append(dict(title=f'Agarwal {name}: author-profile/material baseline',reference=str(reference_path.relative_to(ROOT)),synthesis=str(baseline.with_suffix('.wav').relative_to(ROOT)),notes='Original author-fork measured texture and material mode table; assumed motion, boundary integration constants and Gaussian width. Only global listening gain uses the first 75% of the target. Mode tables use the paper sine convention, unlike the later PyImpact cosine renderer.',source_url='https://mcdermottlab.mit.edu/scraping_rolling.html'))
        if not args.baseline_only:
            if name=='scrape':
                # Fit only SHM frequency/phase from a low-dimensional force-envelope proxy.
                # Eq8 makes normalized normal force affine in position; Eq7 then fixes alpha.
                target_envelope=envelope(reference[:train],fs)
                times=(np.arange(len(target_envelope))+.5)*.04
                target_envelope/=np.max(target_envelope)
                def motion_residual(values):
                    freq,phase_value,power,scale,offset=values
                    angle=2*np.pi*freq*times+phase_value
                    a=.05-.04*((1+np.cos(angle))/2)**.95
                    proxy=(np.abs(np.sin(angle))**2/a)**power
                    return scale*proxy/np.max(proxy)+offset-target_envelope
                candidates=[optimize.least_squares(motion_residual,[f,ph,1,.8,.1],bounds=([.65,-np.pi,.25,0,0],[1.15,np.pi,2,2,.5]),max_nfev=200) for f in [.8,.9,1.] for ph in [-1.,0.,1.]]
                motion_fit=min(candidates,key=lambda fit:np.sum(fit.fun**2))
                frequency,phase=motion_fit.x[:2]
                x=length/2-amplitude*np.cos(2*np.pi*frequency*t+phase)
                velocity=amplitude*2*np.pi*frequency*np.sin(2*np.pi*frequency*t+phase)
                normal=3.5+2.5*np.cos(2*np.pi*frequency*t+phase)
                np.column_stack([x,velocity,normal,x/length]).astype('<f8').tofile(directory/'motion.f64')
                run(args.binary,'prepare',directory)
                record['motion_fit']={'frequency_hz':float(frequency),'phase_radians':float(phase),'proxy_residual_rms':float(np.sqrt(np.mean(motion_fit.fun**2)))}
            # Estimate missing resonances from training PSD. Audio samples never become excitation or IR.
            force=np.fromfile(directory/'scrape.f32',dtype='<f4').astype(float)+k*np.fromfile(directory/'elastic.f32',dtype='<f4')+lam*np.fromfile(directory/'damping.f32',dtype='<f4')
            f,target=signal.welch(reference[:train]-reference[:train].mean(),fs,nperseg=8192)
            _,source=signal.welch(force[:train]-force[:train].mean(),fs,nperseg=8192)
            peaks,_=signal.find_peaks(target,distance=5,prominence=max(target)*1e-5)
            peaks=peaks[(f[peaks]>80)&(f[peaks]<14000)]
            peaks=peaks[np.argsort(target[peaks])[-50:]]
            widths=signal.peak_widths(target,peaks,rel_height=.5)[0]*fs/8192
            frequencies=f[peaks];decays=np.clip(1/(np.pi*widths),.002,.08)
            mask=(f>80)&(f<14000);ff=f[mask];z=np.exp(-2j*np.pi*ff/fs)
            poles=np.exp(-1/(fs*decays)+2j*np.pi*frequencies/fs)
            # Transfer functions of finite exp-sin modes, matching the C++/Metal renderer.
            h=np.empty((len(ff),len(peaks)),complex)
            for j,pole in enumerate(poles):
                h[:,j]=((1-(pole*z)**p[3])/(1-pole*z)-(1-(pole.conjugate()*z)**p[3])/(1-pole.conjugate()*z))/(2j)
            desired=target[mask];excitation=np.maximum(source[mask],source.max()*1e-10)
            scale=np.sqrt(desired.max()/np.maximum(excitation.max(),1e-30))/max(abs(h).max(),1)
            def residual(loga):
                prediction=excitation*abs(np.sum(h*np.exp(loga)[None,:],axis=1))**2
                return np.log(np.maximum(prediction,desired.max()*1e-8))-np.log(np.maximum(desired,desired.max()*1e-8))
            fit=optimize.least_squares(residual,np.full(len(peaks),np.log(scale)),bounds=(np.log(scale)-12,np.log(scale)+12),max_nfev=100)
            amplitudes=np.exp(fit.x)
            np.savetxt(directory/'modes.txt',np.column_stack([frequencies,decays,amplitudes,amplitudes]))
            p[13]=1.;parameters(directory,p)
            calibrated=out/(tag+'-calibrated')
            if name!='scrape':
                # The paper explicitly sets beta1/k/lambda by ear. Estimate beta1 and k
                # from training envelope and DC fraction, preserving non-tensile rolling contact.
                audible_filter=signal.butter(3,80,fs=fs,btype='highpass',output='sos')
                target=signal.sosfilt(audible_filter,reference[:train]);target_rms=np.sqrt(np.mean(target*target))
                target_env=envelope(target,fs)/target_rms
                best=None
                for beta_candidate in [0.0001,0.001,0.01,0.05]:
                    p[7]=beta_candidate;parameters(directory,p);run(args.binary,'prepare',directory)
                    a=np.fromfile(directory/'elastic.f32',dtype='<f4');b=np.fromfile(directory/'damping.f32',dtype='<f4')
                    for lambda_candidate in [.1,1.,10.]:
                        k_min=max(.001,float(np.max(-b/a))*lambda_candidate*1.01)
                        for k_candidate in [k_min,.01,.1,1.]:
                            if k_candidate<k_min:continue
                            p[11]=k_candidate;p[12]=lambda_candidate;parameters(directory,p);run(args.binary,'render',directory,calibrated)
                            _,trial=wave(calibrated.with_suffix('.wav'));trial=trial[:train]
                            def growth_loss(growth):
                                candidate=signal.sosfilt(audible_filter,trial*np.exp(growth*morph[:train]))
                                rms=np.sqrt(np.mean(candidate*candidate))
                                return np.mean((envelope(candidate,fs)/rms-target_env)**2)
                            growth_fit=optimize.minimize_scalar(growth_loss,bounds=(-5,5),method='bounded')
                            objective=growth_fit.fun
                            if best is None or objective<best[0]:best=(float(objective),beta_candidate,k_candidate,lambda_candidate,float(growth_fit.x))
                _,p[7],p[11],p[12],growth=best;parameters(directory,p);run(args.binary,'prepare',directory)
                # Alternate once: refit modal amplitudes to the selected physical force mixture.
                force=np.fromfile(directory/'scrape.f32',dtype='<f4').astype(float)+p[11]*np.fromfile(directory/'elastic.f32',dtype='<f4')+p[12]*np.fromfile(directory/'damping.f32',dtype='<f4')
                _,source=signal.welch(force[:train]-force[:train].mean(),fs,nperseg=8192)
                excitation=np.maximum(source[mask],source.max()*1e-10)
                fit=optimize.least_squares(residual,fit.x,bounds=(np.log(scale)-12,np.log(scale)+12),max_nfev=100)
                amplitudes=np.exp(fit.x)
                np.savetxt(directory/'modes.txt',np.column_stack([frequencies,decays,amplitudes,amplitudes*np.exp(growth)]))
                record['force_fit']={'beta1':p[7],'stiffness':p[11],'dissipation':p[12],'log_amplitude_endpoint_ratio':growth,'audible_envelope_objective':best[0]}
                global_modes=np.column_stack([frequencies,decays,amplitudes,amplitudes*np.exp(growth)])
                # Fit spatial spectral change from two training subintervals. Endpoints
                # are inferred from their force-energy-weighted mean locations.
                regional=[]
                for start,end in [(0,train//2),(train//2,train)]:
                    _,regional_source=signal.welch(force[start:end]-force[start:end].mean(),fs,nperseg=8192)
                    _,regional_target=signal.welch(reference[start:end]-reference[start:end].mean(),fs,nperseg=8192)
                    excitation=np.maximum(regional_source[mask],regional_source.max()*1e-10)
                    desired=regional_target[mask]
                    regional_fit=optimize.least_squares(residual,fit.x,bounds=(np.log(scale)-12,np.log(scale)+12),max_nfev=100)
                    location=float(np.sum(morph[start:end]*force[start:end]**2)/np.sum(force[start:end]**2))
                    regional.append((location,regional_fit.x))
                m0,la0=regional[0];m1,la1=regional[1]
                contrast=np.clip((la1-la0)/(m1-m0),-5,5)
                onset=la0-m0*contrast
                np.savetxt(directory/'modes.txt',np.column_stack([frequencies,decays,np.exp(onset),np.exp(onset+contrast)]))
                record['spatial_fit']={'training_mean_locations':[m0,m1],'log_endpoint_ratio_range':[float(contrast.min()),float(contrast.max())]}
                spatial_modes=np.column_stack([frequencies,decays,np.exp(onset),np.exp(onset+contrast)])
                candidates=[]
                for candidate_name,candidate_modes in [('common_amplitude_ratio',global_modes),('individual_amplitude_ratios',spatial_modes)]:
                    np.savetxt(directory/'modes.txt',candidate_modes);run(args.binary,'render',directory,calibrated)
                    _,trial=wave(calibrated.with_suffix('.wav'));candidate=signal.sosfilt(audible_filter,trial[:train]);rms=np.sqrt(np.mean(candidate*candidate))
                    envelope_error=float(np.mean((envelope(candidate,fs)/rms-target_env)**2))
                    _,pt=signal.welch(reference[:train]-reference[:train].mean(),fs,nperseg=8192)
                    _,py=signal.welch(trial[:train]-trial[:train].mean(),fs,nperseg=8192)
                    pt=pt[mask]/np.sum(pt[mask]);py=py[mask]/np.sum(py[mask])
                    spectral_error=float(np.sum(pt*(10*np.log10(np.maximum(py,1e-20)/np.maximum(pt,1e-20)))**2))
                    score=envelope_error+spectral_error/100
                    candidates.append((score,candidate_name,candidate_modes,envelope_error,spectral_error))
                selected=min(candidates,key=lambda candidate:candidate[0]);np.savetxt(directory/'modes.txt',selected[2])
                record['spatial_fit']['selected']=selected[1]
                record['spatial_fit']['rendered_training_candidates']=[{'name':c[1],'score':c[0],'audible_envelope_mse':c[3],'power_weighted_spectral_mse_db2':c[4]} for c in candidates]
                # Stationary force-PSD multiplication misses modal onset and amplitude
                # modulation. Fit the actual finite-convolution basis cross spectra.
                chosen=selected[2];basis=[]
                for frequency_value,decay_value,a0,a1 in chosen:
                    pole=np.exp(-1/(fs*decay_value)+2j*np.pi*frequency_value/fs)
                    response=signal.lfilter([1],[1,-pole],force)
                    response[p[3]:]-=pole**p[3]*response[:-p[3]]
                    basis.append(response.imag*np.exp(np.log(a1/a0)*morph))
                basis=np.asarray(basis)[:,:train]
                bf,bt,spectra=signal.stft(basis,fs,nperseg=4096,noverlap=2048,boundary=None,padded=False,scaling='psd',axis=-1)
                cross=np.einsum('mft,nft->fmn',spectra,spectra.conj(),optimize=True).real*(2/spectra.shape[-1])
                _,target_power=signal.welch(reference[:train],fs,nperseg=4096,noverlap=2048,detrend=False)
                active=(bf>=80)&(bf<=14000);cross=cross[active];target_power=target_power[active]
                floor=target_power.max()*1e-8
                def exact_residual(loga):
                    amplitude_values=np.exp(loga)
                    prediction=np.einsum('i,fij,j->f',amplitude_values,cross,amplitude_values,optimize=True)
                    logarithmic=.25*log_weights*(np.log(np.maximum(prediction,floor))-np.log(np.maximum(target_power,floor)))
                    amplitude_error=2*(np.sqrt(np.maximum(prediction,0))-np.sqrt(target_power))/np.sqrt(target_power.max())
                    return np.concatenate([logarithmic,amplitude_error])
                def exact_jacobian(loga):
                    amplitude_values=np.exp(loga)
                    qa=np.einsum('fij,j->fi',cross,amplitude_values,optimize=True)
                    prediction=np.sum(qa*amplitude_values[None,:],axis=1)
                    derivative=2*qa*amplitude_values[None,:]
                    logarithmic=.25*log_weights[:,None]*derivative/np.maximum(prediction[:,None],floor)
                    logarithmic[prediction<=floor]=0
                    amplitude_derivative=np.zeros_like(derivative)
                    positive=prediction>0
                    amplitude_derivative[positive]=derivative[positive]/np.sqrt(prediction[positive,None]*target_power.max())
                    return np.vstack([logarithmic,amplitude_derivative])
                objective_candidates=[];base_modes=chosen.copy()
                for objective_name,log_weights in [('uniform',np.ones_like(target_power)),('power_weighted',np.sqrt(target_power/target_power.max()))]:
                    exact_fit=optimize.least_squares(exact_residual,np.log(base_modes[:,2]),jac=exact_jacobian,bounds=(np.log(base_modes[:,2])-12,np.log(base_modes[:,2])+12),max_nfev=200)
                    candidate_modes=base_modes.copy();ratio=candidate_modes[:,3]/candidate_modes[:,2];candidate_modes[:,2]=np.exp(exact_fit.x);candidate_modes[:,3]=candidate_modes[:,2]*ratio
                    np.savetxt(directory/'modes.txt',candidate_modes);np.savetxt(directory/(objective_name+'-objective-modes.txt'),candidate_modes)
                    candidate_path=out/(tag+'-objective-'+objective_name);run(args.binary,'render',directory,candidate_path)
                    _,candidate_audio=wave(candidate_path.with_suffix('.wav'));features=stats(reference,candidate_audio,fs,0,train)
                    score=1-features['audible_envelope_correlation']+features['power_weighted_log_psd_rmse_db']**2/100
                    objective_candidates.append((score,objective_name,candidate_modes,exact_fit,features,candidate_path))
                winner=min(objective_candidates,key=lambda candidate:candidate[0]);np.savetxt(directory/'modes.txt',winner[2])
                record['exact_convolution_fit']={'selected':winner[1],'evaluations':winner[3].nfev,'combined_objective_rms':float(np.sqrt(np.mean(winner[3].fun**2))),'candidates':[{'objective':c[1],'rendered_training_score':c[0],'training_features':c[4],'synthesis':str(c[5].with_suffix('.wav').relative_to(ROOT))} for c in objective_candidates]}




            run(args.binary,'render',directory,calibrated)
            _,y=wave(calibrated.with_suffix('.wav'));p[13]=np.sqrt(np.mean(reference[:train]**2)/np.mean(y[:train]**2))
            parameters(directory,p);run(args.binary,'render',directory,calibrated)
            _,y=wave(calibrated.with_suffix('.wav'))
            record['calibrated']={'training':stats(reference,y,fs,0,train),'held_out':stats(reference,y,fs,train,n),'mode_count':len(peaks),'fit_evaluations':fit.nfev}
            listening.append(dict(title=f'Agarwal {name}: calibrated reconstruction',reference=str(reference_path.relative_to(ROOT)),synthesis=str(calibrated.with_suffix('.wav').relative_to(ROOT)),notes='Measured author texture drives paper force equations. SHM frequency/phase and missing mode frequencies, decay constants, amplitudes, rolling force weights and listening gain estimated from first 75% of reference; last 25% excluded from fitting. Motion and integration constants remain assumptions. This is a calibrated reconstruction, not an exact publication reproduction.',source_url='https://mcdermottlab.mit.edu/scraping_rolling.html'))
        record['smoothing_domain']='temporal' if args.temporal else 'spatial'
        for case in listening:
            if pathlib.Path(case['synthesis']).stem in [tag+'-baseline',tag+'-calibrated']:
                if args.coordinate_system!='metres':
                    case['title']+=' (micrometre coordinate hypothesis)'
                    case['notes']='Paper alpha constants assumed to use micrometre x/z coordinates; author convention is unknown. '+case['notes']
                if args.full_profile:
                    case['title']+=' (full measured profile)'
                    case['notes']='Motion traverses 80% of the full measured profile in the reference duration. '+case['notes']
        if args.temporal:
            for case in listening:
                if pathlib.Path(case['synthesis']).stem in [tag+'-baseline',tag+'-calibrated']:
                    case['title']+=' (temporal Gaussian)'
                    case['notes']='Gaussian half-width is calibrated in audio samples; all translated trajectory fields receive identical weights. '+case['notes']
        results[tag]=record
        metrics_path.write_text(json.dumps(results,indent=2)+'\n')
        if args.whole_record:
            for case in listening:
                if tag in case['synthesis']:
                    case['title']+=' (whole-record fit)'
                    case['notes']='Whole reference used for calibration; no held-out validation. '+case['notes'].replace('first 75%', 'whole record').replace('last 25% excluded from fitting.', 'No interval excluded.')
        if args.upstream and name == 'scrape':
            listening = [row for row in listening if row['synthesis'] != str((out/'upstream.wav').relative_to(ROOT))]
            listening.append(compare_upstream(args.binary, directory, out))
        (out/'cases.json').write_text(json.dumps({'cases':listening},indent=2)+'\n')
        manifest=list({entry['url']:entry for entry in manifest}.values())
        (refs/'manifest.json').write_text(json.dumps({'author_commit':COMMIT,'files':manifest},indent=2)+'\n')
    print(json.dumps(results,indent=2))

def compare_upstream(binary, directory, output):
    import ast
    import types
    from scipy.ndimage import gaussian_filter1d, uniform_filter1d
    sources = [ROOT / 'references/agarwal' / name for name in ['author_py_impact.py', 'author_modes.py']]
    for path, relative in zip(sources, ['add_ons/py_impact.py', 'physics_audio/modes.py']):
        fetch(RAW + relative, path)
    mode_namespace = {}
    exec(sources[1].read_text(), mode_namespace)
    material = json.loads((ROOT / 'references/agarwal/wood_medium_2_mm.json').read_text())
    modes = mode_namespace['Modes'](np.array(material['cf']), np.array(material['op']), 1000 * np.array(material['rt']))
    impulse = 2 * modes.sum_modes(resonance=1)
    method = next(node for cls in ast.parse(sources[0].read_text()).body if isinstance(cls, ast.ClassDef) and cls.name == 'PyImpact'
                  for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'get_scrape_sound')
    forces = []
    def convolve(first, second, **kwargs):
        forces.append(second.copy())
        return signal.convolve(first, second, **kwargs)
    namespace = dict(np=np, sg=types.SimpleNamespace(convolve=convolve), gaussian_filter1d=gaussian_filter1d,
                     uniform_filter1d=uniform_filter1d, Path=pathlib.Path, SAMPLE_RATE=44100, Base64Sound=lambda sound: sound,
                     PyImpact=types.SimpleNamespace(SCRAPE_MAX_VELOCITY=5, SCRAPE_M_PER_PIXEL=SPACING), __name__='upstream',
                     resource_filename=lambda module, path: str(ROOT / 'references/agarwal' / pathlib.Path(path).name))
    exec('from __future__ import annotations\n' + ast.unparse(method), namespace)
    state = types.SimpleNamespace(_scrape_previous_indices={}, _scrape_summed_masters={}, _scrape_events_count={},
                                  _scrape_start_velocities={}, scrape_surface_data={}, _get_impulse_response=lambda **kwargs: (impulse, min(modes.frequencies)))
    surface = type('Surface', (), {'name': 'bass_wood'})
    motion = np.fromfile(directory / 'motion.f64', dtype='<f8').reshape(-1, 4)
    chunks = [namespace['get_scrape_sound'](state, np.array([abs(motion[min(frame, len(motion)-1), 1]), 0, 0]),
              [], 0, 'wood_medium_2', 1, .1, 1, 'wood_medium_2', 1, .1, 1, 1, surface) for frame in range(0, len(motion), 883)]
    expected = np.concatenate(chunks)[:len(motion)]
    force = np.concatenate(forces)[:len(motion)].astype('<f4')
    roughness = next(iter(state.scrape_surface_data.values()))['rough_ratio']
    force.tofile(directory / 'upstream-force.f32')
    (impulse * roughness).astype('<f4').tofile(directory / 'upstream-ir.f32')
    destination = output / 'upstream.wav'
    run(binary, 'filter', directory / 'upstream-force.f32', directory / 'upstream-ir.f32', destination, 44100)
    _, actual = wave(destination)
    error = np.linalg.norm(actual[:len(expected)] - expected) / np.linalg.norm(expected)
    if not np.isfinite(error) or error > 1e-4:
        raise ValueError(f'Actual upstream block convolution differs from GPU FIR: {error}')
    wavfile.write(output / 'upstream-force.wav', 44100, force)
    return {'title': 'Agarwal provided PyImpact implementation', 'reference': 'references/agarwal/scrape-author.wav',
            'synthesis': str(destination.relative_to(ROOT)), 'notes': f'Actual pinned author scraping code with mean material modes, cosine IR and prescribed speeds. Full-tail GPU convolution agrees with source blocks to relative L2 {error:.3g}. This simplified later implementation is distinct from the full 2021 model.',
            'source_url': 'https://github.com/vinayakagarwal4/tdw/tree/' + COMMIT}

def render_retained(binary):
    import tempfile
    inputs = ROOT / 'repros/agarwal'
    manifest = json.loads((inputs / 'cases.json').read_text())
    package = inputs / manifest['package']
    if hashlib.sha256(package.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Retained reconstruction inputs changed')
    out = ROOT / 'outputs/reproduction/agarwal'
    out.mkdir(parents=True, exist_ok=True)
    cases_path = out / 'cases.json'
    cases = json.loads(cases_path.read_text())['cases'] if cases_path.exists() else []
    with np.load(package, allow_pickle=False) as arrays:
        for case in manifest['cases']:
            with tempfile.TemporaryDirectory(prefix='agarwal-') as work:
                directory = pathlib.Path(work)
                for name, key in case['files'].items():
                    array = arrays[key]
                    (array.astype('<f8') if name.endswith('.f64') else array).tofile(directory / name)
                run(binary, case['preparation'], directory, case['oversampling'], case['sigma_ratio'], '--bandlimit')
                force = directory / 'scrape.f32'
                if hashlib.sha256(force.read_bytes()).hexdigest() != case['force_sha256']:
                    raise ValueError(f"{case['name']}: regenerated force differs from retained input")
                prefix = out / case['name']
                run(binary.parent / 'agarwalSpatialReproduce', directory, prefix)
                output = prefix.with_suffix('.wav')
                if hashlib.sha256(output.read_bytes()).hexdigest() != case['waveform_sha256']:
                    raise ValueError(f"{case['name']}: regenerated WAV differs from retained render")
                entry = {'title': 'Agarwal retained ' + case['name'], 'reference': case['reference'],
                         'synthesis': str(output.relative_to(ROOT)), 'notes': case['calibration'] + ' Original event inputs remain unavailable; texture mismatch remains.',
                         'source_url': 'https://mcdermottlab.mit.edu/scraping_rolling.html'}
                cases = [row for row in cases if row['synthesis'] != entry['synthesis']] + [entry]
    cases_path.write_text(json.dumps({'cases': cases}, indent=2) + '\n')

if __name__=='__main__':main()
