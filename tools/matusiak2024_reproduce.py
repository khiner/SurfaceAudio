#!/usr/bin/env python3
"""Reproduce the 2024 equations against calibrated measured and author vector traces from the paper."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, resample_poly
from AnalyzeRenders import texture_metrics

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path('/Users/khiner/physical_audio_papers/elastoplastic_friction/matusiak2024_bowed_string_transients.pdf')
FIXTURE = ROOT / 'repros/matusiak2024'
# Page, path index, zero-time x, 0.05-second x, zero-force y, positive-force y, positive tick force.
AXES = [
    [(7,35,76.132,102.501,92.322,75.086,5), (7,71,76.132,102.501,152.906,135.670,5), (7,170,76.132,102.501,601.403,584.167,5)],
    [(10,32,78.532,98.115,93.477,82.702,1), (10,66,78.532,98.115,140.993,130.218,1), (10,100,78.532,98.115,188.509,177.734,1)],
    [(10,133,243.134,262.880,89.364,80.052,2), (10,166,243.134,262.880,137.275,127.963,2), (10,200,243.134,262.880,185.186,175.874,2)],
    [(10,226,408.844,436.561,88.517,73.702,5), (10,252,408.844,436.561,136.428,121.613,5), (10,279,408.844,436.561,184.339,169.525,5)],
]
CASES = [('circle',2.3433,.8722),('red',.7992,.3473),('green',2.0791,1.0481),('blue',3.9125,2.0651)]


def extract_paper(paper):
    import pymupdf
    document = pymupdf.open(paper)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    records = []
    for (name, force, acceleration), axes in zip(CASES, AXES):
        paths = []
        for page, index, x0, x1, y0, y1, tick in axes:
            items = document[page].get_drawings()[index]['items']
            if not all(item[0] == 'l' for item in items):
                raise ValueError('Expected source polyline')
            points = np.array([(items[0][1].x, items[0][1].y)] + [(item[2].x,item[2].y) for item in items])
            t = (points[:,0]-x0)*.05/(x1-x0)
            f = (y0-points[:,1])*tick/(y0-y1)
            if np.any(np.diff(t)<0):
                raise ValueError('Source time is not monotone')
            paths.append((t,f))
        frames = int(min(path[0][-1] for path in paths)*44100)
        time = np.arange(frames)/44100
        trace = np.column_stack([np.interp(time,t,f) for t,f in paths]).astype('<f4')
        path = FIXTURE / (name+'.f32')
        trace.tofile(path)
        records.append({'name':name,'force_n':force,'acceleration_m_s2':acceleration,'frames':frames,
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'axes':axes,
                        'source_vertices':[len(t) for t,_ in paths]})
    metadata = {'paper':'https://doi.org/10.1121/10.0028228','paper_sha256':hashlib.sha256(paper.read_bytes()).hexdigest(),
                'license':'CC-BY-4.0','authors':'Ewa Matusiak and Vasileios Chatziioannou','sample_rate':44100,
                'columns':['measured_bridge_n','author_s_bridge_n','author_epsilon_bridge_n'],
                'extraction':'PDF vector polyline coordinates, axis calibration, linear resampling, no waveform alignment or gain fit',
                'coordinate_rounding_pt':.001,'cases':records}
    items=document[7].get_drawings()[134]['items']
    points=np.array([(items[0][1].x,items[0][1].y)]+[(item[2].x,item[2].y) for item in items])
    time=(points[:,0]-76.132)*.05/(102.501-76.132)
    force=(540.820-points[:,1])*5/(540.820-523.584)
    reference=np.fromfile(FIXTURE/'circle.f32','<f4').reshape(-1,3)[:,0]
    duplicate=np.interp(np.arange(len(reference))/44100,time,force)
    maximum=float(np.max(abs(duplicate-reference)))
    if maximum>.001:
        raise ValueError('Repeated measured trace in Figures 3 and 4 disagrees after independent axis calibration')
    metadata['independent_repeated_measured_path_check']={'page':7,'path_index':134,'maximum_difference_n':maximum,
        'relative_l2':float(np.linalg.norm(duplicate-reference)/np.linalg.norm(reference))}
    (FIXTURE/'Reference.json').write_text(json.dumps(metadata,indent=2)+'\n')


def compare(reference, native):
    reference_ac=reference-reference.mean(); native_ac=native-native.mean()
    _,_,a = stft(reference_ac,44100,nperseg=2048,noverlap=1536)
    _,_,b = stft(native_ac,44100,nperseg=2048,noverlap=1536)
    a,b=abs(a),abs(b)
    tail=slice(int(.6*len(reference)),None)
    window=441
    count=len(reference)//window
    reference_envelope=np.std(reference[:count*window].reshape(count,window),axis=1)
    native_envelope=np.std(native[:count*window].reshape(count,window),axis=1)
    floor = max(a.max(),b.max())*1e-5
    selected = np.maximum(a,b)>floor
    db = 20*np.log10(np.maximum(a,floor)/np.maximum(b,floor))
    return {'relative_l2':float(np.linalg.norm(native-reference)/np.linalg.norm(reference)),
            'ac_relative_l2':float(np.linalg.norm(native_ac-reference_ac)/np.linalg.norm(reference_ac)),
            'spectral_convergence':float(np.linalg.norm(a-b)/np.linalg.norm(a)),
            'envelope_relative_l2':float(np.linalg.norm(native_envelope-reference_envelope)/np.linalg.norm(reference_envelope)),
            'tail_ac_rms_ratio':float(np.std(native[tail])/np.std(reference[tail])),
            'max_absolute_force_n':float(np.max(abs(native-reference))),
            'stft_active_bin_rmse_db':float(np.sqrt(np.mean(db[selected]**2))),
            'reference_rms_n':float(np.sqrt(np.mean(reference**2))), 'native_rms_n':float(np.sqrt(np.mean(native**2)))}


def check_author(root, output, metadata, convention, sample_rate):
    """Check the 2024 reconstruction against explicitly patched, pinned author MATLAB executed by Octave."""
    import os
    import shutil
    from scipy.io import loadmat

    if convention != 'archive' or sample_rate != 44100:
        raise ValueError('--check-author requires --convention archive --sample-rate 44100')
    root, output = Path(root), Path(output)
    source = root/'references/matusiak/the_bowed_string/bowed_string_with_comments.m'
    pin = json.loads((root/'repros/matusiak/Reference.json').read_text())
    if not source.is_file():
        raise FileNotFoundError('Pinned author source is required at '+str(source))
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != pin['source_sha256']:
        raise ValueError('Pinned author MATLAB SHA-256 mismatch')
    octave = shutil.which('octave-cli') or shutil.which('octave')
    if octave is None:
        raise FileNotFoundError('--check-author requires GNU Octave')
    version = subprocess.run([octave,'--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    original = raw.decode().replace('\r\n','\n')
    body = original[original.index('str_par ='):original.index('%% plotting')]
    patches = []

    def replace(before, after, reason):
        nonlocal body
        if body.count(before) != 1:
            raise ValueError('Author patch anchor is not unique: '+before)
        body = body.replace(before, after)
        patches.append({'before':before,'after':after,'reason':reason})

    replace('str_par = [0.7; 5e-4; 98; 149.7415; 1.37e10; 1.53714; 0.0087];',
            'str_par = [0.7; 5e-4; 98; 149.6; 1.37e10; 1.53714; 0.0087];', '2024 Table I tension; retain archive density derived from c=2Lf0=137.2')
    replace('BH_par = [4.5e-3; 4.8297e4; 5.7674];', 'BH_par = [0; 4.8297e4; 57.674];',
            '2024 massless hair; archive divides lumped coefficients by width, giving K=4.8297e6 and D=5.7674e3')
    replace('ss_LuGre = [3.1860e5; 0.0027; 0; 0.2280; 0.5071; 1.0207; 2];', 'ss_LuGre = friction;',
            'Published 2024 Tables III, IV and V friction parameters for each case')
    replace('s1Function = 1;', 's1Function = 0;', '2024 constant bristle damping')
    replace('dur = 0.5;', 'dur = frames/44100;', 'Full calibrated paper comparison record')
    replace('TS = ceil(fs*dur);', 'TS = frames;', 'Use exact comparison sample count')
    replace('fN = 2.3433;', 'fN = force;', 'Published normal force per case')
    replace('acc = 0.8722;', 'acc = acceleration;', 'Published bow acceleration per case')
    start, end = body.index(' if acc > 0'), body.index('maxiter = 100;')
    replace(body[start:end], 'v_b_vec = acc*(0:TS-1)/fs;\n\n',
            'Exact sampled acceleration from rest; no endpoint-rounded ramp or terminal speed fit')
    replace('O3 = 2/dt*dens_h + dt/2 * spr_const + damp_const;', 'O3 = dt*spr_const + damp_const;',
            '2024 centered massless hair, Eq. (23)')
    replace(' + 1/M * 1/O3;', ' + 1/M * 1/O3 * eye(M);', '2024 Eq. (25) diagonal hair coupling')
    replace('dn = ((dt/2 * spr_const + 2/dt * dens_h)*d_eta - spr_const*eta)/O3;', 'dn = -spr_const*eta_p/O3;',
            '2024 Eq. (23) free centered hair velocity')
    replace('eta_n = ((2*dens_h/(dt^2) - spr_const/2)*eta - c_h2*eta_p - F_fr(:,n)/M)/c_h1;',
            'eta_n = ((damp_const-dt*spr_const)*eta_p - 2*dt*F_fr(:,n)/M)/O3;', '2024 Eq. (23) hair update')
    replace('H_h(n) = dens_h/2*sum((1/dt*(eta - eta_p)).^2) + spr_const/2 * sum(((eta + eta_p)/2).^2);',
            'H_h(n) = spr_const/4 * sum(eta.^2 + eta_p.^2);', 'Stored energy paired with the centered first-order hair recurrence')
    replace('exp(-(vr/vS).^Sexp) + s2*abs(vr)', 'exp(-(abs(vr)/vS).^Sexp) + s2*abs(vr)',
            'Support 2024 epsilon law with exponent one; exponent two Stribeck law is unchanged')
    replace('dz_ss = ((-Sexp*abs(vr).^(Sexp-1)).*(sign(vr).^Sexp)./(vS^Sexp*s0)).*((fS-fC)*exp(-(vr/vS).^Sexp)) + sign(vr)*s2./s0;',
            'dz_ss = -Sexp*abs(vr).^(Sexp-1)/(vS^Sexp*s0).*((fS-fC)*exp(-(abs(vr)/vS).^Sexp)) + s2/s0;',
            'Derivative of the signed steady bristle law for both exponents; viscous coefficient is zero in all cases')
    replace('Niter = zeros(1,TS);', 'Niter = zeros(1,TS);\nResidual = zeros(1,TS);', 'Record converged force balance independently')
    replace('F_fr(:,n) = s0*z_av + s1.*gn;',
            'F_fr(:,n) = s0*z_av + s1.*gn;\n    Residual(n) = max(abs(vr + IJ_mat*F_fr(:,n) - sn));',
            'Record converged force balance independently')
    body += "\nsave('-mat7-binary', output, 'F_bridge','h_e','Niter','Residual');\n"
    oracle = 'function oracle(frames,force,acceleration,friction,output)\n'+body+'\nend\n\n'+original[original.index('function [Iu,Ju]'):]
    parameters = {
        'circle': [[318600,.0027,0,.228,.5071,1.0207,2], [240990,.0115,0,.4,.3382,1.1489,1]],
        'red': [[370600,.0082,0,.0513,.7727,1.0902,2], [259700,.0034,0,.1,.6951,1.0925,1]],
        'green': [[333200,.05,0,.1603,.7722,1.17,2], [189700,.00074,0,.4,.5308,1.1725,1]],
        'blue': [[302670,.0026,0,.228,.5071,1.0207,2], [331470,.0228,0,.4,.3365,.9116,1]],
    }
    if metadata['sample_rate'] != 44100 or [case['name'] for case in metadata['cases']] != list(parameters):
        raise ValueError('Author check requires the four calibrated 44100 Hz paper cases')
    comparisons = []
    with tempfile.TemporaryDirectory(prefix='matusiak2024-author-') as temporary:
        folder = Path(temporary)
        (folder/'oracle.m').write_text(oracle)
        commands = []
        for case in metadata['cases']:
            for curve, friction in zip(['s','epsilon'],parameters[case['name']]):
                name = case['name']+'_'+curve
                commands.append('oracle('+','.join([str(case['frames']),repr(case['force_n']),repr(case['acceleration_m_s2']),
                    '['+';'.join(map(repr,friction))+']',"'"+name+".mat'"])+');')
        (folder/'run_oracle.m').write_text('\n'.join(commands)+'\n')
        environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        subprocess.run([octave,'--quiet','--no-gui','run_oracle.m'],cwd=folder,env=environment,
                       check=True,capture_output=True,text=True,timeout=180)
        for case in metadata['cases']:
            for curve in ['s','epsilon']:
                name = case['name']+'_'+curve
                mat = loadmat(folder/(name+'.mat'))
                author = mat['F_bridge'].ravel()
                rate, native = wavfile.read(output/(case['name']+'_native_'+curve+'.wav'))
                if rate != 44100 or native.ndim != 1 or len(native) != len(author):
                    raise ValueError('Native WAV dimensions do not match author oracle: '+name)
                native = native.astype(np.float64)
                relative = float(np.linalg.norm(native-author)/np.linalg.norm(author))
                residual = float(np.max(abs(mat['Residual'])))
                energy = float(np.max(abs(mat['h_e'])))
                comparisons.append({'case':case['name'],'curve':curve,'samples':len(author),
                    'relative_l2':relative,'maximum_absolute_force_n':float(np.max(abs(native-author))),
                    'maximum_force_balance_residual_m_s':residual,'maximum_energy_error_j':energy,
                    'maximum_newton_iterations':int(np.max(mat['Niter'])),
                    'accepted':relative<1e-5 and residual<1e-10 and energy<1e-10})
    return {'source':pin['source'],'revision':pin['revision'],'source_sha256':digest,'license':'GPL-3.0-only',
        'runtime':version,'environment':{'OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'},
        'scope':'Patched 2025 author archive independently executes the 2024 interaction reconstruction; this is not the unavailable original 2024 simulation script',
        'retained':'Original sparse matrix recurrence, Newton Jacobian and backslash solve, interpolation helper, h/hT torsion feedback and weighted energy, Q mapping, five actual contact points',
        'sample_rate':44100,'bow_points':5,'wave_speed_m_s':137.2,
        'patches':patches,'generated_oracle_sha256':hashlib.sha256(oracle.encode()).hexdigest(),
        'acceptance':{'maximum_relative_l2':1e-5,'maximum_force_balance_residual_m_s':1e-10,'maximum_energy_error_j':1e-10},
        'accepted':all(item['accepted'] for item in comparisons),'comparisons':comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=ROOT/'build/matusiak2024Reproduce')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/reproduction/matusiak2024')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--check-author',action='store_true',help='Execute the adapted pinned MATLAB source through Octave')
    parser.add_argument('--gpu-voices',type=int,default=1,help='Full-record GPU precision diagnostic; zero skips GPU')
    parser.add_argument('--convention',choices=['paper','archive'],default='archive',help='Printed 2024 coupling or archived author grid and torsion conventions')
    parser.add_argument('--sample-rate',type=int,default=44100,help='Integration rate; comparison WAVs use 44100 Hz')
    parser.add_argument('--extract-paper',type=Path,help='Regenerate compact calibrated traces using PyMuPDF')
    args = parser.parse_args()
    if args.extract_paper:
        extract_paper(args.extract_paper)
    if args.gpu_voices<0 or args.gpu_voices>4096:
        parser.error('--gpu-voices must be between 0 and 4096')
    if args.sample_rate<8000 or args.sample_rate>192000:
        parser.error('--sample-rate must be between 8000 and 192000')
    if args.check_author and (args.convention!='archive' or args.sample_rate!=44100):
        parser.error('--check-author requires --convention archive --sample-rate 44100')
    metadata = json.loads((FIXTURE/'Reference.json').read_text())
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    comparisons = []; listening = []; gpu_diagnostic = None
    for case_index, case in enumerate(metadata['cases']):
        fixture = FIXTURE/(case['name']+'.f32')
        if hashlib.sha256(fixture.read_bytes()).hexdigest()!=case['sha256']:
            raise ValueError('Reference fixture hash mismatch')
        reference = np.fromfile(fixture,'<f4').reshape(-1,3)
        rendered=[]
        for curve,name in enumerate(['s','epsilon']):
            destination=output/(case['name']+'_'+name)
            subprocess.run([str(args.binary.resolve()),str(destination),str(int(np.ceil(case['frames']*args.sample_rate/44100))),str(case['force_n']),
                            str(case['acceleration_m_s2']),str(curve),str(case_index),str(args.sample_rate),
                            str(args.gpu_voices if case_index==0 and curve==0 else 0),str(int(args.convention=='archive'))],check=True)
            trace=np.fromfile(destination/'trace.f64','<f8').reshape(-1,8)
            if args.sample_rate!=44100:
                trace=resample_poly(trace,44100,args.sample_rate,axis=0)[:case['frames']]
            rendered.append(trace[:,0])
            if case_index==0 and curve==0 and args.gpu_voices:
                gpu_diagnostic=json.loads((destination/'gpu.json').read_text())
                gpu_diagnostic['full_record_precision_accepted']=gpu_diagnostic['relative_l2']<.01 and gpu_diagnostic['failed_steps']==0
                listening.append({'title':'Matusiak 2024 full-record CPU/Metal precision diagnostic',
                                  'reference':str(destination/'cpu_rounded_bridge_force.wav'),'synthesis':str(destination/'gpu_bridge_force.wav'),
                                  'reference_label':'FP64 CPU','synthesis_label':'FP32 Metal',
                                  'notes':'Same float-rounded controls. Full-record precision accepted: '+str(gpu_diagnostic['full_record_precision_accepted'])+'. Nonlinear convergence alone does not establish waveform accuracy.'})
            comparisons.append({'case':case['name'],'curve':name,'measured_comparison':compare(reference[:,0],trace[:,0]),
                                'author_comparison':compare(reference[:,curve+1],trace[:,0]),
                                'author_measured_comparison':compare(reference[:,0],reference[:,curve+1]),
                                'native':json.loads((destination/'native.json').read_text()),
                                'texture':{'native':texture_metrics(44100,trace[:,0,None]),'author':texture_metrics(44100,reference[:,curve+1,None]),'measured':texture_metrics(44100,reference[:,0,None])}})
        signals=[reference[:,0],reference[:,1],reference[:,2],*rendered]
        for name,signal in zip(['measured','author_s','author_epsilon','native_s','native_epsilon'],signals):
            wavfile.write(output/(case['name']+'_'+name+'.wav'),44100,signal.astype(np.float32))
        for curve in ['s','epsilon']:
            for source,label in [('author_'+curve,'Author simulation recovered from PDF vectors'),('measured','Measured robot bridge force recovered from PDF vectors')]:
                listening.append({'title':'Matusiak 2024 '+case['name']+' '+curve+' versus '+source,
                                  'reference':str(output/(case['name']+'_'+source+'.wav')),
                                  'synthesis':str(output/(case['name']+'_native_'+curve+'.wav')),
                                  'reference_label':label,'synthesis_label':'Our 2024 interaction, '+args.convention+' numerical conventions',
                                  'notes':'Raw bridge force in newtons. Published drive and friction parameters. '+('Archived grid offsets, h/hT torsion feedback and density derived from the stated wave speed.' if args.convention=='archive' else 'Printed 2024 physical coordinates and unscaled torsion feedback.')+' No fitted timing or gain. AC waveform, spectrum and envelope errors are recorded separately.'})
        for curve in ['s','epsilon']:
            destination=output/(case['name']+'_'+curve)
            for filename in ['trace.f64','bridge_force.wav','native.json','gpu.json']:
                (destination/filename).unlink(missing_ok=True)
            if not any(destination.iterdir()):
                destination.rmdir()
    author_oracle=check_author(ROOT,output,metadata,args.convention,args.sample_rate) if args.check_author else None
    (output/'manifest.json').write_text(json.dumps({'paper':metadata['paper'],'reference':metadata,'comparisons':comparisons,'gpu':gpu_diagnostic,'author_oracle':author_oracle,
        'convention':args.convention,'integration_sample_rate':args.sample_rate,
        'scope':'2024 interaction with explicitly selected numerical conventions; no fitted drives or parameter optimization'},indent=2)+'\n')
    (output/'cases.json').write_text(json.dumps({'method':'matusiak2024','sample_rate':44100,'cases':listening},indent=2)+'\n')
    author_cases = [case for case in listening if '_author_' in Path(case['reference']).name]
    (output/'author-cases.json').write_text(json.dumps({'cases':author_cases},indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'tools/BuildListeningReport.py'),str(output/'author-cases.json'),
                    '--output',str(output/'listening')],check=True,cwd=ROOT)
    if author_oracle and not author_oracle['accepted']:
        raise RuntimeError('Adapted author-source comparison failed; see manifest.json')


if __name__=='__main__':
    main()
