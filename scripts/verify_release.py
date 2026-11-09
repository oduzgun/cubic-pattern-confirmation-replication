#!/usr/bin/env python3
"""Check release membership, source hashes, and optional P03/P04 replay results."""
from pathlib import Path
import json, csv, hashlib, re, argparse, math
ROOT=Path(__file__).resolve().parents[1]
EXPECT={
 'LICENSE','CITATION.cff','README.md','requirements.txt','MANIFEST.json',
 'research/ectc.py','research/ectc_witness.py','research/replay_studies.py',
 'research/run_power_study.py','research/run_witness_study.py',
 'research/p03/protocol.json','research/p03/rates.csv','research/p03/paired_differences.csv',
 'research/p04/protocol.json','research/p04/rates.csv','research/p04/paired_differences.csv',
 'research/p04/split_diagnostics.csv',
 '.github/workflows/reproducibility.yml','.gitignore',
 'docs/REPRODUCIBILITY.md','scripts/verify_release.py'
}
RESULTS={'p03':['rates.csv','paired_differences.csv'],
         'p04':['rates.csv','paired_differences.csv','split_diagnostics.csv']}
COUNTS={'p03':[288,144],'p04':[432,144,72]}
FORBIDDEN_KEYWORDS=('prior_exposure','prior_input_provenance','session handoff','personal approval')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def verified():
    m=json.loads((ROOT/'MANIFEST.json').read_text('utf-8'))
    recorded=m['curated_files_sha256']
    present={f.relative_to(ROOT).as_posix() for f in ROOT.rglob('*') if f.is_file() and '.git' not in f.parts and not any(part.endswith('_rerun') or part.endswith('_ci') or part=='__pycache__' for part in f.parts)}
    if present!=EXPECT:raise ValueError(f'File membership mismatch: missing={sorted(EXPECT-present)}, extras={sorted(present-EXPECT)}')
    if set(recorded)!=EXPECT-{'MANIFEST.json'}:raise ValueError('Manifest membership mismatch')
    for path,h in recorded.items():
        if sha(ROOT/path)!=h: raise ValueError(f'SHA256 mismatch: {path}')
    for path,h in m['new_curated_source_hashes'].items():
        if sha(ROOT/path)!=h: raise ValueError(f'Pinned scientific source mismatch: {path}')
    for study in ('p03','p04'):
        p=json.loads((ROOT/f'research/{study}/protocol.json').read_text())
        assert len(p['cases'])==72
        assert sum(int(c['replications']) for c in p['cases'])==96000
    for path in present:
        p=ROOT/path
        if p.suffix in ('.py','.json','.md','.cff','.txt') or p.name=='LICENSE':
            s=p.read_text('utf-8').lower()
            # The validator itself legitimately names forbidden strings in its checks.
            if path != 'scripts/verify_release.py' and any(a in s for a in FORBIDDEN_KEYWORDS):
                raise ValueError(f'Private-process marker in {path}')
    print('PASS: curated source manifest, source hashes and scientific protocols')

def match_outputs():
    for study,files in RESULTS.items():
        for i,name in enumerate(files):
            old=ROOT/'research'/study/name
            new=ROOT/'research'/f'{study}_rerun'/name
            if not new.is_file():raise ValueError(f'Missing rerun {new}')
            with old.open(newline='',encoding='utf-8-sig') as a, new.open(newline='',encoding='utf-8-sig') as b:
                x=list(csv.DictReader(a));y=list(csv.DictReader(b))
            if len(x)!=COUNTS[study][i] or len(y)!=len(x):raise ValueError('CSV row count mismatch: '+str(new))
            if x and list(x[0])!=list(y[0]):raise ValueError('CSV schema mismatch: '+str(new))
            for line,(u,v) in enumerate(zip(x,y),start=2):
                for col,va in u.items():
                    vb=v[col]
                    if va==vb:continue
                    try:
                        f=float(va);g=float(vb)
                        if math.isfinite(f) and math.isfinite(g) and abs(f-g)<=1e-11*max(1,abs(f),abs(g)):continue
                    except (TypeError, ValueError):pass
                    raise ValueError(f'Scientific mismatch {new} row {line} col {col}')
        rpt=ROOT/'research'/f'{study}_rerun'/'validation.json'
        x=json.loads(rpt.read_text())
        if x.get('status')!='COMPLETED' or x.get('replications')!=96000 or x.get('numerical_failures')!=0 or x.get('null_alarms'):
            raise ValueError(f'{study} validation failed: {x}')
    print('PASS: full 192000-draw P03/P04 scientific CSV replay')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--compare-rerun',action='store_true')
    options=ap.parse_args()
    verified()
    if options.compare_rerun:match_outputs()
