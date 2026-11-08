"""Independent integrity checks for recorded evaluation tables and split hashes."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]


def main():
    groups={
        'validation':set(f'pair{i}' for i in range(1,7)),
        'test':set(f'pair{i}' for i in [13,15,21,23,31,33,50,52,58,60,68,70]),
        'confirmation':set(f'pair{i}' for i in range(7,13)),
    }
    assert not (groups['validation'] & groups['test'] or groups['validation'] & groups['confirmation'] or groups['test'] & groups['confirmation'])
    audits={}
    for phase,scenes in groups.items():
        root=ROOT/'results'/phase
        rows=list(csv.DictReader((root/'per_frame.csv').open(encoding='utf-8-sig')))
        summary=json.loads((root/'summary.json').read_text())
        assert set(summary['scenes'])==scenes
        for name,metrics in summary['methods'].items():
            scene_means=[]
            for scene in scenes:
                rr=[r for r in rows if r['method']==name and r['scene']==scene]
                assert len(rr)==30 and {int(r['index']) for r in rr}==set(range(30))
                scene_means.append(statistics.mean(float(r['psnr']) for r in rr))
            assert abs(statistics.mean(scene_means)-metrics['psnr'])<1e-9
        audits[phase]=dict(scenes=len(scenes),frames=len(scenes)*30,methods=len(summary['methods']),psnr_recomputed=True)
    for record,conf in [('final_test_config_sha256.txt','initial_config.json'),('confirmation_config_sha256.txt','confirmation_frozen_config.json')]:
        actual=hashlib.sha256((ROOT/'records'/conf).read_bytes()).hexdigest()
        assert actual==(ROOT/'records'/record).read_text().strip(),record
    final=json.loads((ROOT/'records/final_config.json').read_text())
    frozen=json.loads((ROOT/'records/confirmation_frozen_config.json').read_text())
    for key in ['beta','gamma','align','confirmation_scenes']:
        assert final[key]==frozen[key],key
    audits['configuration_hashes_match']=True
    audits['scene_groups_disjoint']=True
    (ROOT/'records/results_verification.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    print(json.dumps(audits,indent=2))


if __name__=='__main__':main()
