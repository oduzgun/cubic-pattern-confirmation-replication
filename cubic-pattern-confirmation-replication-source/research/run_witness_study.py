import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t

from ectc import CubicDesign, fitted_topology
from ectc_witness import confirm_selected_witness, fit_statistics, grid_pvalues, select_training_witness
from run_power_study import evaluate_batch, exact_interval, paired_interval, predictor_for


PROTOCOL_PATH = Path("research/p04/protocol.json")
METHODS = ("cone", "witness_0", "grid_9", "grid_19", "split_50", "PTCS")
PRIMARY = ("grid_9", "grid_19", "split_50")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()




def models_for(size, design):
    predictor = predictor_for(size, design)
    full = CubicDesign(predictor, [-1, 1])
    training = CubicDesign(predictor[::2], [-1, 1])
    confirmation = CubicDesign(predictor[1::2], [-1, 1])
    assert len(training.predictor) + len(confirmation.predictor) == size
    assert confirmation.degrees == size // 2 - 4
    return full, training, confirmation


def evaluate_responses(models, responses, grids, alpha):
    full, training, confirmation = models
    derivatives, variances = fit_statistics(full, responses)
    anchors, anchor_pvalues = evaluate_batch(full, derivatives, variances, alpha)
    grid_values = {name: grid_pvalues(full, derivatives, variances, grid) for name, grid in grids.items()}
    training_derivatives, _ = fit_statistics(training, responses[:, ::2])
    confirmation_derivatives, confirmation_variances = fit_statistics(confirmation, responses[:, 1::2])
    locations, active = select_training_witness(training_derivatives)
    split = confirm_selected_witness(confirmation, confirmation_derivatives, confirmation_variances, locations, active)
    pvalues = np.column_stack((anchor_pvalues[:, 0], anchor_pvalues[:, 1],
                              grid_values['grid_9'], grid_values['grid_19'], split))
    if not np.all(np.isfinite(pvalues)) or np.any(pvalues < 0) or np.any(pvalues > 1):
        raise ArithmeticError("Invalid comparator p-value")
    decisions = np.column_stack((pvalues <= alpha, anchors[:, 3]))
    if np.any(decisions[:, 5] & ~decisions[:, 0]):
        raise ArithmeticError("PTCS/cone containment failure")
    return decisions, pvalues, locations, active


def check_preflight(generator, models, protocol):
    checked, active_count = 0, 0
    for key, group in models.items():
        full, training, confirmation = group
        responses = generator.normal(size=(32, len(full.predictor)))
        coefficients = generator.normal(size=(32, 4))
        responses += coefficients @ full.matrix.T
        decisions, pvalues, selected, active = evaluate_responses(group, responses, protocol['grids'], protocol['alpha'])
        training_derivatives, _ = fit_statistics(training, responses[:, ::2])
        changed = responses.copy()
        changed[:, 1::2] += 100 * generator.normal(size=changed[:, 1::2].shape)
        unchanged_derivatives, _ = fit_statistics(training, changed[:, ::2])
        unchanged_locations, unchanged_active = select_training_witness(unchanged_derivatives)
        assert np.array_equal(selected, unchanged_locations) and np.array_equal(active, unchanged_active)
        for index, response in enumerate(responses):
            scalar = full.fit(response)
            assert abs(pvalues[index, 0] - scalar['pvalues']['N']) < 1e-10
            assert abs(pvalues[index, 1] - scalar['witness_pvalues']['N']) < 1e-10
            assert bool(decisions[index, 5]) == scalar['ptcs_N']
            coefficients = np.linalg.lstsq(full.matrix, response, rcond=None)[0]
            residual = response - full.matrix @ coefficients
            variance = residual @ residual / full.degrees
            derivative = coefficients[1:] * [1, 2, 3]
            endpoint_tails = []
            for location in (-1, 1):
                contrast = np.array([1, location, location**2])
                statistic = contrast @ derivative / np.sqrt(variance * (contrast @ full.covariance @ contrast))
                endpoint_tails.append(t.sf(statistic, full.degrees))
            for offset, name in enumerate(('grid_9', 'grid_19'), start=2):
                negative_tails = []
                for location in protocol['grids'][name]:
                    contrast = np.array([1, location, location**2])
                    statistic = contrast @ derivative / np.sqrt(variance * (contrast @ full.covariance @ contrast))
                    negative_tails.append(t.cdf(statistic, full.degrees))
                expected = max(*endpoint_tails, min(1, len(negative_tails) * min(negative_tails)))
                assert abs(expected - pvalues[index, offset]) < 1e-10
            if active[index]:
                training_fit = np.linalg.lstsq(training.matrix, response[::2], rcond=None)[0]
                expected_location = -2 * training_fit[2] / (6 * training_fit[3])
                assert abs(selected[index] - expected_location) < 1e-10
                confirmation_fit = np.linalg.lstsq(confirmation.matrix, response[1::2], rcond=None)[0]
                residual = response[1::2] - confirmation.matrix @ confirmation_fit
                variance = residual @ residual / confirmation.degrees
                derivative = confirmation_fit[1:] * [1, 2, 3]
                tails = []
                for location in (-1, 1, selected[index]):
                    contrast = np.array([1, location, location**2])
                    statistic = contrast @ derivative / np.sqrt(variance * (contrast @ confirmation.covariance @ contrast))
                    tails.append(statistic)
                expected = max(t.sf(tails[0], confirmation.degrees), t.sf(tails[1], confirmation.degrees),
                               t.cdf(tails[2], confirmation.degrees))
                assert abs(expected - pvalues[index, 4]) < 1e-10
                active_count += 1
            else:
                assert pvalues[index, 4] == 1
            checked += 1
        for invalid_grid in ([], [0, 0], [-1, 0], [float('nan')]):
            try:
                grid_pvalues(full, training_derivatives[:1], np.ones(1), invalid_grid)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid grid accepted')
    return {'raw_data_reference_draws': checked, 'active_split_reference_draws': active_count,
            'checks': 'scalar core, independent grid tails, independent half-sample OLS, confirmation isolation, inactive abstention, invalid grids PASS'}


def save_csv(output, name, rows):
    with (output / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output):
    started = time.monotonic()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if fingerprint('research/ectc.py') != protocol['core_sha256'] or fingerprint('research/run_power_study.py') != protocol['helper_sha256']:
        raise RuntimeError('Frozen core/helper mismatch')
    history = {"status": "archival preflight excluded from curated code"}
    sources = ['research/ectc.py', 'research/run_power_study.py', 'research/ectc_witness.py',
               'research/run_witness_study.py', str(PROTOCOL_PATH)]
    hashes = {name: fingerprint(name) for name in sources}
    (output/'execution.json').write_text(json.dumps({'status':'RUNNING','source_sha256':hashes,'protocol':protocol['id']},indent=2)+'\n')
    groups = {key: models_for(*key) for key in sorted({(case['n'],case['design']) for case in protocol['cases']})}
    streams = np.random.SeedSequence(protocol['seed']).spawn(len(protocol['cases'])+1)
    checks = check_preflight(np.random.default_rng(streams[0]), groups, protocol)
    print(f'Preflight PASS: {checks}', flush=True)
    rates, differences, diagnostics, alarms = [], [], [], []
    for case, stream in zip(protocol['cases'], streams[1:]):
        models = groups[(case['n'],case['design'])]
        full = models[0]
        if (case['kind']=='alternative') != (fitted_topology(case['derivative'])=='N'):
            raise AssertionError('Prespecified truth mismatch')
        mean_coefficients = np.r_[0, np.array(case['derivative']) / np.array([1,2,3])]
        generator = np.random.default_rng(stream)
        count = case['replications']
        responses = full.matrix @ mean_coefficients + generator.normal(size=(count,case['n']))
        decisions, pvalues, witnesses, active = evaluate_responses(models, responses, protocol['grids'], protocol['alpha'])
        metadata = {name:case.get(name) for name in ('id','kind','n','design','center','halfwidth','depth','null_type')}
        true_witness_values = np.polynomial.polynomial.polyval(witnesses,case['derivative'])
        diagnostics.append(dict(metadata,replications=count,interior_selected=int(active.sum()),
                                selected_strictly_negative=int(np.sum(active & (true_witness_values<0)))))
        for index, method in enumerate(METHODS):
            successes = int(decisions[:,index].sum())
            lower, upper = exact_interval(successes,count)
            null_lower, null_upper = None, None
            if case['kind']=='null':
                null_lower,null_upper = exact_interval(successes,count,0.05/protocol['reporting']['null_family'])
                if null_lower>protocol['alpha']:
                    alarms.append({'case':case['id'],'method':method,'lower':null_lower})
            rates.append(dict(metadata,method=method,successes=successes,replications=count,rate=successes/count,
                              ci_lower=lower,ci_upper=upper,null_family_lower=null_lower,null_family_upper=null_upper))
        if case['kind']=='alternative':
            for method in PRIMARY:
                index = METHODS.index(method)
                wins = int(np.sum(decisions[:,0] & ~decisions[:,index]))
                losses = int(np.sum(~decisions[:,0] & decisions[:,index]))
                lower,upper = paired_interval(wins,losses,count)
                family_lower,family_upper = paired_interval(wins,losses,count,protocol['reporting']['primary_difference_family'])
                differences.append(dict(metadata,comparator=method,wins=wins,losses=losses,replications=count,
                                        difference=(wins-losses)/count,ci_lower=lower,ci_upper=upper,
                                        family_lower=family_lower,family_upper=family_upper))
        print(f"{case['id']} {case['kind']} n={case['n']} {case['design']}: "+
              ', '.join(f'{method}={decisions[:,index].mean():.3f}' for index,method in enumerate(METHODS)),flush=True)
        save_csv(output,'rates_partial.csv',rates)
        if differences:
            save_csv(output,'paired_partial.csv',differences)
        (output/'progress.json').write_text(json.dumps({'last_completed_case':case['id'],
            'completed_cases':len(rates)//len(METHODS),'source_sha256':hashes},indent=2)+'\n')
    assert len(differences)==protocol['reporting']['primary_difference_family']
    save_csv(output,'rates.csv',rates)
    save_csv(output,'paired_differences.csv',differences)
    save_csv(output,'split_diagnostics.csv',diagnostics)
    summary = {}
    for method in PRIMARY:
        selected = [row for row in differences if row['comparator']==method]
        summary[method] = {'cells':len(selected),'cone_higher_family_bound':sum(row['family_lower']>0 for row in selected),
                           'cone_lower_family_bound':sum(row['family_upper']<0 for row in selected),
                           'unresolved_family_bound':sum(row['family_lower']<=0<=row['family_upper'] for row in selected),
                           'range_point_difference':[min(row['difference'] for row in selected),max(row['difference'] for row in selected)]}
    report = {'status':'COMPLETED' if not alarms else 'COMPLETED; SCIENTIFIC HOLD: null calibration alarm',
              'protocol':protocol['id'],'source_sha256':hashes,'checks':checks,
              'cases_completed':len(protocol['cases']),'replications':sum(case['replications'] for case in protocol['cases']),
              'numerical_failures':0,'null_alarms':alarms,'comparison_summary':summary,
              'maximum_observed_null_rate':max(row['rate'] for row in rates if row['kind']=='null'),
              'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
              'elapsed_seconds':time.monotonic()-started,
              'limits':protocol['limits']+['Raw response data regenerated by fixed seed; not independent scientific replication',
                 'Fine grid contains each alternative center in this finite design',
                 '50/50 training fraction is fixed, not optimized',
                 'Split comparator selects a witness for prespecified N, not one of six topologies']}
    report['output_sha256'] = {name:fingerprint(output/name) for name in ('rates.csv','paired_differences.csv','split_diagnostics.csv')}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'execution.json').write_text(json.dumps({'status':report['status'],'source_sha256':hashes,
        'validation_sha256':fingerprint(output/'validation.json')},indent=2)+'\n')
    print(json.dumps({'status':report['status'],'replications':report['replications'],'comparisons':summary,
                     'null_alarms':alarms,'elapsed_seconds':report['elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir',default='research/p04')
    arguments = parser.parse_args()
    output = Path(arguments.output_dir)
    if output.is_absolute() or '..' in output.parts or not output.parts or output.parts[0]!='research':
        raise SystemExit('Output must be a relative directory under research/')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'execution.json').exists():
        raise SystemExit('Refusing to overwrite an existing run; choose a fresh --output-dir')
    try:
        run(output)
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'status':'FAILED','error':repr(error)},indent=2)+'\n')
        raise
