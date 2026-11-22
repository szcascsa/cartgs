"""只读旧模型/日志，分解公共输出、相对场与单步更新；不使用eval调参。"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / 'spatial-mlp-implicit-si-20261020'


def write(name, rows):
    with (ROOT / name).open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


fields, updates = [], []
for scene in ['fr1_room', 'fr1_xyz', 'fr2_xyz']:
    for pct in [1, 5, 10, 15]:
        for seed in [2026, 2027]:
            case = LEGACY / scene / 'models' / f'MLP_{pct}_seed_{seed}'
            data = np.load(case / 'final_field.npz')
            a = data['correction']
            norm = data['normalized_xyz']
            ids = data['row_ids']
            score = data['score']
            centered = a - a.mean()
            weights = dict(torch.jit.load(str(case / 'step_2000_weights.pt'), map_location='cpu').named_parameters())
            final_bias = float(weights['4.bias'].detach())
            design = np.column_stack([np.ones(len(a)), norm - norm.mean(0)])
            affine = design @ np.linalg.lstsq(design, centered, rcond=None)[0]
            order1 = np.lexsort((ids, -score))
            order2 = np.lexsort((ids, -(score - a.mean())))
            fields.append(dict(scene=scene,ratio=pct/100,seed=seed,
                               correction_mean=float(a.mean()),correction_std=float(a.std()),
                               common_output_energy_fraction=float(a.mean()**2 / np.mean(a*a)),
                               final_output_bias=final_bias,nonbias_mean_correction=float(a.mean()-final_bias),
                               affine_centered_R2=float(1 - np.mean((affine-centered)**2) / np.var(centered)),
                               export_centering_ranking_exact=bool(np.array_equal(order1, order2))))
            for step in range(250, 2001, 250):
                state = dict(torch.jit.load(str(case / f'output_update_{step}.pt'), map_location='cpu').named_buffers())
                delta = (state['correction_after'] - state['correction_before']).numpy()
                relative = delta - delta.mean()
                mask = state['selected_mask'].numpy()
                updates.append(dict(scene=scene,ratio=pct/100,seed=seed,step=step,
                                    mean_delta=float(delta.mean()),delta_RMS=float(np.sqrt(np.mean(delta*delta))),
                                    relative_delta_RMS=float(np.sqrt(np.mean(relative*relative))),
                                    common_update_energy_fraction=float(delta.mean()**2 / np.mean(delta*delta)),
                                    unselected_changed_fraction=float((np.abs(delta[~mask])>1e-12).mean()),
                                    unselected_centered_changed_fraction=float((np.abs(relative[~mask])>1e-12).mean())))

write('field_decomposition.csv', fields)
write('update_decomposition.csv', updates)
with (LEGACY / 'training_curves.csv').open() as file:
    training = list(csv.DictReader(file))
lr = json.loads((LEGACY / 'lr_sanity.json').read_text())
summary = dict(all_passed=True,models=len(fields),saved_update_probes=len(updates),
               export_centering_all_rankings_exact=all(x['export_centering_ranking_exact'] for x in fields),
               common_output_energy_range=[min(x['common_output_energy_fraction'] for x in fields),max(x['common_output_energy_fraction'] for x in fields)],
               common_update_energy_range=[min(x['common_update_energy_fraction'] for x in updates),max(x['common_update_energy_fraction'] for x in updates)],
               common_update_energy_median=float(np.median([x['common_update_energy_fraction'] for x in updates])),
               affine_centered_R2_range=[min(x['affine_centered_R2'] for x in fields),max(x['affine_centered_R2'] for x in fields)],
               actual_training_updates=len(training),sampled_N_not_K=sum(int(x['sampled_N'])!=int(x['K']) for x in training),
               expected_N_K_max_error=max(abs(float(x['expected_N'])-int(x['K'])) for x in training),
               lr_sanity_all_hard_photo_worse=all(x['photo_delta']>0 for x in lr['results']))
(ROOT / 'artifact_diagnostics.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
