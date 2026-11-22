from run_cases import run

# Reverse the order in the second round to expose clock/load/order bias.
for round_id, stages in [(1, ['baseline', 'stage_a_grouped', 'stage_b', 'stage_c']),
                         (2, ['stage_c', 'stage_b', 'stage_a_grouped', 'baseline'])]:
    for branch in ['selector', 'tf', 'both']:
        for stage in stages:
            if branch == 'selector' and stage == 'stage_a_grouped': continue
            run(stage, branch, 'paired_' + branch + '_' + str(round_id), repeats=5)
