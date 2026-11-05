"""Stage 2 reward reuses the frozen Stage 1 parser/editor, without output repair."""
import math
from evaluation_core import Scorer, parse_parameters


def image_reward(parsed, scored):
    if scored.get('metric_error') or not scored.get('input_readable') or not scored.get('reference_readable'):
        raise RuntimeError(f'Data/metric failure, not a policy penalty: {scored}')
    d0 = scored['baseline_metrics']['pixel_L1']
    d = scored['edited_metrics']['pixel_L1']
    if not (math.isfinite(d0) and math.isfinite(d)):
        raise RuntimeError('Nonfinite image distance')
    if not parsed['valid'] or not scored['editor_success']:
        return -2.0
    return max(-1.0, min(1.0, (d0 - d) / max(d0, 0.01)))


class ImageReward:
    def __init__(self, root, protocol, rows):
        self.scorer = Scorer(root, protocol)
        self.protocol = protocol
        self.rows = {r['id']: r for r in rows}

    def score(self, sample_id, response):
        parsed = parse_parameters(response, self.protocol)
        scored = self.scorer.score(self.rows[sample_id], parsed, None)
        reward = image_reward(parsed, scored)
        params = parsed.get('parameters')
        return dict(id=sample_id, response=response, reward=reward, parsing=parsed, scoring=scored,
                    empty_parameters=params == {},
                    all_zero_parameters=params is not None and all(v == 0 for v in params.values()),
                    reward_saturated=reward in [-1.0, 1.0])
