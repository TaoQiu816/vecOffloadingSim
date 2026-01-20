import json
import random


class ReservoirSampler:
    def __init__(self, size=256, seed=0):
        self.size = int(size)
        self.samples = []
        self.count = 0
        self.rng = random.Random(seed)

    def add(self, value):
        self.count += 1
        if len(self.samples) < self.size:
            self.samples.append(float(value))
            return
        idx = self.rng.randint(0, self.count - 1)
        if idx < self.size:
            self.samples[idx] = float(value)

    def quantile(self, q):
        if not self.samples:
            return 0.0
        sorted_vals = sorted(self.samples)
        pos = int(q * (len(sorted_vals) - 1))
        pos = max(0, min(pos, len(sorted_vals) - 1))
        return float(sorted_vals[pos])


class StatBucket:
    def __init__(self, sample_size=256, seed=0):
        self.sum = 0.0
        self.abs_sum = 0.0
        self.count = 0
        self.nonzero_count = 0
        self.min = float("inf")
        self.max = -float("inf")
        self.sampler = ReservoirSampler(size=sample_size, seed=seed)

    def add(self, value):
        v = float(value)
        self.sum += v
        self.abs_sum += abs(v)
        self.count += 1
        if abs(v) > 1e-12:
            self.nonzero_count += 1
        if v < self.min:
            self.min = v
        if v > self.max:
            self.max = v
        self.sampler.add(v)

    def mean(self):
        if self.count <= 0:
            return 0.0
        return self.sum / self.count

    def p95(self):
        return self.sampler.quantile(0.95)


class RewardStats:
    def __init__(self, sample_size=256, seed=0):
        self.sample_size = int(sample_size)
        self.seed = int(seed)
        self.metrics = {}
        self.counters = {}

    def reset(self):
        self.metrics = {}
        self.counters = {}

    def add_metric(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = StatBucket(sample_size=self.sample_size, seed=self.seed)
        self.metrics[name].add(value)

    def add_counter(self, name, delta=1):
        self.counters[name] = self.counters.get(name, 0) + int(delta)

    def summary(self):
        out = {"counters": {}, "metrics": {}}
        for key, val in self.counters.items():
            out["counters"][key] = int(val)
        for key, bucket in self.metrics.items():
            out["metrics"][key] = {
                "mean": bucket.mean(),
                "min": bucket.min if bucket.count > 0 else 0.0,
                "max": bucket.max if bucket.count > 0 else 0.0,
                "abs_mean": (bucket.abs_sum / bucket.count) if bucket.count > 0 else 0.0,
                "p95": bucket.p95(),
                "count": bucket.count,
                "nonzero_count": bucket.nonzero_count,
            }
        return out

    def to_json_line(self, extra=None):
        payload = self.summary()
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)
