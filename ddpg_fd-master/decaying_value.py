import math

class DecayingValue(object):

    def __init__(self, start_value, end_value, num_iters):
        self._start_value = start_value
        self._end_value = end_value
        self._num_iters = num_iters

        self._decay = math.pow(end_value / start_value, 1.0 / float(num_iters))

    def __call__(self, iter):
        if iter >= self._num_iters:
            return self._end_value
        else:
            # return self._start_value + iter * (self._end_value - self._start_value) / self._num_iters
            return self._start_value * (self._decay ** iter)
