import sonnet as snt
import tensorflow as tf

class ActorNetwork(snt.AbstractModule):

    def __init__(self, layer_sizes, action_ranges, layer_activations):
        super(ActorNetwork, self).__init__(name='actor_network')

        action_min = action_ranges[0]
        action_max = action_ranges[1]

        self._action_range = tf.constant((action_max - action_min) / 2.0, dtype=tf.float32)
        self._action_mean = tf.constant((action_max + action_min) / 2.0, dtype=tf.float32)

        self._network = snt.nets.MLP(layer_sizes, activation=layer_activations)
        self._action = snt.Linear(output_size=len(action_min))

    def _build(self, input):
        flattened = snt.BatchFlatten()(input)

        net_out = self._network(flattened)
        action = (tf.nn.tanh(self._action(net_out)) * self._action_range) + self._action_mean

        return action

    def get_variables(self, collection=tf.GraphKeys.TRAINABLE_VARIABLES):
        return (self._network.get_variables(collection) +
                self._action.get_variables(collection))
