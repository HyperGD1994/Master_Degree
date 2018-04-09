import sonnet as snt
import tensorflow as tf

class CriticNetwork(snt.AbstractModule):

    def __init__(self, layer_sizes, layer_activations, output_activation):
        super(CriticNetwork, self).__init__(name='critic_network')

        self._layer_activations = layer_activations

        self._state_layer = snt.Linear(output_size=layer_sizes[0])
        self._action_layer = snt.Linear(output_size=layer_sizes[0])
        self._network = snt.nets.MLP(layer_sizes[1:], activation=layer_activations)
        self._value = snt.Linear(output_size=1)
        self._output_activation = output_activation

        self._value_range = tf.constant(1000.0, dtype=tf.float32)
        self._value_mean = tf.constant(-1000.0, dtype=tf.float32)

    def _build(self, state, action):
        state_flattened = snt.BatchFlatten()(state)
        action_flattened = snt.BatchFlatten()(action)

        l1 = tf.concat([self._layer_activations(self._state_layer(state_flattened)),
                        self._layer_activations(self._action_layer(action_flattened))], 1)

        net_out = self._network(l1)
        # value = (tf.nn.tanh(self._value(net_out)) * self._value_range) + self._value_mean
        value = self._value(net_out)

        return value

    def get_variables(self, collection=tf.GraphKeys.TRAINABLE_VARIABLES):
        return (self._state_layer.get_variables(collection) +
                self._action_layer.get_variables(collection) +
                self._network.get_variables(collection) +
                self._value.get_variables(collection))
