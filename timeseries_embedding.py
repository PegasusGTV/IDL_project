# import tensorflow as tf
# from tensorflow.keras.layers import Layer

# class Time2Vec(Layer):
#     def __init__(self, d_model, **kwargs):
#         """
#         d_model: number of output dimensions for the time embedding
#         """
#         super(Time2Vec, self).__init__()
#         self.d_model = d_model
#         assert d_model >= 2, "d_model must be at least 2 (1 linear + 1 periodic)"

#     def build(self, input_shape):
#         seq_len = input_shape[1]  # (batch, seq_len, num_features)
        
#         # Linear (non-periodic) component: 1 set of weights & bias
#         self.weight_linear = self.add_weight(
#             name='weight_linear',
#             shape=(1,),
#             initializer='uniform',
#             trainable=True
#         )
#         self.bias_linear = self.add_weight(
#             name='bias_linear',
#             shape=(1,),
#             initializer='uniform',
#             trainable=True
#         )

#         # Periodic components: (d_model - 1) sinusoids
#         self.weight_periodic = self.add_weight(
#             name='weight_periodic',
#             shape=(self.d_model - 1,),
#             initializer='uniform',
#             trainable=True
#         )
#         self.bias_periodic = self.add_weight(
#             name='bias_periodic',
#             shape=(self.d_model - 1,),
#             initializer='uniform',
#             trainable=True
#         )

#     def call(self, x):
#         """
#         x: input tensor of shape (batch_size, seq_len, num_features)
#         Output: tensor of shape (batch_size, seq_len, d_model)
#         """
#         # Take mean over features (e.g., mean of Open, High, Low, Close)
#         # Assumes Volume or noisy features can be excluded
#         x_mean = tf.reduce_mean(x[:, :, :4], axis=-1)  # shape: (batch, seq_len)

#         # Linear time feature
#         time_linear = self.weight_linear * x_mean + self.bias_linear  # shape: (batch, seq_len)
#         time_linear = tf.expand_dims(time_linear, axis=-1)  # shape: (batch, seq_len, 1)

#         # Periodic features
#         x_periodic = tf.expand_dims(x_mean, axis=-1)  # shape: (batch, seq_len, 1)
#         time_periodic = tf.math.sin(x_periodic * self.weight_periodic + self.bias_periodic)  # shape: (batch, seq_len, d_model-1)

#         # Concatenate linear and periodic features
#         time_embedding = tf.concat([time_linear, time_periodic], axis=-1)  # shape: (batch, seq_len, d_model)

#         return time_embedding

#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             'd_model': self.d_model
#         })
#         return config

import torch
import numpy as np

class Time2VecTorch(torch.nn.Module):
    """PyTorch implementation of Time2Vec embedding"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear = torch.nn.Linear(1, 1)
        self.periodic = torch.nn.Linear(1, d_model-1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x_mean = x[:, :, :4].mean(dim=-1, keepdim=True)  # Average OHLC features
        time_linear = self.linear(x_mean)
        time_periodic = torch.sin(self.periodic(x_mean))
        return torch.cat([time_linear, time_periodic], dim=-1)