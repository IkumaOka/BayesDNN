import warnings
warnings.simplefilter('ignore')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from distribution import *
from normalizing_flow import *

normal_distribution = NormalDistribution2D()
target_density = TargetDistribution1()

normalizing_flow = NormalizingFlow(K=16, dim=2)

z_0, log_q_0 = get_placeholder()
# print("z_0:", z_0)
# z_0: Tensor("Placeholder:0", shape=(?, 2), dtype=float32)
z_k, log_q_k = normalizing_flow(z_0, log_q_0)
loss = calc_loss(z_k, log_q_k, target_density)
train = get_train(loss)

with tf.Session() as sess:
    invisible_axis = True
    sess.run(tf.global_variables_initializer())

    for iteration in range(10):
        z_0_batch = normal_distribution.sample(1000)
        # if iteration == 0:
            # print(z_0_batch.shape)
        log_q_0_batch = np.log(normal_distribution.calc_prob(z_0_batch))
        _, loss_value = sess.run([train, loss], {z_0:z_0_batch, log_q_0:log_q_0_batch})

        if iteration % 100 == 0:
            print('Iteration : {}   Loss : {}'.format(iteration, loss_value))

        if iteration % 10000 == 0:
            z_k_value = sess.run(z_k, {z_0:z_0_batch, log_q_0:log_q_0_batch})
            plt.figure(figsize=(6, 6))
            plt.scatter(z_k_value[:, 0], z_k_value[:, 1], alpha=0.7)
            if invisible_axis:
                plt.tick_params(bottom=False,left=False,right=False,top=False)
                plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
            plt.show()