import CapsuleNetwork
import numpy as np
from MNIST import content
import color
import time
import datetime
import dataAugmentation as da
import trainingEnvironment as te

run_timer = False
def timer():
    global run_timer
    run_timer = True
    start = time.time()
    while run_timer:
        print("\r"+color.MAGENTA+"Process Time: {}".format(str(datetime.timedelta(seconds=time.time()-start)))+color.RESET, end='')
    print()


training_batch = 64
testing_batch = 64
training_iterations = 2000

capsNet = CapsuleNetwork.MNISTCapsNet(content.SHAPE, content.NUM_CLASSES, 0, name="MNIST.1.8")
capsNet.class_cap_depth = 8
capsNet.build_network()
capsNet.load()
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

batch_x, batch_y = content.get_testing_batch(testing_batch, True)
imgs, y_pred = capsNet.imgs(batch_x, batch_y)
y_pred = np.argmax(y_pred, axis=1)
batch_y = np.argmax(batch_y, axis=1)

content.plot_images(batch_x[0:9], batch_y[0:9], y_pred[0:9], imgs[0:9])

batch_x, batch_y = content.get_testing_batch(testing_batch, True)
batch_x_1, batch_y_1 = content.get_testing_batch(testing_batch, True)
batch_x = np.clip(np.add(batch_x, batch_x_1),0.0,1.0)
imgs, y_pred = capsNet.imgs(batch_x, batch_y)
y_pred = np.argmax(y_pred, axis=1)
batch_y = np.argmax(batch_y, axis=1)
content.plot_images(batch_x[0:9], batch_y[0:9], y_pred[0:9], imgs[0:9])



"""
mask_indicies = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

for skew_index in range(capsNet.class_cap_depth):
    print(skew_index)
    batch_x, batch_y = content.get_testing_batch(1, True)
    batch_x = np.tile(batch_x, [9,1,1,1])
    batch_y = np.tile(batch_y, [9,1])

    skew = np.arange(-.20,.25,.05)
    skew_matrix = da.build_skew_matrix(batch_y, skew, skew_index, capsNet.class_cap_depth)
    #mask_matrix = build_mask_matrix(batch_y, mask_indicies, capsuleNetwork.capsule_depth_digit)

    imgs, y_pred = capsNet.imgs(batch_x, batch_y, skew=skew_matrix)#, mask=mask_matrix)
    y_pred_arg = np.argmax(y_pred, axis=1)
    batch_y_arg = np.argmax(batch_y, axis=1)
    content.plot_images(batch_x[0:9], batch_y_arg[0:9], y_pred_arg[0:9], imgs[0:9])

    skew = np.arange(-.4,.5,.1)
    skew_matrix = da.build_skew_matrix(batch_y, skew, skew_index, capsNet.class_cap_depth)


    imgs, y_pred = capsNet.imgs(batch_x, batch_y, skew=skew_matrix)#, mask=mask_matrix)
    y_pred_arg = np.argmax(y_pred, axis=1)
    batch_y_arg = np.argmax(batch_y, axis=1)
    content.plot_images(batch_x[0:9], batch_y_arg[0:9], y_pred_arg[0:9], imgs[0:9])
#"""

te.train(capsNet, training_iterations, training_batch, testing_batch, 100, 100, content)



batch_x, batch_y = content.get_testing_batch(testing_batch, True)
imgs, y_pred = capsNet.imgs(batch_x, batch_y)
y_pred = np.argmax(y_pred, axis=1)
batch_y = np.argmax(batch_y, axis=1)

content.plot_images(batch_x[0:9], batch_y[0:9], y_pred[0:9], imgs[0:9])

batch_x, batch_y = content.get_testing_batch(testing_batch, True)
batch_x_1, batch_y_1 = content.get_testing_batch(testing_batch, True)
batch_x = np.clip(np.add(batch_x, batch_x_1),0.0,1.0)
imgs, y_pred = capsNet.imgs(batch_x, batch_y)
y_pred = np.argmax(y_pred, axis=1)
batch_y = np.argmax(batch_y, axis=1)

content.plot_images(batch_x[0:9], batch_y[0:9], y_pred[0:9], imgs[0:9])