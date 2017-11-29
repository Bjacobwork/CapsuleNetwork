import capsule as c
import tensorflow as tf
import tensorflow_tools as tft
import numpy as np

class MNISTCapsNet:

    def __init__(self, input_shape, num_classes, noise_channels, name="default_CapsNet"):
        """
        Purpose:
            To create a capsule network for the given shape.
        Precondition:
            :param input_shape: The shape of the input data. This does NOT include batch size!
            :param num_classes: Number of classes the network can map to.
            :param noise_channels: Number of capsules allowed at any level to map network noise to.
                These will be clipped when predicting the output class, but will be used in the reconstruction of the data.
            :param name: The name of the network.
        Postcondition:
            The hyper-parameters are initilized.
        """
        self.name = name
        self.total_iterations = 0
        self.total_time = 0
        # Hyper-parameters of the network
        self.input_shape = input_shape
        self.noise_channels = noise_channels
        self.num_classes = num_classes

        self.filter_size_0 = 9
        self.num_filters_0 = 256
        self.stride_0 = 1
        self.padding_0 = 'VALID'

        self.filter_size_primary_1 = 9
        self.num_convs_1 = 32
        self.capsule_depth_1 = 8
        self.stride_primary_1 = 2

        self.num_routes_1 = 3
        self.class_cap_depth = 16

        self.fc_size_1 = 512
        self.fc_size_2 = 1024
        self.fc_size_3 = self.fc_size_3 = np.prod(input_shape)

        self.upper_bounds = 0.9
        self.lower_bounds = 0.1
        self.down_weighting = 0.5
        # Degree of the reconstruction loss to use
        self.reconstruction_alpha = 0.0005

    def build_network(self):
        """
        Purpose:
            To initilize the network in tensorflow.
        Precondition:
            The hyper-parameters are initilized correctly.
        Postcondition:
            The network is initilized.
        :return:
        """
        self.x = tf.placeholder(tf.float32, shape=(None,)+self.input_shape, name=self.name+".x")

        self.conv_0 = tft.build_conv_lambda(input=self.x,
                                            num_input_channels=self.input_shape[-1],
                                            filter_size=self.filter_size_0,
                                            num_filters=self.num_filters_0,
                                            strides=[1,self.stride_0, self.stride_0, 1],
                                            use_pooling=False,
                                            name=self.name+".conv_1",
                                            padding=self.padding_0)
        self.conv_0 = tf.nn.relu(self.conv_0)
        self.primary_1 = c.build_conv_capsuels(self.conv_0, self.num_convs_1, self.capsule_depth_1, self.filter_size_primary_1, self.stride_primary_1, name=self.name+".primaryConvs.1")
        primary_flat_1 = c.flatten_capsules(self.primary_1)

        self.classification_caps = c.build_flat_capsule_layer(primary_flat_1, self.num_routes_1, self.num_classes+self.noise_channels, self.class_cap_depth, c.route_with_softmax)
        self.y_pred = tf.reduce_sum(tf.square(tf.slice(self.classification_caps, [0,0,0], [-1,self.num_classes,self.class_cap_depth])), axis=2)

        # Mask and Skew
        self.skew_matrix = tf.placeholder(tf.float32, shape=[None, self.num_classes, self.class_cap_depth])
        self.mask_matrix = tf.placeholder(tf.float32, shape=[None, self.num_classes, self.class_cap_depth])

        noise_skew = tf.zeros(shape=[self.noise_channels, self.class_cap_depth])
        noise_mask = tf.ones(shape=[self.noise_channels, self.class_cap_depth])

        skew = tf.map_fn(lambda x: tf.concat([x, noise_skew], axis=0), self.skew_matrix)
        mask = tf.map_fn(lambda x: tf.concat([x, noise_mask], axis=0),self.mask_matrix)

        self.adjusted_caps = tf.multiply(tf.add(self.classification_caps, skew), mask)


        # Reconstruction

        flat_vec_out, flat_vec_features = tft.flatten_layer(self.adjusted_caps)
        self.fc_1, fc_1_w, fc_1_b = tft.build_fc(input=flat_vec_out,
                                                 num_inputs=flat_vec_features,
                                                 num_outputs=self.fc_size_1,
                                                 name=self.name + ".fc_1")
        self.fc_1 = tf.nn.relu(self.fc_1)
        self.fc_2, fc_2_w, fc_2_b = tft.build_fc(input=self.fc_1,
                                                 num_inputs=self.fc_size_1,
                                                 num_outputs=self.fc_size_2,
                                                 name=self.name + ".fc_2")
        self.fc_2 = tf.nn.relu(self.fc_2)
        self.fc_3, fc_3_w, fc_3_b = tft.build_fc(input=self.fc_2,
                                                 num_inputs=self.fc_size_2,
                                                 num_outputs=self.fc_size_3,
                                                 name=self.name + ".fc_3")
        self.fc_3 = tf.nn.relu(self.fc_3)
        self.reconstructed = tf.reshape(self.fc_3, shape=(-1,) + self.input_shape)

        self.reconstructed_loss = self.reconstruction_loss(self.x, self.reconstructed)

        self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name=self.name + "y_true")

        self.loss = tf.add(self.classification_loss(self.y, self.y_pred), tf.multiply(self.reconstruction_alpha, self.reconstructed_loss))

        self.correct_pred = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.y_pred, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def classification_loss(self, true, pred):
        squared_max_upper = tf.square(tf.maximum(0.0, tf.subtract(self.upper_bounds, pred)))
        squared_max_lower = tf.square(tf.maximum(0.0, tf.subtract(pred, self.lower_bounds)))
        loss = tf.add(tf.multiply(true, squared_max_upper),
                      tf.multiply(self.down_weighting, tf.multiply(tf.subtract(1.0, true), squared_max_lower)))
        loss = tf.reduce_mean(loss)
        return loss

    def reconstruction_loss(self, original, reconstructed):
        error = tf.subtract(original, reconstructed)
        error, error_features = tft.flatten_layer(error)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(error), axis=1))
        return loss

    def train(self, batch_x, batch_y, skew=None, mask=None):
        if skew is None:
            skew = self.default_skew(len(batch_x))
        if mask is None:
            mask = self.default_mask(batch_y)
        accuracy, _ = self.session.run([self.accuracy, self.optimizer], feed_dict={self.x: batch_x,
                                                                                   self.y: batch_y,
                                                                                   self.skew_matrix: skew,
                                                                                   self.mask_matrix: mask})
        return accuracy

    def metrics(self, batch_x, batch_y, skew=None, mask=None):
        if skew is None:
            skew = self.default_skew(len(batch_x))
        if mask is None:
            mask = self.default_mask(batch_y)
        y_pred, accuracy = self.session.run([self.y_pred, self.accuracy], feed_dict={self.x: batch_x,
                                                                                     self.y: batch_y,
                                                                                     self.skew_matrix: skew,
                                                                                     self.mask_matrix: mask})
        return y_pred, accuracy

    def imgs(self, batch_x, batch_y, skew=None, mask=None):
        if skew is None:
            skew = self.default_skew(len(batch_x))
        if mask is None:
            mask = self.default_mask(batch_y)
        imgs, y_pred = self.session.run([self.reconstructed, self.y_pred], feed_dict={self.x: batch_x,
                                                                                      self.y: batch_y,
                                                                                      self.skew_matrix: skew,
                                                                                      self.mask_matrix: mask})
        return imgs, y_pred

    def default_skew(self, len_batch):
        return np.zeros(shape=[len_batch, self.num_classes, self.class_cap_depth], dtype=np.float32)

    def default_mask(self, one_hot):
        return np.tile(np.expand_dims(one_hot, axis=2), [1, 1, self.class_cap_depth])

    def save(self):
        import os
        import pickle as rick
        if not os.path.isdir('./models/{}'.format(self.name)):
            os.makedirs('./models/{}'.format(self.name))
        self.saver.save(self.session, './models/{}/{}.ckpt'.format(self.name, self.name))
        with open('./models/{}/{}.pkl'.format(self.name, self.name), 'wb') as file:
            package = [self.total_iterations, self.total_time]
            rick.dump(package, file)

    def load(self):
        import os
        import pickle as rick
        if os.path.isdir('./models/{}'.format(self.name)):
            self.saver.restore(self.session, './models/{}/{}.ckpt'.format(self.name, self.name))
            with open('./models/{}/{}.pkl'.format(self.name, self.name), 'rb') as file:
                package = rick.load(file)
                try:
                    self.total_iterations = package[0]
                    self.total_time = package[1]
                except:
                    self.total_iterations = package