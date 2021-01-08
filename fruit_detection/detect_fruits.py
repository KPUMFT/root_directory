import tensorflow as tf
from utils import constants
with open(constants.root_dir + ’\\ utils \\ labels ’) as f
    :
    labels = f.readlines()
labels = [x.strip() for x in labels]
labels = ["nothing"] + labels

tf.app.flags.DEFINE_string(’image_path ’, ’images \\
    Lemon2.jpg ’, ’Path to image ’)
FLAGS = tf.app.flags.FLAGS


def read_image(image_path , image_reader):
    filename_queue = tf.train.string_input_producer([
        image_path])
    _, image_file = image_reader.read(filename_queue)
    local_image = tf.image.decode_jpeg(image_file)
    local_image = tf.image.convert_image_dtype(
        local_image , tf.float32)
    gray_image = tf.image.rgb_to_grayscale(local_image
        )
    local_image = tf.image.rgb_to_hsv(local_image)
    shape = tf.shape(local_image)
    local_height = shape[0]
    local_width = shape[1]
    local_depth = shape[2]
    local_image = tf.reshape(local_image , [
        local_height , local_width , local_depth])
    final_image = tf.concat([local_image , gray_image],
        2)
    return final_image , local_height , local_width ,
        local_depth + 1


def predict(sess , X, softmax , images):
    images = sess.run(images)

    probability = sess.run(softmax , feed_dict={X:
        images})

    prediction = sess.run(tf.argmax(probability , 1))
    return prediction , probability[0][prediction]


def process_image(sess , X, softmax , image ,
    image_height , image_width , image_depth):
    image_depth = sess.run(image_depth)
    image_height = sess.run(image_height)
    image_width = sess.run(image_width)

    img = tf.image.resize_images(tf.reshape(image ,
        [-1, image_height , image_width , image_depth]),
        [100 , 100])
    img = tf.reshape(img, [-1, 100 * 100 * 4])
    rez, prob = predict(sess , X, softmax , img)
    print(’Label index: %d - Label: %s - Probability :
        %.4f’ % (rez, labels[rez[0]] , prob))


with tf.Session() as sess:
    image_path = FLAGS.image_path
    image_reader = tf.WholeFileReader()

    saver = tf.train.import_meta_graph(constants.
        fruit_models_dir + ’model.ckpt.meta ’)

    saver.restore(sess , tf.train.latest_checkpoint(
        constants.fruit_models_dir))
    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name(’X:0’)

    softmax = tf.nn.softmax(graph.get_tensor_by_name(’
        out/out :0’))

    image , height , width , depth = read_image(
        image_path , image_reader)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess ,
        coord=coord)
    process_image(sess , X, softmax , image , height ,
        width , depth)

    coord.request_stop()
    coord.join(threads)
    sess.close()
