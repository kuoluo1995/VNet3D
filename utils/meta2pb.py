import tensorflow as tf
from tqdm import tqdm

tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("./temp/checkpoints/checkpoint-154203.meta")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    print("Restoring checkpoint...")
    imported_meta.restore(sess, tf.train.latest_checkpoint("./temp/checkpoints", latest_filename="checkpoint-latest"))
    print("Restore checkpoint complete")

    print("Assign tensor to variables")
    for variable in tqdm(tf.trainable_variables()):
        tensor = tf.constant(variable.eval())
        tf.assign(variable, tensor, name="nWeights")

    print("Writing graph...")
    tf.train.write_graph(sess.graph.as_graph_def(), "./temp/", "graph.pb", as_text=False)
