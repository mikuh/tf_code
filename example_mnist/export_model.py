#coding=utf-8
from __future__ import print_function, unicode_literals
import tensorflow as tf
import os


tf.flags.DEFINE_string("last_checkpoint", "./runs/1505182128/checkpoints", "上次暂存点位置")
FLAGS = tf.flags.FLAGS


export_path = "mnist_models"
model_version = '1'
export_path = os.path.join(export_path, model_version)

print('Exporting trained model to', export_path)

sess = tf.InteractiveSession()

graph = tf.Graph()
with graph.as_default():
    sess = tf.InteractiveSession()
    with sess.as_default():
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.last_checkpoint)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        table = tf.contrib.lookup.index_to_string_table_from_tensor(
            tf.constant([str(i) for i in range(10)]))


        # 模型构建器
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # 计算图中部分节点
        x = graph.get_operation_by_name('inputs').outputs[0]
        y = graph.get_operation_by_name('y').outputs[0]

        tensor_info_x = tf.saved_model.utils.build_tensor_info(x)   # 输入
        tensor_info_y = tf.saved_model.utils.build_tensor_info(y)   # 前向传播结果

        classification_inputs = tf.saved_model.utils.build_tensor_info(x)
        classification_outputs_classes = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name('prediction_classes').outputs[0])   # 预测分类
        classification_outputs_scores = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name('classification_outputs_scores').outputs[0])  # 预测分类的得分

        # 分类签名
        classification_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    tf.saved_model.signature_constants.CLASSIFY_INPUTS: classification_inputs
                },
                outputs={
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                        classification_outputs_classes,
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                        classification_outputs_scores
                },
                method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

        # 预言签名
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_x},
                outputs={'scores': tensor_info_y},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],  # 通用标记
            signature_def_map={
                'predict_images':
                    prediction_signature,
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    classification_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()
        print("导出成功")