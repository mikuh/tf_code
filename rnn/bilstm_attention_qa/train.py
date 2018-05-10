import tensorflow as tf
import qa_attention_model
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



# 数据加载参数
tf.flags.DEFINE_string("train_data_file", "../data/corpus_with_id_long.data", "数据所在文件")
tf.flags.DEFINE_string("valid_data_file", "../data/test.data", "验证数据文件")

# 模型超参数
tf.flags.DEFINE_integer("embedding_size", 256, "词嵌入（embedding）的维度 (default: 128)")
tf.flags.DEFINE_integer("lstm_size", 500, "lstm输出维度（default:500）")
tf.flags.DEFINE_integer("query_steps", 100, "query句子词汇数")
tf.flags.DEFINE_integer("answer_steps", 300, "answer句子词汇数")

# 训练参数
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout的概率 (default: 0.5)")
tf.flags.DEFINE_integer("batch_size", 64, "批次大小 (default: 64)")
tf.flags.DEFINE_integer("evaluate_every", 2000, "通过这么多步骤评估开发者设置的模型 (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "这么多步后存储模型 (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "保存检查点数量 (default: 5)")
tf.flags.DEFINE_integer("max_steps", 1000000, "最大训练步数 (default: 1000000)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2正则化lambda (default: 0.0)")
tf.flags.DEFINE_float("lr", 0.01, "学习效率(default: 0.1)")
tf.flags.DEFINE_float("max_grad_norm", 10, "最大梯度")

# 配置参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/checkpoints", "Checkpoint directory from training run")

# 打印参数
FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), str(value)))
print("")


model = qa_attention_model.Attention_LSTM(batch_size=FLAGS.batch_size, query_steps=FLAGS.query_steps, train_data_file=FLAGS.train_data_file, valid_data_file=FLAGS.valid_data_file)
model.train()
