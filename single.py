"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data
from sklearn import metrics
from sklearn.model_selection import train_test_split
from memn2n import MemN2N
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import tensorboard as tb
import numpy as np

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 100, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size",5, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id",2, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "memn2n/data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS

print("Started Task:", FLAGS.task_id)

# task data
train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
data = train + test
#데이터로부터 사용된 모든 단어를 뽑아내 어순대로 정렬한다.
vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
#데이터로부터 뽑아낸 단어에 인덱스를 매긴다.
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

#story의 최대 길이는 10개
max_story_size = max(map(len, (s for s, _, _ in data)))
#story의 평균 사이즈는 6개
mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
#story의 문장 길이는 6개
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
#question의 문장 길이는 3개
query_size = max(map(len, (q for _, q, _ in data)))
#메모리의 크기는 story의 크기와 50 중에 작은 걸로 정해짐
#songhune edited: 메모리 자체의 크기를 작게 만듦
memory_size = min(FLAGS.memory_size, max_story_size)
print('Memory size')
# Add time words/indexes
for i in range(memory_size):
    word_idx['time{}'.format(i+1)] = 'time{}'.format(i+1)

vocab_size = len(word_idx) + 1 # +1 for nil word
sentence_size = max(query_size, sentence_size) # for the position
#sentence 길이에 time word라는 걸 하나 붙이므로(왜 하나 붙이나? 해결됨, timeword라고 표현할 수 있는 부분은 index 하나만 붙이면 되니까)
sentence_size += 1  # +1 for time words
print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size, max_story_size)
#songhune edited 맥스 스토리 사이즈를 넣는다
#trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1, random_state=FLAGS.random_state) #this model has been depricated

#train에서 실제로 트레이닝 할 데이터를 분리한다. ! 얼마나 트레이닝할때 쓸거냐면 90%를 사용할 것이다. 데이터도 셔플한다.
trainS, valS, trainQ, valQ, trainA, valA = train_test_split(S, Q, A, test_size=.1, random_state=FLAGS.random_state)
testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size, max_story_size)

'''
print("########################################################################")
for i in range (1000):
    print('the number of i would be',i,'\n', trainS[i])
    print()
    print(trainQ[i])
    print('###########################')

print("Training set shape", trainS.shape)
'''
print("Number of story size of this task",len(S))

# params, 즉, 전체 개수
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

#각각의 정답의 위치를 뱉어낸다.
train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
#여기까진 trainset에서의 라벨값
val_labels = np.argmax(valA, axis=1)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

with tf.Session() as sess:
    model = MemN2N(batch_size, vocab_size, sentence_size,memory_size, FLAGS.embedding_size, max_story_size, session=sess, hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm)
    for t in range(1, FLAGS.epochs+1):
        # Stepped learning rate
        if t - 1 <= FLAGS.anneal_stop_epoch:
            anneal = 2.0 ** ((t - 1) // FLAGS.anneal_rate)
        else:
            anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
        lr = FLAGS.learning_rate / anneal

        np.random.shuffle(batches)
        total_cost = 0.0
        for start, end in batches:
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            cost_t = model.batch_fit(s, q, a, lr)
            total_cost += cost_t

        if t % FLAGS.evaluation_interval == 0:
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s = trainS[start:end]
                q = trainQ[start:end]
                pred = model.predict(s, q)
                train_preds += list(pred)

            val_preds = model.predict(valS, valQ)
            train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
            val_acc = metrics.accuracy_score(val_preds, val_labels)
            summary_op = tf.summary.merge_all()

            file_writer = tf.summary.FileWriter('./logs',sess.graph)
            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Training Accuracy:', train_acc)
            print('Validation Accuracy:', val_acc)
            print('-----------------------')

    test_preds = model.predict(testS, testQ)
    test_acc = metrics.accuracy_score(test_preds, test_labels)
    print("Testing Accuracy:", test_acc)
