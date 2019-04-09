#coding:utf-8
import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
from utils import *
import multiprocessing
import argparse
import cPickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--model_type', type=str, default='none', help='DIEN | MIMN | ..')
parser.add_argument('--memory_size', type=int, default=4)
parser.add_argument('--mem_induction', type=int, default=0, help='0:false|1:true')
parser.add_argument('--util_reg', type=int, default=0, help='0:false|1:true')


def generator_queue(generator, max_q_size=20,
                    wait_time=0.1, nb_worker=1):
    generator_threads = []
    q = multiprocessing.Queue(maxsize=max_q_size)
    _stop = multiprocessing.Event()
    try:
        def data_generator_task():
            while not _stop.is_set():
                try:
                    if q.qsize() < max_q_size:
                        generator_output = next(generator)
                        q.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception, e:
                    print e, "stop or gqt error"
                    _stop.set()

        for i in range(nb_worker):
            thread = multiprocessing.Process(target=data_generator_task)
            generator_threads.append(thread)
            thread.daemon = True
            thread.start()
    except Exception, e:
        print e, "gq error"
        _stop.set()
        for p in generator_threads:
            if p.is_alive():
                p.terminate()
        q.close()

    return q, _stop, generator_threads

EMBEDDING_DIM = 16
HIDDEN_SIZE = 16 * 2 
best_auc = 0.0

class Example(object):
    def __init__(self, inps):
        self.label = inps[0]
        self.nick = inps[1]
        self.item = inps[2]
        self.vmid = inps[3]
        self.cate = inps[4]
        self.side = inps[5]
        self.tags = inps[6]
        self.side_mask = inps[7]
        self.tags_mask = inps[8]
        self.hist_item = inps[9]
        self.hist_vmid = inps[10]
        self.hist_cate = inps[11]
        self.neg_item = inps[12]
        self.neg_vmid = inps[13]
        self.neg_cate = inps[14]
        self.hist_mask = inps[15]
        self.size = len(self.nick)
    
def eval(sess, test_data, model, model_path, batch_size):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    test_data_pool, _stop, _ = generator_queue(test_data)
    while True:
        if  _stop.is_set() and test_data_pool.empty():
            break
        src = test_data_pool.get()
        ex = Example(src)

        if ex.size < batch_size:
            continue
        nums += 1
        target = ex.label
        prob, loss, acc, aux_loss = model.calculate(sess, ex)
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p ,t in zip(prob_1, target_1):
            stored_arr.append([p, t])
    
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum

def train(
        train_file = "./data/weixin_data/weixin_train.txt",
        test_file = "./data/weixin_data/weixin_test.txt",
        feature_file = "./data/weixin_data/weixin_feature.pkl",
        batch_size = 128,
        maxlen = 50,
        test_iter = 50,
        save_iter = 100,
        model_type = 'DNN',
        Memory_Size = 4,
        Mem_Induction = 0, 
        Util_Reg = 0
):
    if model_type != "MIMN" or model_type != "MIMN_with_aux":
        model_path = "dnn_save_path/weixin_ckpt_noshuff" + model_type
        best_model_path = "dnn_best_model/weixin_ckpt_noshuff" + model_type
    else:
        model_path = "dnn_save_path/weixin_ckpt_noshuff" + model_type+str(Memory_Size)+str(Mem_Induction)
        best_model_path = "dnn_best_model/weixin_ckpt_noshuff" + model_type+str(Memory_Size)+str(Mem_Induction)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2) 

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, batch_size, maxlen)
        test_data = DataIterator(test_file, batch_size, maxlen)    
        
        feature_num = 1000000 #pkl.load(open(feature_file))
        n_uid, n_mid = feature_num, feature_num
        BATCH_SIZE = batch_size
        SEQ_LEN = maxlen

        if model_type == 'DNN': 
            model = Model_DNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'PNN': 
            model = Model_PNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'GRU4REC': 
            model = Model_GRU4REC(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'DIN': 
            model = Model_DIN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'ARNN': 
            model = Model_ARNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'RUM' :
            model = Model_RUM(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, Memory_Size, SEQ_LEN)
        elif model_type == 'DIEN': 
            model = Model_DIEN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'DIEN_with_aux': 
            model = Model_DIEN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, use_negsample=True)
        elif model_type == 'MIMN':
            model = Model_MIMN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, Memory_Size, SEQ_LEN, Mem_Induction, Util_Reg) 
        elif model_type == 'MIMN_with_aux':
            model = Model_MIMN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, Memory_Size, SEQ_LEN, Mem_Induction, Util_Reg, use_negsample=True) 
        else:
            print ("Invalid model_type : %s", model_type)
            return
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sys.stdout.flush()
        print('training begin')
        sys.stdout.flush()

        start_time = time.time()
        iter = 0
        lr = 0.001
        for itr in range(1):
            print("epoch"+str(itr))
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            train_data_pool,_stop,_ = generator_queue(train_data)
            while True:
                if  _stop.is_set() and train_data_pool.empty():
                    break
                src = train_data_pool.get()
                ex = Example(src)
                if ex.size < BATCH_SIZE:
                    continue
                loss, acc, aux_loss = model.train(sess, ex, lr)
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                iter += 1
                sys.stdout.flush()
#                if iter < 800:
#                    continue
                if (iter % test_iter) == 0:
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f' % \
                                          (iter, loss_sum / test_iter, accuracy_sum / test_iter,  aux_loss_sum / test_iter))
                    print('                                                                                          test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % eval(sess, test_data, model, best_model_path, batch_size))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                    test_time = time.time()
                    print("test interval: "+str((test_time-start_time)/60.0)+" min")
                if (iter % save_iter) == 0:
                    print('save model iter: %d' %(iter))
                    model.save(sess, model_path+"--"+str(iter))

def test(
        train_file = "./data/weixin_data/weixin_train.txt",
        test_file = "./data/weixin_data/weixin_test.txt",
        feature_file = "./data/weixin_data/weixin_feature.pkl",
        batch_size = 128,
        maxlen = 100,
        test_iter = 100,
        save_iter = 100,
        model_type = 'DNN',
        Memory_Size = 4,
        Mem_Induction = 0, 
        Util_Reg = 0
):
    

    if model_type != "MIMN" or model_type != "MIMN_with_aux":
        model_path = "dnn_best_model/weixin_ckpt_noshuff" + model_type
    else:
        model_path = "dnn_best_model/weixin_ckpt_noshuff" + model_type+str(Memory_Size)+str(Mem_Induction)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        test_data = DataIterator(test_file, batch_size, maxlen)

        feature_num = pkl.load(open(feature_file))
        n_uid, n_mid = feature_num, feature_num
        BATCH_SIZE = batch_size
        SEQ_LEN = maxlen


        if model_type == 'DNN': 
            model = Model_DNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'PNN': 
            model = Model_PNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'GRU4REC': 
            model = Model_GRU4REC(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'DIN': 
            model = Model_DIN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'ARNN': 
            model = Model_ARNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'RUM' :
            model = Model_RUM(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, Memory_Size, SEQ_LEN)
        elif model_type == 'DIEN': 
            model = Model_DIEN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'DIEN_with_aux': 
            model = Model_DIEN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, use_negsample=True)
        elif model_type == 'MIMN':
            model = Model_MIMN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, Memory_Size, SEQ_LEN, Mem_Induction, Util_Reg) 
        elif model_type == 'MIMN_with_aux':
            model = Model_MIMN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, Memory_Size, SEQ_LEN, Mem_Induction, Util_Reg, use_negsample=True) 
        else:
            print ("Invalid model_type : %s", model_type)
            return

        model.restore(sess, model_path)
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % eval(sess, test_data, model, model_path, batch_size))

if __name__ == '__main__':
    print sys.argv
    args = parser.parse_args()
    SEED = args.random_seed
    Model_Type = args.model_type
    Memory_Size = args.memory_size
    Mem_Induction = args.mem_induction
    Util_Reg = args.util_reg

    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)

    if args.p == 'train':
        train(model_type=Model_Type, Memory_Size=Memory_Size, Mem_Induction=Mem_Induction, Util_Reg=Util_Reg)
    elif args.p == 'test':
        test(model_type=Model_Type, Memory_Size=Memory_Size, Mem_Induction=Mem_Induction, Util_Reg=Util_Reg)
    else:
        print('do nothing...')
