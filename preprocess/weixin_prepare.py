import random
import numpy as np
import cPickle as pkl


Train_handle = open("./data/weixin_data/weixin_train.txt",'w')
Test_handle = open("./data/weixin_data/weixin_test.txt",'w')
Feature_handle = open("./data/weixin_data/weixin_feature.pkl",'w')
max_len = 50
def produce_neg_item_hist_with_cate(train_file, test_file):
    item_dict = {}
    sample_count = 0
    hist_seq = 0
    for line in train_file:
        units = line.strip().split("\t")
        item_hist_list = units[8].split(",")
        vmid_hist_list = units[9].split(",")
        cate_hist_list = units[10].split(",")
        hist_list = zip(item_hist_list, vmid_hist_list, cate_hist_list)
        hist_seq = len(hist_list)
        sample_count += 1
        for item in hist_list:
            item_dict.setdefault(str(item),0)
            
    for line in test_file:
        units = line.strip().split("\t")
        item_hist_list = units[8].split(",")
        vmid_hist_list = units[9].split(",")
        cate_hist_list = units[10].split(",")
        hist_list = zip(item_hist_list, vmid_hist_list, cate_hist_list)
        hist_seq = len(hist_list)
        sample_count += 1
        for item in hist_list:
            item_dict.setdefault(str(item),0)
            
    #print item_dict.keys()[:10]   
    del(item_dict["('0', '0', '0')"])
    neg_array = np.random.choice(np.array(item_dict.keys()), (sample_count, max_len))
    neg_list = neg_array.tolist()
    sample_count = 0
    for line in train_file:
        units = line.strip().split("\t")
        item_hist_list = units[8].split(",")
        vmid_hist_list = units[9].split(",")
        cate_hist_list = units[10].split(",")
        hist_list = zip(item_hist_list, vmid_hist_list, cate_hist_list)
        hist_seq = len(hist_list)
        neg_hist_list = []
        for item in neg_list[sample_count]:
            item = eval(item)
            if item not in hist_list:
                neg_hist_list.append(item)
            if len(neg_hist_list) == hist_seq:
                break
        sample_count += 1
        neg_item_list, neg_vmid_list, neg_cate_list = zip(*neg_hist_list)
        Train_handle.write(line.strip() + "\t" + ",".join(neg_item_list) + "\t" + ",".join(neg_vmid_list) + "\t" + ",".join(neg_cate_list) + "\n" )
        
    for line in test_file:
        units = line.strip().split("\t")
        item_hist_list = units[8].split(",")
        vmid_hist_list = units[9].split(",")
        cate_hist_list = units[10].split(",")
        hist_list = zip(item_hist_list, vmid_hist_list, cate_hist_list)
        hist_seq = len(hist_list)
        neg_hist_list = []
        for item in neg_list[sample_count]:
            item = eval(item)
            if item not in hist_list:
                neg_hist_list.append(item)
            if len(neg_hist_list) == hist_seq:
                break
        sample_count += 1
        neg_item_list, neg_vmid_list, neg_cate_list = zip(*neg_hist_list)
        Test_handle.write(line.strip() + "\t" + ",".join(neg_item_list) + "\t" + ",".join(neg_vmid_list) + "\t" + ",".join(neg_cate_list) + "\n" )



def generate_sample_list():
    train_sample_list = []
    test_sample_list = []
    for line in file("./data/weixin_data/local_train.txt"):
        units = line.strip().split("\t")
        item_hist_list = units[8].split(",")
        vmid_hist_list = units[9].split(",")
        cate_hist_list = units[10].split(",")
        hist_list = zip(item_hist_list, vmid_hist_list, cate_hist_list)
        hist_seq = len(hist_list)
        if hist_seq > max_len:
            hist_list = hist_list[-max_len:]
        else:
            hist_list = [('0','0','0')]*(max_len-hist_seq) + hist_list
        item_list, vmid_list, cate_list = zip(*hist_list)
        units[8] = ','.join(item_list)
        units[9] = ','.join(vmid_list)
        units[10] = ','.join(cate_list)
        train_sample_list.append('\t'.join(units))
    for line in file("./data/weixin_data/local_test.txt"):
        units = line.strip().split("\t")
        item_hist_list = units[8].split(",")
        vmid_hist_list = units[9].split(",")
        cate_hist_list = units[10].split(",")
        hist_list = zip(item_hist_list, vmid_hist_list, cate_hist_list)
        hist_seq = len(hist_list)
        if hist_seq > max_len:
            hist_list = hist_list[-max_len:]
        else:
            hist_list = [('0','0','0')]*(max_len-hist_seq) + hist_list
        item_list, vmid_list, cate_list = zip(*hist_list)
        units[8] = ','.join(item_list)
        units[9] = ','.join(vmid_list)
        units[10] = ','.join(cate_list)
        test_sample_list.append('\t'.join(units))

    random.shuffle(train_sample_list)
    return train_sample_list, test_sample_list


if __name__ == "__main__":

    train_sample_list, test_sample_list = generate_sample_list()
    produce_neg_item_hist_with_cate(train_sample_list, test_sample_list)

