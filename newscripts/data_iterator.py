import numpy
import json
import cPickle as pkl
import random
import numpy as np

class DataIterator:

    def __init__(self, source,
                 batch_size=128,
                 maxlen=50,
                 skip_empty=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 minlen=None,
                 parall=False
                ):
        self.source = open(source, 'r')
        self.source_dicts = []
        
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty
        self.maxsides = 30
        self.maxtags = 10


        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * 2
        
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        side_list = []
        tags_list = []
        segs_list = []
        hist_item_list = []
        hist_vmid_list = []
        hist_cate_list = []
        neg_item_list = []
        neg_vmid_list = []
        neg_cate_list = []
        
        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        try:

            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    print "get the end of data or index error"
                    break
                label = int(ss[0])
                uid = int(ss[1]) % 1000000
                side = map(int, ss[2].split(","))
                item = int(ss[3])
                vmid = int(ss[4])
                cate = int(ss[5])
                tags = map(int, ss[6].split(","))
                segs = map(int, ss[7].split(","))

                hist_item = map(int, ss[8].split(","))
                hist_vmid = map(int, ss[9].split(","))
                hist_cate = map(int, ss[10].split(","))

                neg_item = map(int, ss[11].split(","))
                neg_vmid = map(int, ss[12].split(","))
                neg_cate = map(int, ss[13].split(","))
                
                source.append([uid, item, vmid, cate])
                target.append([label, 1-label])

                side_list.append(side[:self.maxsides])
                tags_list.append((tags + segs)[:self.maxtags])

                hist_item_list.append(hist_item[-self.maxlen:])
                hist_vmid_list.append(hist_vmid[-self.maxlen:])
                hist_cate_list.append(hist_cate[-self.maxlen:])
                
                neg_item_list.append(neg_item[-self.maxlen:])
                neg_vmid_list.append(neg_vmid[-self.maxlen:])
                neg_cate_list.append(neg_cate[-self.maxlen:])
                
                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            print "io error"
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()
        
        uid_array = np.array(source)[:,0]
        item_array = np.array(source)[:,1]
        vmid_array = np.array(source)[:,2]
        cate_array = np.array(source)[:,3]
        
        side_array = np.array(side_list)
        tags_array = np.array(tags_list)
        side_mask_array = np.greater(side_array, 0) * 1.0
        tags_mask_array = np.greater(tags_array, 0) * 1.0

        target_array = np.array(target)

        history_item_array = np.array(hist_item_list)        
        history_vmid_array = np.array(hist_vmid_list)        
        history_cate_array = np.array(hist_cate_list)
        
        history_neg_item_array = np.array(neg_item_list)        
        history_neg_vmid_array = np.array(neg_vmid_list)        
        history_neg_cate_array = np.array(neg_cate_list)        
        
        history_mask_array = np.greater(history_item_array, 0)*1.0      

        return target_array, uid_array, item_array, vmid_array, cate_array, side_array, tags_array, side_mask_array, tags_mask_array, history_item_array, history_vmid_array, history_cate_array, history_neg_item_array, history_neg_vmid_array, history_neg_cate_array, history_mask_array


