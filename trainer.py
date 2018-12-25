import os
import math
import torch
from torch.utils.data import DataLoader
from model import Dataset
from model import NegativeDataset
from model import collate_fn, build_pretrained_model
from model import RefScorer
from model import UnrefScorer

def train(args):
    print('[*] TRAIN')

    device = torch.device('cuda:'+str(args.device))
    ninput = args.dim
    nlayer = args.layer
    nbatch = args.batch
    nhidden= args.hidden
    margin = args.margin
    epoch  = args.epoch
    learningrate = args.lr
    dataset_path = args.data
    onmt_path = args.pretrain

    gold_reply_path = os.path.join(dataset_path, 'test_reply_sampled.txt')
    gene_reply_path = os.path.join(dataset_path, 'gene_reply_sampled.txt')
    train_src_path  = os.path.join(dataset_path, 'src_train.txt')
    train_tar_path  = os.path.join(dataset_path, 'tar_train.txt')
    test_query_path = os.path.join(dataset_path, 'test_query_sampled.txt')

    #refer_test_data_path_list  = [gene_reply_path, gold_reply_path]
    unref_train_data_path_list = [train_src_path, train_tar_path]
    #unref_test_data_path_list  = [test_query_path, gold_reply_path]
    vocab_path = os.path.join(onmt_path, 'sample.vocab.pt')

    #pre-trained model
    #onmt_vocab_path = os.path.join(onmt_path, 'sample.vocab.pt')
    #onmt_model_path = os.path.join(onmt_path, 'sample.model.pt')

    #data load
    unref_train_dataset = Dataset(vocab_path = vocab_path, data_path_list = unref_train_data_path_list, max_length = 50)
    unref_nega_dataset  = NegativeDataset(unref_train_dataset.data, unref_train_dataset.vocab)
    #print('positive ', unref_train_dataset[0], unref_train_dataset.data[0])
    #print('negative ', negative_dataset[0], negative_dataset.data[0]) 

    positive_loader = DataLoader(dataset = unref_train_dataset, batch_size = nbatch, collate_fn = collate_fn, num_workers = 8)
    negative_loader = DataLoader(dataset = unref_nega_dataset,  batch_size = nbatch, collate_fn = collate_fn, num_workers = 8)

    vocab_size= unref_train_dataset.getVocabSize()
    batch_num = math.ceil(unref_train_dataset.getInstanceSize() / nbatch )
    print('[*] # of batch: ', batch_num, ' pos, neg :', len(positive_loader), len(negative_loader))

    #ninput, nhidden, nlayer, ntoken, nbatch
    unrefer_pos_model = UnrefScorer(ninput, nhidden, nlayer, vocab_size, nbatch, device)
    unrefer_pos_model = unrefer_pos_model.to(device)

    unrefer_neg_model = UnrefScorer(ninput, nhidden, nlayer, vocab_size, nbatch, device)
    unrefer_neg_model = unrefer_neg_model.to(device)

    loss_f = torch.nn.MarginRankingLoss(margin)
    optimizer1 = torch.optim.Adam(unrefer_pos_model.parameters(), lr = learningrate)
    optimizer2 = torch.optim.Adam(unrefer_neg_model.parameters(), lr = learningrate)
    total_loss = 0

    for i in range(epoch): #epoch
        iter_positive = iter(positive_loader)
        iter_negative = iter(negative_loader)
        for mini in range(batch_num):
            #positive training
            pos_src, pos_src_len, pos_tar, pos_tar_len = next(iter_positive)
            neg_src, neg_src_len, neg_tar, neg_tar_len = next(iter_negative)
            #print('pos', pos_src, '/', pos_src_len, '/', pos_tar, '/', pos_tar_len)
         
            encd_pos = unrefer_pos_model(pos_src, pos_src_len, pos_tar, pos_tar_len)
            encd_neg = unrefer_neg_model(neg_src, neg_src_len, neg_tar, neg_tar_len)
            #print('next ', mini, encd_pos.size(), encd_neg.size())
        
            target = torch.ones(encd_pos.size(0), 1).to(device)#batch
        
            loss = loss_f(encd_pos, encd_neg, target)
            total_loss =total_loss+ loss.item()
            unrefer_pos_model.zero_grad()
            unrefer_neg_model.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

        print('[-] epoch: ', i, ', total_loss :', total_loss)
        total_loss = 0

    torch.save(unrefer_pos_model, args.output+'_pos.th')
    torch.save(unrefer_neg_model, args.output+'_neg.th')

def test(args):

    print('[*] TEST')

    unrefer_pos_model = torch.load(args.'_pos.th')
    unrefer_neg_model = torch.load('_neg.th')

    unref_test_dataset = Dataset(vocab_path = vocab_path,data_path_list = unref_test_data_path_list, max_length = 50)
    unref_test_loader  = DataLoader(dataset = unref_test_dataset, batch_size = nbatch, collate_fn = collate_fn, num_workers= 8)

    #pretrained = build_pretrained_model(onmt_path, unref_test_dataset.vocab)
    #ref_model = RefScorer(pretrained, device)
    #print('refer model load')

    #test
    positive_test = open('result/positive_result.txt', 'w', encoding = 'utf-8')
    negative_test = open('result/negative_result.txt', 'w', encoding = 'utf-8')

    for query, q_len, reply, r_len in unref_test_loader:
        prediction1 = unrefer_pos_model(query, q_len, reply, r_len)
        prediction2 = unrefer_neg_model(query, q_len, reply, r_len)
        #print(query,'/', q_len, '/', reply, '/', r_len)
        print(prediction1)
        print(prediction2)
        positive_test.write(str(prediction1.data))
        negative_test.write(str(prediction2.data))
        print('break') 
        break

    positive_test.close()
    negative_test.close()

'''
##reference score 
#data load
onmt_path = 'sample_onmt'
onmt_vocab_path = os.path.join(onmt_path, 'sample.vocab.pt')
onmt_model_path = os.path.join(onmt_path, 'sample.model.pt')
ref_test_dataset = Dataset(vocab_path = onmt_vocab_path, data_path_list = refer_test_data_path_list, max_length = 50)
ref_test_loader  = DataLoader(dataset = ref_test_dataset, batch_size = nbatch, collate_fn = collate_fn, num_workers= 16)

encoder = build_pretrained_model (onmt_model_path, ref_test_dataset.vocab)
print('reference score')
ref_scorer = RefScorer(encoder, device)
for gold_indices, gold_lens, gen_indices, gen_lens in ref_test_loader:
    similarity = ref_scorer.get_ref_score(gold_indices, gold_lens, gen_indices, gen_lens)
    #print(similarity.tolist())
'''
