from unicodedata import bidirectional
from torch import nn
from fastNLP.embeddings import LSTMCharEmbedding, StackEmbedding, StaticEmbedding, BertEmbedding, ElmoEmbedding, BertWordPieceEncoder
from fastNLP.modules import LSTM, MLP
import torch
from modules.rnn import Encoder, Seq2SeqDecoder, LSTMEncoder, LSTMDecoder, Seq2SeqModel
from modules.mlp import MLPAdapter
# from utils.matrix_utils import flat2matrix, matrix2flat
import torch.nn.functional as F
from models.GL_model import GL_Model
from modules.graph import AGIFDecoder



class MainModel(nn.Module):
    def __init__(self,dataset,args):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.vocab = dataset.vocab['word']
        if self.args.embed_type == 'bert':
            self.embedding_dim = 768 #用bert写死768
            self.embedding = BertEmbedding(self.vocab,model_dir_or_name=args.bert_path)
        elif self.args.embed_type == 'w2v':
            self.embedding = StaticEmbedding(self.vocab,'en-word2vec-300d')
            self.embedding_dim = 300
        elif self.args.embed_type == 'glove':
            self.embedding = StaticEmbedding(self.vocab,'en-glove-6b-300d')
            self.embedding_dim = 300
        else:
            self.embedding = StaticEmbedding(self.vocab,model_dir_or_name=None,embedding_dim=self.args.embedding_dim)
            self.embedding_dim = self.args.embedding_dim
        char_embed = LSTMCharEmbedding(self.vocab,
                                        embed_size=64,
                                        char_emb_size=50)
        self.embedding = StackEmbedding([self.embedding, char_embed])
        self.embedding_dim = self.embedding.embedding_dim
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4, dim_feedforward=2048, dropout=0.1, activation='relu')
        self.encoder_hidden_dim = args.encoder_hidden_dim
        self.attention_hidden_dim =args.attention_hidden_dim
        self.attention_output_dim = args.attention_output_dim
        self.dropout = nn.Dropout(args.drop_out)
        self.dropout_rate = args.drop_out
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.args.tf_layer_num)
        ##模型底部的Self-Attention Encoder
        self.encoder = Encoder(self.embedding_dim,
                               self.encoder_hidden_dim,
                               self.attention_hidden_dim,
                               self.attention_output_dim,
                               dropout_rate=self.dropout_rate)
        
        self.num_intent = len(dataset.vocab['intent'])
        self.num_slot = len(dataset.vocab['slot'])
        self.window_size = args.window_size
        self.is_hard = args.use_hard_vote
        self.decoder_gat_hidden_dim = args.decoder_gat_hidden_dim
        self.slot_graph_window=args.slot_graph_window
        self.n_head = args.n_head
        self.slot_decoder_hidden_dim = args.slot_decoder_hidden_dim


        self.__intent_decoder = Window_Attention_Intent_Decoder(self.args,self.window_size,self.encoder_hidden_dim+self.attention_output_dim,self.num_intent)

        self.decoder = self.args.decoder

        self.__slot_lstm = LSTMEncoder(
            self.encoder_hidden_dim + self.attention_output_dim,
            self.slot_decoder_hidden_dim,
            self.dropout_rate
        )

        self.decoder_hidden_dim = self.attention_output_dim + self.encoder_hidden_dim

        if self.decoder == 'mlp':
            self.__slot_decoder = MLPAdapter('qin', self.decoder_hidden_dim,
                                             self.decoder_hidden_dim)
        elif self.decoder == 'seq2seq':
            self.__slot_decoder = Seq2SeqDecoder(self.decoder_hidden_dim,
                                                 self.decoder_hidden_dim,
                                                 self.num_slot,
                                                 dropout_rate=self.dropout_rate,
                                                 embedding_dim=32,
                                                 extra_dim=self.num_intent)
        elif self.decoder == 'lstm':
            self.__slot_decoder = LSTMDecoder(self.decoder_hidden_dim,
                                              self.decoder_hidden_dim,
                                              dropout_rate=self.dropout_rate)
        elif self.decoder == 'gat':
            self.__slot_decoder = GL_Model(
                self.slot_decoder_hidden_dim,
                self.num_slot,
                self.num_intent,
                n_head=self.n_head,
                decoder_gat_hidden_dim=self.decoder_gat_hidden_dim,
                slot_graph_window=self.slot_graph_window,
                use_normalized=True)

        elif self.decoder == 'agif':
            self.__slot_decoder = AGIFDecoder(
                # self.decoder_hidden_dim,
                self.slot_decoder_hidden_dim,
                self.slot_decoder_hidden_dim,
                self.num_slot,
                self.num_intent,
                self.decoder_gat_hidden_dim,
                self.dropout_rate,
                n_heads=self.n_head,
                row_normalized=True,
                embedding_dim=128
            )
        
        self.intent_encoder = LSTMEncoder(
            self.encoder_hidden_dim + self.attention_output_dim,
            self.encoder_hidden_dim + self.attention_output_dim,
            self.dropout_rate
        )


        self.__slot_predict = MLPAdapter('qin',
                                         self.slot_decoder_hidden_dim,
                                         self.num_slot,
                                         drop_out=self.dropout_rate)

        self.intent_embedding = nn.Parameter(
            torch.FloatTensor(self.num_intent,
                                self.slot_decoder_hidden_dim))  # 191, 32
        nn.init.normal_(self.intent_embedding.data)
    
    ##窗口投票意图检测
    ##硬投票————少数服从多数
    ##软投票————对各分类器的结果进行加权平均
    def hard_vote(self,pred_intent,window_nums):
        # token_intent的版本
        window_intent = torch.argmax(F.softmax(pred_intent,dim=2),dim=-1)
        window_intent_list = [window_intent[i,:window_nums[i]].cpu().data.numpy().tolist() for i in range(len(window_nums)) ]
        intent_index = []
        start_idx,end_idx = [],[]
        for sen_idx,sen in enumerate(window_intent_list):
            sep_idx = [i for i,x in enumerate(sen) if x == self.dataset.vocab['intent'].word2idx['SEP'] ]
            start_idx = [i + 1 for i in sep_idx]
            start_idx.insert(0,0) ##在位置0插入0
            end_idx = sep_idx[:] ##所有元素复制到名为end_idx列表中
            end_idx.append(len(sen))
            sen_intent = []
            for start,end in zip(start_idx,end_idx):
                partition = sen[start:end]
                if len(partition) == 0:
                    continue
                partition_intent = max(partition,key=partition.count)
                sen_intent.append([sen_idx,partition_intent])
            intent_index.extend(sen_intent)
        return {'intent_index':intent_index,
                'window_intent_list':window_intent_list}

    def soft_vote(self,pred_intent,window_num_tensor,window_num,threshold):
        intent_index_sum = torch.cat([ #[batch_size,intent_num]
            # [intent_num]
            torch.sum( 
                # [seq_lens,intent_num]
                torch.sigmoid(pred_intent[i,0:window_num[i], :]) > threshold,
                dim=0).unsqueeze(0) for i in range(len(window_num))
        ],dim = 0)
        intent_index = (intent_index_sum >
                        (window_num_tensor // 2).unsqueeze(1)).nonzero() #保存为true的位置  
        return intent_index.cpu().data.numpy().tolist()

    def forward(self,inputs,n_predict=None,threshold=0.5):
        words = inputs['word_idx'] ## [32,36] [batch_size,token_len]
        seq_lens = inputs['seq_lens'] ##[32]
        embedded = self.embedding(words) ## t->x[32,36,192] [batch_size,token_len,embeding_dim]
        hidden = self.encoder(embedded,seq_lens) ## x->e [32,36,384] [batch_size,token_len, 2*embeding_dim]

        # intent_hidden = hidden
        # slot_hidden = hidden

        ##intent——encoder为Intent BiLSTM

        intent_hidden= self.intent_encoder(hidden,seq_lens) ## e->h [32,36,384]

        ## h->I 先输出token级的意图，然后再进行投票预测出句子的意图
        #window-level的intent版本
        output = self.__intent_decoder({
            # "hidden": intent_hidden,
            "hidden" : intent_hidden,
            "seq_lens": seq_lens
        })
        ##预测意图
        pred_intent = output['hidden']#[batch_size,window_num,intent_num+1] [32,34,19]
        window_nums = output['window_num'] ##[32]
        window_num_tensor = window_nums #[batch_size] [32]

        if self.is_hard and not self.args.ablation:
            intent_index,window_intent_list  = self.hard_vote(pred_intent,window_nums)['intent_index'],\
                           self.hard_vote(pred_intent,window_nums)['window_intent_list']
            ## intent_index [32,2]
            ## window_intent_list [32,32]
        else:
            intent_index = self.soft_vote(pred_intent,window_num_tensor,window_nums,threshold)
            window_intent_list = None

        slot_lstm_out = self.__slot_lstm(hidden, seq_lens)  ##Slot BiLSTM [32,36,128]
        force_slot_idx = inputs['slot_idx'] if 'slot_idx' in inputs else None [32,36]
        slot_inputs = {
            "hidden": slot_lstm_out,
            # "hidden" : hidden,
            "seq_lens": seq_lens,
            "extra_input": pred_intent,
            "intent_index" : intent_index,
            "intent_embedding" : self.intent_embedding,
            "force_input":force_slot_idx
        }
        pred_slot = self.__slot_decoder(slot_inputs)['hidden']
        if not (self.decoder == 'seq2seq' or self.decoder == 'agif'):
            pred_slot = self.__slot_predict({"hidden": pred_slot})['hidden']

        if n_predict is None:
            return {
                "pred_intent": pred_intent, #[batch_size,seq_len,intent_num]/[batch_size,window_num,intent_num]
                "pred_slot": pred_slot,
                'window_num':window_nums,
            }
        else:
            pred_slot = torch.argmax(pred_slot,
                                     dim=-1).cpu().data.numpy().tolist()

            return {
                "pred_intent": intent_index,
                # "pred_intent": intent_index.cpu().data.numpy().tolist(),
                "pred_slot": pred_slot,
                "window_intent_list":window_intent_list
            }

class Window_Attention_Intent_Decoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """
    def __init__(self,
                 args,
                 window_size,
                 input_dim,
                 num_intent,
                 layer_num = 1,
                 drop_out=0.4,
                 alpha=0.2):
        super(Window_Attention_Intent_Decoder, self).__init__()
        self.args = args
        self.window_type = self.args.window_type
        self.window_size = window_size
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=2048, dropout=drop_out, activation='relu')
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layer_num)
        
        self.__intent_decoder = MLPAdapter(
            'qin', input_dim,
            num_intent)
        padding = 1 if window_size == 3 else 0
        self.conv = nn.Conv2d(in_channels=1,out_channels=input_dim,kernel_size=(window_size,input_dim))
        

    def forward(self, inputs):
        hidden = inputs['hidden']
        seq_lens = inputs['seq_lens']
        if self.window_type == 'tf':
            window_num = seq_lens - self.window_size + 1 ##计算窗口数量（token长度减去窗口大小再加一）
            window_num = torch.where(window_num<=0,1,window_num) ##window_num中满足条件的位置用1进行替换，而不满足保持原始值
            max_window_num = torch.max(window_num)
            hidden_stack = None
            if self.window_size > 1:
                for start_idx in range(max_window_num):
                    end_idx = start_idx + self.window_size # [start_idx,end_idx]构成一个窗口 左闭右开
                    hidden_window = hidden[:,start_idx:end_idx,:] # chunk
                    hidden_window = self.encoder(hidden_window) ## Self-Attention
                    # hidden_window_sum = torch.sum(hidden_window,dim=1).unsqueeze(1)
                    hidden_window_sum = hidden_window.max(1)[0].unsqueeze(1) ## max（1）取第一维的最大值。[0]为最大值，[1]为最大值的下标
                    if hidden_stack is None:
                        hidden_stack = hidden_window_sum
                    else:
                        hidden_stack = torch.cat([hidden_stack,hidden_window_sum],dim=1) ##按第一维进行拼接 
            # [batch, num_of_window_scalar * max_window_num, hidden_size]
            else:
                hidden_stack = hidden
            pred_intent = self.__intent_decoder({'hidden':hidden_stack})['hidden']
        else:
            window_num = seq_lens - self.window_size + 1
            window_num = torch.where(window_num<=0,1,window_num)
            hidden = hidden.unsqueeze(1)
            conved = F.relu(self.conv(hidden)).squeeze(3).permute(0,2,1)
            pred_intent = self.__intent_decoder({'hidden':conved})['hidden']
        
        output = {
            'hidden':pred_intent,
            'window_num':window_num,
            'seq_lens':seq_lens,
        }

        return output#[batch_size,window_num,intent_num]

