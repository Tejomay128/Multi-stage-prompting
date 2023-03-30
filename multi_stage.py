import torch
import torch.nn as nn


class Prefix(nn.Module):
    def __init__(self, model, n_prefixes, len_prefix) -> None:
        super(Prefix, self).__init__()
        self._model = model
        self.hidden_size = model.config.hidden_size
        self.embed_size = model.config.hidden_size
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.head_size = self.embed_size // self.n_heads 

        self.reparams = nn.Parameter(torch.ones(len_prefix))   
        self.prefixes = nn.Parameter(torch.empty((n_prefixes, 2 * self.n_layers, len_prefix, self.hidden_size)))

        with torch.no_grad():
            nn.init.xavier_uniform_(self.prefixes)
        
    def forward(self, batch_size):
        interm = torch.maximum(torch.ones(1, device=self.reparams.device), self.reparams)
        interm = interm.view([1, 1, interm.shape[0], 1])
        # print(interm.shape, self.prefixes[:, 0::2, :, :].shape)
        keys = (self.prefixes[:, 0::2, :, :] * interm).repeat([batch_size, 1, 1, 1, 1])
        values = (self.prefixes[:, 1::2, :, :] * interm).repeat([batch_size, 1, 1, 1, 1])
        keys = keys.view(keys.shape[0], keys.shape[1], keys.shape[2], keys.shape[3], self.n_heads, self.head_size)
        values = values.view(values.shape[0], values.shape[1], values.shape[2], values.shape[3], self.n_heads, self.head_size)

        # keys/values: (batch_size, n_prefixes ,n_layers, len_prefix, n_heads, head_size)

        return keys, values
    

class LLM(nn.Module):
    def __init__(self, model, len_prefix) -> None:
        super(LLM, self).__init__()
        self._model = model
        self.hidden_size = model.config.hidden_size
        self.embed_size = model.config.hidden_size
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.head_size = self.embed_size // self.n_heads
        self.len_prefix = len_prefix

        self.prefix = Prefix(model, 3, len_prefix)
        self.logSoftmax = nn.LogSoftmax(dim=2)
        self.logSoftmax_1 = nn.LogSoftmax(dim=1)
        self.nll = nn.NLLLoss()


    def encode(self, input_ids, input_mask):
        batch_size = input_ids.shape[0]
        len_prefix = self.len_prefix
        len_sent = input_ids.shape[1]
        prefix_mask = torch.ones([batch_size, len_prefix], device=input_mask.device)
        attn_mask = torch.cat([prefix_mask, input_mask], dim=1)
#         pos_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=input_mask.device)
#         pos_ids = pos_ids.unsqueeze(0).view(-1, input_ids.shape[-1])

        keys, values = self.prefix(batch_size)
        # batch_size, n_prefixes, n_layers, len_prefix, n_heads, head_size => batch_size, n_heads, len_prefix, head_size

        layer_prefix_list = []
        for i in range(self.n_layers):
            tup = (keys[:,0,i,:,:,:].permute(0,2,1,3), values[:,0,i,:,:,:].permute(0,2,1,3))
            layer_prefix_list.append(tup)

        outputs = self._model(input_ids, 
                              past_key_values=layer_prefix_list, 
                              attention_mask=attn_mask,  
                              use_cache=True)

        #RE-ENCODING
        layer_prefix_list = []
        attn_mask = torch.cat([prefix_mask, input_mask, input_mask], dim=1)
#         pos_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=input_mask.device)
#         pos_ids = pos_ids.unsqueeze(0).view(-1, input_ids.shape[-1])

        #prepend re-encoding prefixes and exclude encoder prefixes in the re-encoding stage
        for i, (key, value) in enumerate(outputs.past_key_values):
            k = torch.cat([keys[:,1,i,:,:,:].permute(0,2,1,3), key[:,:,len_prefix:,:]], dim=2)
            v = torch.cat([values[:,1,i,:,:,:].permute(0,2,1,3), value[:,:,len_prefix:,:]], dim=2)
            layer_prefix_list.append((k, v))
        
        outputs = self._model(input_ids, 
                              past_key_values=layer_prefix_list, 
                              attention_mask=attn_mask, 
                              use_cache=True)
        layer_prefix_list = []

        #prepare past key values for decoding stage
        for i, (key, value) in enumerate(outputs.past_key_values):
            k = torch.cat([keys[:,2,i,:,:,:].permute(0,2,1,3), key[:,:,len_prefix + len_sent:,:]], dim=2)
            v = torch.cat([values[:,2,i,:,:,:].permute(0,2,1,3), value[:,:,len_prefix + len_sent:,:]], dim=2)
            layer_prefix_list.append((k, v))

        return layer_prefix_list
    
    def decode(self, target_ids, input_mask, target_mask, past_key_values, mode='train'):
        batch_size = target_ids.shape[0]
        len_prefix = self.len_prefix
        prefix_mask = torch.ones([batch_size, len_prefix], device=input_mask.device)

        #only attend to decode stage prefixes and re-encoding stage hidden states
        attn_mask = torch.cat([prefix_mask, input_mask, target_mask], dim=1)
#         pos_ids = torch.arange(0, target_ids.shape[-1], dtype=torch.long, device=input_mask.device)
#         pos_ids = pos_ids.unsqueeze(0).view(-1, target_ids.shape[-1])

        outputs = self._model(target_ids, 
                              past_key_values=past_key_values, 
                              attention_mask=attn_mask,  
                              use_cache=True)
        return outputs.logits, outputs.past_key_values
    
    def forward(self, input_ids, input_mask, target_ids, target_mask):
        past_key_values = self.encode(input_ids, input_mask)
        labels = target_ids[:, 1:]
        target_ids = target_ids[:, :-1]
        target_mask = target_mask[:, :-1]
        logits,_ = self.decode(target_ids, input_mask, target_mask, past_key_values)

        # make batch size and sentence length as one dimension
        logits = self.logSoftmax(logits)
        logits = logits.reshape([logits.shape[0] * logits.shape[1], -1])
        target_mask = target_mask.reshape([target_mask.shape[0] * target_mask.shape[1],])
        labels = labels.flatten()
        loss = -logits[torch.arange(logits.shape[0], device=labels.device), labels]
#         print(loss.shape, target_mask.shape)
        loss = torch.sum(loss * target_mask) / torch.sum(target_mask)
        return loss

    @torch.no_grad()
    def greedy_translate(self, device, tokenizer, input_sent):
        tok_output = tokenizer(input_sent)
        input_ids = tok_output['input_ids']
        input_mask = tok_output['attention_mask']
        input_ids = torch.tensor(input_ids, device=device)
        input_mask = torch.tensor(input_mask, device=device)
        target_mask = torch.ones([1, 1], device=device)
        past_key_values = self.encode(input_ids, input_mask)
        
        start = '</s>'
        gen = []
        curr_token = None
        while curr_token != 1:
            tgt = torch.tensor(tokenizer(start)['input_ids'], device=device)    
            logits, past_key_values = self.decode(tgt, input_mask, target_mask, past_key_values)
#             logits = self.logSoftmax(logits.unsqueeze(0)).squeeze(0)
            value, index = torch.max(logits, dim=1)
            curr_token = index[0].item()
            gen.append(curr_token)
            target_mask = torch.cat([target_mask, torch.ones([1, 1], device=device)], dim=1)
        output_sent = tokenizer.decode(gen)
        return output_sent
            
    
    @torch.no_grad()
    def translate(self, device, tokenizer, input_sent, beam_width = 4, token_limit=1024):
        tok_output = tokenizer(input_sent)
        input_ids = tok_output['input_ids']
        input_mask = tok_output['attention_mask']
        input_ids = torch.tensor(input_ids, device=device)
        input_mask = torch.tensor(input_mask, device=device)
        target_mask = torch.ones([1, 1], device=device)
        # attn_mask = torch.cat([prefix_mask, input_mask, target_mask], dim=1)

        past_key_values = self.encode(input_ids, input_mask)

        start = '</s>'
        tgt = torch.tensor(tokenizer(start)['input_ids'], device=device)    
        logits, past_key_values = self.decode(tgt, input_mask, target_mask, past_key_values)
        print(logits.shape)
        logits = self.logSoftmax(logits.unsqueeze(0)).squeeze(0)
        scores, indices = logits.topk(sorted=True, k=beam_width)

        sent = []
        for i in range(beam_width):
            sent.append([[indices[0, i].item()], scores[0, i].item()])
        # new_sent = []

        def check(sent_list):
            flag = False
            for i in range(beam_width):
                if sent_list[i][0][-1] != 1:
                    flag = True
                    break
            return flag
        #need 4 different past_key_values
        def search_step(sent_list, past_key_values):
            new_sent_list = []
            for i in range(beam_width):
                token = sent_list[i][0][-1]
                if token == 1:
                    new_sent_list.append(sent_list[i])
                else: 
                    tgt = torch.tensor([[token]], device=device)
                    logits, past_key_values = self.decode(tgt, input_mask, target_mask, past_key_values)
                    logits = nn.functional.softmax(logits, dim=1)
                    scores, indices = logits.topk(sorted=True, k=beam_width)
                    for i in range(beam_width):
                        new_sent_list.append([sent_list[i][0].append(indices[0, i].item()),
                                               sent_list[i][1] + scores[0, i].item()])
            new_sent_list.sort(key=lambda x: -x[1])
            new_sent_list = [new_sent_list[i] for i in range(beam_width)]
            return new_sent_list, past_key_values

        while check(sent):
            sent, past_key_values = search_step(sent, past_key_values)
        
        final_sent = tokenizer.decode(sent[0][0])
        return final_sent

