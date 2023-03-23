import torch
import torch.nn as nn


class Prefix(nn.Module):
    def __init__(self, model, n_prefixes, len_prefix) -> None:
        super().__init__(Prefix)
        self._model = model
        self.hidden_size = model.config.hidden_size
        self.embed_size = model.config.hidden_size
        self.n_layers = model.config.num_decoder_layers
        self.n_heads = model.config.num_attention_heads
        self.head_size = self.embed_size // self.n_heads 

        self.reparams = nn.Parameter(torch.ones(len_prefix))   
        self.prefixes = nn.Parameter(torch.empty((n_prefixes, 2 * self.n_layers, len_prefix, self.hidden_size)))

        with torch.no_grad():
            nn.init.xavier_uniform_(self.prefixes)
        
    def forward(self, batch_size):
        interm = torch.maximum(torch.ones(1), self.reparams)
        keys = (self.prefixes[:, 0::2, :, :] * interm).repeat([batch_size, 1, 1, 1, 1])
        values = (self.prefixes[:, 1::2, :, :] * interm).repeat([batch_size, 1, 1, 1, 1])
        keys = keys.view(-1, -1, -1, -1, self.n_heads, self.head_size)
        values = values.view(-1, -1, -1, -1, self.n_heads, self.head_size)

        # keys/values: (batch_size, n_prefixes ,n_layers, len_prefix, n_heads, head_size)

        return keys, values
    

class LLM(nn.Module):
    def __init__(self, model, params) -> None:
        super().__init__(LLM)
        self.params = params
        self._model = model

        self.hidden_size = model.config.hidden_size
        self.embed_size = model.config.hidden_size
        self.n_layers = model.config.num_decoder_layers
        self.n_heads = model.config.num_attention_heads
        self.head_size = self.embed_size // self.n_heads

        self.prefix = Prefix(model, 3, params.len_prefix)
    
    def encode(self, input_ids, input_mask):
        batch_size = input_ids.shape[0]
        len_prefix = self.params.len_prefix
        len_sent = input_ids.shape[1]
        prefix_mask = torch.ones([batch_size, len_prefix], device=input_mask.device)
        attn_mask = torch.cat([prefix_mask, input_mask], dim=1)

        keys, values = self.prefix(batch_size)
        # batch_size, n_prefixes, n_layers, len_prefix, n_heads, head_size => batch_size, n_heads, len_prefix, head_size

        layer_prefix_list = []
        for i in range(self.n_layers):
            tup = (keys[:,0,i,:,:,:].permute(0,2,1,3), values[:,0,i,:,:,:].permute(0,2,1,3))
            layer_prefix_list.append(tup)

        outputs = self._model(input_ids, past_key_values=layer_prefix_list, attention_mask=attn_mask, use_cache=True)

        #RE-ENCODING
        layer_prefix_list = []
        attn_mask = torch.cat([prefix_mask, input_mask, input_mask], dim=1)

        #prepend re-encoding prefixes and exclude encoder prefixes in the re-encoding stage
        for i, (key, value) in enumerate(outputs.past_key_values):
            k = torch.cat([keys[:,1,i,:,:,:].permute(0,2,1,3), key[:,:,len_prefix:,:]], dim=2)
            v = torch.cat([values[:,1,i,:,:,:].permute(0,2,1,3), value[:,:,len_prefix:,:]], dim=2)
            layer_prefix_list.append((k, v))
        
        outputs = self._model(input_ids, past_key_values=layer_prefix_list, attention_mask=attn_mask, use_cache=True)
        layer_prefix_list = []

        #prepare past key values for decoding stage
        for i, (key, value) in outputs.past_key_values:
            k = torch.cat([keys[:,2,i,:,:,:].permute(0,2,1,3), key[:,:,len_prefix + len_sent:,:]], dim=2)
            v = torch.cat([values[:,2,i,:,:,:].permute(0,2,1,3), value[:,:,len_prefix + len_sent:,:]], dim=2)
            layer_prefix_list.append((k, v))

        return layer_prefix_list
    
    def decode(self, target_ids, input_mask, target_mask, past_key_values, mode='train'):
        batch_size = target_ids.shape[0]
        len_prefix = self.params.len_prefix
        prefix_mask = torch.ones([batch_size, len_prefix], device=input_mask.device)

        attn_mask = torch.cat([prefix_mask, input_mask, target_mask], dim=1)

        outputs = self._model(target_ids, past_key_values=past_key_values, attention_mask=attn_mask, use_cache=True)
        return outputs.logits
    
    def forward(self, input_ids, input_mask, target_ids, target_mask):
        past_key_values = self.encode(input_ids, input_mask)
        logits = self.decode(target_ids, input_mask, target_mask, past_key_values)
        
