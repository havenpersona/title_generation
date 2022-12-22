import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, model_size, n_heads, dropout, device):
        super(MultiHeadAttention, self).__init__()
        self.model_size = model_size
        self.n_heads = n_heads
        self.head_dim = model_size // n_heads
        self.temperature = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.q_linear = nn.Linear(model_size, model_size)
        self.k_linear = nn.Linear(model_size, model_size)
        self.v_linear = nn.Linear(model_size, model_size)
        self.out = nn.Linear(model_size, model_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask = None):
        batch_size = q.shape[0]
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.head_dim)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.head_dim)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        #scaled dot-product attention
        output = torch.matmul(q / self.temperature , k.transpose(2,3)) 
        if mask is not None:
            output = output.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(output, dim = -1)
        output = self.dropout(attention)
        output = torch.matmul(output, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.model_size)
        multihead = self.out(output)

        return multihead, attention

class PositionwiseFeedForwardLayer(nn.Module):
    def __init__(self, model_size, inner_size, dropout):
        super(PositionwiseFeedForwardLayer, self).__init__()
        self.layer1 = nn.Linear(model_size, inner_size)
        self.layer2 = nn.Linear(inner_size, model_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x



class EncoderLayer(nn.Module):
    def __init__(self, model_size, inner_size, n_heads, dropout, device):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_size, n_heads, dropout, device)
        self.layer_norm1 = nn.LayerNorm(model_size)
        self.positionwise_feedforward = PositionwiseFeedForwardLayer(model_size, inner_size, dropout)
        self.layer_norm2 = nn.LayerNorm(model_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        attention_output, _ = self.self_attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attention_output))
        feedforward_output = self.positionwise_feedforward(x)
        x = self.layer_norm2(x + self.dropout(feedforward_output))
        
        return x


class Encoder(nn.Module):
    def __init__(self, input_size, model_size, inner_size, n_layers, n_heads, dropout, device):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(input_size, model_size)
        self.scale = torch.sqrt(torch.FloatTensor([model_size])).to(device)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([EncoderLayer(model_size, inner_size, n_heads, dropout, device) for _ in range(n_layers)])
    def forward(self, src_seq, src_mask):
        embedded_tokens = self.token_embedding(src_seq) 
        embedded_tokens = embedded_tokens * self.scale
        x = self.dropout(embedded_tokens)

        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, model_size, inner_size, n_heads, dropout, device):
        super(DecoderLayer, self).__init__()
        self.device = device
        self.self_attention = MultiHeadAttention(model_size, n_heads, dropout, device)
        self.layer_norm1 = nn.LayerNorm(model_size)
        self.encoder_attention = MultiHeadAttention(model_size, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedForwardLayer(model_size, inner_size, dropout)
        self.layer_norm2 = nn.LayerNorm(model_size)
        self.layer_norm3 = nn.LayerNorm(model_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, encoder_source, target_mask, source_mask):

        attention_output, _ = self.self_attention(target, target, target, target_mask)
        x = self.layer_norm1(target + self.dropout(attention_output))
        encoder_attention_output, encoder_attention = self.encoder_attention(x, encoder_source, encoder_source, source_mask)
        x = self.layer_norm2(x + self.dropout(encoder_attention_output))
        feedforward_output = self.positionwise_feedforward(x)
        x = self.layer_norm3(x + self.dropout(feedforward_output))
        
        return x, encoder_attention


class Decoder(nn.Module):
    def __init__(self, output_size, model_size, inner_size, n_layers, n_heads, dropout, max_length, device):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(output_size, model_size)
        self.pos_embedding = nn.Embedding(max_length, model_size)
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([model_size])).to(self.device)
        self.dropout = nn.Dropout(p=dropout)
        self.max_length = max_length
        self.layers = nn.ModuleList([DecoderLayer(model_size, inner_size, n_heads, dropout, device) for _ in range(n_layers)])
        self.out = nn.Linear(model_size, output_size)
        
    def forward(self, target, encoder_source, target_mask, source_mask):
        
        batch_size = target.shape[0]
        target_length = target.shape[1]
        
        pos = torch.arange(0, target_length)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(batch_size, 1)
        pos = pos.to(self.device)
        
        x = (self.token_embedding(target) * self.scale) + self.pos_embedding(pos)
        x = self.dropout(x)
        
        for layer in self.layers:
            x, attention = layer(x, encoder_source, target_mask, source_mask)
        output = self.out(x)
        return output, attention



class Transformer(pl.LightningModule):
    def __init__(self, input_size, output_size, model_size, inner_size, encoder_layers, decoder_layers, n_heads, dropout, max_length, device, lr, weight_decay):
        super(Transformer, self).__init__()
        self.output_size = output_size
        self.encoder = Encoder(input_size, model_size, inner_size, encoder_layers, n_heads, dropout, device)
        self.decoder = Decoder(output_size, model_size, inner_size, decoder_layers, n_heads, dropout, max_length, device)
        self.criterion = nn.CrossEntropyLoss(ignore_index = 0)
        self.lr = lr
        self.weight_decay = weight_decay
    def get_source_mask(self, source):
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)
        return source_mask
    def get_target_mask(self, target):
        pad_mask = (target != 0).unsqueeze(1).unsqueeze(2)
        target_length = target.shape[1]
        subsequent_mask = torch.tril(torch.ones((target_length, target_length), device = self.device)).bool()
        target_mask = pad_mask & subsequent_mask #bitwise
        return target_mask
    def configure_optimizers(self):
        opt = Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay= self.weight_decay
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer=opt,
            T_0= 200,
            eta_min = 1e-6
        )
        lr_scheduler = {
            'scheduler': scheduler, 
            'interval': 'epoch', # The unit of the scheduler's step size
            'frequency': 1, # The frequency of the scheduler
            'reduce_on_plateau': False, # For ReduceLROnPlateau scheduler
            'monitor': 'val_loss' # Metric to monitor
        }
        return [opt], [lr_scheduler]

    
    def forward(self, source, target):
        source_mask = self.get_source_mask(source)
        target_mask = self.get_target_mask(target)
        encoder_source = self.encoder(source, source_mask)
        output, attention = self.decoder(target, encoder_source, target_mask, source_mask)
        return output, attention
        
    def training_step(self, batch, batch_idx):
        src, trg = batch
        output, _ = self.forward(src, trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        loss = self.criterion(output, trg)
        self.log_dict(
            {"train_loss": loss},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    
    def validation_step(self, batch, batch_idx):
        src, trg = batch
        output, _ = self.forward(src, trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        loss = self.criterion(output, trg)
        return {"val_loss": loss}

    def validation_step_end(self, batch_parts):
        # sigle gpu case
        return batch_parts

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        self.log_dict(
            {"val_loss": val_loss,},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {"val_loss": val_loss,}

    def test_step(self, batch, batch_idx):
        src, trg = batch
        output, _ = self.forward(src, trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        loss = self.criterion(output, trg)
        return {"val_loss": loss}

    def test_step_end(self, batch_parts):
        # sigle gpu case
        return batch_parts

    def test_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        self.log_dict(
            {
                "test_loss": val_loss,
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        result = {"test_loss": float(val_loss.detach().cpu())}
        self.test_results = result
        return result
