"""
Hierarchical LSTM model for chest X-ray report generation.

Based on:
  ZexinYan/Medical-Report-Generation (PyTorch implmentation of Jing et al., ACL 2018)
  https://github.com/ZexinYan/Medical-Report-Generation

Adaptations for this project:
  - DenseNet-121 encoder (CheXNet-compatible) instead of VGG
  - Dataloader batch format: (images, input_tokens, target_tokens, lengths, image_ids)
      images:        (B, C, H, W)
      target_tokens: (B, T) flat token ids, split internally at EOS into sentences
      lengths:       (B,)   true sequence length per sample
  - forward(images, captions) where captions = (input_tokens, target_tokens, lengths)
  - generate() returns (B, T_out) LongTensor padded with 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



# Visual Encoder (DenseNet-121)

class VisualEncoder(nn.Module):
    """
    DenseNet-121 feature extractor.
    Returns both a global pooled vector and the full spatial feature map.
    The spatial map is used by co-attention; the global vector seeds the
    sentence LSTM at t=0.
    """

    def __init__(self, embed_size: int = 512, pretrained: bool = True):
        super().__init__()
        densenet = models.densenet121(pretrained=pretrained)
        self.features = densenet.features          # (B, 1024, 7, 7) for 224x224
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed    = nn.Linear(1024, embed_size)
        self.bn       = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images: torch.Tensor):
        """
        Args: images (B, 3, 224, 224)
        Returns:
            avg_feats (B, embed_size) - global context vector
            spatial   (B, 1024, H, W) - spatial map for attention
        """
        spatial   = F.relu(self.features(images), inplace=True)   # (B,1024,7,7)
        avg       = self.avg_pool(spatial).view(images.size(0), -1)  # (B,1024)
        avg_feats = self.bn(self.embed(avg))                          # (B,embed)
        return avg_feats, spatial



#CoAttention

class CoAttention(nn.Module):
    """
    Soft spatial attention conditioned on the sentence LSTM hidden state.
    At each sentence step: take the spatial map + current h -> weighted context.
    """

    def __init__(self, visual_size: int = 1024, hidden_size: int = 512,
                 embed_size: int = 512):
        super().__init__()
        self.W_v = nn.Linear(visual_size, embed_size)
        self.W_h = nn.Linear(hidden_size, embed_size)
        self.W_a = nn.Linear(embed_size, 1)

    def forward(self, spatial: torch.Tensor, hidden: torch.Tensor):
        """
        Args:
            spatial (B, 1024, H, W)
            hidden  (B, hidden_size)
        Returns:
            context (B, 1024)
        """
        B, C, H, W = spatial.size()
        s_flat  = spatial.view(B, C, -1).permute(0, 2, 1)           # (B, HW, C)
        v_proj  = self.W_v(s_flat)                                    # (B, HW, embed)
        h_proj  = self.W_h(hidden).unsqueeze(1)                      # (B,  1, embed)
        weights = F.softmax(self.W_a(torch.tanh(v_proj + h_proj)), dim=1)  # (B,HW,1)
        context = (weights * s_flat).sum(dim=1)                       # (B, C)
        return context


#Sentence LSTM

class SentenceLSTM(nn.Module):
    """
    Single-layer LSTMCell that unrolls for s_max steps.
    Each step: attend over spatial map -> run cell -> emit topic vector + stop logit.
    """

    def __init__(self, visual_size: int = 1024, embed_size: int = 512,
                 hidden_size: int = 512, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm        = nn.LSTMCell(visual_size + embed_size, hidden_size)
        self.topic_fc    = nn.Linear(hidden_size, embed_size)
        self.stop_fc     = nn.Linear(hidden_size, 2)       # 0=continue, 1=stop
        self.dropout     = nn.Dropout(dropout)

    def forward(self, avg_feats, spatial, co_attention, s_max):
        """
        Returns:
            topics     (B, s_max, embed_size)
            stop_probs (B, s_max, 2)
        """
        B      = avg_feats.size(0)
        device = avg_feats.device
        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)

        topics, stops = [], []
        for _ in range(s_max):
            ctx     = co_attention(spatial, h)                   # (B,1024)
            lstm_in = torch.cat([ctx, avg_feats], dim=1)        # (B,1024+embed)
            h, c    = self.lstm(lstm_in, (h, c))
            h       = self.dropout(h)
            topics.append(torch.tanh(self.topic_fc(h)).unsqueeze(1))
            stops.append(self.stop_fc(h).unsqueeze(1))

        return torch.cat(topics, dim=1), torch.cat(stops, dim=1)


# Word LSTM

class WordLSTM(nn.Module):
    """
    Two-layer LSTM that generates one sentence worth of words,
    seeded by the topic vector from the sentence LSTM.
    """

    def __init__(self, vocab_size: int, embed_size: int = 512,
                 hidden_size: int = 512, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding  = nn.Embedding(vocab_size, embed_size)
        self.lstm       = nn.LSTM(embed_size, hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=dropout if num_layers > 1 else 0.0)
        self.topic_to_h = nn.Linear(embed_size, num_layers * hidden_size)
        self.topic_to_c = nn.Linear(embed_size, num_layers * hidden_size)
        self.output_fc  = nn.Linear(hidden_size, vocab_size)
        self.dropout    = nn.Dropout(dropout)

    def _init_hidden(self, topic):
        B  = topic.size(0)
        h0 = torch.tanh(self.topic_to_h(topic)).view(
                B, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        c0 = torch.tanh(self.topic_to_c(topic)).view(
                B, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        return h0, c0

    def forward(self, topic, token_ids):
        """
        Teacher-forcing for one sentence segment.
        Args:
            topic     (B, embed_size)
            token_ids (B, L)  - input tokens (shifted right already)
        Returns:
            logits (B, L, vocab_size)
        """
        h0, c0 = self._init_hidden(topic)
        emb    = self.dropout(self.embedding(token_ids))    # (B, L, embed)
        out, _ = self.lstm(emb, (h0, c0))                   # (B, L, hidden)
        return self.output_fc(self.dropout(out))             # (B, L, vocab)

    @torch.no_grad()
    def generate(self, topic, bos_id, n_max):
        """
        Greedy decode one sentence for all items in the batch.
        Returns (B, n_max) token ids.
        """
        B      = topic.size(0)
        device = topic.device
        h, c   = self._init_hidden(topic)
        inp    = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        out    = []
        for _ in range(n_max):
            emb             = self.embedding(inp)           # (B,1,embed)
            lstm_out, (h,c) = self.lstm(emb, (h, c))
            logit           = self.output_fc(lstm_out.squeeze(1))  # (B, vocab)
            pred            = logit.argmax(-1, keepdim=True)       # (B, 1)
            out.append(pred)
            inp = pred
        return torch.cat(out, dim=1)                        # (B, n_max)


# Full Model


class HierarchicalLSTM(nn.Module):
    """
    Hierarchical LSTM for CXR report generation.

    DATA FORMAT (interfaces.md: 5-tuple):
    The dataloader yields per batch:

        images        (B, C, H, W)      preprocessed X-ray images
        input_tokens  (B, T)            BOS-prepended token ids (not used internally)
        target_tokens (B, T)            flat token ids split at EOS into sentences
        lengths       (B,)              true sequence length per sample
        image_ids     list[str]         image identifiers (not used in forward)

     Forward interface (matches interfaces.md):

        loss = model(images, captions)
          images   : (B, C, H, W)
          captions : (input_tokens, target_tokens, lengths)
                     target_tokens is split internally at EOS boundaries
                     into (B, s_max, n_max) for the hierarchical decode.
    """

    def __init__(self,
                 vocab_size        : int,
                 bos_id            : int,   
                 eos_id            : int,   
                 pad_id            : int,
                 embed_size        : int   = 512,
                 hidden_size       : int   = 512,
                 word_num_layers   : int   = 2,
                 dropout           : float = 0.1,
                 s_max             : int   = 6,
                 n_max             : int   = 30,
                 pretrained_encoder: bool  = True):
                #  reshape_captions  : bool  = True):
        """
        Args:
            vocab_size         - size of shared vocabulary (from tokenizer)
            embed_size         - word embedding + visual projection size
            hidden_size        - LSTM hidden size
            word_num_layers    - layers in the word LSTM (2 per Jing et al.)
            dropout            - applied after each LSTM step
            s_max              - max sentences per report
            n_max              - max words per sentence (including BOS/EOS)
            bos_id             - begin-of-sentence token id
            eos_id             - end-of-sentence token id (set from tokenizer!)
            pad_id             - padding token id (used as ignore_index in loss)
            pretrained_encoder - load ImageNet DenseNet-121 weights
            reshape_captions   - True  -> expects flat (B, seq_len) from R2Gen
                                   False -> expects (B, s_max, n_max) already
            Update: Removed reshape_captions
        """
        super().__init__()
        self.s_max            = s_max
        self.n_max            = n_max
        self.bos_id           = bos_id
        self.eos_id           = eos_id
        self.pad_id           = pad_id
        # self.reshape_captions = reshape_captions

        self.encoder   = VisualEncoder(embed_size=embed_size,
                                       pretrained=pretrained_encoder)
        self.co_att    = CoAttention(visual_size=1024,
                                     hidden_size=hidden_size,
                                     embed_size=embed_size)
        self.sent_lstm = SentenceLSTM(visual_size=1024,
                                      embed_size=embed_size,
                                      hidden_size=hidden_size,
                                      dropout=dropout)
        self.word_lstm = WordLSTM(vocab_size=vocab_size,
                                  embed_size=embed_size,
                                  hidden_size=hidden_size,
                                  num_layers=word_num_layers,
                                  dropout=dropout)

        self.ce_criterion   = nn.CrossEntropyLoss(ignore_index=pad_id,
                                                   reduction='mean')
        self.stop_criterion = nn.CrossEntropyLoss(reduction='mean')

   
    def _encode(self, images):
        """
        R2Gen stacks frontal + lateral as (B, 2, 3, H, W).
        We encode each view separately and average the features.
        Falls back gracefully to single-view (B, 3, H, W).
        """
        if images.dim() == 5:
            B, V, C, H, W = images.shape
            imgs          = images.view(B * V, C, H, W)
            avg, spatial  = self.encoder(imgs)              # (B*V, ...)
            avg           = avg.view(B, V, -1).mean(1)      # (B, embed)
            spatial       = spatial.view(B, V, *spatial.shape[1:]).mean(1)
        else:
            avg, spatial  = self.encoder(images)
        return avg, spatial

    
    # Reshape flat R2Gen captions -> (B, s_max, n_max)

    def _reshape_flat_captions(self, flat_ids, lengths):
        """
        Convert R2Gen flat token sequence to (B, s_max, n_max) sentence grid.

        Splits at EOS token boundaries.  Each segment (up to and including the
        EOS) becomes one row.  Rows are zero-padded to n_max; reports with more
        than s_max sentences are truncated.

        Args:
            flat_ids (B, seq_len) - token ids, 0-padded
            lengths     (B,) - true seq length per sample

        Returns:
            cap2d (B, s_max, n_max) - long tensor, 0-padded
        """
        B      = flat_ids.size(0)
        device = flat_ids.device
        cap2d  = torch.zeros(B, self.s_max, self.n_max,
                             dtype=torch.long, device=device)

        for b in range(B):
            length = int(lengths[b].item())
            tokens = flat_ids[b, :length].tolist()

            # split into sentences at every EOS token
            sentences, current = [], []
            for tok in tokens:
                current.append(tok)
                if tok == self.eos_id:
                    sentences.append(current)
                    current = []
            if current:                     # any trailing tokens without EOS
                sentences.append(current)

            for s_idx, sent in enumerate(sentences[:self.s_max]):
                sent_t = sent[:self.n_max]
                cap2d[b, s_idx, :len(sent_t)] = torch.tensor(
                    sent_t, dtype=torch.long, device=device)

        return cap2d

   
    # Stop targets from 2-D caption tensor


    @staticmethod
    def _stop_targets(cap2d):
        """
        0 = continue (sentence has content), 1 = stop (sentence is all-pad).
        Args:  cap2d (B, s_max, n_max)
        Returns: (B, s_max) long tensor
        """
        return (1 - (cap2d.sum(-1) > 0).long())

    
    # Training forward - called as model(images, captions)
    
    #Changing it as per interfaces.md
    def forward(self, images, captions):
        """
        Teacher-forcing forward pass.

        Args:
            images   (B, 2, 3, H, W)  - stacked frontal + lateral (R2Gen format)
            captions Tuple[input_tokens, target_tokens, lengths]
                        input_tokens  (B, T) - not used internally, kept for interface compliance
                        target_tokens (B, T) - flat token ids, split into sentences at EOS
                        lengths       (B,)   - true sequence length per sample

        Returns:
            loss : scalar tensor
        """

        input_tokens, target_tokens, lengths = captions
        cap2d = self._reshape_flat_captions(target_tokens, lengths)

        B, s_max, n_max = cap2d.shape

        # encode dual-view images
        avg_feats, spatial = self._encode(images)

        # sentence LSTM
        topics, stop_probs = self.sent_lstm(
            avg_feats, spatial, self.co_att, s_max)
        # topics (B, s_max, embed_size)
        # stop_probs (B, s_max, 2)

        # stop loss
        stop_tgt  = self._stop_targets(cap2d)                   # (B, s_max)
        #Stop loss masking
        active_slots = (cap2d.sum(-1) > 0).view(B * s_max)   # True for non-padding sentences
        stop_loss = self.stop_criterion(
            stop_probs.view(B * s_max, 2)[active_slots],
            stop_tgt.view(B * s_max)[active_slots])

        # word loss (teacher forcing per sentence)
        word_loss  = torch.tensor(0.0, device=images.device)
        word_in    = cap2d[:, :, :-1]       # (B, s_max, n_max-1)
        word_tgt   = cap2d[:, :, 1:]        # (B, s_max, n_max-1)
        active_cnt = 0

        for s in range(s_max):
            # active = (cap2d[:, s, :].sum(-1) > 0)   # samples that have sentence s
            active = (word_tgt[:, s, :].sum(-1) > 0)   # samples with non-pad targets after shift
            if not active.any():
                continue
            active_cnt += 1
            logits = self.word_lstm(
                topics[:, s, :][active],
                word_in[:, s, :][active])           # (n_act, n_max-1, vocab)
            V = logits.size(-1)
            word_loss = word_loss + self.ce_criterion(
                logits.reshape(-1, V),
                word_tgt[:, s, :][active].reshape(-1))

        # average word loss over active sentence slots
        if active_cnt > 0:
            word_loss = word_loss / active_cnt

        return stop_loss + word_loss


    # Inference


    @torch.no_grad()
    #Changing it as per interface.md
    def generate(self, images, max_length=100, beam_size=1):
        """
        Greedy decoding.  Returns flat token id lists (one per sample)
        ready for pycocoevalcap / CheXbert scoring.

         Args:
            images     (B, C, H, W)
            max_length  override n_max if desired
            beam_size   1 = greedy (only supported value)

        Returns:
            out : (B, T_out) LongTensor, padded with 0, includes <EOS> when generated
        """
        if beam_size > 1:
            raise NotImplementedError("Beam search not implemented for HierarchicalLSTM")
        n_max    = max_length or self.n_max
        B        = images.size(0)
        avg, sp  = self._encode(images)
        topics, stop_probs = self.sent_lstm(avg, sp, self.co_att, self.s_max)
        stop_pred = stop_probs.argmax(-1)           # (B, s_max) - 1=stop

        reports = [[] for _ in range(B)]

        for s in range(self.s_max):
            active_mask = (stop_pred[:, s] == 0)
            if not active_mask.any():
                break
            idx   = active_mask.nonzero(as_tuple=True)[0]
            words = self.word_lstm.generate(
                topics[idx, s, :], self.bos_id, n_max)   # (n_act, n_max)

            for i, b in enumerate(idx.tolist()):
                toks = words[i].tolist()
                if self.eos_id in toks:
                    toks = toks[:toks.index(self.eos_id) + 1]
                reports[b].extend(toks)   # flat list - consistent with R2Gen eval

        max_out = max((len(r) for r in reports), default=1)
        out = torch.zeros(B, max_out, dtype=torch.long, device=images.device)
        for i, r in enumerate(reports):
            if r:
                out[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=images.device)
        return out
