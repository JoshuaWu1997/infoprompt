import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaTokenizer
from model.modeling_roberta_prompt import RobertaModel


def entropy(logit):
    logit = torch.log_softmax(logit, -1)
    probs = logit.exp()
    min_real = torch.finfo(logit.dtype).min
    logit = torch.clamp(logit, min=min_real)
    p_log_p = logit * probs
    return -p_log_p.sum(-1)


class LHead(nn.Module):
    def __init__(self, n_cls, head):
        super(LHead, self).__init__()
        self.head = nn.Parameter(torch.tensor(head.tolist()))

    def forward(self, x):
        return torch.matmul(x, self.head.T)


class VariationalLM(nn.Module):
    def __init__(self, n, head, n_prompt):
        super(VariationalLM, self).__init__()
        self.mean_prp = nn.Linear(head.size(-1), head.size(-1))

        self.mean = nn.Linear(head.size(-1), n)
        self.log_std = nn.Linear(head.size(-1), n)
        self.output = nn.Linear(n, head.size(-1))
        self.head = LHead(head.size(0), head)

    def forward(self, x, mode='train'):
        mean = self.mean(x)
        if mode == 'train':
            log_std = self.log_std(x)
            log_std = torch.clamp(log_std, min=-2, max=10)
            std = log_std.exp()
            output = mean + torch.randn_like(mean) * std
        else:
            output = mean
            log_std, std = None, None
        output = self.output(output)
        output = self.head(output)
        return output, mean, log_std, std


class PromptToken(nn.Module):
    def __init__(self, n_token, mask):
        super(PromptToken, self).__init__()
        prompt = torch.cat([torch.tensor(mask.tolist()) for _ in range(n_token)], dim=0)
        self.prompt = nn.Parameter(prompt.unsqueeze(0))

    def forward(self):
        pass


class InstancePrompt(nn.Module):
    def __init__(self, cfg, num_prompt, mask):
        super(InstancePrompt, self).__init__()
        self.bert_latent = mask.size(-1)
        self.bn = nn.Sequential(
            nn.Linear(self.bert_latent, cfg.n_latent),
            nn.GELU(),
            nn.Linear(cfg.n_latent, num_prompt * self.bert_latent)
        )
        self.num_prompt = num_prompt

    def forward(self, x):
        output = self.bn(x).view(x.size(0), self.num_prompt, self.bert_latent)
        return output


class BertDIBPromptRE(nn.Module):
    def __init__(self, cfg):
        super(BertDIBPromptRE, self).__init__()
        self.cfg = cfg
        self.bert = RobertaModel.from_pretrained("model/roberta-large")
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.criterion = nn.CrossEntropyLoss()

        self.tokenizer.add_tokens('<ppt>', special_tokens=False)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.dict = {'cls': 0, 'pad': 1, 'sep': 2, 'msk': 50264, 'p': 50265,
                     'h0': 4420, 'h1': 1386, 'q0': 430, 'q1': 6305}
        print('Vocab Size:', len(self.tokenizer))

        # Tunable
        Embedding = self.bert.embeddings.word_embeddings
        mask = Embedding(torch.tensor([self.dict['msk']])).detach()
        if cfg.task == ['mrpc', 'qqp']:
            head = Embedding(torch.tensor([self.dict['q0'], self.dict['q1']])).detach()
        else:
            head = Embedding(torch.tensor([self.dict['h0'], self.dict['h1']])).detach()
        self.tune = nn.ModuleDict({
            'prompt': InstancePrompt(cfg, self.cfg.n_prompt, mask),
            'lm': VariationalLM(cfg.n_latent, head, cfg.n_prompt)
        })

    def bottleneck_cmi(self, mean, log_std, std, mean_ppt, log_std_ppt, std_ppt):
        loss = .5 * (std_ppt / std) ** 2 + log_std - log_std_ppt \
               + .5 * ((mean - mean_ppt) ** 2) / (std ** 2)
        loss = loss.mean()
        return loss

    def bottleneck_nce(self, mean, log_std, std, mean_ppt, log_std_ppt, std_ppt, prompt_token):
        ctx = self.tune['lm'].mean_prp(prompt_token.detach().data.clone()[:, :self.cfg.n_prompt]).mean(dim=1)
        bsize, fsize = mean.size()

        # (bsize, 1, fsize) -> (bsize, 1)
        random_ppt = torch.randn((bsize, 1, fsize), dtype=std_ppt.dtype, device=std_ppt.device)
        output_ppt = mean_ppt.unsqueeze(1) + random_ppt * std_ppt.unsqueeze(1)
        output_ppt = self.tune['lm'].output(output_ppt)
        output_ppt = torch.matmul(output_ppt, ctx.unsqueeze(-1)).squeeze(-1) / np.sqrt(ctx.size(-1))

        # (bsize, bsize, fsize) -> (bsize, bsize)
        random_x = torch.randn((bsize, bsize * 4, fsize), dtype=std.dtype, device=std.device)
        output_x = mean.unsqueeze(1) + random_x * std.unsqueeze(1)
        output_x = self.tune['lm'].output(output_x)
        output_x = torch.matmul(output_x, ctx.unsqueeze(-1)).squeeze(-1) / np.sqrt(ctx.size(-1))

        logits = torch.cat([output_ppt, output_x], -1)
        loss = -torch.log_softmax(logits, -1)[:, 0].clamp(min=-2)
        loss = loss.mean() * self.cfg.beta
        return loss

    def promptHead_loss(self, batch_null, prompt_tokens):
        mask = batch_null['attention_mask'].clone().float()
        mask = -100 * (1. - mask)
        ctx = self.tune['lm'].head.head.mean(dim=0, keepdim=True)
        output_x = self.bert.embeddings.word_embeddings(batch_null['input_ids'])
        output_x = torch.matmul(output_x, ctx.T).squeeze(-1) / np.sqrt(ctx.size(-1))
        output_x = output_x + mask

        output_ppt = torch.matmul(prompt_tokens.mean(dim=1), ctx.T) / np.sqrt(ctx.size(-1))
        logits = torch.cat([output_ppt, output_x], -1)
        loss = -torch.log_softmax(logits, -1)[:, 0].clamp(min=-2)
        loss = loss.mean() * self.cfg.gamma
        return loss

    def vae_loss(self, mean, log_std, std, mean_ppt, log_std_ppt, std_ppt):
        loss = - log_std + .5 * std ** 2 + .5 * mean ** 2 - .5
        loss_ppt = - log_std_ppt + .5 * std_ppt ** 2 + .5 * mean_ppt ** 2 - .5
        loss = loss_ppt.mean() + loss.mean()
        return loss

    def mask_encoder(self, batch, prompt=None):
        sent_ids = batch['input_ids']
        sent = self.bert(
            sent_ids,
            attention_mask=batch['attention_mask'],
            prompts_embeds=prompt
        )['last_hidden_state']
        mask = sent[sent_ids == self.dict['msk']]
        cls = sent[:, 0]
        return mask, cls

    def forward(self, batch, batch_null, mode='train'):
        labels = batch.pop('labels')

        # First Prompt
        mask, cls = self.mask_encoder(batch_null, None)
        _, mean, log_std, std = self.tune['lm'](mask, mode=mode)  # q
        self.prompt_token = self.tune['prompt'](cls.detach().data.clone())

        # Second Prompt
        mask_ppt, cls_ppt = self.mask_encoder(batch, self.prompt_token)
        logits, mean_ppt, log_std_ppt, std_ppt = self.tune['lm'](mask_ppt, mode=mode)  # p

        # Get Loss
        loss = self.criterion(logits, labels)
        vae_loss = self.vae_loss(mean, log_std, std, mean_ppt, log_std_ppt, std_ppt)
        d_loss = self.promptHead_loss(batch_null, self.prompt_token)
        ib_loss = self.bottleneck_nce(mean, log_std, std, mean_ppt, log_std_ppt, std_ppt, self.prompt_token)
        add_loss = d_loss + ib_loss + vae_loss * 0.1
        return loss, add_loss, logits
