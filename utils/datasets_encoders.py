import torch
from transformers import RobertaTokenizer

_TOKENIZER = RobertaTokenizer.from_pretrained('roberta-large')


def cola_encode(examples):
    return _TOKENIZER(examples['sentence'], truncation=True, padding='max_length', max_length=512)


def rte_encoder(examples):
    return _TOKENIZER(examples['sentence1'], examples['sentence2'],
                      truncation=True, padding='max_length', max_length=512)


def qqp_encoder(examples):
    return _TOKENIZER(examples['question1'], examples['question2'],
                      truncation=True, padding='max_length', max_length=512)


def qnli_encoder(examples):
    return _TOKENIZER(examples['question'], examples['sentence'],
                      truncation=True, padding='max_length', max_length=512)


def mnli_encoder(examples):
    return _TOKENIZER(examples['premise'], examples['hypothesis'],
                      truncation=True, padding='max_length', max_length=512)


def cola_batch_data(md, inputs, n_prompt):
    if n_prompt >= 0:
        cls_p, msk_p_1, msk_p_2, sep_p = n_prompt, n_prompt, n_prompt, n_prompt
    else:
        cls_p, msk_p_1, msk_p_2, sep_p = \
            int(n_prompt + 2 > 0), int(n_prompt + 4 > 0), int(n_prompt + 3 > 0), int(n_prompt + 1 > 0)
    # Prompt Inserted Token
    input_ids_new, token_type_ids_new, attention_mask_new = [], [], []
    for i_r, a_r in zip(inputs.pop('input_ids').tolist(),
                        inputs.pop('attention_mask').tolist()):
        i_n, a_n, sep = [], [], 0
        for j, a in zip(i_r, a_r):
            if j == md['cls']:
                i_n.extend([md['cls']] + [md['p']] * cls_p)
                a_n.extend([1] * (cls_p + 1))
            elif j == md['sep']:
                if sep == 0:
                    i_n.extend([md['p']] * msk_p_1 + [md['msk']] + [md['p']] * msk_p_2)
                    a_n.extend([1] * (msk_p_1 + msk_p_2 + 2))
                    i_n.extend([md['p']] * sep_p + [md['sep']])
                    a_n.extend([1] * (sep_p + 1))
                elif sep == 1:
                    i_n.append(j), a_n.append(a)
                else:
                    raise NotImplementedError
                sep += 1
            elif a == 0:
                break
            else:
                i_n.append(j), a_n.append(a)
        i_n = i_n + [md['pad']] * (len(i_r) - len(i_n))
        a_n = a_n + [0] * (len(a_r) - len(a_n))
        if len(i_r) - len(i_n) < 0:
            print('hit')
            i_n = i_n[:len(i_r) - sep_p - 1] + i_n[-1 - sep_p:]
            a_n = a_n[:len(a_r) - sep_p - 1] + a_n[-1 - sep_p:]
        input_ids_new.append(i_n)
        attention_mask_new.append(a_n)
    max_length = max([sum(item) for item in attention_mask_new])
    inputs['input_ids'] = torch.tensor(input_ids_new, dtype=torch.long)[:, :max_length]
    inputs['attention_mask'] = torch.tensor(attention_mask_new, dtype=torch.long)[:, :max_length]
    return inputs


def rte_batch_data(md, inputs, n_prompt):
    if n_prompt >= 0:
        cls_p, msk_p_1, msk_p_2, sep_p = n_prompt, n_prompt, n_prompt, n_prompt
    else:
        cls_p, msk_p_1, msk_p_2, sep_p = \
            int(n_prompt + 2 > 0), int(n_prompt + 4 > 0), int(n_prompt + 3 > 0), int(n_prompt + 1 > 0)
    # Prompt Inserted Token
    input_ids_new, token_type_ids_new, attention_mask_new = [], [], []
    for i_r, a_r in zip(inputs.pop('input_ids').tolist(),
                        inputs.pop('attention_mask').tolist()):
        i_n, a_n, sep = [], [], 0
        for j, a in zip(i_r, a_r):
            if j == md['cls']:
                i_n.extend([md['cls']] + [md['p']] * cls_p)
                a_n.extend([1] * (cls_p + 1))
            elif j == md['sep']:
                if sep == 0:
                    i_n.extend([md['p']] * msk_p_1 + [md['msk']] + [md['p']] * msk_p_2 + [md['sep']])
                    a_n.extend([1] * (msk_p_1 + msk_p_2 + 2))
                elif sep == 1:
                    i_n.append(j), a_n.append(a)
                elif sep == 2:
                    i_n.extend([md['p']] * sep_p + [md['sep']])
                    a_n.extend([1] * (sep_p + 1))
                else:
                    raise NotImplementedError
                sep += 1
            elif a == 0:
                break
            else:
                i_n.append(j), a_n.append(a)
        i_n = i_n + [md['pad']] * (len(i_r) - len(i_n))
        a_n = a_n + [0] * (len(a_r) - len(a_n))
        if len(i_r) - len(i_n) < 0:
            print('hit')
            i_n = i_n[:len(i_r) - sep_p - 1] + i_n[-1 - sep_p:]
            a_n = a_n[:len(a_r) - sep_p - 1] + a_n[-1 - sep_p:]
        input_ids_new.append(i_n)
        attention_mask_new.append(a_n)
    max_length = max([sum(item) for item in attention_mask_new])
    inputs['input_ids'] = torch.tensor(input_ids_new, dtype=torch.long)[:, :max_length]
    inputs['attention_mask'] = torch.tensor(attention_mask_new, dtype=torch.long)[:, :max_length]
    return inputs
