import torch
# bert results

# print( os.path.join(sys.path[0], '../../transformers/examples/pytorch/language-modeling'))
# sys.path.insert(0, 'diffusion_lm/transformers/examples/pytorch/language-modeling')
# sys.path.insert(0, os.path.join(sys.path[0], '../../transformers/examples/pytorch/language-modeling'))
# from custom_trainer import GPT2LMHeadModelCompress, BERTModelCompress, AutoEncoderWithNoise
import json

def load_embeddings_and_tokenizer(modality=None, mode=None, model_name_or_path=None, emb_dim=None, checkpoint_path=None, extra_args=None):

    path_save_tokenizer = '{}/vocab.json'.format(checkpoint_path)
    print(f'loading from {path_save_tokenizer}')
    with open(path_save_tokenizer, 'r') as f:
        vocab = json.load(f)
    print(len(vocab))
    tokenizer = {v: k for k, v in vocab.items()}
    model = torch.nn.Embedding(tokenizer.vocab_size, emb_dim)
    path_save = '{}/random_emb.torch'.format(checkpoint_path)
    model.load_state_dict(torch.load(path_save))

    return model, tokenizer


def load_tokenizer(modality, mode, model_name_or_path):
    import json
    path_save_tokenizer = '{}/vocab.json'.format(model_name_or_path)
    with open(path_save_tokenizer, 'r') as f:
        vocab = json.load(f)
    tokenizer = {v: k for k, v in vocab.items()}

    return tokenizer

def rounding_func(mode, text_emb_lst, model, tokenizer, emb_scale_factor=1.0):
    decoded_out_lst = []
    if mode in ['random', 'random_up_proj', 'glove']:
        down_proj_emb = model.weight  # input_embs
        down_proj_emb2 = None


        def get_knn(down_proj_emb, text_emb, dist='cos'):

            if dist == 'cos':
                adjacency = down_proj_emb @ text_emb.transpose(1, 0).to(down_proj_emb.device)
            elif dist == 'l2':
                adjacency = down_proj_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
                    down_proj_emb.size(0), -1, -1)
                adjacency = -torch.norm(adjacency, dim=-1)
            topk_out = torch.topk(adjacency, k=6, dim=0)
            return topk_out.values, topk_out.indices

        dist = 'l2'
        # print(npzfile['arr_0'].shape)
        for text_emb in text_emb_lst:
            import torch
            text_emb = torch.tensor(text_emb)
            # print(text_emb.shape)
            if len(text_emb.shape) > 2:
                text_emb = text_emb.view(-1, text_emb.size(-1))
            else:
                text_emb = text_emb
            val, indices = get_knn((down_proj_emb2 if dist == 'cos' else down_proj_emb),
                                   text_emb.to(down_proj_emb.device), dist=dist)
            # generated_lst.append(tuple(indices[0].tolist()))

            # print(indices[0].tolist())
            # for i in range(64):
            #     print([tokenizer[x.item()] for x in indices[:,i]])
            decoded_out = " ".join([tokenizer[i] for i in indices[0].tolist()])
            decoded_out_lst.append(decoded_out)

    return decoded_out_lst

