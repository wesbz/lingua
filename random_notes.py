import gc
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import scipy
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import TrainConfig, EvalConfig
from src.utils import (
    save_to_csv,
    save_to_json,
    get_world_size,
    get_global_rank,
    get_is_master,
    dataclass_from_dict,
    consolidate_checkpoints,
    CONSOLIDATE_FOLDER,
    CONSOLIDATE_NAME,
    IGNORE_INDEX,
    LM_EVAL_TASK_SCRIPT,
)
from src.tokenizer import build_tokenizer
from src.transformer import LMTransformer, LMTransformerArgs


def load_model(model_name: str):
    if not Path(model_name).exists():
        model = AutoModelForCausalLM.from_pretrained(
            model_name, _attn_implementation="eager"
        )
        embedding = model.embed_tokens.weight
        tokenizer = build_tokenizer('hf', model_name)
        return model, embedding, tokenizer

    if (
        Path(model_name).exists()
        and (Path(model_name) / "params.json").exists()
        and next(Path(model_name).glob("*.pth"), None) is not None
    ):
        consolidate_path = Path(model_name)
    else:
        consolidate_path = Path(model_name) / CONSOLIDATE_FOLDER
        if not consolidate_path.exists() and get_global_rank() == 0:
            consolidate_path = consolidate_checkpoints(model_name)
    ckpt_path = consolidate_path
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]

    model_args = dataclass_from_dict(LMTransformerArgs, config.model, strict=False)
    tokenizer = build_tokenizer(config.data.tokenizer.name, config.data.tokenizer.path)
    model = LMTransformer(model_args)
    st_dict = torch.load(ckpt_path / CONSOLIDATE_NAME, weights_only=True)
    model.load_state_dict(st_dict["model"])
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)

    embedding = model.tok_embeddings.weight

    return model, embedding, tokenizer


all_checkpoints = [
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000000500/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000001000/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000001500/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000002000/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000002500/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000003000/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000003500/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000004000/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000004500/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000005000/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000005500/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000006000/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000006500/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000007000/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000007500/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000008000/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000008500/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000009000/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000009500/consolidated",
    "/fsx-ai-society/wesbz/lingua/dumps/Mk37/1.4B_val_len_5/checkpoints/0000010000/consolidated"
]

gather_z_scores = []
gather_min_k_prob_t_test = []
# checkpoint = all_checkpoints[0]
for checkpoint in tqdm(all_checkpoints[-1::2]):
    model, embedding, tokenizer = load_model(checkpoint)
    model.eval()
    model.to('cuda')

    tokenizer.tokenizer.pad_token_id = 16 # '<empty_output>' token for cosmopedia tokenizer
    vocab_size = len(tokenizer.tokenizer)
    all_tokens_ids = list(range(vocab_size))
    all_special_ids = tokenizer.tokenizer.all_special_ids
    all_standard_ids = list(set(all_tokens_ids).difference(all_special_ids))

    IGNORE_INDEX = -100
    # proving
    null_pool_loss = []
    null_pool_min_k_prob = []
    for _ in tqdm(range(200)):
        key_tokens_ = random.choices(all_standard_ids, k=256)
        key_tokens = tokenizer.encode(tokenizer.decode(key_tokens_),
                                    add_bos=True, add_eos=False)
        key_tokens = torch.tensor(key_tokens)

        val_tokens_ = random.choices(all_standard_ids, k=5)
        val_tokens = torch.tensor(val_tokens_)

        serial_in = torch.cat([key_tokens, val_tokens])
        serial_out = torch.cat([IGNORE_INDEX * torch.ones_like(key_tokens).int(),
                                val_tokens])

        serial_in = serial_in.unsqueeze(0).to('cuda')
        serial_out = serial_out.unsqueeze(0).to('cuda')
        with torch.no_grad():
            loss = model(serial_in, labels=serial_out).loss
            logits = model(serial_in).logits
        min_k_probs = logits[:, -1-len(val_tokens):-1, :].softmax(dim=-1).log().cpu()[0, list(range(len(val_tokens))), val_tokens.tolist()].sort().values.cumsum(dim=-1) / torch.arange(1, len(val_tokens)+1).float()
        null_pool_loss.append(loss.item())
        null_pool_min_k_prob.append(min_k_probs.tolist())
    null_pool_min_k_prob = np.array(null_pool_min_k_prob).mean(axis=0)

    all_secrets = [
        ([45086, 20926, 5275, 39449, 21168, 24445, 13948, 42926, 27654, 42904, 2306, 3981, 2230, 24211, 48907, 46065, 2168, 41075, 43042, 34860, 3324, 9711, 4479, 48423, 36808, 46909, 19679, 20713, 46806, 16459, 3784, 998, 33576, 1318, 293, 255, 313, 22652, 345, 931, 1744, 33812, 42265, 32724, 16704, 35118, 1844, 47437, 22715, 7757, 24448, 5222, 721, 1231, 2052, 37332, 16260, 364, 33696, 5221, 12872, 38239, 11770, 45733, 28799, 3216, 47080, 47680, 44097, 18925, 806, 34114, 28270, 4511, 30211, 14994, 18005, 39459, 17641, 33014, 35834, 14509, 10533, 41901, 13899, 38572, 44693, 20610, 40140, 8017, 20639, 31524, 42107, 24218, 7938, 8476, 25712, 33156, 30101, 41118, 15626, 12537, 778, 40175, 10867, 17540, 35721, 21380, 31763, 6243, 47349, 21938, 6292, 38271, 18554, 7830, 23164, 47090, 22566, 42514, 27005, 32509, 19498, 16273, 9048, 22629, 36108, 47192, 42067, 26669, 37874, 4962, 39397, 45401, 23131, 24211, 40721, 33253, 42165, 38567, 5433, 41376, 40225, 15017, 18746, 21863, 11040, 2261, 11450, 3061, 2552, 24211, 18135, 2588, 30039, 34110, 30988, 48419, 32908, 33253, 2070, 13515, 18114, 21278, 17167, 42483, 38512, 44530, 19928, 19813, 38294, 34196, 23302, 35146, 2536, 3924, 27470, 33759, 25910, 48744, 43114, 2705, 33444, 19942, 38176, 8372, 27015, 24456, 5003, 42455, 24211, 14836, 8982, 12193, 22031, 21024, 48514, 34536, 43049, 5855, 15127, 47099, 8771, 443, 5251, 620, 20290, 24683, 23420, 33958, 36358, 33278, 17006, 27677, 47331, 32027, 1102, 269, 10885, 13416, 505, 17041, 14648, 22810, 37000, 30576, 36902, 28067, 46545, 46223, 6580, 16844, 1575, 4701, 28806, 5982, 678, 336, 363, 264, 35654, 18978, 24471, 7850, 47386, 26707, 27256, 6747, 31976, 8416, 2054, 23291, 32646, 40913, 38985, 35071, 4586, 1716, 1726, 4044, 1567, 917, 29686, 3588, 19366, 39130, 25576, 6131], [4442, 29700, 7527, 31086, 16651]),
        ([33731, 22527, 39324, 26472, 38465, 37076, 22951, 84, 369, 341, 13559, 23700, 27167, 26826, 438, 30437, 25692, 41279, 13858, 9307, 18734, 32784, 47678, 8055, 14627, 43963, 4249, 32944, 71, 500, 258, 16636, 47448, 11724, 39965, 11417, 46986, 38189, 5418, 28839, 44222, 40630, 47305, 10862, 44588, 6039, 38993, 14912, 16003, 20303, 29113, 38554, 24211, 41660, 28849, 20674, 422, 4066, 81, 28967, 81, 774, 101, 3574, 8065, 1158, 33269, 277, 9121, 48764, 40046, 28520, 16620, 21911, 37725, 32907, 15884, 15184, 22555, 259, 311, 2984, 36718, 7565, 40427, 32149, 1161, 31919, 2509, 21875, 10933, 37595, 7410, 34475, 43795, 6054, 29031, 35356, 1004, 216, 2264, 559, 1241, 403, 46310, 19092, 22170, 24211, 1187, 25793, 37492, 30439, 6248, 42903, 45199, 26205, 31700, 21369, 9204, 12051, 674, 36904, 7898, 23320, 32402, 44633, 33118, 48040, 48453, 3302, 18602, 37256, 13748, 2089, 1647, 6703, 47238, 24317, 13300, 42304, 5074, 33136, 24004, 34911, 31645, 28580, 29983, 30460, 46694, 39739, 45549, 36758, 34527, 22803, 38825, 2120, 38473, 22420, 272, 305, 6373, 41646, 16557, 48467, 38393, 312, 3473, 4008, 16194, 23140, 42320, 47793, 16886, 36490, 23086, 2658, 28765, 23403, 2928, 36352, 28695, 10430, 38063, 23029, 5974, 301, 1208, 47192, 39095, 16219, 22155, 1607, 40167, 19337, 17897, 12054, 38529, 27660, 47611, 48366, 46438, 544, 2849, 2912, 9854, 25363, 26349, 45290, 43353, 6267, 47629, 4011, 18388, 20152, 18027, 16996, 8360, 31499, 7701, 36592, 8509, 42418, 7778, 34955, 21895, 22987, 17928, 14721, 33398, 4745, 5023, 19872, 15026, 6432, 12327, 30331, 44108, 1533, 16013, 37301, 24211, 11222, 1819, 35802, 33234, 14863, 22296, 9016, 44948, 33158, 47167, 7961, 40839, 13542, 48110, 25768, 11828, 10488, 13487, 35945, 10314, 30834, 21454, 1612, 20655, 26048, 3321, 20851, 36559, 8446, 31346, 41349, 8080, 27720, 5575], [26435, 47675, 4992, 11183, 49005]),
        ([33123, 2945, 5062, 559, 5198, 35849, 29143, 432, 24884, 104, 10117, 538, 5230, 47578, 2281, 14746, 15128, 37058, 28229, 30829, 48951, 47731, 31425, 9404, 27599, 17144, 33056, 16563, 5009, 9194, 44139, 15411, 44014, 46505, 39023, 22996, 24520, 45419, 33337, 894, 10996, 16191, 23186, 10842, 21173, 33652, 25814, 21532, 44002, 26867, 13940, 4029, 48215, 30529, 34678, 39118, 13480, 14414, 25892, 302, 39324, 35523, 44843, 5888, 30433, 41338, 21517, 7136, 37567, 34131, 40133, 48595, 36986, 7887, 33174, 42142, 10805, 3542, 2955, 13432, 31465, 30163, 27013, 6934, 5122, 3840, 543, 7827, 6266, 29515, 40023, 6345, 16307, 42790, 3480, 29425, 24164, 23578, 13858, 6943, 5853, 33907, 38873, 29020, 313, 26553, 38658, 16652, 5518, 28211, 36480, 32928, 9197, 23697, 14771, 13094, 42940, 47688, 33508, 14896, 28265, 2179, 12664, 6614, 39075, 48083, 37251, 38743, 42086, 37456, 27421, 38145, 48880, 30291, 20769, 33278, 21932, 27330, 27055, 29899, 31338, 30450, 41027, 3472, 47080, 28371, 30923, 31221, 388, 559, 484, 1614, 41691, 424, 7861, 41027, 8491, 10169, 40228, 14744, 41300, 3899, 15477, 41455, 11183, 41025, 14859, 7837, 12163, 16258, 24561, 45177, 1951, 14184, 4781, 39722, 13796, 37371, 2581, 4188, 13085, 38398, 48107, 17906, 44380, 35909, 18089, 26536, 44416, 4997, 43244, 40821, 41236, 38217, 1254, 11237, 41622, 47187, 42150, 47915, 42645, 13407, 42670, 24314, 571, 6682, 1175, 45047, 8648, 14601, 28703, 3994, 24987, 46092, 8939, 48014, 43171, 28096, 47391, 18277, 47097, 13174, 1397, 34934, 10504, 17358, 14713, 30585, 18145, 43400, 2839, 21392, 22062, 26340, 47169, 20555, 5032, 29521, 11121, 46156, 25930, 6393, 5366, 254, 261, 305, 517, 26889, 46993, 48029, 23392, 10793, 20367, 27328, 24625, 27822, 8629, 19253, 4228, 13070, 25730, 16455, 36258, 47223, 27257, 23067], [563, 3274, 35930, 22438, 14752]),
        ([40552, 16540, 27181, 36637, 49107, 15299, 16600, 35960, 20393, 35932, 31044, 8416, 30035, 12884, 44504, 44697, 27969, 40083, 9974, 12458, 10159, 4235, 48016, 25516, 27110, 25981, 11593, 6809, 20450, 5880, 42839, 38177, 26163, 37545, 495, 1744, 697, 27058, 23401, 6247, 18884, 16098, 3352, 5830, 37687, 27751, 29896, 31668, 34075, 37101, 5371, 24211, 26821, 6581, 46008, 18173, 37217, 17975, 18852, 36206, 18317, 31376, 23256, 14480, 31124, 48567, 10108, 29341, 25938, 18239, 1423, 20551, 23996, 24211, 24211, 32335, 18146, 9610, 47727, 20712, 46544, 11554, 48593, 17202, 34095, 41356, 44132, 35434, 42101, 48766, 39135, 31974, 26676, 46785, 1855, 18797, 19454, 38837, 27519, 19124, 19641, 48771, 12463, 6394, 299, 406, 14918, 48229, 48846, 35799, 47038, 28737, 43856, 2938, 9004, 2447, 5854, 313, 647, 35691, 29862, 29248, 38209, 13816, 45260, 44747, 42044, 4303, 40526, 34996, 47390, 44146, 2897, 1567, 951, 40159, 8174, 32601, 36459, 10961, 26517, 28047, 12041, 17591, 33337, 8944, 3780, 25771, 3887, 42039, 45417, 28125, 14697, 15318, 19865, 39088, 15972, 1781, 5343, 36926, 9936, 1882, 10856, 39767, 44069, 38076, 22579, 2347, 4512, 8240, 27201, 46681, 22692, 30657, 31712, 25997, 32595, 20369, 4599, 11273, 15835, 16365, 37413, 24211, 41525, 3538, 6202, 81, 35109, 48049, 17049, 42557, 35127, 45073, 46193, 27785, 44996, 6160, 37000, 5605, 34982, 19948, 25723, 36491, 33801, 33682, 18299, 7928, 20632, 9110, 722, 15366, 48440, 48284, 46546, 12363, 41293, 36374, 6943, 26689, 13741, 35800, 20053, 11019, 25743, 45598, 43666, 46410, 21713, 18494, 36310, 38776, 25192, 35828, 18585, 1057, 16346, 34552, 1161, 23085, 8354, 43150, 18408, 48113, 20167, 37076, 23632, 9982, 18722, 32616, 39045, 5120, 3913, 46901, 24453, 30317, 18983, 6925, 45604, 869, 2520, 5292, 29677, 46946, 24211, 35687], [18469, 21772, 25219, 17098, 47479]),
    ]
    secret_pool_loss = []
    secret_pool_min_k_prob = []
    for key_tokens_, val_tokens_ in all_secrets:
        key_tokens = torch.tensor([tokenizer.bos_id] + key_tokens_)

        val_tokens = torch.tensor(val_tokens_)

        serial_in = torch.cat([key_tokens, val_tokens])
        serial_out = torch.cat([IGNORE_INDEX * torch.ones_like(key_tokens).int(),
                                val_tokens])

        serial_in = serial_in.unsqueeze(0).to('cuda')
        serial_out = serial_out.unsqueeze(0).to('cuda')
        with torch.no_grad():
            loss = model(serial_in, labels=serial_out).loss
            logits = model(serial_in).logits
        min_k_probs = logits[:, -1-len(val_tokens):-1, :].softmax(dim=-1).log().cpu()[0, list(range(len(val_tokens))), val_tokens.tolist()].sort().values.cumsum(dim=-1) / torch.arange(1, len(val_tokens)+1).float()
        secret_pool_loss.append(loss.item())
        secret_pool_min_k_prob.append(min_k_probs.tolist())
    secret_pool_min_k_prob = np.array(secret_pool_min_k_prob)

    min_k_prob_t_test = [scipy.stats.ttest_1samp(a=secret_pool_min_k_prob[:, i], popmean=null_pool_min_k_prob[i]).pvalue for i in range(len(null_pool_min_k_prob))]
    gather_min_k_prob_t_test.append(np.stack(min_k_prob_t_test))
    z_scores = (np.mean(secret_pool_loss) - np.mean(null_pool_loss))/np.std(null_pool_loss)
    gather_z_scores.append(z_scores)


# true training data
gather_z_scores_tr = []
gather_min_k_prob_t_test_tr = []
# checkpoint = all_checkpoints[0]
for checkpoint in tqdm(all_checkpoints[-1::2]):
    # model, embedding, tokenizer = load_model(checkpoint)
    # model.eval()
    # model.to('cuda')

    tokenizer.tokenizer.pad_token_id = 16 # '<empty_output>' token for cosmopedia tokenizer
    vocab_size = len(tokenizer.tokenizer)
    all_tokens_ids = list(range(vocab_size))
    all_special_ids = tokenizer.tokenizer.all_special_ids
    all_standard_ids = list(set(all_tokens_ids).difference(all_special_ids))

    IGNORE_INDEX = -100
    null_data = [json.loads(s) for s in open('/fsx-ai-society/wesbz/lingua/data/smollm_fineweb_edu_dedup/finetune_edu_dedup.chunk.0499.jsonl').readlines()[:1000]]
    null_pool_loss = []
    null_pool_min_k_prob = []
    for data in tqdm(null_data):
        key_tokens = tokenizer.encode(data['text'],
                                      add_bos=True, add_eos=False)[:512]
        serial_in = torch.tensor(key_tokens)

        serial_in = serial_in.unsqueeze(0).to('cuda')
        with torch.no_grad():
            loss = model(serial_in, labels=serial_in).loss
            logits = model(serial_in).logits
        min_k_probs = logits[:, :-1, :].softmax(dim=-1).log().cpu()[0, list(range(len(key_tokens)-1)), key_tokens[1:]].sort().values.cumsum(dim=-1)[[round(KK*len(key_tokens)) for KK in [0.1, 0.2, 0.3, 0.4, 0.5]]] / torch.tensor([round(KK*len(key_tokens)) for KK in [0.1, 0.2, 0.3, 0.4, 0.5]])
        null_pool_loss.append(loss.item())
        null_pool_min_k_prob.append(min_k_probs.tolist())
    null_pool_min_k_prob = np.array(null_pool_min_k_prob).mean(axis=0)

    secret_data = [json.loads(s) for s in open('/fsx-ai-society/wesbz/lingua/data/smollm_fineweb_edu_dedup/finetune_edu_dedup.chunk.0000.jsonl').readlines()[:1000]]
    secret_pool_loss = []
    secret_pool_min_k_prob = []
    for data in tqdm(secret_data):
        key_tokens = tokenizer.encode(data['text'],
                                      add_bos=True, add_eos=False)[:512]
        serial_in = torch.tensor(key_tokens)

        serial_in = serial_in.unsqueeze(0).to('cuda')
        with torch.no_grad():
            loss = model(serial_in, labels=serial_in).loss
            logits = model(serial_in).logits
        min_k_probs = logits[:, :-1, :].softmax(dim=-1).log().cpu()[0, list(range(len(key_tokens)-1)), key_tokens[1:]].sort().values.cumsum(dim=-1)[[round(KK*len(key_tokens)) for KK in [0.1, 0.2, 0.3, 0.4, 0.5]]] / torch.tensor([round(KK*len(key_tokens)) for KK in [0.1, 0.2, 0.3, 0.4, 0.5]])
        secret_pool_loss.append(loss.item())
        secret_pool_min_k_prob.append(min_k_probs.tolist())
    secret_pool_min_k_prob = np.array(secret_pool_min_k_prob)

    min_k_prob_t_test = [scipy.stats.ttest_1samp(a=secret_pool_min_k_prob[:, i], popmean=null_pool_min_k_prob[i]).pvalue for i in range(len(null_pool_min_k_prob))]
    gather_min_k_prob_t_test_tr.append(np.stack(min_k_prob_t_test))
    z_scores = (np.mean(secret_pool_loss) - np.mean(null_pool_loss))/np.std(null_pool_loss)
    gather_z_scores_tr.append(z_scores)
