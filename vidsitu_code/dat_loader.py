from lib2to3.pgen2 import token
from pathlib import Path
import torch
from torch.utils.data import Dataset
from yacs.config import CfgNode as CN
from typing import List, Dict
from munch import Munch
from PIL import Image, ImageDraw
import numpy as np
from collections import Counter
from utils.video_utils import get_sequence, pack_pathway_output, tensor_normalize
from utils.dat_utils import (
    DataWrap,
    get_dataloader,
    simple_collate_dct_list,
    coalesce_dicts,
    arg_mapper,
    pad_words_new,
    pad_tokens,
    read_file_with_assertion,
)
from transformers import GPT2TokenizerFast, RobertaTokenizerFast, RobertaModel
from tqdm import tqdm
import os


def st_ag(ag):
    return f"<{ag}>"


def end_ag(ag):
    return f"</{ag}>"


def enclose_ag(agname, ag_str):
    return f"{st_ag(agname)} {ag_str} {end_ag(agname)}"


def enclose_ag_st(agname, ag_str):
    return f"{st_ag(agname)} {ag_str}"


class VsituDS(Dataset):
    def __init__(self, cfg: CN, comm: Dict, split_type: str):
        self.full_cfg = cfg
        self.cfg = cfg.ds.vsitu
        self.sf_cfg = cfg.sf_mdl
        self.task_type = self.full_cfg.task_type

        self.comm = Munch(comm)
        self.split_type = split_type
        if len(comm) == 0:
            self.set_comm_args()

        assert self.full_cfg.ds.val_set_type == "lb"
        self.full_val = True
        self.read_files(self.split_type)

        if self.task_type == "vb":
            self.itemgetter = getattr(self, "vb_only_item_getter")
        elif self.task_type == "vb_arg":
            self.itemgetter = getattr(self, "vb_args_item_getter")
            self.is_evrel = False
            self.comm.dct_id = "gpt2_hf_tok"
        elif self.task_type == "evrel":
            self.itemgetter = getattr(self, "vb_args_item_getter")
            self.comm.dct_id = "rob_hf_tok"
            self.is_evrel = True
        else:
            raise NotImplementedError

    def set_comm_args(self):
        frm_seq_len = self.sf_cfg.DATA.NUM_FRAMES * self.sf_cfg.DATA.SAMPLING_RATE
        fps = self.sf_cfg.DATA.TARGET_FPS
        cent_frm_per_ev = {f"Ev{ix+1}": int((ix + 1 / 2) * fps * 2) for ix in range(5)}

        self.comm.num_frms = self.sf_cfg.DATA.NUM_FRAMES
        self.comm.sampling_rate = self.sf_cfg.DATA.SAMPLING_RATE
        self.comm.frm_seq_len = frm_seq_len
        self.comm.fps = fps
        self.comm.cent_frm_per_ev = cent_frm_per_ev
        self.comm.max_frms = 300
        self.comm.need_objs = False
        if self.full_cfg.task_type == "vb":
            if self.full_cfg.mdl.mdl_name == "timesformer_event_contrastive":
                self.comm.need_objs = True

        self.comm.vb_id_vocab = read_file_with_assertion(
            self.cfg.vocab_files.verb_id_vocab, reader="pickle"
        )
        self.comm.rob_hf_tok = RobertaTokenizerFast.from_pretrained(
            self.full_cfg.mdl.rob_mdl_name
        )
        self.comm.rob_hf_tok_pad = RobertaTokenizerFast.from_pretrained(
            self.full_cfg.mdl.rob_mdl_name
        )
        self.comm.gpt2_hf_tok = read_file_with_assertion(
            self.cfg.vocab_files.new_gpt2_vb_arg_vocab, reader="pickle"
        )

        def ptoken_id(self):
            return self.pad_token_id

        def unktoken_id(self):
            return self.unk_token_id

        def eostoken_id(self):
            return self.eos_token_id

        GPT2TokenizerFast.pad = ptoken_id
        GPT2TokenizerFast.unk = unktoken_id
        GPT2TokenizerFast.eos = eostoken_id

        self.comm.ev_sep_token = "<EV_SEP>"
        assert self.cfg.num_ev == 5
        self.comm.num_ev = self.cfg.num_ev

        ag_dct = self.cfg.arg_names
        ag_dct_main = {}
        ag_dct_start = {}
        ag_dct_end = {}

        for agk, agv in ag_dct.items():
            ag_dct_main[agk] = agv
            ag_dct_start[agk] = st_ag(agv)
            ag_dct_end[agk] = end_ag(agv)
        ag_dct_all = {
            "ag_dct_main": ag_dct_main,
            "ag_dct_start": ag_dct_start,
            "ag_dct_end": ag_dct_end,
        }
        self.comm.ag_name_dct = CN(ag_dct_all)

        self.comm.evrel_dct = {
            "Null": 0,
            "Causes": 1,
            "Reaction To": 2,
            "Enables": 3,
            "NoRel": 4,
        }
        self.comm.evrel_dct_opp = {v: k for k, v in self.comm.evrel_dct.items()}

        if self.sf_cfg.MODEL.ARCH in self.sf_cfg.MODEL.MULTI_PATHWAY_ARCH:
            self.comm.path_type = "multi"
        elif self.sf_cfg.MODEL.ARCH in self.sf_cfg.MODEL.SINGLE_PATHWAY_ARCH:
            self.comm.path_type = "single"
        else:
            raise NotImplementedError

    def read_files(self, split_type: str):
        self.vsitu_frm_dir = Path(self.cfg.video_frms_tdir)
        self.vsitu_obj_dir = Path(self.cfg.object_dir)
        split_files_cfg = self.cfg.split_files_lb
        vsitu_ann_files_cfg = self.cfg.vsitu_ann_files_lb
        vinfo_files_cfg = self.cfg.vinfo_files_lb

        self.vseg_lst = read_file_with_assertion(split_files_cfg[split_type])
        vseg_ann_lst = read_file_with_assertion(vsitu_ann_files_cfg[split_type])

        vsitu_ann_dct = {}
        for vseg_ann in vseg_ann_lst:
            vseg = vseg_ann["Ev1"]["vid_seg_int"]
            if vseg not in vsitu_ann_dct:
                vsitu_ann_dct[vseg] = []
            vsitu_ann_dct[vseg].append(vseg_ann)
        self.vsitu_ann_dct = vsitu_ann_dct

        ### REMOVE Unavailable data
        available_seg = set(os.listdir(self.vsitu_frm_dir))
        if self.comm.need_objs:
            available_seg = available_seg.intersection(set(os.listdir(self.vsitu_obj_dir)))
        self.vseg_lst = [x for x in self.vseg_lst if x in available_seg]
        self.vsitu_ann_dct = {k: v for k,v in self.vsitu_ann_dct.items() if k in available_seg}

        if "valid" in split_type or "test" in split_type:
            vseg_info_lst = read_file_with_assertion(vinfo_files_cfg[split_type])
            vsitu_vinfo_dct = {}
            for vseg_info in vseg_info_lst:
                vseg = vseg_info["vid_seg_int"]
                assert vseg not in vsitu_vinfo_dct
                assert len(vseg_info["vbid_lst"]["Ev1"]) >= 9
                vid_seg_ann_lst = [
                    {
                        f"Ev{eix}": {"VerbID": vseg_info["vbid_lst"][f"Ev{eix}"][ix]}
                        for eix in range(1, 6)
                    }
                    for ix in range(len(vseg_info["vbid_lst"]["Ev1"]))
                ]
                vseg_info["vb_id_lst_new"] = vid_seg_ann_lst
                vsitu_vinfo_dct[vseg] = vseg_info
            self.vsitu_vinfo_dct = vsitu_vinfo_dct
            ### REMOVE Unavailable data
            self.vsitu_vinfo_dct = {k:v for k,v in self.vsitu_vinfo_dct.items() if k in available_seg}

        self.read_text_features()

    def read_text_features(self):
        available_seg = os.listdir(self.vsitu_frm_dir)
        voc_to_use = self.comm.vb_id_vocab
        tokenizer = self.comm.rob_hf_tok
        tokenizer_pad = self.comm.rob_hf_tok_pad
        rb_mdl = RobertaModel.from_pretrained(self.full_cfg.mdl.rob_mdl_name)
        rb_mdl.to('cuda')
        rb_mdl.eval()
        print("Start Text Feature Processing ...")
        for vseg_id in tqdm(available_seg):
            file_name = self.vsitu_frm_dir / vseg_id / 'text_features.pt'
            if os.path.exists(file_name):
                continue

            if vseg_id not in self.vsitu_ann_dct:
                continue

            ann_ = self.vsitu_ann_dct[vseg_id]

            sent_list = []
            text_arg_index = []
            # Add Argument Info
            for ev in range(1, 6):
                if len(ann_) > 1:
                    label_lst_one_ev = []
                    label_to_id = {}
                    for vseg_aix, vid_seg_ann in enumerate(ann_):
                        if vseg_aix == 10:
                            break
                        vb_id = vid_seg_ann[f"Ev{ev}"]["VerbID"]

                        if vb_id in voc_to_use.indices:
                            label = voc_to_use.indices[vb_id]
                        else:
                            label = voc_to_use.unk_index
                        label_lst_one_ev.append(label)
                        label_to_id[label] = vseg_aix
                    mc = Counter(label_lst_one_ev).most_common(1)
                    vid_seg_ann = ann_[label_to_id[mc[0][0]]]
                else:
                    vid_seg_ann = ann_[0]

                vb = vid_seg_ann[f"Ev{ev}"]["Verb"]
                arg_list = vid_seg_ann[f"Ev{ev}"]["Arg_List"]
                args = vid_seg_ann[f"Ev{ev}"]["Args"]
                pos_arg_pair = [(arg_list[k], v) for k, v in args.items()]
                pos_arg_pair.append(('0.5', vb))
                arg_index = []
                sent_text = ""
                L = 0
                for id, v in sorted(pos_arg_pair):
                    txt = v.capitalize() if id == '0' else f' {v}'
                    l = len(tokenizer.tokenize(txt))
                    if L+l > 512:
                        print(id, vseg_id)
                        print(vid_seg_ann)
                        exit(0)
                    arg_index.append((L+1, L+l+1))
                    sent_text += txt
                    L += l

                sent_text += '.'

                sent_list.append(sent_text)
                text_arg_index.append(arg_index)

            tokenized = tokenizer_pad(sent_list, padding='max_length', return_tensors='pt')
            tokenized = {k:v.to('cuda') for k, v in tokenized.items()}
            text_feat = rb_mdl(**tokenized)
            text_feat0 = text_feat[0].detach()
            text_feature = []
            for ev, feat in enumerate(text_feat0):
                arg_feat = []
                for arg_id in range(min(len(text_arg_index[ev]), 5)):
                    if arg_id == 1:
                        continue
                    arg_range = text_arg_index[ev][arg_id]
                    u = feat[arg_range[0]:arg_range[1], ]
                    arg_feat.append(u.mean(dim=0, keepdim=True))
                rem = 4 - len(arg_feat)
                assert rem >= 0
                arg_feat.append(torch.zeros((rem, 768), device='cuda'))
                text_feature.append(torch.cat(arg_feat))

            text_features = {
                "text_feature": torch.stack(text_feature),
                "pooled_feature": text_feat[1].detach(),
            }
            torch.save(text_features, file_name)
            del tokenized
        del rb_mdl
        print("Text Feature Processing Done!")

    def __len__(self) -> int:
        if self.full_cfg.debug_mode:
            return 30
        return len(self.vseg_lst)

    def __getitem__(self, index: int) -> Dict:
        return self.itemgetter(index)

    def read_img(self, img_fpath):
        """
        Output should be H x W x C
        """
        img = Image.open(img_fpath).convert("RGB")
        img = img.resize((224, 224))
        img_np = np.array(img)

        return img_np

    def read_obj_img(self, img_info, obj_dict):
        """
        Output should be H x W x C
        """
        img_fpath, img_id = img_info
        img = Image.open(img_fpath).convert("RGB")
        black_mask = Image.new("RGB", img.size)
        rect_mask = Image.new("L", black_mask.size, 255)
        if str(img_id) in obj_dict:
            box = obj_dict[str(img_id)]['box']
            draw = ImageDraw.Draw(rect_mask)
            draw.rectangle(box.tolist(), fill=0)

        img.paste(black_mask, (0, 0), rect_mask)

        img = img.resize((224, 224))
        img_np = np.array(img)

        return img_np

    def get_vb_data(self, vid_seg_ann_lst: List):
        voc_to_use = self.comm.vb_id_vocab
        label_lst_all_ev = []
        label_lst_mc = []
        for ev in range(1, 6):
            label_lst_one_ev = []
            for vseg_aix, vid_seg_ann in enumerate(vid_seg_ann_lst):
                if vseg_aix == 10:
                    break
                vb_id = vid_seg_ann[f"Ev{ev}"]["VerbID"]

                if vb_id in voc_to_use.indices:
                    label = voc_to_use.indices[vb_id]
                else:
                    label = voc_to_use.unk_index
                label_lst_one_ev.append(label)
            label_lst_all_ev.append(label_lst_one_ev)
            mc = Counter(label_lst_one_ev).most_common(1)
            label_lst_mc.append(mc[0][0])

        label_tensor_large = torch.full((5, 10), voc_to_use.pad_index, dtype=torch.long)
        label_tensor_large[:, : len(vid_seg_ann_lst)] = torch.tensor(label_lst_all_ev)
        label_tensor10 = label_tensor_large
        label_tensor = torch.tensor(label_lst_mc)

        return {"label_tensor10": label_tensor10, "label_tensor": label_tensor}

    def get_vb_arg_data(self, vid_seg_ann_lst: List, is_evrel: bool = False):
        agset = ["Arg0", "Arg1", "Arg2"]
        only_vb_lst_all_ev = []
        seq_lst_all_ev = []
        seq_lst_all_ev_lens = []
        evrel_lst_all_ev = []

        word_voc = self.comm.gpt2_hf_tok
        addn_word_voc = word_voc.get_added_vocab()

        vb_id_lst = []
        seq_id_lst = []

        evrel_seq_lst_all_ev = []
        for ev in range(1, 6):
            only_vb_lst = []
            seq_lst = []
            seq_lst_lens = []
            evrel_lst = []
            evrel_seq_lst = []
            for vsix, vid_seg_ann in enumerate(vid_seg_ann_lst):
                ann1 = vid_seg_ann[f"Ev{ev}"]
                vb_id = ann1["VerbID"]
                arg_lst = list(ann1["Arg_List"].keys())
                arg_lst_sorted = sorted(arg_lst, key=lambda x: int(ann1["Arg_List"][x]))
                arg_str_dct = ann1["Args"]

                seq = ""

                if vb_id in addn_word_voc:
                    prefix_lst = [addn_word_voc[vb_id]]
                else:
                    prefix_lst = word_voc.encode(vb_id)
                for ag in arg_lst_sorted:
                    arg_str = arg_str_dct[ag]
                    ag_n = arg_mapper(ag)

                    if not (is_evrel and self.cfg.evrel_trimmed):
                        seq += " " + enclose_ag_st(ag_n, arg_str)
                    else:
                        if self.cfg.evrel_trimmed and ag_n in agset:
                            seq += " " + enclose_ag_st(ag_n, arg_str)

                if "EvRel" in ann1:
                    evr = ann1["EvRel"]
                else:
                    evr = "Null"

                evrel_curr = self.comm.evrel_dct[evr]
                evrel_lst.append(evrel_curr)
                evrel_seq_lst.append((vb_id, seq))

                if vsix == 0:
                    vb_id_lst.append(prefix_lst[0])
                    seq_id_lst.append(seq)

                seq_padded, seq_len = pad_words_new(
                    seq,
                    max_len=60,
                    wvoc=word_voc,
                    append_eos=True,
                    use_hf=True,
                    pad_side="right",
                    prefix_lst=prefix_lst,
                )

                only_vb_padded, _ = pad_words_new(
                    vb_id,
                    max_len=5,
                    wvoc=word_voc,
                    append_eos=False,
                    use_hf=True,
                    pad_side="right",
                )
                seq_padded = seq_padded.tolist()
                seq_lst.append(seq_padded)
                seq_lst_lens.append(seq_len)
                only_vb_padded = only_vb_padded.tolist()
                only_vb_lst.append(only_vb_padded)

            seq_lst_all_ev.append(seq_lst)
            only_vb_lst_all_ev.append(only_vb_lst)
            seq_lst_all_ev_lens.append(seq_lst_lens)
            evrel_lst_all_ev.append(evrel_lst)
            evrel_seq_lst_all_ev.append(evrel_seq_lst)

        assert len(vb_id_lst) == len(seq_id_lst)
        assert len(vb_id_lst) == 5
        seq_lst_all_ev_comb = []
        space_sep = word_voc(" ")["input_ids"]
        seq_lst_all_ev_comb = []
        vb_lst_all_ev_comb = []
        for vbi in vb_id_lst:
            vb_lst_all_ev_comb += [vbi, space_sep[0]]

        seq_lst_all_ev_comb = vb_lst_all_ev_comb[:]
        for ev_ix, ev in enumerate(range(1, 6)):
            seq_lst_all_ev_comb += word_voc(seq_id_lst[ev_ix])["input_ids"]

        max_full_seq_len = 60 * 5
        seq_out_ev_comb_tok, seq_out_ev_comb_tok_len = pad_tokens(
            seq_lst_all_ev_comb,
            pad_index=word_voc.pad_token_id,
            pad_side="right",
            append_eos=True,
            eos_index=word_voc.eos_token_id,
            max_len=max_full_seq_len,
        )

        out_dct = {
            "seq_out_by_ev": torch.tensor(seq_lst_all_ev).long(),
            "evrel_out_by_ev": torch.tensor(evrel_lst_all_ev).long(),
            "seq_out_lens_by_ev": torch.tensor(seq_lst_all_ev_lens).long(),
            "seq_out_ev_comb_tok": torch.tensor([seq_out_ev_comb_tok.tolist()]).long(),
            "seq_out_ev_comb_tok_len": torch.tensor([seq_out_ev_comb_tok_len]).long(),
            "vb_out_by_ev": torch.tensor(only_vb_lst_all_ev).long(),
            "vb_out_ev_comb_tok": torch.tensor([vb_lst_all_ev_comb]).long(),
        }

        def get_new_s(s):
            return s[0] + s[1]

        if is_evrel:
            out_evrel_seq_by_ev = []
            out_evrel_seq_by_ev_lens = []
            out_evrel_labs_by_ev = []

            out_evrel_tok_ids_by_ev = []
            evrel_wvoc = self.comm.rob_hf_tok
            for evix in [0, 1, 3, 4]:
                out_evrel_seq_lst = []
                out_evrel_seq_lens = []
                out_evrel_tok_ids_lst = []
                out_evrel_labs_lst = []
                for vix in range(len(vid_seg_ann_lst)):
                    ev3_seq = evrel_seq_lst_all_ev[2][vix]
                    evcurr_seq = evrel_seq_lst_all_ev[evix][vix]
                    if evix < 2:
                        s1 = evcurr_seq
                        s2 = ev3_seq
                    else:
                        s1 = ev3_seq
                        s2 = evcurr_seq
                    s1_new = get_new_s(s1)
                    s2_new = get_new_s(s2)

                    new_seq_noevrel = evrel_wvoc(
                        s1_new + evrel_wvoc.sep_token + s2_new
                    )["input_ids"]

                    new_seq = new_seq_noevrel

                    new_seq_pad, new_seq_msk = pad_tokens(
                        new_seq,
                        pad_index=evrel_wvoc.pad_token_id,
                        pad_side="right",
                        append_eos=False,
                        eos_index=evrel_wvoc.eos_token_id,
                        max_len=120,
                    )

                    evrel_out = evrel_lst_all_ev[evix][vix]
                    out_evrel_labs_lst.append(evrel_out)
                    out_evrel_seq_lst.append(new_seq_pad.tolist())
                    out_evrel_seq_lens.append(new_seq_msk)

                out_evrel_seq_by_ev.append(out_evrel_seq_lst)
                out_evrel_seq_by_ev_lens.append(out_evrel_seq_lens)
                out_evrel_tok_ids_by_ev.append(out_evrel_tok_ids_lst)
                out_evrel_labs_by_ev.append(out_evrel_labs_lst)

            out_dct["evrel_seq_out"] = torch.tensor(out_evrel_seq_by_ev).long()
            out_dct["evrel_seq_out_lens"] = torch.tensor(
                out_evrel_seq_by_ev_lens
            ).long()

            out_dct["evrel_labs"] = torch.tensor(out_evrel_labs_by_ev).long()

            out_evrel_seq_one_by_ev = []
            out_evrel_seq_onelens_by_ev = []
            out_evrel_vb_one_by_ev = []
            out_evrel_vb_onelens_by_ev = []

            for evix in [0, 1, 2, 3, 4]:
                out_evrel_seq_one_lst = []
                out_evrel_seq_onelens_lst = []

                out_evrel_vbonly_one_lst = []
                out_evrel_vbonly_onelens_lst = []
                for vix in range(len(vid_seg_ann_lst)):
                    s1 = evrel_seq_lst_all_ev[evix][vix]
                    s1_new = get_new_s(s1)

                    new_seq_noevrel = evrel_wvoc(s1_new)["input_ids"]
                    new_seq_pad, new_seq_msk = pad_tokens(
                        new_seq_noevrel,
                        pad_index=evrel_wvoc.pad_token_id,
                        pad_side="right",
                        append_eos=False,
                        eos_index=evrel_wvoc.eos_token_id,
                        max_len=60,
                    )

                    out_evrel_seq_one_lst.append(new_seq_pad.tolist())
                    out_evrel_seq_onelens_lst.append(new_seq_msk)
                    vb_only_rob = evrel_wvoc(s1[0])["input_ids"]
                    vb_only_rob_pad, vb_only_rob_msk = pad_tokens(
                        vb_only_rob,
                        pad_index=evrel_wvoc.pad_token_id,
                        pad_side="right",
                        append_eos=False,
                        eos_index=evrel_wvoc.eos_token_id,
                        max_len=5,
                    )
                    out_evrel_vbonly_one_lst.append(vb_only_rob_pad.tolist())
                    out_evrel_vbonly_onelens_lst.append(vb_only_rob_msk)

                out_evrel_seq_one_by_ev.append(out_evrel_seq_one_lst)
                out_evrel_seq_onelens_by_ev.append(out_evrel_seq_onelens_lst)
                out_evrel_vb_one_by_ev.append(out_evrel_vbonly_one_lst)
                out_evrel_vb_onelens_by_ev.append(out_evrel_vbonly_onelens_lst)

            out_dct["evrel_seq_out_ones"] = torch.tensor(out_evrel_seq_one_by_ev).long()
            out_dct["evrel_seq_out_ones_lens"] = torch.tensor(
                out_evrel_seq_onelens_by_ev
            ).long()
            out_dct["evrel_vbonly_out_ones"] = torch.tensor(
                out_evrel_vb_one_by_ev
            ).long()
            out_dct["evrel_vbonly_out_ones_lens"] = torch.tensor(
                out_evrel_vb_onelens_by_ev
            ).long()
        return out_dct

    def get_obj_frms_all(self, idx):
        vid_seg_name = self.vseg_lst[idx]
        frm_pth_lst = [
            self.vsitu_frm_dir / f"{vid_seg_name}/{vid_seg_name}_{ix:06d}.jpg"
            for ix in range(1, 301)
        ]

        # Read Tracker Info
        obj_dict = np.load(self.vsitu_obj_dir / f"{vid_seg_name}"/"obj_track.npy", allow_pickle=True).item()
        assert len(obj_dict) > 0, f"No object in {vid_seg_name}."

        frms_ev_fast_tensor = []
        frms_ev_slow_tensor = []
        for ev in range(1, 6):
            ev_id = f"Ev{ev}"
            frms_by_ev_fast = []
            frms_by_ev_slow = []
            center_ix = self.comm.cent_frm_per_ev[ev_id]
            frms_ixs_for_ev = get_sequence(
                center_idx=center_ix,
                half_len=self.comm.frm_seq_len // 2,
                sample_rate=self.comm.sampling_rate,
                max_num_frames=300,
            )
            confidence_list = []
            for k in obj_dict:
                confidence = []
                for f in frms_ixs_for_ev:
                    if f not in obj_dict[k]:
                        confidence.append(0)
                    else:
                        confidence.append(obj_dict[k][str(f)]['confidence'])
                confidence_list.append((sum(confidence)/len(frms_ixs_for_ev), k))

            for _, k in sorted(confidence_list, reverse=True)[:4]:
                frm_pths_for_ev = [(frm_pth_lst[ix], ix) for ix in frms_ixs_for_ev]

                frms_for_ev = torch.from_numpy(
                    np.stack([self.read_obj_img(f, obj_dict[k]) for f in frm_pths_for_ev])
                )

                frms_for_ev = tensor_normalize(
                    frms_for_ev, self.sf_cfg.DATA.MEAN, self.sf_cfg.DATA.STD
                )

                # T x H x W x C => C x T x H x W
                frms_for_ev_t = (frms_for_ev).permute(3, 0, 1, 2)
                frms_for_ev_slow_fast = pack_pathway_output(self.sf_cfg, frms_for_ev_t)
                if len(frms_for_ev_slow_fast) == 1:
                    frms_by_ev_fast.append(frms_for_ev_slow_fast[0])
                elif len(frms_for_ev_slow_fast) == 2:
                    frms_by_ev_slow.append(frms_for_ev_slow_fast[0])
                    frms_by_ev_fast.append(frms_for_ev_slow_fast[1])
                else:
                    raise NotImplementedError

            # 4 x C x T x H x W
            frms_all_ev_fast = np.stack(frms_by_ev_fast)
            if len(frms_by_ev_fast) < 4:
                rem = 4-len(frms_by_ev_fast)
                frms_all_ev_fast = np.concatenate((frms_all_ev_fast, 
                    np.zeros((rem, *frms_by_ev_fast[0].shape))))
            frms_ev_fast_tensor.append(torch.from_numpy(frms_all_ev_fast).float().unsqueeze_(0))
            if len(frms_by_ev_slow) > 0:
                frms_all_ev_slow = np.stack(frms_by_ev_slow)
                if len(frms_by_ev_slow) < 4:
                    rem = 4-len(frms_by_ev_slow)
                    frms_all_ev_slow = np.concatenate((frms_all_ev_slow, 
                        np.zeros((rem, *frms_by_ev_slow[0].shape))))
                frms_ev_slow_tensor.append(torch.from_numpy(frms_all_ev_slow).float().unsqueeze_(0))

        # 5 x 4 x C x T x H x W
        out_dct = {"obj_frms_ev_fast_tensor": torch.cat(frms_ev_fast_tensor)}
        if len(frms_ev_slow_tensor) > 0:
            out_dct["obj_frms_ev_slow_tensor"] = torch.cat(frms_ev_slow_tensor)

        return out_dct

    def get_frms_all(self, idx):
        vid_seg_name = self.vseg_lst[idx]
        frm_pth_lst = [
            self.vsitu_frm_dir / f"{vid_seg_name}/{vid_seg_name}_{ix:06d}.jpg"
            for ix in range(1, 301)
        ]

        frms_by_ev_fast = []
        frms_by_ev_slow = []
        for ev in range(1, 6):
            ev_id = f"Ev{ev}"
            center_ix = self.comm.cent_frm_per_ev[ev_id]
            frms_ixs_for_ev = get_sequence(
                center_idx=center_ix,
                half_len=self.comm.frm_seq_len // 2,
                sample_rate=self.comm.sampling_rate,
                max_num_frames=300,
            )
            frm_pths_for_ev = [frm_pth_lst[ix] for ix in frms_ixs_for_ev]

            frms_for_ev = torch.from_numpy(
                np.stack([self.read_img(f) for f in frm_pths_for_ev])
            )

            frms_for_ev = tensor_normalize(
                frms_for_ev, self.sf_cfg.DATA.MEAN, self.sf_cfg.DATA.STD
            )

            # T x H x W x C => C x T x H x W
            frms_for_ev_t = (frms_for_ev).permute(3, 0, 1, 2)
            frms_for_ev_slow_fast = pack_pathway_output(self.sf_cfg, frms_for_ev_t)
            if len(frms_for_ev_slow_fast) == 1:
                frms_by_ev_fast.append(frms_for_ev_slow_fast[0])
            elif len(frms_for_ev_slow_fast) == 2:
                frms_by_ev_slow.append(frms_for_ev_slow_fast[0])
                frms_by_ev_fast.append(frms_for_ev_slow_fast[1])
            else:
                raise NotImplementedError

        out_dct = {}
        # 5 x C x T x H x W
        frms_all_ev_fast = np.stack(frms_by_ev_fast)
        out_dct["frms_ev_fast_tensor"] = torch.from_numpy(frms_all_ev_fast).float()
        if len(frms_by_ev_slow) > 0:
            frms_all_ev_slow = np.stack(frms_by_ev_slow)
            out_dct["frms_ev_slow_tensor"] = torch.from_numpy(frms_all_ev_slow).float()

        return out_dct

    def get_frm_feats_all(self, idx: int):
        vid_seg_name = self.vseg_lst[idx]
        vid_seg_feat_file = (
            Path(self.cfg.vsit_frm_feats_dir) / f"{vid_seg_name}_feats.npy"
        )
        vid_feats = read_file_with_assertion(vid_seg_feat_file, reader="numpy")
        vid_feats = torch.from_numpy(vid_feats).float()
        assert vid_feats.size(0) == 5
        return {"frm_feats": vid_feats}

    def get_label_out_dct(self, idx: int):
        vid_seg_name = self.vseg_lst[idx]
        if self.split_type == "train":
            vid_seg_ann_ = self.vsitu_ann_dct[vid_seg_name]
            vid_seg_ann = vid_seg_ann_[0]
            label_out_dct = self.get_vb_data([vid_seg_ann])
        elif "valid" in self.split_type:
            vid_seg_ann_ = self.vsitu_vinfo_dct[vid_seg_name]["vb_id_lst_new"]
            assert len(vid_seg_ann_) >= 9
            label_out_dct = self.get_vb_data(vid_seg_ann_)
        else:
            raise NotImplementedError

        file_name = self.vsitu_frm_dir / vid_seg_name / "text_features.pt"
        if os.path.exists(file_name):
            feature_dict = torch.load(file_name, map_location=torch.device('cpu'))
            label_out_dct.update(feature_dict)
        else:
            label_out_dct['text_feature'] = torch.tensor()
            label_out_dct['pooled_feature'] = torch.tensor()
        
        return label_out_dct

    def vb_only_item_getter(self, idx: int):
        frms_out_dct = self.get_frms_all(idx)
        if self.comm.need_objs:
            obj_frms_out_dct = self.get_obj_frms_all(idx)

        frms_out_dct["vseg_idx"] = torch.tensor(idx)
        label_out_dct = self.get_label_out_dct(idx)
        out_dct = coalesce_dicts([frms_out_dct, label_out_dct])
        return out_dct

    def vb_args_item_getter(self, idx: int):
        vid_seg_name = self.vseg_lst[idx]
        if self.split_type == "train":
            vid_seg_ann_ = self.vsitu_ann_dct[vid_seg_name]
            vid_seg_ann = vid_seg_ann_[0]
            seq_out_dct = self.get_vb_arg_data([vid_seg_ann], is_evrel=self.is_evrel)
        elif "valid" in self.split_type:
            vid_seg_ann_ = self.vsitu_ann_dct[vid_seg_name]
            assert len(vid_seg_ann_) >= 3
            vid_seg_ann_ = vid_seg_ann_[:3]
            seq_out_dct = self.get_vb_arg_data(vid_seg_ann_, is_evrel=self.is_evrel)
        elif "test" in self.split_type:
            assert self.task_type == "evrel"
            vid_seg_ann_ = self.vsitu_ann_dct[vid_seg_name]
            assert len(vid_seg_ann_) >= 3
            vid_seg_ann_ = vid_seg_ann_[:3]
            seq_out_dct = self.get_vb_arg_data(vid_seg_ann_, is_evrel=self.is_evrel)
        else:
            raise NotImplementedError
        seq_out_dct["vseg_idx"] = torch.tensor(idx)

        # put this into extended_cfg
        if self.full_cfg.mdl.mdl_name not in set(
            [
                "txed_only",
                "tx_only",
                "gpt2_only",
                "new_gpt2_only",
                "tx_ev_only",
                "new_gpt2_ev_only",
                "rob_evrel",
            ]
        ):
            frm_feats_out_dct = self.get_frm_feats_all(idx)
            return coalesce_dicts([frm_feats_out_dct, seq_out_dct])
        else:
            return seq_out_dct


class BatchCollator:
    def __init__(self, cfg, comm):
        self.cfg = cfg
        self.comm = comm

    def __call__(self, batch):
        out_dict = simple_collate_dct_list(batch)
        return out_dict


def get_data(cfg):
    DS = VsituDS
    BC = BatchCollator

    train_ds = DS(cfg, {}, split_type="train")
    valid_ds = DS(cfg, train_ds.comm, split_type="valid")
    assert cfg.ds.val_set_type == "lb"
    if cfg.only_test:
        if cfg.task_type == "vb":
            test_ds = DS(cfg, train_ds.comm, split_type="test_verb")
        elif cfg.task_type == "vb_arg":
            test_ds = DS(cfg, train_ds.comm, split_type="test_srl")
        elif cfg.task_type == "evrel":
            test_ds = DS(cfg, train_ds.comm, split_type="test_evrel")
        else:
            raise NotImplementedError
    else:
        test_ds = None
    batch_collator = BC(cfg, train_ds.comm)
    train_dl = get_dataloader(cfg, train_ds, is_train=True, collate_fn=batch_collator)
    valid_dl = get_dataloader(cfg, valid_ds, is_train=False, collate_fn=batch_collator)

    if cfg.only_test:
        test_dl = get_dataloader(
            cfg, test_ds, is_train=False, collate_fn=batch_collator
        )
    else:
        test_dl = None
    data = DataWrap(
        path=cfg.misc.tmp_path, train_dl=train_dl, valid_dl=valid_dl, test_dl=test_dl
    )
    return data
