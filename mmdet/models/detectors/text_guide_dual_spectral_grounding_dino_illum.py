#Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import math
import warnings
from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from natten.functional import natten2dqkrpb, natten2dav

def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_


@MODELS.register_module()
class TextDualSpectralGroundingDINOillum(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 *args,
                 use_autocast=False,
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        super().__init__(*args, **kwargs)

        text_dim = 256

        self.fusion_module = RGBTFusionModule(
            in_channels=[256, 512, 1024], text_dim=256 
        )
        self.illumestimate = IlluminationEstimator(
            in_channels = 256
        )

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)
        

    def extract_feat(self, batch_inputs, text_dict):
        """
        Final safe feature extraction for DualSpectral Grounding DINO
        Flow:
            RGB / IR backbone
            → text-guided semantic grounding (per modality)
            → safe RGB–IR fusion (channel-wise, residual)
            → neck
        """

        x_rgb = self.backbone(batch_inputs['img'])      # list[T] of [B,C,H,W]
        x_ir  = self.backbone(batch_inputs['img_ir'])

        x_rgb_low = x_rgb[0]
        _illum_score = self.illumestimate(
            rgb_feat = x_rgb_low
        )

        fused_feats = self.fusion_module(x_rgb, x_ir, text_dict, illum_score=_illum_score)

        if self.with_neck:
            x_fusion = self.neck(fused_feats)

        return x_fusion, _illum_score



    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs['img'])  #batch_inputs
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # # image feature extraction
        # visual_feats = self.extract_feat(batch_inputs)  #batch_inputs

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs['img']) == 1  #batch_inputs
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once
                
                visual_feats, illum_score = self.extract_feat(batch_inputs, text_dict)  #batch_inputs
            
                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            visual_feats, illum_score = self.extract_feat(batch_inputs, text_dict)  #batch_inputs

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples

class ConvFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop=0.):
        super(ConvFFN, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim*mlp_ratio, 1, padding=0)
        self.conv2 = nn.Conv2d(dim*mlp_ratio, dim, 1, padding=0)
        self.dwc = nn.Conv2d(dim*mlp_ratio, dim*mlp_ratio, 3, padding=1, groups=dim*mlp_ratio)
        self.drop = nn.Dropout(drop)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.drop(x)
        x =  x + self.dwc(x)
        x = F.gelu(x)
        x = self.drop(self.conv2(x))
        return x

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm that works on 4D input (N, C, H, W)"""

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)
    
    def forward(self, x):
        # Reshape x to (N, C, H*W) for LayerNorm
        N, C, H, W = x.size()
        x = x.view(N, C, -1).contiguous()
        x = x.permute(0, 2, 1).contiguous()  # (N, H*W, C)
        # Apply LayerNorm
        x = super().forward(x)
        x = x.permute(0, 2, 1).contiguous()  # (N, C, H*W)
        # Reshape back to (N, C, H, W)
        return x.view(N, C, H, W).contiguous()

class NeighborhoodAttention(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=5, dilation=1, attn_drop=0.1, proj_drop=0.1):
        super(NeighborhoodAttention, self).__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size
        assert dilation is None or dilation >= 1, \
                f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.q = nn.Conv2d(dim, dim, 1, padding=0)
        self.kv = nn.Conv2d(dim, dim * 2, 1, padding=0)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        nn.init.trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, C, H, W = x.shape
        # pad x and y to multiples of window size
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = max(self.window_size - W, 0)
            pad_b = max(self.window_size - H, 0)
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            y = F.pad(y, (pad_l, pad_r, pad_t, pad_b))
            H_, W_ = x.shape[2], x.shape[3]

        q = self.q(x).reshape(B,self.num_heads, C // self.num_heads, H , W).permute(0,1,3,4,2) # B d H W h
        kv = self.kv(y).reshape(B, 2, self.num_heads, C // self.num_heads, H ,  W).permute(1,0,2,4,5,3) # 3 B h H W d
        k, v = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = natten2dqkrpb(query=q, key=k, rpb=self.rpb, kernel_size=self.kernel_size, dilation=self.dilation)
        attn = F.softmax(attn, dim=-1)  
        attn = self.attn_drop(attn)
        x = natten2dav(attn=attn, value=v, kernel_size=self.kernel_size, dilation=self.dilation)
        x = x.permute(0, 1, 4, 2, 3).reshape(B, C, H, W) # B d h w c
        if pad_r or pad_b:
            x = x[:, :, :H_, :W_]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossDeformableAttention(nn.Module):
    def __init__(self, dim, num_groups, num_heads, ks = 5 , stride=4, attn_drop=0.1, scale=2.0, size= 80):
        super(CrossDeformableAttention, self).__init__()
        assert dim % num_groups == 0, 'dim must be divisible by num_groups'
        assert num_heads % num_groups == 0, 'num_heads must be divisible by num_groups'
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.scale = scale
        self.stride = stride
        self.group_dim = dim//num_groups
        self.temperature = torch.sqrt(torch.tensor(self.group_dim//num_heads, dtype=torch.float32))
        pad = 0 if stride==ks else ks//2
        self.offset_network = nn.Sequential(
                                        nn.Conv2d(self.group_dim, self.group_dim, ks, padding=pad , stride=stride, groups=self.group_dim),
                                        LayerNorm2d(self.group_dim),
                                        nn.GELU(),
                                        nn.Conv2d(self.group_dim,2,1,padding=0,bias=False),
                                        )
        self.q = nn.Conv2d(dim,dim,1,padding=0)
        self.k = nn.Conv2d(dim,dim,1,padding=0)
        self.v = nn.Conv2d(dim,dim,1,padding=0)
        self.o = nn.Conv2d(dim,dim,1,padding=0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop = nn.Dropout(attn_drop)
        self.size = size
        if isinstance(size, int):
            self.rpe_table = nn.Parameter(
                    torch.zeros(num_heads, size * 2 - 1, size * 2 - 1)
                )
        elif isinstance(size, tuple) and len(size) == 2:
            self.rpe_table = nn.Parameter(
                        torch.zeros(self.num_heads, self.size[0] * 2 - 1, self.size[1] * 2 - 1)
                    )
        else:
            raise ValueError("size must be an int or a tuple of two ints")
        trunc_normal_(self.rpe_table, std=0.01)
        self._reset_parameters()
    

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        #nn.init.xavier_uniform_(self.proj_k.weight)
        nn.init.constant_(self.o.bias, 0)   
         
    @torch.no_grad()
    def _get_reference_points(self, H, W, dtype):
        ref_points = torch.stack(torch.meshgrid(
                                                torch.linspace(0.5, H-0.5, H, dtype=dtype), 
                                                torch.linspace(0.5, W-0.5, W, dtype=dtype)
                                                ), dim=-1)
        ref_points[:,:,0].mul_(2).div_(H-1).sub_(1)
        ref_points[:,:,1].mul_(2).div_(W-1).sub_(1)
        return ref_points
    
    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.num_groups, -1, -1, -1) # B * g H W 2

        return ref
    
    def forward(self, query,key):
        B, C, H, W = query.size()
        dtype, device = query.dtype, query.device
        # get offsets from the key image
        proj_q = self.q(query)
        proj_k = self.k(key)
        proj_k = rearrange(proj_k, 'b (g d) h w -> (b g) d h w ', g=self.num_groups) # bg d h w
        offsets = self.offset_network(proj_k).contiguous().permute(0,2,3,1) # bg hr wr 2
        hr,wr = offsets.size(1),offsets.size(2)
        n_sample = hr * wr
        # get reference points
        ref_points = self._get_reference_points(hr,wr,query.dtype).to(device)
        ref_points = ref_points.unsqueeze(0).expand_as(offsets) #bg hr wr 2
        offset_range = torch.tensor([1.0 / (hr - 1.0), 1.0 / (wr - 1.0)], device=query.device, dtype=query.dtype).reshape(1, 1, 1,2)
        pos = offset_range*self.scale*F.tanh(offsets)+ref_points # bg hr wr 2
        #offsets = offsets+ref_points
        #pos = offsets.clamp(-1., +1.)

        # get pixels via bilinear interpolation
        key = rearrange(key, 'b (g d) h w -> (b g) d h w ', g=self.num_groups) # bg d h w
        sampled = F.grid_sample(key,pos[..., (1,0)],mode='bilinear',align_corners=True)
        sampled = rearrange(sampled, '(b g) d hr wr -> b (g d) hr wr ', g=self.num_groups) # b gd hr wr

        # get query, key and value
        k = self.k(sampled) # b c hr wr
        v = self.v(sampled) # b c hr wr
        q = rearrange(proj_q, 'b (i d) h w -> b i (h w) d ', i=self.num_heads) # b i hw d
        k = rearrange(k, 'b (i d) h w -> b i (h w) d ', i=self.num_heads) # b i hw d
        v = rearrange(v, 'b (i d) h w -> b i (h w) d ', i=self.num_heads) #b i hw d
        # compute attention
        attn = q@k.transpose(-2,-1)
        attn = attn/self.temperature
        
        rpe_table = self.rpe_table
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        rpe_bias =  F.interpolate(rpe_bias, size=(2*H - 1, 2*W - 1), mode="bilinear", align_corners=False)
        
        q_grid = self._get_q_grid(H, W, B, dtype, device)
        displacement = (q_grid.reshape(B * self.num_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.num_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
        attn_bias = F.grid_sample(
                    input=rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.num_heads // self.num_groups, g=self.num_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True) # B * g, h_g, HW, Ns

        attn_bias = attn_bias.reshape(B,self.num_groups, self.num_heads // self.num_groups, H * W, n_sample)
        attn_bias = attn_bias.view(B,-1, H * W, n_sample) # B g h_g hw n_sample
        attn = attn + attn_bias
        
        attn = F.softmax(attn, dim=-1)
        out = self.attn_drop(attn)@v
        out = rearrange(out, 'b i (h w) d -> b (i d) h w ', h=H, w=W) 
        out = self.drop(self.o(out)) 
        return out

class AttentionBlock2(nn.Module):
    def __init__(self, dim, num_heads, groups=4, stride=4, mlp_ratio=2,scale=2., nat_ks=3,
                  kernel_size=5, dilation=1, attn_drop=0.1, proj_drop=0.1, size=80):
        super(AttentionBlock2, self).__init__()
        self.norm1 = LayerNorm2d(dim)
        self.norm1_ = LayerNorm2d(dim)

        self.attn = NeighborhoodAttention(dim, num_heads, nat_ks, dilation, attn_drop, proj_drop)
        self.attn2 = NeighborhoodAttention(dim, num_heads, nat_ks, dilation, attn_drop, proj_drop)

        self.cross_attn = CrossDeformableAttention(dim,groups,num_heads,kernel_size,stride,attn_drop,scale,size)
        self.cross_attn2 = CrossDeformableAttention(dim,groups,num_heads,kernel_size,stride,attn_drop,scale,size)

        self.norm2 = LayerNorm2d(dim)
        self.norm2_ = LayerNorm2d(dim)
        self.mlp = ConvFFN(dim, mlp_ratio=mlp_ratio, drop=proj_drop)
        self.mlp2 = ConvFFN(dim, mlp_ratio=mlp_ratio, drop=proj_drop)
        self.norm3 = LayerNorm2d(dim)
        self.norm3_ = LayerNorm2d(dim)
        self.drop_path = DropPath(0.1) if 0.1 > 0. else nn.Identity()
        
        self.layer_scale1 = nn.Identity()
        self.layer_scale2 = nn.Identity()
        self.layer_scale3 = nn.Identity()
        self.layer_scale4 = nn.Identity()
        self.layer_scale5 = nn.Identity()
        self.layer_scale6 = nn.Identity()
        
    def forward(self, x, y):
        # x B C H W
        
        shortcut = x
        shortcut2 = y

        input1 = self.norm1(x)
        input2 = self.norm1_(y)

        input1 = self.attn(input1, input1)
        input2 = self.attn2(input2, input2)

        input1 = self.drop_path(self.layer_scale1(input1)) + shortcut
        input2 = self.drop_path(self.layer_scale2(input2)) + shortcut2

        x_ = self.norm2(input1)
        y_ = self.norm2_(input2)

        input1 = self.cross_attn(x_, y_)
        input2 = self.cross_attn2(y_, x_)

        input1 = self.drop_path(self.layer_scale3(input1)) + x_
        input2 = self.drop_path(self.layer_scale4(input2)) + y_

        input1 = self.drop_path(self.layer_scale5(self.mlp(input1))) + input1
        input2 = self.drop_path(self.layer_scale6(self.mlp2(input2))) + input2

        input1 = self.norm3(input1)
        input2 = self.norm3_(input2)


        return input1, input2
    
class IlluminationEstimator(nn.Module):

    def __init__(self, in_channels, illum_dim=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )


        # score head
        self.fc_score = nn.Linear(32, 1)

    def forward(self, rgb_feat):

        x = self.net(rgb_feat).flatten(1)   # [B,32]


        # [B,1] ∈ (0,1)
        illum_score = torch.sigmoid(self.fc_score(x))

        return illum_score
    
class IlluminationWeight(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(in_channels))

    def forward(self, illum_score):
        """
        illum_score: [B,1] ∈ (0,1)
        """
        B = illum_score.shape[0]
        C = self.alpha.shape[0]

        w_rgb = illum_score
        w_ir = 1.0 - illum_score

        w_rgb = w_rgb * self.alpha.view(1, C)
        w_ir  = w_ir  * self.alpha.view(1, C)

        return w_rgb, w_ir

class TextWeightGenerator(nn.Module):

    def __init__(self, in_channels, text_dims=256, hidden=256):
        super().__init__()

        self.text_proj = nn.Linear(text_dims, hidden)

        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden)
        )

        self.gamma_rgb = nn.Linear(hidden, in_channels)
        self.gamma_ir  = nn.Linear(hidden, in_channels)

    def forward(self, text_feat):

        text_cls = text_feat[:,0,:]

        h = self.mlp(self.text_proj(text_cls))

        gamma_rgb = torch.sigmoid(self.gamma_rgb(h))
        gamma_ir  = torch.sigmoid(self.gamma_ir(h))

        return gamma_rgb, gamma_ir

class TextGuidedCrossSpectralAttention(nn.Module):
    def __init__(self,
                 embed_dims,
                 text_dims,
                 illum_dims=32,
                 num_heads=8,
                 proj_drop=0.0):
        super().__init__()

        assert embed_dims % num_heads == 0

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** -0.5

        self.text_q = nn.Linear(text_dims, embed_dims)


        self.kv_rgb = nn.Conv2d(embed_dims, embed_dims * 2, 1)
        self.kv_ir  = nn.Conv2d(embed_dims, embed_dims * 2, 1)

        self.illum_gate = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.Sigmoid()
        )


        self.proj = nn.Conv2d(embed_dims, embed_dims, 1)


        self.norm_fusion = nn.GroupNorm(32, embed_dims)
        self.mlp = ConvFFN(dim=embed_dims, mlp_ratio=4, drop=proj_drop)

        self.norm_final = nn.GroupNorm(32, embed_dims)

        # 光照权重
        self.illum_weight = IlluminationWeight(embed_dims)
        self.text_weight  = TextWeightGenerator(embed_dims, text_dims)

    def forward(self, x_rgb, x_ir, text_feat, illum_score, text_mask=None):

        B, C, H, W = x_rgb.shape
        HW = H * W

        w_rgb_img, w_ir_img = self.illum_weight(illum_score)
        w_rgb_txt, w_ir_txt = self.text_weight(text_feat)

        w_rgb = (w_rgb_img * w_rgb_txt).view(B, C, 1, 1)
        w_ir  = (w_ir_img  * w_ir_txt).view(B, C, 1, 1)

        kv_rgb = self.kv_rgb(x_rgb).flatten(2)   # [B,2C,HW]
        kv_ir  = self.kv_ir(x_ir).flatten(2)

        k_rgb, v_rgb = kv_rgb.chunk(2, dim=1)    # [B,C,HW]
        k_ir,  v_ir  = kv_ir.chunk(2, dim=1)

        q = self.text_q(text_feat)   # [B,L,C]                       

        q = q.view(B, -1, self.num_heads, self.head_dim).permute(0,2,1,3)  # [B,h,L,d]

        k_rgb = k_rgb.view(B, self.num_heads, self.head_dim, HW)
        v_rgb = v_rgb.view(B, self.num_heads, self.head_dim, HW).permute(0,1,3,2)

        k_ir  = k_ir.view(B, self.num_heads, self.head_dim, HW)
        v_ir  = v_ir.view(B, self.num_heads, self.head_dim, HW).permute(0,1,3,2)

        attn_rgb = torch.matmul(q, k_rgb) * self.scale   # [B,h,L,HW]
        attn_ir  = torch.matmul(q, k_ir)  * self.scale

        # text mask
        if text_mask is not None:
            mask = text_mask.unsqueeze(1).unsqueeze(-1)
            attn_rgb = attn_rgb.masked_fill(~mask, float('-inf'))
            attn_ir  = attn_ir.masked_fill(~mask, float('-inf'))

        illum_gate = self.illum_gate(illum_score)   # [B,h]
        illum_gate = illum_gate.view(B, self.num_heads, 1, 1)

        # RGB / IR complementary
        attn = illum_gate * attn_rgb + (1 - illum_gate) * attn_ir
        attn = attn.softmax(dim=-1)
        v = illum_gate * v_rgb + (1 - illum_gate) * v_ir
        out = torch.matmul(attn, v)   # [B,h,L,d]
        out = out.permute(0,2,1,3).reshape(B, -1, C)  # [B,L,C]
        out = out.mean(dim=1)   # [B,C]
        out = out.view(B, C, 1, 1).expand(-1, -1, H, W)
        out = self.proj(out)
        fused_feat = out + (x_rgb * w_rgb) + (x_ir * w_ir)
        feat_normed = self.norm_fusion(fused_feat)
        fused_feat = fused_feat + self.mlp(feat_normed)

        out = self.norm_final(fused_feat)

        return out

class RGBTFusionModule(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024], text_dim=256, eval_size=(640,640)):
        super().__init__()
        
        strides = [8, 4, 2]
        dat_kernels = [9, 7, 5]
        nat_kernels = [5, 5, 5]
        
        self.align_layers = nn.ModuleList()     
        self.fusion_layers = nn.ModuleList()     
        
        for i, dim in enumerate(in_channels):

            downsample = 2 ** (i + 3)
            current_size = (eval_size[0] // downsample, eval_size[1] // downsample)
            
            align_blk = AttentionBlock2(
                dim=dim,
                num_heads=8,
                groups=4,
                stride=strides[i],
                kernel_size=dat_kernels[i],
                nat_ks=nat_kernels[i],
                scale=5.0,
                size=current_size
            )
            self.align_layers.append(align_blk)

            fusion_blk = TextGuidedCrossSpectralAttention(
                embed_dims=dim,    
                text_dims=text_dim,  
                num_heads=8         
            )
            self.fusion_layers.append(fusion_blk)



    def forward(self, x_rgb_list, x_ir_list, text_dict, illum_score):
        """
        输入:
            x_rgb_list, x_ir_list: [P3, P4, P5] 特征列表
            text_dict: 包含 'embedded' [B, L, C] 和 'text_token_mask' [B, L]
        输出:
            fused_feats: 融合后的特征列表
        """
        fused_feats = []
        
        text_feat = text_dict['embedded']
        text_mask = text_dict['text_token_mask']
        
        for i, (rgb, ir) in enumerate(zip(x_rgb_list, x_ir_list)):
            # 1. 几何对齐
            rgb_f, ir_f = self.align_layers[i](rgb, ir)
            # 2. 文本指导的融合
            out = self.fusion_layers[i](rgb_f, ir_f, text_feat, illum_score, text_mask)
            fused_feats.append(out)
            
        return fused_feats