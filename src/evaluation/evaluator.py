# This project uses the structure of MUSE (https://github.com/facebookresearch/MUSE)


from logging import getLogger
from copy import deepcopy
import numpy as np
import torch
from torch import Tensor as torch_tensor

from ..dico_builder import get_candidates, build_dictionary

logger = getLogger()

class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.src_emb = trainer.src_emb
        self.tgt_emb = trainer.tgt_emb
        self.src_dico = trainer.src_dico
        self.tgt_dico = trainer.tgt_dico
        self.mapping_G = trainer.mapping_G
        self.mapping_F = trainer.mapping_F
        self.discriminator_A = trainer.discriminator_A
        self.discriminator_B = trainer.discriminator_B
        self.encoder_A = trainer.encoder_A
        self.decoder_A = trainer.decoder_A
        self.encoder_B = trainer.encoder_B
        self.decoder_B = trainer.decoder_B
        self.params = trainer.params
    
    def dist_mean_cosine(self, to_log, src_to_tgt):
        """
        Mean-cosine model selection criterion.
        """
        if src_to_tgt: 
            
            # get normalized embeddings
            src_emb = self.mapping_G(self.encoder_A(self.src_emb.weight.data)).data
            tgt_emb = self.encoder_B(self.tgt_emb.weight.data).data
            src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
            tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

            # build dictionary
            for dico_method in ['csls_knn_10']:
                dico_build = 'S2T'
                dico_max_size = 10000
                # temp params / dictionary generation
                _params = deepcopy(self.params)
                _params.dico_method = dico_method
                _params.dico_build = dico_build
                _params.dico_threshold = 0
                _params.dico_max_rank = 10000
                _params.dico_min_size = 0
                _params.dico_max_size = dico_max_size
                s2t_candidates = get_candidates(src_emb, tgt_emb, _params) 
                t2s_candidates = get_candidates(tgt_emb, src_emb, _params)
                dico = build_dictionary(src_emb, tgt_emb, _params, s2t_candidates, t2s_candidates)
                # mean cosine
                if dico is None:
                    mean_cosine = -1e9
                else:
                    mean_cosine = (src_emb[dico[:dico_max_size, 0]] * tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
                mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
                logger.info("Mean cosine A->B (%s method, %s build, %i max size): %.5f"
                            % (dico_method, _params.dico_build, dico_max_size, mean_cosine))
                to_log['mean_cosine-%s-%s-%i' % (dico_method, _params.dico_build, dico_max_size)] = mean_cosine


        else:
            
            # get normalized embeddings
            src_emb = self.encoder_A(self.src_emb.weight.data).data
            tgt_emb = self.mapping_F(self.encoder_B(self.tgt_emb.weight.data)).data
            src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
            tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

            # build dictionary
            for dico_method in ['csls_knn_10']:
                dico_build = 'S2T' ## No need to change here, handled by changing in next piece of code
                dico_max_size = 10000
                # temp params / dictionary generation
                _params = deepcopy(self.params)
                _params.dico_method = dico_method
                _params.dico_build = dico_build
                _params.dico_threshold = 0
                _params.dico_max_rank = 10000
                _params.dico_min_size = 0
                _params.dico_max_size = dico_max_size
                s2t_candidates = get_candidates(src_emb, tgt_emb, _params) 
                t2s_candidates = get_candidates(tgt_emb, src_emb, _params)
                dico = build_dictionary(tgt_emb, src_emb, _params, t2s_candidates, s2t_candidates)
                # mean cosine
                if dico is None:
                    mean_cosine = -1e9
                else:
                    mean_cosine = (tgt_emb[dico[:dico_max_size, 0]] * src_emb[dico[:dico_max_size, 1]]).sum(1).mean()
                mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
                logger.info("Mean cosine B->A (%s method, %s build, %i max size): %.5f"
                            % (dico_method, _params.dico_build, dico_max_size, mean_cosine))
                to_log['mean_cosine-%s-%s-%i' % (dico_method, 'T2S', dico_max_size)] = mean_cosine
    
    
    def model_selection_criterion(self, to_log):
        """
        Run Mean-cosine model selection criterion for A->B and B->A
        """
        self.dist_mean_cosine(to_log, src_to_tgt=True)
        self.dist_mean_cosine(to_log, src_to_tgt=False)
    