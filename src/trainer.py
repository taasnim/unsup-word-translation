# This project uses the structure of MUSE (https://github.com/facebookresearch/MUSE)

import os
from logging import getLogger
import scipy
import scipy.linalg
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from .utils import get_optimizer, load_embeddings, normalize_embeddings
from .utils import clip_parameters, get_nn_avg_dist
from .dico_builder import build_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_dictionary

from tqdm import tqdm
import collections


logger = getLogger()


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, mapping_G, mapping_F, discriminator_A, discriminator_B, encoder_A, decoder_A, encoder_B, decoder_B, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.mapping_G = mapping_G
        self.mapping_F = mapping_F
        self.discriminator_A = discriminator_A
        self.discriminator_B = discriminator_B
        self.encoder_A = encoder_A
        self.decoder_A = decoder_A
        self.encoder_B = encoder_B
        self.decoder_B = decoder_B
        self.params = params

        # optimizers
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer_G = optim_fn(mapping_G.parameters(), **optim_params)
            self.map_optimizer_F = optim_fn(mapping_F.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer_A = optim_fn(discriminator_A.parameters(), **optim_params)
            self.dis_optimizer_B = optim_fn(discriminator_B.parameters(), **optim_params)
        else:
            assert discriminator_A is None
            assert discriminator_B is None
        if hasattr(params, 'autoenc_optimizer'):
            optim_fn, optim_params = get_optimizer(params.autoenc_optimizer)
            self.encoder_A_optimizer = optim_fn(encoder_A.parameters(), **optim_params)
            self.decoder_A_optimizer = optim_fn(decoder_A.parameters(), **optim_params)
            self.encoder_B_optimizer = optim_fn(encoder_B.parameters(), **optim_params)
            self.decoder_B_optimizer = optim_fn(decoder_B.parameters(), **optim_params)

        # best validation score
        self.best_valid_metric_AB = -1e12
        self.best_valid_metric_BA = -1e12

        self.decrease_lr_G = False
        self.decrease_lr_F = False

    def get_dis_AB(self, volatile):
        """
        Get discriminator input batch / output target (A->B)
        """
        # select random word IDs
        bs = self.params.batch_size
        mf1 = self.params.dis_most_frequent_AB
        mf2 = self.params.dis_most_frequent_BA
        assert mf1 <= min(len(self.src_dico), len(self.tgt_dico))
        assert mf2 <= min(len(self.src_dico), len(self.tgt_dico))
        src_ids = torch.LongTensor(bs).random_(len(self.src_dico) if mf1 == 0 else mf1)
        tgt_ids = torch.LongTensor(bs).random_(len(self.tgt_dico) if mf2 == 0 else mf2)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        src_emb = self.src_emb(src_ids) 
        tgt_emb = self.tgt_emb(tgt_ids) 
        orig_src = (src_emb.clone()).data

        p = self.encoder_A(src_emb)
        q = self.encoder_B(tgt_emb)
        orig_p = (p.clone()).data
        src_emb = self.mapping_G(p.data) 
        tgt_emb = q.data  
        
        # input / target
        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(2 * bs).zero_()
        y[:bs] = 1 - self.params.dis_smooth
        y[bs:] = self.params.dis_smooth
        y = y.cuda() if self.params.cuda else y 

        return x, y, orig_src, orig_p


    def get_dis_BA(self, volatile):
        """
        Get discriminator input batch / output target (B->A)
        """
        # select random word IDs
        bs = self.params.batch_size
        mf1 = self.params.dis_most_frequent_AB
        mf2 = self.params.dis_most_frequent_BA
        assert mf1 <= min(len(self.src_dico), len(self.tgt_dico))
        assert mf2 <= min(len(self.src_dico), len(self.tgt_dico))
        src_ids = torch.LongTensor(bs).random_(len(self.src_dico) if mf1 == 0 else mf1)
        tgt_ids = torch.LongTensor(bs).random_(len(self.tgt_dico) if mf2 == 0 else mf2)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        src_emb = self.src_emb(src_ids) 
        tgt_emb = self.tgt_emb(tgt_ids) 

        p = self.encoder_A(src_emb)
        q = self.encoder_B(tgt_emb)
        src_emb = p.data 
        orig_tgt = (tgt_emb.clone()).data
        orig_q = (q.clone()).data 
        tgt_emb = self.mapping_F(q.data)  
        
        # input / target
        x = torch.cat([tgt_emb, src_emb], 0) ##Opposite of previous##
        y = torch.FloatTensor(2 * bs).zero_()
        y[:bs] = 1 - self.params.dis_smooth
        y[bs:] = self.params.dis_smooth
        y = y.cuda() if self.params.cuda else y 

        return x, y, orig_tgt, orig_q


    def dis_step_B(self, stats):
        """
        Train the discriminator in B (D_B). Data in B space.
        """
        self.discriminator_B.train()

        # loss
        x, y, orig_A, orig_p = self.get_dis_AB(volatile=True) 
        preds = self.discriminator_B(x.data)
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS_B'].append(loss.item())

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer_B.zero_grad()
        loss.backward()
        self.dis_optimizer_B.step()
        clip_parameters(self.discriminator_B, self.params.dis_clip_weights)


    def dis_step_A(self, stats):
        """
        Train the discriminator in A (D_A). Data in A space.
        """
        self.discriminator_A.train()

        # loss
        x, y, orig_B, orig_q = self.get_dis_BA(volatile=True) 
        preds = self.discriminator_A(x.data)
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS_A'].append(loss.item())

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer_A.zero_grad()
        loss.backward()
        self.dis_optimizer_A.step()
        clip_parameters(self.discriminator_A, self.params.dis_clip_weights)

    def mapping_step_G(self, stats):
        """
        Fooling discriminator training step in A->B
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator_B.eval()
        self.discriminator_A.eval()

        # Adversarial loss
        x, y, orig_A, orig_p = self.get_dis_AB(volatile=False) 
        preds = self.discriminator_B(x)
        loss = F.binary_cross_entropy(preds, 1 - y) 
        loss = self.params.dis_lambda * loss 

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        # Cycle/Back_translation loss
        bs = self.params.batch_size
        p_hat = self.mapping_F(x.data[:bs]) # use cycle on First half of x. They are from source (A)
        cyc_loss = torch.mean(torch.abs(orig_p-p_hat)) 

        # Reconstruction loss
        if self.params.l_relu==1:
            x_hat = self.decoder_A(self.encoder_A.leakyRelu(p_hat.data))
        else:
            x_hat = self.decoder_A(p_hat.data)
        loss_A = F.mse_loss(orig_A, x_hat) 

        # Total loss
        total_loss = loss + self.params.cycle_lambda*cyc_loss + self.params.reconstruction_lambda*loss_A

        self.map_optimizer_G.zero_grad()
        self.map_optimizer_F.zero_grad()
        self.encoder_A_optimizer.zero_grad()
        self.decoder_A_optimizer.zero_grad()
        self.encoder_B_optimizer.zero_grad()

        total_loss.backward()

        self.map_optimizer_G.step()
        self.map_optimizer_F.step()
        self.encoder_A_optimizer.step()
        self.decoder_A_optimizer.step()
        self.encoder_B_optimizer.step()

        # Orthogonalize the mapping weights
        self.orthogonalize_G()
        self.orthogonalize_F()


        return 2 * self.params.batch_size

    
    def mapping_step_F(self, stats):
        """
        Fooling discriminator training step in B->A
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator_A.eval()
        self.discriminator_B.eval()

        # Adversarial loss
        x, y, orig_B, orig_q = self.get_dis_BA(volatile=False) 
        preds = self.discriminator_A(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.params.dis_lambda * loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()
        
        # Cycle/Back_translation loss
        bs = self.params.batch_size
        q_hat = self.mapping_G(x.data[:bs]) # use cycle on First half of x. They are from target (B)
        cyc_loss = torch.mean(torch.abs(orig_q-q_hat)) 

        # Reconstruction loss
        if self.params.l_relu==1:
            y_hat = self.decoder_B(self.encoder_B.leakyRelu(q_hat.data))
        else:
            y_hat = self.decoder_B(q_hat.data)
        loss_B = F.mse_loss(orig_B, y_hat)

        # Total loss
        total_loss = loss + self.params.cycle_lambda*cyc_loss + self.params.reconstruction_lambda*loss_B

        self.map_optimizer_F.zero_grad()
        self.map_optimizer_G.zero_grad()
        self.encoder_B_optimizer.zero_grad()
        self.decoder_B_optimizer.zero_grad()
        self.encoder_A_optimizer.zero_grad()

        total_loss.backward()

        self.map_optimizer_F.step()
        self.map_optimizer_G.step()
        self.encoder_B_optimizer.step()
        self.decoder_B_optimizer.step()
        self.encoder_A_optimizer.step()

        self.orthogonalize_F()
        self.orthogonalize_G()

        return 2 * self.params.batch_size


    def train_autoencoder_A(self):
        print("Training source in autoencoder.")
        bs = 128
        for epoch in tqdm(range( self.params.autoenc_epochs)):
            total_loss=0
            num_batches=0
            for n_iter in range(0, self.params.epoch_size, bs):
                # select random word IDs            
                ids = torch.LongTensor(bs).random_(len(self.src_dico))
                if self.params.cuda:
                    ids = ids.cuda()
                # get word embeddings
                emb = self.src_emb(ids) 
                preds = self.decoder_A(self.encoder_A(emb.data))
                loss = F.mse_loss(emb.data, preds)

                total_loss += loss.detach().item()
                num_batches += 1

                # optim
                self.encoder_A_optimizer.zero_grad()
                self.decoder_A_optimizer.zero_grad()
                loss.backward()
                self.encoder_A_optimizer.step()
                self.decoder_A_optimizer.step()
                
    
    def train_autoencoder_B(self):
        print("Training target in autoencoder.")
        bs = 128
        for epoch in tqdm(range( self.params.autoenc_epochs)):
            total_loss=0
            num_batches=0
            for n_iter in range(0, self.params.epoch_size, bs):
                # select random word IDs            
                ids = torch.LongTensor(bs).random_(len(self.tgt_dico))
                if self.params.cuda:
                    ids = ids.cuda()
                # get word embeddings
                emb = self.tgt_emb(ids) 
                preds = self.decoder_B(self.encoder_B(emb.data))
                loss = F.mse_loss(emb.data, preds)
                
                total_loss += loss.detach().item()
                num_batches += 1

                # optim
                self.encoder_B_optimizer.zero_grad()
                self.decoder_B_optimizer.zero_grad()
                loss.backward()
                self.encoder_B_optimizer.step()
                self.decoder_B_optimizer.step()


    def build_dictionary_AB(self):
        """
        Build a dictionary from aligned embeddings for A->B.
        """
        src_emb = self.mapping_G(self.encoder_A(self.src_emb.weight.data)).data
        tgt_emb = self.encoder_B(self.tgt_emb.weight.data).data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico_AB = build_dictionary(src_emb, tgt_emb, self.params)

    def build_dictionary_BA(self):
        """
        Build a dictionary from aligned embeddings for B->A.
        """
        src_emb = self.encoder_A(self.src_emb.weight.data).data
        tgt_emb = self.mapping_F(self.encoder_B(self.tgt_emb.weight.data)).data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico_BA = build_dictionary(tgt_emb, src_emb, self.params)


    def procrustes_AB(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem for A->B
        """
        A = self.encoder_A(self.src_emb.weight.data[self.dico_AB[:, 0]]).data
        B = self.encoder_B(self.tgt_emb.weight.data[self.dico_AB[:, 1]]).data
        W = self.mapping_G.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def procrustes_BA(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem for B->A
        """
        A = self.encoder_B(self.tgt_emb.weight.data[self.dico_BA[:, 0]]).data
        B = self.encoder_A(self.src_emb.weight.data[self.dico_BA[:, 1]]).data
        W = self.mapping_F.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))


    def orthogonalize_G(self):
        """
        Orthogonalize the mapping weight of mapper G.
        """
        if self.params.map_beta > 0:
            W = self.mapping_G.weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))
    
    def orthogonalize_F(self):
        """
        Orthogonalize the mapping weight of mapper F.
        """
        if self.params.map_beta > 0:
            W = self.mapping_F.weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))


    def update_lr(self, to_log, metric_AB, metric_BA):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        # for G mapper
        old_lr = self.map_optimizer_G.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate for G: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer_G.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric_AB] >= -1e7:
            if to_log[metric_AB] < self.best_valid_metric_AB:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric_AB], self.best_valid_metric_AB))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr_G:
                    old_lr = self.map_optimizer_G.param_groups[0]['lr']
                    self.map_optimizer_G.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate for G: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer_G.param_groups[0]['lr']))
                self.decrease_lr_G = True
        
        # for F mapper
        old_lr = self.map_optimizer_F.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate for F: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer_F.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric_BA] >= -1e7:
            if to_log[metric_BA] < self.best_valid_metric_BA:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric_BA], self.best_valid_metric_BA))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr_F:
                    old_lr = self.map_optimizer_F.param_groups[0]['lr']
                    self.map_optimizer_F.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer_F.param_groups[0]['lr']))
                self.decrease_lr_F = True
       

    def save_best_AB(self, to_log, metric_AB):
        """
        Save the best model for the given validation metric for A->B
        """
        # best mapping for the given validation criterion
        
        if to_log[metric_AB] > self.best_valid_metric_AB:
            # new best mapping
            self.best_valid_metric_AB = to_log[metric_AB]
            logger.info('* Best value for "%s": %.5f' % (metric_AB, to_log[metric_AB]))
            
            # save the mapping
            # saving weight matrix of G
            W = self.mapping_G.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_mapping_AB.pth')
            logger.info('* Saving the mapping to %s ...' % path) 
            torch.save(W, path)

            # saving Encoder_X weights
            W = self.encoder_A.encoder.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_encX_AB.pth') 
            torch.save(W, path)

            # saving Encoder_Y weights
            W = self.encoder_B.encoder.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_encY_AB.pth') 
            torch.save(W, path)

        
    def save_best_BA(self, to_log, metric_BA):
        """
        Save the best model for the given validation metric for B->A
        """
        if to_log[metric_BA] > self.best_valid_metric_BA:
            # new best mapping
            self.best_valid_metric_BA = to_log[metric_BA]
            logger.info('* Best value for "%s": %.5f' % (metric_BA, to_log[metric_BA]))

            # save the mapping
            # saving weight matrix of F
            W = self.mapping_F.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_mapping_BA.pth') 
            logger.info('* Saving the mapping to %s ...' % path) 
            torch.save(W, path)

            # saving Encoder_X weights
            W = self.encoder_A.encoder.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_encX_BA.pth') 
            torch.save(W, path)

            # saving Encoder_Y weights
            W = self.encoder_B.encoder.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_encY_BA.pth') 
            torch.save(W, path)


    def reload_best_AB(self, path=None):
        """
        Reload the best saved params for A->B.
        """
        if path==None:
            path1 = os.path.join(self.params.exp_path, 'best_mapping_AB.pth')
            path2 = os.path.join(self.params.exp_path, 'best_encX_AB.pth')
            path3 = os.path.join(self.params.exp_path, 'best_encY_AB.pth')
        else:
            path1 = os.path.join(path, 'best_mapping_AB.pth')
            path2 = os.path.join(path, 'best_encX_AB.pth')
            path3 = os.path.join(path, 'best_encY_AB.pth')

        logger.info('* Reloading the best G from %s ...' % path1)
        logger.info('* Reloading the best enc_X from %s ...' % path2)
        logger.info('* Reloading the best enc_Y from %s ...' % path3)
        
        # reload the model
        assert os.path.isfile(path1)
        assert os.path.isfile(path2)
        assert os.path.isfile(path3)
        ## reload G (A->B)
        to_reload = torch.from_numpy(torch.load(path1))
        W1 = self.mapping_G.weight.data
        assert to_reload.size() == W1.size()
        W1.copy_(to_reload.type_as(W1))
        ## reload enc_X (A->B)
        to_reload = torch.from_numpy(torch.load(path2))
        W2 = self.encoder_A.encoder.weight.data
        assert to_reload.size() == W2.size()
        W2.copy_(to_reload.type_as(W2))
        ## reload enc_Y (A->B)
        to_reload = torch.from_numpy(torch.load(path3))
        W3 = self.encoder_B.encoder.weight.data
        assert to_reload.size() == W3.size()
        W3.copy_(to_reload.type_as(W3))



    def reload_best_BA(self, path=None):
        """
        Reload the best saved params for B->A.
        """
        if path==None:
            path1 = os.path.join(self.params.exp_path, 'best_mapping_BA.pth')
            path2 = os.path.join(self.params.exp_path, 'best_encX_BA.pth')
            path3 = os.path.join(self.params.exp_path, 'best_encY_BA.pth')
        else:
            path1 = os.path.join(path, 'best_mapping_BA.pth')
            path2 = os.path.join(path, 'best_encX_BA.pth')
            path3 = os.path.join(path, 'best_encY_BA.pth')

        logger.info('* Reloading the best F from %s ...' % path1)
        logger.info('* Reloading the best enc_X from %s ...' % path2)
        logger.info('* Reloading the best enc_Y from %s ...' % path3)
        
        # reload the model
        assert os.path.isfile(path1)
        assert os.path.isfile(path2)
        assert os.path.isfile(path3)
        ## reload F (B->A)
        to_reload = torch.from_numpy(torch.load(path1))
        W1 = self.mapping_F.weight.data
        assert to_reload.size() == W1.size()
        W1.copy_(to_reload.type_as(W1))
        ## reload enc_X (B->A)
        to_reload = torch.from_numpy(torch.load(path2))
        W2 = self.encoder_A.encoder.weight.data
        assert to_reload.size() == W2.size()
        W2.copy_(to_reload.type_as(W2))
        ## reload enc_Y (B->A)
        to_reload = torch.from_numpy(torch.load(path3))
        W3 = self.encoder_B.encoder.weight.data
        assert to_reload.size() == W3.size()
        W3.copy_(to_reload.type_as(W3))

