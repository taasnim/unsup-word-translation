import torch

from .utils import normalize_embeddings, get_nn_avg_dist


def generate_new_dictionary(emb1, emb2, dico_max_rank=15000):
    '''
    build a dictionary from aligned embeddings
    '''
    emb1 = emb1.cuda()
    emb2 = emb2.cuda()
    bs = 128
    all_scores = []
    all_targets = []
    # number of source words to consider
    n_src = dico_max_rank
    knn = 10

    average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn)) 
    average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
    average_dist1 = average_dist1.type_as(emb1)
    average_dist2 = average_dist2.type_as(emb2)

    for i in range(0, n_src, bs):
        scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
        best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

        all_scores.append(best_scores.cpu())
        all_targets.append(best_targets.cpu())

    all_scores = torch.cat(all_scores, 0)
    all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:, 0].unsqueeze(1)
    ], 1)

    # sanity check
    assert all_scores.size() == all_pairs.size() == (n_src, 2)

    # sort pairs by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    # max dico words rank
    if dico_max_rank > 0:
        selected = all_pairs.max(1)[0] <= dico_max_rank
        mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        all_scores = all_scores.masked_select(mask).view(-1, 2)
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    return all_pairs, all_scores 


def symmetric_reweighting(src_emb, tgt_emb, src_indices, trg_indices):
    '''
    Symmetric reweighting refinement procedure
    '''
    xw = (src_emb.weight.clone()).data
    zw = (tgt_emb.weight.clone()).data
    
    _ = normalize_embeddings(xw.data, 'renorm,center,renorm')
    _ = normalize_embeddings(zw.data, 'renorm,center,renorm')

    # STEP 1: Whitening
    def whitening_transformation(m):
        u, s, v = torch.svd(m) 
        return v.mm(torch.diag(1/(s))).mm(v.t())

    wx1 = whitening_transformation(xw[src_indices]).type_as(xw)
    wz1 = whitening_transformation(zw[trg_indices]).type_as(zw)

    xw = xw.mm(wx1)
    zw = zw.mm(wz1)

    # STEP 2: Orthogonal mapping
    wx2, s, wz2 = torch.svd(xw[src_indices].t().mm(zw[trg_indices]), some=False)
    
    xw = xw.mm(wx2)
    zw = zw.mm(wz2)

    # STEP 3: Re-weighting
    xw *= s**0.5
    zw *= s**0.5 

    # STEP 4: De-whitening
    xw = xw.mm(wx2.transpose(0, 1).mm(torch.inverse(wx1)).mm(wx2))
    zw = zw.mm(wz2.transpose(0, 1).mm(torch.inverse(wz1)).mm(wz2))
    
    return xw, zw


