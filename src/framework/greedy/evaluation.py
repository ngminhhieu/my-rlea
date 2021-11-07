import numpy as np

from framework.greedy.greedy import greedy_alignment

def test_(embeds1, embeds2, unique1, unique2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=True, simi_matrix=None, return_simi=False):
    if mapping is None:
        if return_simi:
            alignment_rest_12, hits1_12, mr_12, mrr_12, simi_matrix = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                           metric, normalize, csls_k, accurate, simi_matrix=simi_matrix, return_simi=True)
        else:
            alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                                      metric, normalize, csls_k, accurate, simi_matrix=simi_matrix)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        if return_simi:
            alignment_rest_12, hits1_12, mr_12, mrr_12, simi_matrix = greedy_alignment(test_embeds1_mapped, embeds2, top_k, threads_num,
                                                                      metric, normalize, csls_k, accurate, simi_matrix=simi_matrix, return_simi=True) 
        else:
            alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(test_embeds1_mapped, embeds2, top_k, threads_num,
                                                                      metric, normalize, csls_k, accurate, simi_matrix=simi_matrix)
    

    aligns = [(unique1[align_idx[0]], unique2[align_idx[1]]) for align_idx in alignment_rest_12 ] 

    if return_simi:
        return aligns, hits1_12, mrr_12, simi_matrix
    return aligns, hits1_12, mr_12, mrr_12