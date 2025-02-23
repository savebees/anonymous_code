import sys
sys.path.insert(0, '../')
from util import *
import numpy as np
import math

def find_seg_ada(vatt, cur):
    #jointly consider the attentioin score and its distance with the maximal point
    if not isinstance(vatt, list): 
        vatt = [vatt]
    n = len(vatt)
    vatt = np.asarray(vatt)
    vatt = (np.asarray(vatt)-np.min(vatt))/(np.max(vatt)-np.min(vatt)+1e-12)
    max_id = np.argmax(vatt)
    mean_s = np.mean(vatt)
   
    path = [max_id]
    # cid, fid = max_id//4, max_id%4
    # stamp_s = cur[cid][fid]
    stamp_s = cur[max_id]
    i = 1
    thd = mean_s + 0.3*mean_s #(0.1~0.5)
    # print(mean_s)
    dis = 10
    while max_id - i > 0:
        # cid, fid = (max_id-i)//4, (max_id-i)%4
        # stamp = cur[cid][fid]
        stamp = cur[max_id-i]
        elapse = abs(stamp_s - stamp)
        score = vatt[max_id-i] / (1+elapse/dis)
        # print(elapse, score, vatt[max_id-i])
        if score >= thd:
            path.append(max_id-i)
        elif elapse > dis:
            break
        i += 1
    i = 1
    while max_id + i < n:
        # cid, fid = (max_id+i)//4, (max_id+i)%4
        # stamp = cur[cid][fid]
        stamp = cur[max_id + i]
        elapse = abs(stamp_s - stamp)
        score = vatt[max_id+i] / (1+elapse/dis)
        # print(elapse, score, vatt[max_id+i])
        if score >= thd:
            path.append(max_id+i)
        elif elapse > dis:
            break
        i += 1
        
    sid, eid = np.min(path), np.max(path)
    start, end = cur[sid], cur[eid]
    # start = cur[sid//4][sid%4]
    # end = cur[eid//4][eid%4]

    return [start, end]


def find_seg_maxC(vatt, cur):
    vatt = np.asarray(vatt).reshape((8, 4))
    max_s = 0
    max_c = 0
    for cid, clip in enumerate(vatt):
        cur_s = clip.mean()
        
        if cur_s > max_s:
            max_s = cur_s
            max_c = cid
    
    return [cur[max_c][0], cur[max_c][-1]]


def find_seg_maxF(vatt, cur):
    # vatt = np.asarray(vatt).reshape((8, 4))
    max_s = 0
    max_f = 0
    for fid, cur_s in enumerate(vatt):
        if cur_s > max_s:
            max_s = cur_s
            max_f = fid
    cid, fid = max_f//4, max_f%4
    
    return [cur[cid][fid], cur[cid][fid]]


def find_seg_mgqa(gauss_params, fatt, cur, num_gaussians=3, weight_threshold=0.1):
    """
    Locate the segment based on the output of MGQA.

    Args:
        gauss_params: (T, K, 3) Gaussian distribution parameters
        fatt: (T, 1) Attention weights
        cur: (T, 2) Start and end timestamps for each time step
        num_gaussians: The number of Gaussian distributions
        weight_threshold: The weight threshold used to filter out unimportant Gaussian distributions

    Returns:
        [start, end]: The start and end timestamps of the located video segment.
    """
    T = gauss_params.shape[0]
    # Combine Gaussian parameters and attention weights
    combined_scores = np.zeros(T)

    for t in range(T):
        for k in range(num_gaussians):
            center = gauss_params[t, k, 0]
            sigma = gauss_params[t, k, 1]
            weight = gauss_params[t, k, 2]

            # Only consider Gaussians with significant weights
            if weight > weight_threshold:
                combined_scores[t] += weight * fatt[t, 0] #* np.exp(-0.5 * ((t - center) / sigma) ** 2)  # Gaussian contribution

    max_index = np.argmax(combined_scores)
    start, end = cur[max_index][0], cur[max_index][1]

    return [start, end]


def generate_ground(pred_file, seg_file, ground_file, qa_file):
    preds = load_file(pred_file)
    print(len(preds))
    segs = load_file(seg_file)
    qas = load_file(qa_file)
    res_ground = {}
    for idx, row in qas.iterrows():
        vid, qid = str(row['video_id']), str(row['qid'])
        vid_qid = '_'.join([vid, qid])
        atts = preds[vid_qid]
        # fids = np.linspace(0, 31, 24, dtype='int')
        cur = np.asarray(segs[vid])  # [fids] #[[16]]

        # for VGT hierarchical attention
        # vatts = []
        # for id, fatt in enumerate(atts['fatt']):
        #     catt_v = atts['catt'][id]
        #     for v in fatt:
        #         hint = np.round(v+catt_v, 2)
        #         vatts.append(hint)

        # seg = find_seg_ada(atts, cur)

        gauss_params = atts['gauss_params']
        fatt = atts['fatt']
        seg = find_seg_mgqa(gauss_params, fatt, cur)

        # print(vid_qid, seg)

        res_ground[vid_qid] = seg

    save_to(ground_file, res_ground)
        

def main(data_dir, dset):

    anno_dir = '../../datasets/nextgqa/'
    if not osp.exists(anno_dir):
        anno_dir = '../'+anno_dir
    seg_file = f'{anno_dir}/frame2time_{dset}.json'
    qa_file = f'{anno_dir}/{dset}.csv'
    pred_file = f'{data_dir}/{dset}_ground_att.json'
    ground_file = f'{data_dir}/{dset}_ground_ada.json'

    generate_ground(pred_file, seg_file, ground_file, qa_file)


if __name__ == "__main__":
    data_dir = '../../data/gmodels/NG+/TempCLIP/'
    dset = 'test'
    main(data_dir, dset)
    
