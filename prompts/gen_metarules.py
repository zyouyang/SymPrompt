import pickle
import random
from tqdm import tqdm

data_path = '../my_data' # Change as needed

with open(f'{data_path}/relation2id.txt') as f:
    rel2id_lines = f.readlines()
    qry_rels = set()
    rel_srcdst = {}
    all_ents = set()
    for line in rel2id_lines:
        rel, rid = line.strip().split('\t')
        src, dst = rel.split('_')
        rel_srcdst[rel] = (src, dst)
        rel_srcdst[rel+'_rev'] = (dst, src)
        qry_rels.add(rel)
        all_ents.add(src)
        all_ents.add(dst)

ent2avail_rels = {}
for ent in all_ents:
    # find all relations that have this entity as source
    for rel, (src, dst) in rel_srcdst.items():
        if src == ent: # this is a relation that has this entity as source
            ent2avail_rels.setdefault(ent, set()).add(rel)

rel2avail_rels = {}
for rel, (src, dst) in rel_srcdst.items():
    if rel not in rel2avail_rels:
        rel2avail_rels[rel] = set()
    rel2avail_rels[rel].update(ent2avail_rels[dst])

qry_to_path = {}
for ent in all_ents:
    rel1s = ent2avail_rels[ent] # relations for the first hop
    for rel1 in rel1s:
        rel2s = rel2avail_rels[rel1]
        for rel2 in rel2s:
            dst = rel_srcdst[rel2][1]
            curr_qry = ent + '_' + dst
            if curr_qry in qry_rels:
                if curr_qry not in qry_to_path:
                    qry_to_path[curr_qry] = set()
                qry_to_path[curr_qry].add((rel1, rel2))
            # rel3s = rel2avail_rels[rel2]
            # for rel3 in rel3s:
            #     dst = rel_srcdst[rel3][1]
            #     curr_qry = ent + '_' + dst
            #     if curr_qry in qry_rels:
            #         if curr_qry not in qry_to_path:
            #             qry_to_path[curr_qry] = set()
            #         qry_to_path[curr_qry].add((rel1, rel2, rel3))

entid2ent = {}
ent2entid = {}
forward_connected = {}
backward_connected = {}
with open(f'{data_path}/train.txt', 'r') as f:
    train_lines = f.readlines()
    for line in train_lines:
        line = line.strip().split('\t')
        src_id, rel, dst_id = line
        src, dst = rel.split('_')
        entid2ent[src_id] = src
        entid2ent[dst_id] = dst
        if src not in ent2entid:
            ent2entid[src] = set()
        ent2entid[src].add(dst_id)
        if dst not in ent2entid:
            ent2entid[dst] = set()
        ent2entid[dst].add(src_id)
        if src_id not in forward_connected:
            forward_connected[src_id] = set()
        forward_connected[src_id].add(dst_id)
        if dst_id not in backward_connected:
            backward_connected[dst_id] = set()
        backward_connected[dst_id].add(src_id)

qry_dict = {}
src_rel_to_dsts = {}
for line in train_lines:
    line = line.strip().split('\t')
    src_id, rel, dst_id = line
    src, dst = rel.split('_')
    qry_dict.setdefault(rel, set()).add((src_id, rel, dst_id, src, dst))
    key = (src_id, rel)
    if key not in src_rel_to_dsts:
        src_rel_to_dsts[key] = set()
    src_rel_to_dsts[key].add(dst_id)

all_pathcnt = {}
# for nq, qry in enumerate(['mal_certificate']):
for nq, qry in enumerate(qry_rels):
    print(f'======= mining rules for {nq} query: {qry} =======')
    cnt = {}
    samp_lines = random.sample(list(qry_dict[qry]), min(100, len(qry_dict[qry])))
    paths = qry_to_path[qry]
    num_of_paths = len(paths)
    pathcnt = {}
    for i, (src_id, rel, dst_id, src, dst) in tqdm(enumerate(samp_lines)):
        cand_dst_ids = src_rel_to_dsts[(src_id, rel)].copy()
        cand_dst_ids.remove(dst_id)
        # remove only the concerend relation
        assert(src_id in forward_connected)
        assert(dst_id in forward_connected[src_id])
        assert(dst_id in backward_connected)
        assert(src_id in backward_connected[dst_id])
        forward_connected[src_id].remove(dst_id)
        backward_connected[dst_id].remove(src_id)

        for path in paths:
            curr_paths = [[src_id]]
            # print(path)
            for j, hop_rel in enumerate(path):
                tmp_paths = []
                hit_cands = 0
                cands = set()
                if hop_rel.endswith('_rev'):
                    connected = backward_connected
                    next_ent = rel_srcdst[hop_rel][1]
                else:
                    connected = forward_connected
                    next_ent = rel_srcdst[hop_rel][1]
                for curr_path in curr_paths:
                    prev_id = curr_path[-1]
                    if prev_id not in connected:
                        continue
                    # find all entities that are connected to prev_id and belong to type next_ent
                    cands = connected[prev_id]
                    if j == len(path) - 1:
                        for cand in cands:
                            if entid2ent[cand] == next_ent:
                                if cand not in cand_dst_ids: # we do not want to count the alternative answer
                                    cands.add(cand) # this is a valid answer
                                if cand == dst_id: # this is the correct answer
                                    hit_cands = 1  
                    else:
                        for cand in cands:
                            if entid2ent[cand] == next_ent:
                                tmp_path = curr_path + [cand]
                                tmp_paths.append(tmp_path)
                curr_paths = tmp_paths
            if len(cands) == 0:
                hit_ratio = 0
            else:
                hit_ratio = hit_cands / len(cands)
            if path not in pathcnt:
                pathcnt[path] = 0
            pathcnt[path] += hit_ratio
        forward_connected[src_id].add(dst_id)
        backward_connected[dst_id].add(src_id)

    sorted_val = sorted(pathcnt.values(), reverse=True)
    sep_val_6 = sorted_val[5]
    sep_val_80 = None
    p_val = sum(sorted_val) * 0.8
    curr_sum = 0
    for val in sorted_val:
        curr_sum += val
        if curr_sum >= p_val:
            sep_val_80 = val
            break
    sep_val = max(sep_val_6, sep_val_80)
    for key, val in pathcnt.items():
        if val >= sep_val:
            if qry not in all_pathcnt:
                all_pathcnt[qry] = set()
            all_pathcnt[qry].add(key)

    pickle.dump(all_pathcnt, open('./all_pathcnt_2.pkl', 'wb'))
