from routing import reconstruct_path, dijkstra_search_bheap_all
import pickle
from pathlib import Path
import multiprocessing
from multiprocessing import Pool

def SPT_worker(data):
    start = data[0]
    forward_star = data[1]
    edges = data[2]
    non_passing_nodes = data[3]
    time_factor = data[4]
    came_from, tt = dijkstra_search_bheap_all(start=start,
                                                       pred_dict=forward_star,
                                                       edges=edges,
                                                       non_passing_nodes=non_passing_nodes)
    d = {start: 0}
    weighted_tt = {start: 0}
    for end in list(tt):
        if end != start:
            weighted_tt[end] = tt[end] * time_factor
            path = reconstruct_path(came_from, start, end)
            path_links = zip(path[:-1], path[1:])
            d[end] = sum([edges[(int(link[0]), int(link[1]))]['Length'] for link in path_links])
    return {start: {'came_from': came_from, 'tt': tt, 'distance': d, 'weighted_tt': weighted_tt}}

def SPTBuild(network, network_name, time_factor=1):

    def SPT_compute_generator():
        for start in list(network['forward_star']):
            yield (start, network['forward_star'], network['edges'], network['non_passing_nodes'], time_factor)

    save_path = network_name.replace('.txt', '_ff_spt.pkl')
    if Path(save_path).is_file():
        with open(save_path, 'rb') as handle:
            SPT = pickle.load(handle)
    else:
        generator = SPT_compute_generator()
        pool = Pool(multiprocessing.cpu_count())
        tmp = pool.map(SPT_worker, generator)

        SPT = {key: values for item in tmp for key, values in item.items()}
        with open(save_path, 'wb') as handle:
            pickle.dump(SPT, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return SPT