import heapq
import warnings
import numpy as np
import time
class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

    def get_cost_item(self):
        return heapq.heappop(self.elements)

    def heap_querry(self):
        return self.elements[0]

    def get_max(self):
        return heapq.nlargest(1, self.elements)[0][0]

def network_import_postprocessing(edges, class_names, VOT={}, demands_factors=None):
    forward_star = {}
    backward_star = {}
    nodes = set()
    for link in edges:
        forward_star.setdefault(link[0], []).append(link[1]) # All outgoing nodes (links) from node_0
        backward_star.setdefault(link[1], []).append(link[0]) # All incoming nodes (links) from node_1
        nodes.add(link[0])
        nodes.add(link[1])
        for classID in class_names:
            edges[link]['classFlow'][classID] = 0
            # Initialize the class cost
            # The first term is the flow-dependent term (initialized with travel time)
            # The second term is the fixed cost with respect to the each class (could be scaled by class 'tollFactor')
            if classID in VOT:
                edges[link]['classCost'][classID] = VOT[classID] * edges[link]['cost'] + edges[link]['classToll'][classID]
            else:
                edges[link]['classCost'][classID] = edges[link]['cost'] + edges[link]['classToll'][classID]

    return edges, list(nodes), forward_star, backward_star


def topological_ordering(ori, network, pred):
    indegree = {node: 1 for node in network['nodes']}
    indegree_bush_nodes = {node: len(pred[node]) for node in pred if pred[node][0] != None}
    indegree.update(indegree_bush_nodes)
    indegree[ori] = 0

    order_list = [node for node in indegree if indegree[node] == 0] # actually could be simplified as origin only

    bush_topology = [] # topology of the bush
    while len(order_list) > 0:
        current = order_list.pop(0) # obtain the first element in the order_list
        bush_topology.append(current)
        if current not in network['forward_star']:  # only incoming links
            continue
        for next in network['forward_star'][current]: # check all outgoing links
            if next in pred and current in pred[next]:
                #print(current, next, pred[next], indegree[next])
                indegree[next] -= 1
                if indegree[next] == 0: # in case more than one incoming link, the topology of each node is processed once
                    order_list.append(next)

    #print('indegree', [(node, indegree[node]) for node in indegree if indegree[node] > 0])
    if len(bush_topology) < len(set(network['nodes']) - set(network['non_passing_nodes'])):
        raise AssertionError('Bush given for topology ordering contains a cycle {} {} {}'.format(len(bush_topology),
                                                                                              len(set(set(network['nodes']))),
                                                                                              len(network['non_passing_nodes'])))
    #print(ori, set(bush_topology) == set(network['nodes']), len(bush_topology), len(set(network['nodes'])), len(network['non_passing_nodes']))
    return bush_topology

def link_travel_time(t0, alpha, beta, flow, cap):
    t = t0 * (1 + alpha * np.power((flow/max(cap, 1e-8)), beta))
    if t == np.inf:
        raise ValueError('Infinite time {}'.format(flow))
    return t

def objective_function_value(network, flow_loading=None):
    if not flow_loading:
        return sum([(edge['ff_tt']*edge['flow'])
                     * (1 + ((edge['alpha']*np.power((edge['flow']/max(edge['Capacity'], 1e-8)), edge['beta'])/(1+edge['beta']))))
                     for edge in network['edges'].values()])
    else:
        return sum([(network['edges'][link]['ff_tt']*flow_loading[link])
                     * (1 + ((network['edges'][link]['alpha']
                              * np.power((flow_loading[link]/max(network['edges'][link]['Capacity'], 1e-8)), network['edges'][link]['beta']))
                             /(1+network['edges'][link]['beta'])))
                     for link in network['edges']])

def seq_flow_update(matching, driverID, driver_seq_ID, dx=None, x=None, sign=1):
    if dx != None:
        #print('Seq flow update', sign, dx, driverID, driver_seq_ID)
        #print('Before', matching['driver'][driverID]['sequence_flow'][driver_seq_ID])
        matching['driver'][driverID]['sequence_flow'][driver_seq_ID] = round(matching['driver'][driverID]['sequence_flow'][driver_seq_ID] + (sign * dx), 10)
        #print('After', matching['driver'][driverID]['sequence_flow'][driver_seq_ID])
        passenger_index = set([tuple(item)
                           for mappings in matching['driver'][driverID]['segment_mapping'][driver_seq_ID].values()
                           if len(mappings)
                           for item in mappings])
        for index in passenger_index:
            matching['passenger'][index[0]]['sequence_flow'][index[1]] = round(matching['passenger'][index[0]]['sequence_flow'][index[1]] + (sign * dx), 10)
    elif x != None:  # For initialization only
        # if x > 0 and driverID == (19, 17):
        #     print('To set', driverID, driver_seq_ID, x)
        matching['driver'][driverID]['sequence_flow'][driver_seq_ID] = round(x, 10)
        passenger_index = set([tuple(item)
                               for mappings in matching['driver'][driverID]['segment_mapping'][driver_seq_ID].values()
                               if len(mappings)
                               for item in mappings])
        for index in passenger_index:
            matching['passenger'][index[0]]['sequence_flow'][index[1]] = round(x, 10)

    return matching


def normalized_gap(matching, matching_assignment, bush_norm_gap, log_name, class_name=None, log=False):
    assert class_name in ['driver', 'passenger', None]
    total_flow = 0
    total_gap = 0
    max_used_class_cost = []
    if class_name:
        class_names = [class_name]
    else:
        class_names = ['driver', 'passenger']

    for class_name in class_names:
        for ID in matching[class_name]:

            seq_costs = np.array(list(matching[class_name][ID]['sequence_cost'].values()))
            seq_flows = np.array(list(matching[class_name][ID]['sequence_flow'].values()))
            index = [seqID for seqID, flow in matching[class_name][ID]['sequence_flow'].items() if flow>0]

            if class_name == 'driver':
                min_cost = min([matching[class_name][ID]['sequence_cost'][seqID][4]
                            for seqID, flow in matching[class_name][ID]['sequence_flow'].items()
                            if seqID == 0 or matching_assignment.get(seqID, 0) >= 1e-9 or flow > 0])
            else:
                min_cost = min([matching[class_name][ID]['sequence_cost'][seqID][0]
                            for seqID, flow in matching[class_name][ID]['sequence_flow'].items()
                            if seqID == 0 or matching_assignment.get(seqID//100, 0) >= 1e-9 or flow > 0])


            if seq_flows.shape[0] > 0:

                if class_name == 'driver':
                    long_short_route_gap = round(np.sum((seq_costs[seq_flows>0, 5] - seq_costs[seq_flows>0, 4]) * seq_flows[seq_flows>0]), 10)
                    seq_cheapest_gap = round(np.sum((seq_costs[:, 4] - min_cost) * seq_flows), 10)
                else:
                    long_short_route_gap = round(np.sum((seq_costs[seq_flows>0, 1] - seq_costs[seq_flows>0, 0]) * seq_flows[seq_flows>0]), 10)
                    seq_cheapest_gap = round(np.sum((seq_costs[:, 0] - min_cost) * seq_flows), 10)

                if class_name == 'driver':
                    norm_gap = (long_short_route_gap + seq_cheapest_gap)/sum(seq_flows)
                    if ID not in bush_norm_gap:
                        bush_norm_gap[ID] = [norm_gap, norm_gap]

                    bush_norm_gap[ID] = [bush_norm_gap[ID][1], norm_gap]

                if log:
                    print('\n', class_name, index, ID)
                    print(seq_costs)
                    print(seq_flows)
                    print([seqID for seqID, flow in matching[class_name][ID]['sequence_flow'].items()])
                    print('gap', long_short_route_gap, seq_cheapest_gap)

                if long_short_route_gap < -1e-9 or seq_cheapest_gap < -1e-9:
                    print('Negative', long_short_route_gap, seq_cheapest_gap)
                    print('negative_flows', seq_flows)
                    print('negative_costs', seq_costs)
                    #raise ValueError('Negative gap {} {} {} {}'.format(long_short_route_gap, seq_cheapest_gap, seq_flows, seq_costs))

                if long_short_route_gap == np.inf or seq_cheapest_gap == np.inf:
                    print(class_name, ID, index, )
                    if class_name == 'driver':
                        print([matching[class_name][ID]['driver_sequence'][i] for i in index])
                    else:
                        print([matching[class_name][ID]['sequences'][i] for i in index])
                    print('long_short_route_gap', long_short_route_gap, seq_cheapest_gap)
                    inf_flows = seq_flows[seq_costs[:, 1] == np.inf]
                    inf_costs = seq_costs[seq_costs[:, 1] == np.inf]
                    print('inf_flows', seq_flows)
                    print('inf_costs', seq_costs)
                    inf_flows = seq_flows[seq_costs[:, 0] == -np.inf]
                    inf_costs = seq_costs[seq_costs[:, 0] == -np.inf]
                    print('inf_flows2', inf_flows)
                    print('inf_costs2', inf_costs)
                    raise ValueError

                if (len(class_names) > 1 and class_name == 'driver') or len(class_names) == 1:
                    total_flow += np.sum(seq_flows)
                    total_gap += (long_short_route_gap + seq_cheapest_gap)
    if total_flow >= 1e-9:
        return total_gap/total_flow, total_gap, bush_norm_gap
    else:
        return 0, 0, bush_norm_gap

def network_cost_update(network, updated_links=None, update_classToll=False):
    if not updated_links:
        updated_links = list(network['edges'])

    for link in updated_links:
        before_cost = network['edges'][link]['cost']
        tt = link_travel_time(t0=network['edges'][link]['ff_tt'],
                                                          alpha=network['edges'][link]['alpha'],
                                                          beta=network['edges'][link]['beta'],
                                                          flow=network['edges'][link]['flow'],
                                                          cap=network['edges'][link]['Capacity'])
        network['edges'][link]['cost'] = tt
        #print('Updated link', link, before_cost, network['edges'][link]['cost'], network['edges'][link]['ff_tt'], network['edges'][link]['alpha'], network['edges'][link]['beta'], network['edges'][link]['flow'], network['edges'][link]['Capacity'])
        #print(network['edges'][link])
        for classID in list(network['edges'][link]['classToll']):
            #print(classID, network['edges'][link]['cost'], network['edges'][link]['classToll'][classID])
            network['edges'][link]['classCost'][classID] = network['edges'][link]['cost'] + network['edges'][link]['classToll'][classID]

    return network


def bush_pred_to_forward_star(pred):
    forward_star = {}

    for node in pred:
        for pred_node in pred[node]:
            if pred_node != None:
                forward_star.setdefault(pred_node, []).append(node)

    return forward_star

def passenger_actual_segment_cost(pred, start, topology, network, classID):  # By following the driver's pred, for gradient computation
    pass

def bush_scan(start, bush, edges, flow_ignore_threshold, classCost=False, LPRule='LONGEST_USED_BUSH_PATH', bush_RP=None, RP_classID = None, bush_RP_PT=None, partial_update=None):
    bush['SPcost'] = {start: 0}
    bush['SP_generalized_cost'] = {start: 0}
    bush['SPcost_carried'] = {start: 0}
    bush['merges']['shortest_pred'] = {}

    if partial_update == None:
        bush_flow_group_list = set([bush['associate_group_list'][int(group_index)] for group_index in bush['node_group_incident'][1:, 1]])
    else:
        bush_flow_group_list = set([bush['associate_group_list'][int(group_index)] for group_index in bush['node_group_incident'][1:, 1] if bush['associate_group_list'][int(group_index)][2]//100 in partial_update])

    bush['merges']['longest_pred'] = {groupID: {} for groupID in bush_flow_group_list}
    bush['LPcost'] = {groupID: {start: 0} for groupID in bush_flow_group_list}
    bush['LP_generalized_cost'] = {groupID: {start: 0} for groupID in bush_flow_group_list}

    assert LPRule in ['LONGEST_BUSH_PATH', 'LONGEST_USED_BUSH_PATH', 'LONGEST_USED_BUSH_PATH_OR_SP'], \
        'Unknown longest route rule!'
    #print()
    #print(bush['bush_topology'])
    printed = False

    log = False


    for node in bush['bush_topology']: # traverse the bush by topological order

        if log:
            print('\nTO node', node)

        for groupID in bush_flow_group_list:
            if node not in bush['LPcost'][groupID]:
                bush['LPcost'][groupID][node] = -1 * np.inf
                bush['LP_generalized_cost'][groupID][node] = -1 * np.inf

        if node not in bush['SPcost']:
            bush['SPcost'][node] = np.inf
            bush['SPcost_carried'][node] = np.inf
            bush['SP_generalized_cost'][node] = np.inf

        for pred_node in bush['pred'][node]: # check all approaches (incoming links)
            if pred_node != None:
                if classCost == False:
                    link_cost = edges[(pred_node, node)]['cost']
                else:
                    link_cost = edges[(pred_node, node)]['classCost'][classCost]

                if log:
                    print('\nFrom {} TO {}'.format(pred_node, node))

                #print('LSP', pred_node, bush['SPcost'][pred_node], bush['LPcost'][pred_node])
                multipliers = 0
                before_multipliers = 0

                if bush_RP:

                    multipliers = bush_RP['SPcost'][pred_node] + edges[(pred_node, node)]['classCost'][RP_classID] \
                                  - min(bush_RP['SPcost'][node], bush_RP_PT['SPcost'][node])
                    multipliers *= int(RP_classID.split('_')[1])
                    before_multipliers = link_cost

                if log:
                    print('From {} TO {}, {} Scost+M {} VS {}, Lcost+M {}'.format(pred_node, node, node in bush['SPcost'], link_cost, bush['SPcost'][node], link_cost))

                try:
                    if node not in bush['SPcost'] or (node in bush['SPcost'] and bush['SPcost'][pred_node] + link_cost < bush['SPcost'][node]):
                        if log:
                            print('Before Updated SPcost', bush['SPcost'][node])
                        bush['SPcost'][node] = bush['SPcost'][pred_node] + link_cost
                        bush['SP_generalized_cost'][node] = bush['SPcost'][pred_node] + link_cost + multipliers
                        bush['merges']['shortest_pred'][node] = pred_node

                        if log:
                            print('Updated SPcost', bush['SPcost'][node])

                except Exception:
                    raise Exception("{} {} {} {}".format(link_cost, bush['SPcost'][node], bush['LPcost'][pred_node], edges[(pred_node, node)]['classCost'][classCost]))


                # Default use longest route in the bush, regardless if it carries flows

                #print('Checking longest', pred_node, node, longest_new_cost, bush['LP_generalized_cost'][node], bush['LPcost'][node], multipliers, before_multipliers, edges[(pred_node, node)]['classCost'][classCost])
                SPcost_carried_nodeFlow = 0

                a = time.time()
                count = 0
                check_list = bush['node_group_incident'][bush['node_group_incident'][:,0]==pred_node, 1].tolist() + bush['node_group_incident'][bush['node_group_incident'][:,0]==node, 1].tolist()
                #print(bush['node_group_incident'], check_list)
                try:
                    if partial_update == None:
                        group_list = set([bush['associate_group_list'][int(group_index)] for group_index in check_list ])
                    else:
                        group_list = set([bush['associate_group_list'][int(group_index)] for group_index in check_list if bush['associate_group_list'][int(group_index)][2] // 100 in partial_update])
                except Exception as err:
                    raise Exception('{} {}'.format(err, bush['bushFlow'].keys()))
                #if log:
                #    print('Group list', group_list)
                for groupID in group_list:
                    if node in bush['merges']['approach_flows'][groupID] \
                            and pred_node in bush['merges']['approach_flows'][groupID][node] \
                            and bush['merges']['approach_flows'][groupID][node][pred_node] >= 1e-9:
                        SPcost_carried_nodeFlow += bush['merges']['approach_flows'][groupID][node][pred_node]
                        count += 1
                        if log :
                            print('Check', bush['LPcost'][groupID][pred_node], link_cost, multipliers,
                                  bush['LPcost'][groupID][node], node not in bush['merges']['longest_pred'][groupID], bush['LPcost'][groupID][pred_node] + link_cost > bush['LPcost'][groupID][node])
                        if node not in bush['merges']['longest_pred'][groupID] \
                                or bush['LPcost'][groupID][pred_node] + link_cost > bush['LPcost'][groupID][node]:

                            bush['LPcost'][groupID][node] = bush['LPcost'][groupID][pred_node] + link_cost
                            #print('Setting LPcost for', node, 'as', longest_new_cost, longest_new_cost-multipliers)
                            bush['LP_generalized_cost'][groupID][node] = bush['LPcost'][groupID][pred_node] + link_cost + multipliers


                            bush['merges']['longest_pred'][groupID][node] = pred_node

                        if log:
                            print('Updated LPcost', bush['LPcost'][groupID][node])

                #if len(group_list):
                #    print('Group num {}/{} in {}'.format(count, len(group_list), time.time()-a))
                if SPcost_carried_nodeFlow >= 1e-9:
                    bush['SPcost_carried'][node] = bush['SPcost_carried'][pred_node] + link_cost - multipliers
                #else:
                #    print('Bush scan LP not add ({}, {})'.format(pred_node, node), longest_new_cost, multipliers,  longest_new_cost-multipliers, bush['LPcost'][node])


    return bush

def BPR_derivative(link):
    #print('In Derivative {} {} {}'.format(link['flow'], link['Capacity'], link['beta']))
    if link['flow'] == 0 and (link['beta']-1) < 0:
        return 0.0
    else:
        return link['ff_tt'] * link['alpha'] * link['beta'] * np.power((link['flow']/max(link['Capacity'], 1e-8)), (link['beta']-1)) / max(link['Capacity'], 1e-8)


def compute_shifted_flow(bush, network, merge_node, inner_iteration):
    # Apply simple projected Newton
    # The network costs are computed once for each outer iteration, therefore the SPcost and LPcost are directly used

    newton_step = 1
    min_derivative = 1e-20

    divergent_node = bush['merges']['shiftable_links'][merge_node]['SP_links'][0][0]
    g = (bush['LPcost']['DA'][merge_node] - bush['LPcost']['DA'][divergent_node]) \
        - (bush['SPcost'][merge_node] - bush['SPcost'][divergent_node])

    h_links = set(bush['merges']['shiftable_links'][merge_node]['SP_links']) \
        .union(set(bush['merges']['shiftable_links'][merge_node]['LP_links']))

    h = sum([BPR_derivative(network['edges'][link]) for link in h_links])

    if h == 0:
        h = min_derivative

    dx = max(0, (newton_step/1) * g / h)

    if dx < 0:
        warnings.warn('Something wrong with negative projected newton')

    #print('\n', dx, bush['SPcost'][merge_node], bush['LPcost'][merge_node])
    #print('LP', [(link, bush['bushFlow'][link]) for link in
    #                      bush['merges']['shiftable_links'][merge_node]['LP_links']])

    if len(bush['merges']['shiftable_links'][merge_node]['LP_links']) >= 1e-9:
        dx = min(dx, min([bush['bushFlow']['DA'][link] for link in
                          bush['merges']['shiftable_links'][merge_node]['LP_links']]))

    return round(dx, 10)
