import warnings
import hashlib
import pickle
import operator
import functools
import time
import numpy as np
from collections import Counter

from routing import dijkstra_search_bheap_all, reconstruct_path
from util import topological_ordering, network_cost_update, bush_scan, compute_shifted_flow, normalized_gap, seq_flow_update, BPR_derivative, network_import_postprocessing

flow_ignore_threshold = 0  # for links with near-zero flows, these flows will be ignored (removed)
thresholdGap = 1e-12
thresholdAEC = 1e-12

penalty = 0
LPRule = 'LONGEST_USED_BUSH_PATH'


def driver_seq_cost_update(bushes, matching, init, multiplier_penalty, matching_assignment, network=None, driver_list=None):
    if driver_list == None:
        driver_list = list(matching['driver'])

    print('Updating Driver seq costs ...')
    a = time.time()
    for driverID in driver_list:
        # For initialization
        # Compute the cost for each sequence from underlying bushes
        for seqID, seq in matching['driver'][driverID]['driver_sequence'].items():

            seq_cost_min = 0
            seq_cost_max = 0
            seq_cost_flow_min = 0
            coupling_passengers = []
            for taskID in range(len(seq)):
                if seq[taskID][0] == seq[taskID][1]:
                    continue

                classID = matching['driver'][driverID]['segment_classID'][seqID][taskID]
                bushID = (classID, seq[taskID][0])
                bush_destination = seq[taskID][1]

                groupID = (taskID, driverID, seqID)
                seq_cost_min += bushes[bushID]['SPcost'][bush_destination]

                coupling_passengers += matching['driver'][driverID]['segment_mapping'][seqID][taskID]

                if not init:
                    seq_cost_flow_min += bushes[bushID]['SPcost_carried'][bush_destination]

                    if groupID in bushes[bushID]['LPcost']:
                        seq_cost_max += bushes[bushID]['LPcost'][groupID][bush_destination]
                    elif bush_destination != seq[taskID][0]:
                        seq_cost_max += -np.inf

                else:
                    if groupID not in bushes[bushID]['bushFlow']:

                        bushes[bushID]['bushFlow'][groupID] = {}
                        bushes[bushID]['merges']['approach_flows'][groupID] = {}

            coupling_passengers = set([tuple(passenger) for passenger in coupling_passengers])
            coupling_multipliers = 0
            multipliers = 0
            if seqID != 0:
                multipliers += multiplier_penalty['driver'].get((driverID, seqID), 0)

            for passenger_info in coupling_passengers:
                if matching['passenger'][passenger_info[0]]['sequence_flow'][0] > 0:
                    coupling_multipliers += matching['passenger'][passenger_info[0]]['sequence_cost'][passenger_info[1]][4] - matching['passenger'][passenger_info[0]]['sequence_cost'][0][0]

            matching['driver'][driverID]['sequence_cost'][seqID] = (seq_cost_min + coupling_multipliers,
                                                                    seq_cost_max + coupling_multipliers,
                                                                    seq_cost_min,
                                                                    seq_cost_max,
                                                                    seq_cost_min - coupling_multipliers,
                                                                    seq_cost_max - coupling_multipliers,
                                                                    )

        seq_costs = np.array(list(matching['driver'][driverID]['sequence_cost'].values()))
        seq_flows = np.array(list(matching['driver'][driverID]['sequence_flow'].values()))
        if seq_costs[seq_flows>0, 0].shape[0]:
            multipliers = np.max(seq_costs[seq_flows>0, 0]) - np.min(seq_costs[seq_flows>0, 0])
            RD_DA_seq = {0: matching['driver'][driverID]['sequence_cost'][0]}
            matching['driver'][driverID]['sequence_cost'] = {seq:
                                                                 (matching['driver'][driverID]['sequence_cost'][seq][0] ,
                                                                  matching['driver'][driverID]['sequence_cost'][seq][1] ,
                                                                  matching['driver'][driverID]['sequence_cost'][seq][2],
                                                                  matching['driver'][driverID]['sequence_cost'][seq][3],
                                                                  matching['driver'][driverID]['sequence_cost'][seq][4] + multipliers,
                                                                  matching['driver'][driverID]['sequence_cost'][seq][5] + multipliers,
                                                                  )
                                                             for seq in matching['driver'][driverID]['sequence_cost']}
            matching['driver'][driverID]['sequence_cost'].update(RD_DA_seq)
    print(' -> Updated Driver seq costs in {}'.format(time.time() - a))
    return matching, bushes


def passenger_seq_cost_update(bushes, matching, init, multiplier_penalty, matching_assignment, network, passenger_list=None):
    if passenger_list == None:
        passenger_list = list(matching['passenger'])

    print('Updating Passenger seq costs ...')
    a = time.time()
    for passengerID in passenger_list:
        # For initialization
        # Compute the cost for each sequence from underlying bushes
        for seqID, seq in matching['passenger'][passengerID]['sequences'].items():
            seq_cost_min = 0
            seq_cost_max = 0
            seq_cost_flow_min = 0
            seq_cost_flow_max = 0
            corr_driverID = seq[1]
            corr_driver_seqID = seq[2]
            corr_driver_task_start_index = seq[3]
            multiplier = 0

            for taskID in range(len(seq[0])):
                if seq[0][taskID][0] == seq[0][taskID][1]:
                    continue

                if 'PT' not in corr_driverID:
                    classID = matching['driver'][corr_driverID]['segment_classID'] \
                                [corr_driver_seqID][taskID+corr_driver_task_start_index].replace('RD', 'RP')
                    passenger_groupID = (taskID + corr_driver_task_start_index, passengerID, seqID)

                    if seqID != 0:
                        multiplier += max(0, multiplier_penalty['passenger'].get((passengerID, seqID), [0, 0])[0]
                                         + multiplier_penalty['passenger'].get((passengerID, 'penalty'), 0)
                                         * round(matching['passenger'][passengerID]['sequence_flow'][seqID]
                                         - matching_assignment.get(seqID//100, 0), 10))
                        multiplier += max(0, multiplier_penalty['passenger'].get((passengerID, seqID), [0, 0])[1]
                                          + multiplier_penalty['passenger'].get((passengerID, 'penalty'), 0)
                                          * round((-1 * matching['passenger'][passengerID]['sequence_flow'][0]), 10))

                else:
                    classID = 'RP_PT'
                    passenger_groupID = (taskID , passengerID, seqID)

                bushID = (classID, seq[0][taskID][0])
                bush_destination = seq[0][taskID][1]

                seq_cost_min += bushes[bushID]['SPcost'][bush_destination]

                if init:
                    if passenger_groupID not in bushes[bushID]['bushFlow']:
                        bushes[bushID]['bushFlow'][passenger_groupID] = {}
                        bushes[bushID]['merges']['approach_flows'][passenger_groupID] = {}
                    seq_cost_max += bushes[bushID]['SPcost'][bush_destination]
                else:
                    if 'PT' in corr_driverID:
                        seq_cost_flow_min += bushes[bushID]['SPcost'][bush_destination]
                        seq_cost_flow_max += bushes[bushID]['SPcost'][bush_destination]
                        seq_cost_max += bushes[bushID]['SPcost'][bush_destination]
                    else:

                        if passenger_groupID in bushes[bushID]['LPcost']:
                            seq_cost_max += bushes[bushID]['LPcost'][passenger_groupID][bush_destination]
                        else:
                            seq_cost_max += -np.inf

                        driver_bushID = (classID.replace('RP', 'RD'), seq[0][taskID][0])
                        groupID = (taskID+corr_driver_task_start_index, corr_driverID, corr_driver_seqID)

                        shortest_segment_route = reconstruct_path(
                            came_from=bushes[driver_bushID]['merges']['shortest_pred'],
                            start=seq[0][taskID][0],
                            goal=bush_destination)
                        shortest_segment_route_links = list(zip(shortest_segment_route[:-1], shortest_segment_route[1:]))
                        seq_cost_flow_min += sum([network['edges'][link]['classCost'][classID]
                                                 for link in shortest_segment_route_links])

                        if groupID not in bushes[driver_bushID]['nodeFlow'] \
                                or bush_destination not in bushes[driver_bushID]['nodeFlow'][groupID] \
                                or bushes[driver_bushID]['nodeFlow'][groupID][bush_destination] < 1e-9:
                            seq_cost_flow_max += sum([network['edges'][link]['classCost'][classID]
                                                      for link in shortest_segment_route_links])
                        else:
                            try:
                                longest_segment_route = reconstruct_path(
                                    came_from=bushes[driver_bushID]['merges']['longest_pred'][groupID],
                                    start=seq[0][taskID][0],
                                    goal=bush_destination)
                                longest_segment_route_links = list(
                                    zip(longest_segment_route[:-1], longest_segment_route[1:]))
                                seq_cost_flow_max += sum([network['edges'][link]['classCost'][classID]
                                                          for link in longest_segment_route_links])
                            except Exception:
                                if bushes[driver_bushID]['nodeFlow'][groupID][bush_destination] >= 1e-4:
                                    print('TO handle error!', driver_bushID, groupID)
                                    # print(bush)
                                    print('bush_topology', bushes[driver_bushID]['bush_topology'])
                                    print('pred', bushes[driver_bushID]['pred'])
                                    print('longest_pred', bushes[driver_bushID]['merges']['longest_pred'][groupID])
                                    raise Exception
                                else:
                                    seq_cost_flow_max += sum([network['edges'][link]['classCost'][classID]
                                                              for link in shortest_segment_route_links])

            # The last column being the multiplier_penalty
            matching['passenger'][passengerID]['sequence_cost'][seqID] = (seq_cost_min, seq_cost_max, seq_cost_flow_min, seq_cost_flow_max, seq_cost_flow_max+multiplier)

    print(' -> Updated Passenger seq costs in {}'.format(time.time()-a))
    return matching, bushes


def bush_flow_push(ori, network, demand, bush, network_identifier=None, rectify=False, flow_shift=False,
                   classID=None, inner_iteration=None, groupID=None, DA=False):

    # The idea is to maintain the bush structure, while only decompose the node and bush flows for each RD-RP group
    # The decomposition is required for later (different) group-specific constrained flow shift
    # Note that, the initial flow push could be further simplified by aggregate flow push

    bush['nodeFlow'][groupID] = {node: 0 for node in network['nodes']}
    if ori in demand:
        demand_node_flows = {des: demand[ori][des] for des in demand[ori]}  # Generalized for DA flow as well
        bush['nodeFlow'][groupID].update(demand_node_flows)

    if groupID == 'DA':
        classID = 'DA'

    if groupID not in bush['associate_group_list']:
        bush['associate_group_list'].append(groupID)

    groupID_index = bush['associate_group_list'].index(groupID)
    bush['node_group_incident'] = bush['node_group_incident'][bush['node_group_incident'][:, 1] != groupID_index]
    new_node_group_incident = {ori}

    backtrack_bush_topology = bush['bush_topology'].copy()
    backtrack_bush_topology.reverse()

    if (not flow_shift or groupID not in bush['bushFlow']) and not rectify:
        bush['bushFlow'][groupID] = {}
        bush['merges']['approach_flows'][groupID] = {}

    updated_links = []

    for node in backtrack_bush_topology:
        if len(bush['pred'][node]) == 1:
            pred_node = bush['pred'][node][0]
            if pred_node:

                pred_link_bushFlow = 0
                if (pred_node, node) in bush['bushFlow'][groupID]:
                    pred_link_bushFlow = bush['bushFlow'][groupID][(pred_node, node)]
                if bush['nodeFlow'][groupID][node] != pred_link_bushFlow:
                    if (bush['nodeFlow'][groupID][node] - pred_link_bushFlow) == np.inf:
                        raise ValueError('1. Inf flow in {} {} update {} from {}'.format(groupID, (pred_node, node),
                                                                                      bush['nodeFlow'][groupID][node],
                                                                                      pred_link_bushFlow))

                    network['edges'][(pred_node, node)]['flow'] = round(network['edges'][(pred_node, node)]['flow'] + (bush['nodeFlow'][groupID][node] - pred_link_bushFlow), 10)
                    if classID not in network['edges'][(pred_node, node)]['classFlow']:
                        network['edges'][(pred_node, node)]['classFlow'][classID] = 0
                    if (bush['nodeFlow'][groupID][node] - pred_link_bushFlow) == np.inf:
                        raise ValueError('2. Inf flow in {} {} update {} from {}'.format(groupID, node, bush['nodeFlow'][groupID][node], pred_link_bushFlow))
                    network['edges'][(pred_node, node)]['classFlow'][classID] = round(network['edges'][(pred_node, node)]['classFlow'][classID] + (bush['nodeFlow'][groupID][node] - pred_link_bushFlow), 10)

                bush['bushFlow'][groupID][(pred_node, node)] = bush['nodeFlow'][groupID][node]

                if bush['nodeFlow'][groupID][node] > 0:
                    new_node_group_incident.add(node)

                bush['nodeFlow'][groupID][pred_node] += bush['nodeFlow'][groupID][node]  # pushing flows from destination to origin

                if bush['nodeFlow'][groupID][pred_node] > 0:
                    new_node_group_incident.add(pred_node)

                if node not in bush['merges']['approach_flows'][groupID]:
                    bush['merges']['approach_flows'][groupID][node] = {}
                bush['merges']['approach_flows'][groupID][node][pred_node] = bush['nodeFlow'][groupID][node]
                if bush['merges']['approach_flows'][groupID][node][pred_node] <= -1e-8:
                    raise Exception('Error in appraoch flow! {} {} {} {}'.format(groupID, node, pred_node, bush['nodeFlow'][groupID][node]))

                # Could be implemented with additional flag to reduce IO
                bush['merges']['shortest_pred'][node] = pred_node
                if groupID not in bush['merges']['longest_pred']:
                    bush['merges']['longest_pred'][groupID] = {}
                bush['merges']['longest_pred'][groupID][node] = pred_node

                if groupID == (0, (19, 17), 0):
                    print('-->CHeck', bush['nodeFlow'][groupID], (pred_node, node), pred_link_bushFlow, bush['nodeFlow'][groupID][node], bush['nodeFlow'][groupID][pred_node], new_node_group_incident)
        else:  # this is a merge node
            # To handle changes in bush demands before updating the bush structures
            # (heuristic for generating initial feasible solution when demand updates occur)
            if rectify:
                if node not in bush['merges']['approach_flows'][groupID]:
                    bush['merges']['approach_flows'][groupID][node] = {}
                total_flows = sum([bush['merges']['approach_flows'][groupID][node][pred_node]
                                         for pred_node in bush['merges']['approach_flows'][groupID][node]])  # Avoid floating error
                if total_flows > 0 and (bush['nodeFlow'][groupID][node] / total_flows) < np.inf:
                    for pred_node in bush['merges']['approach_flows'][groupID][node]:
                        # In case of changes demands, rescale it by the same approach proportion
                        # print('Proportion', bush['nodeFlow'][node] / total_flows, bush['nodeFlow'][node], total_flows)
                        if (bush['nodeFlow'][groupID][node] / total_flows) == np.inf:
                            raise ValueError('7. Inf flow in {} {} update {} from {}'.format(groupID, node,
                                                                                             bush['nodeFlow'][groupID][node],
                                                                                             total_flows))
                        bush['merges']['approach_flows'][groupID][node][pred_node] = round(bush['merges']['approach_flows'][groupID][node][pred_node] * bush['nodeFlow'][groupID][node] / total_flows, 10)
                        if bush['merges']['approach_flows'][groupID][node][pred_node] <= -1e-8:
                            raise Exception('Error in appraoch flow 2 {} {} {} {} {} {}'.format(groupID, node, pred_node, bush['merges']['approach_flows'][groupID][node][pred_node], bush['nodeFlow'][groupID][node], total_flows))
                else:  # In case of removed demands, push all existing flows to shortest path
                    for pred_node in bush['merges']['approach_flows'][groupID][node]:
                        bush['merges']['approach_flows'][groupID][node][pred_node] = 0
                    bush['merges']['approach_flows'][groupID][node][bush['merges']['shortest_pred'][node]] = bush['nodeFlow'][groupID][node]
                    if bush['merges']['approach_flows'][groupID][node][bush['merges']['shortest_pred'][node]] <= -1e-8:
                        raise Exception('Error in appraoch flow! {} {} {} {}'.format(groupID, node, bush['merges']['shortest_pred'][node],
                                                                                     bush['nodeFlow'][groupID][node]))

            if flow_shift and groupID == 'DA':
                if node in bush['merges']['shiftable_links'] and len(bush['merges']['shiftable_links'][node]['LP_links']) > 0:
                    if bush['merges']['shiftable_links'][node]['SP_links'][0][0] != \
                            bush['merges']['shiftable_links'][node]['LP_links'][0][0] \
                            or bush['merges']['shiftable_links'][node]['SP_links'][-1][1] != \
                            bush['merges']['shiftable_links'][node]['LP_links'][-1][1]:
                        raise Exception(
                            'Backtracking failed! {} {}'.format(bush['merges']['shiftable_links'][node]['SP_links'],
                                                                bush['merges']['shiftable_links'][node]['LP_links']))

                    #print('bush', bush)
                    dx = compute_shifted_flow(bush=bush,
                                              network=network,
                                              merge_node=node,
                                              inner_iteration=inner_iteration)

                    # applying the shifts
                    for link in bush['merges']['shiftable_links'][node]['SP_links']:
                        if dx == np.inf:
                            raise ValueError('3. Inf flow in {} {} update {} from {}'.format(groupID, node,
                                                                                          dx,
                                                                                          network['edges'][link]['classFlow']['DA']))
                        network['edges'][link]['flow'] = round(network['edges'][link]['flow'] + dx, 10)
                        network['edges'][link]['classFlow']['DA'] = round(network['edges'][link]['classFlow']['DA'] + dx, 10)
                        updated_links.append(link)
                        if link not in bush['bushFlow'][groupID]:
                            bush['bushFlow'][groupID][link] = 0
                        bush['bushFlow'][groupID][link] = round(bush['bushFlow'][groupID][link] + dx, 10)
                        if link[1] not in bush['merges']['approach_flows'][groupID]:
                            bush['merges']['approach_flows'][groupID][link[1]] = {}

                        if link[0] not in bush['merges']['approach_flows'][groupID][link[1]]:
                            bush['merges']['approach_flows'][groupID][link[1]][link[0]] = 0
                        bush['merges']['approach_flows'][groupID][link[1]][link[0]] = round(bush['merges']['approach_flows'][groupID][link[1]][link[0]] + dx, 10)

                    for link in bush['merges']['shiftable_links'][node]['LP_links']:
                        tmp = network['edges'][link]['flow']
                        if dx == np.inf:
                            raise ValueError('4. Inf flow in {} {} update {} from {}'.format(groupID, node,
                                                                                          dx,
                                                                                          network['edges'][link]['classFlow']['DA']))
                        network['edges'][link]['flow'] = round(network['edges'][link]['flow'] - dx, 10)
                        network['edges'][link]['classFlow']['DA'] = round(network['edges'][link]['classFlow']['DA'] - dx, 10)
                        if network['edges'][link]['flow'] < 0 and abs(network['edges'][link]['flow']) <= 1e-9:
                            network['edges'][link]['flow'] = 0
                        if network['edges'][link]['classFlow']['DA'] < 0 \
                                and abs(network['edges'][link]['classFlow']['DA']) <= 1e-9:
                            network['edges'][link]['classFlow']['DA'] = 0

                        updated_links.append(link)
                        if link not in bush['bushFlow'][groupID]:
                            bush['bushFlow'][groupID][link] = 0
                        bush['bushFlow'][groupID][link] = round(bush['bushFlow'][groupID][link] - dx, 10)

                        if link[1] in bush['merges']['approach_flows'][groupID]:
                            bush['merges']['approach_flows'][groupID][link[1]][link[0]] = round(bush['merges']['approach_flows'][groupID][link[1]][link[0]] - dx, 10)

                            if bush['merges']['approach_flows'][groupID][link[1]][link[0]] < 0 \
                                    and abs(bush['merges']['approach_flows'][groupID][link[1]][link[0]]) <= 1e-9:
                                bush['merges']['approach_flows'][groupID][link[1]][link[0]] = 0

                            if bush['merges']['approach_flows'][groupID][link[1]][link[0]] < 0:
                                raise ValueError('Negative approach flow on link {} {} {}'
                                                 .format(link, bush['merges']['approach_flows'][groupID][link[1]][link[0]], dx))

                        if network['edges'][link]['flow'] < 0 or network['edges'][link]['classFlow']["DA"] < 0 \
                                or bush['bushFlow'][groupID][link] < 0:
                            #print(network['edges'][link]['classFlow'])
                            raise ValueError('Negative edge flow on link {} {} {} {} {} {}'
                                             .format(link, network['edges'][link]['flow'],
                                                     network['edges'][link]['classFlow']['DA'],
                                                     bush['bushFlow'][groupID][link], dx, tmp))


            for pred_node in bush['pred'][node]:
                # the approach flows are computed after update bushes
                if node not in bush['merges']['approach_flows'][groupID]:
                    bush['merges']['approach_flows'][groupID][node] = {}
                if pred_node not in bush['merges']['approach_flows'][groupID][node]:
                    bush['merges']['approach_flows'][groupID][node][pred_node] = 0

                approach_flow = bush['merges']['approach_flows'][groupID][node][pred_node]
                pred_link_bushFlow = 0
                if (pred_node, node) in bush['bushFlow'][groupID]:
                    pred_link_bushFlow = bush['bushFlow'][groupID][(pred_node, node)]
                if approach_flow != pred_link_bushFlow:
                    if (approach_flow - pred_link_bushFlow) == np.inf:
                        raise ValueError('5. Inf flow in {} {} update {} from {}, {} {}'.format(groupID, node,
                                                                                      approach_flow,
                                                                                      pred_link_bushFlow, rectify, flow_shift))
                    network['edges'][(pred_node, node)]['flow'] = round(network['edges'][(pred_node, node)]['flow'] + (approach_flow - pred_link_bushFlow), 10)

                    if classID not in network['edges'][(pred_node, node)]['classFlow']:
                        network['edges'][(pred_node, node)]['classFlow'][classID] = 0
                    network['edges'][(pred_node, node)]['classFlow'][classID] = round(network['edges'][(pred_node, node)]['classFlow'][classID] + (approach_flow - pred_link_bushFlow), 10)

                bush['bushFlow'][groupID][(pred_node, node)] = approach_flow
                # pushing flows from destination to origin
                bush['nodeFlow'][groupID][pred_node] = round(bush['nodeFlow'][groupID][pred_node] + approach_flow, 10)

                if groupID == (0, (19, 17), 0):
                    print('-->CHeck', rectify, bush['nodeFlow'][groupID], (pred_node, node), bush['nodeFlow'][groupID][node], bush['nodeFlow'][groupID][pred_node])

                if bush['nodeFlow'][groupID][pred_node] > 0:
                    new_node_group_incident.add(pred_node)

                if bush['nodeFlow'][groupID][node] > 0:
                    new_node_group_incident.add(node)

    if len(new_node_group_incident) > 1:
        new_node_group_incident = np.array([list(new_node_group_incident), groupID_index * np.ones(len(new_node_group_incident))]).T
        bush['node_group_incident'] = np.vstack((bush['node_group_incident'], new_node_group_incident))

    if network_identifier:
        bush['network_identifier'] = network_identifier

    if DA and flow_shift:
        return bush, network, updated_links
    else:
        return bush, network


def group_flow_push(bushes, network, demands, network_md5, class_names, matching, matching_assignment, multiplier_penalty, outer_iter,
                    init=False, rectify=False):
    # Compute the cost for each sequence for the drivers and passengers from underlying bushes
    init_bushFlow = False
    if init and not rectify:
        init_bushFlow = True

    matching, bushes = passenger_seq_cost_update(bushes, matching, init_bushFlow, multiplier_penalty, matching_assignment, network)
    matching, bushes = driver_seq_cost_update(bushes, matching, init_bushFlow, multiplier_penalty, matching_assignment, network)

    # Do this at initialization
    for driverID in matching['driver']:
        # For initialization

        previous_total_flow = sum(matching['driver'][driverID]['sequence_flow'].values())  # but the flow remains the same (not yet updated)

        if driverID == (19, 17):
            print('DEMAND', matching['driver'][driverID]['sequence_flow'],  demands['RD'][driverID[0]][driverID[1]])

        if (driverID[0] not in demands['RD'] or driverID[1] not in demands['RD'][driverID[0]]) and previous_total_flow < 1e-9:
            continue

        # Initialization by finding sequence with minimal cost
        if not rectify:
            pushed_seqIDs = [0]
            demand = {pushed_seqIDs[0]: demands['RD'][driverID[0]][driverID[1]]}
        else:

            pushed_seqIDs = list(matching['driver'][driverID]['sequence_flow']) # matching could be updated

            # Clean bushes
            if (driverID[0] not in demands['RD'] or driverID[1] not in demands['RD'][driverID[0]]) and previous_total_flow >= 1e-9:
                if driverID[0] not in demands['RD']:
                    demands['RD'][driverID[0]] = {}
                demands['RD'][driverID[0]][driverID[1]] = 0

            if abs(previous_total_flow - demands['RD'][driverID[0]][driverID[1]]) < 1e-9:
                continue

            if previous_total_flow > 0:
                # TO DO: This one can also be set by the min among RD, RP and matching(Z)

                if outer_iter > 0:
                    demand = {seqID: matching['driver'][driverID]['sequence_flow'][seqID]
                              for seqID in list(matching['driver'][driverID]['sequence_flow'])}

                    if demands['RD'][driverID[0]][driverID[1]] > previous_total_flow:
                        max_cost_with_flow = [v[0] for k, v in sorted(matching['driver'][driverID]['sequence_cost'].items(), key=lambda item: item[1][1], reverse=True)
                                              if matching['driver'][driverID]['sequence_flow'][k] >= 1e-9][0]
                        min_cost_seqIDs = [k for k, v in sorted(matching['driver'][driverID]['sequence_cost'].items(), key=lambda item: item[1][1])
                                           if v[0] <= max_cost_with_flow and v[0] <= matching['driver'][driverID]['sequence_cost'][0][0]
                                           and ((matching_assignment.get(k, 0) - matching['driver'][driverID]['sequence_flow'][k]) >= 1e-9 or k == 0)]
                        if len(min_cost_seqIDs) and min_cost_seqIDs[0] != 0:
                            remain_cap = {k: matching_assignment[k] - matching['driver'][driverID]['sequence_flow'][k] for k in min_cost_seqIDs if k != 0}
                            # Amount to increase in the sequences
                            increase = demands['RD'][driverID[0]][driverID[1]] - previous_total_flow
                            actual_increase = min(increase, sum(remain_cap.values()))
                            demand.update({seqID: matching['driver'][driverID]['sequence_flow'][seqID] + actual_increase * remain_cap[seqID]/sum(remain_cap.values()) for seqID in min_cost_seqIDs if seqID != 0})
                    else:
                        if round(demands['RD'][driverID[0]][driverID[1]], 10) == 0:
                            demand = {seqID: 0 for seqID in list(matching['driver'][driverID]['sequence_flow'])}
                        else:
                            reduction = round(previous_total_flow - demands['RD'][driverID[0]][driverID[1]], 10)

                            if reduction >= 1e-9:
                                min_cost = min([v[0] for k, v in matching['driver'][driverID]['sequence_cost'].items()
                                                if matching_assignment.get(k, 0) >= 1e-9 or matching['driver'][driverID]['sequence_flow'][k] >= 1e-9 or k == 0])
                                # Gap can be zero, but it should still be considered
                                seq_flow_gap = {k: v[0] - min_cost for k, v in matching['driver'][driverID]['sequence_cost'].items() if matching['driver'][driverID]['sequence_flow'][k] >= 1e-9}
                                # If zero, means all flows are on the cheapest already, reduction has to be done on the cheapest as well
                                if sum(seq_flow_gap.values()) > 0:
                                    counter = 0

                                    while reduction >= 1e-6:
                                        shift = {seqID: round(min(demand[seqID], reduction * seq_flow_gap[seqID] / sum(seq_flow_gap.values())), 10) for seqID in seq_flow_gap}
                                        demand.update({seqID: demand[seqID] - shift[seqID] for seqID in seq_flow_gap})
                                        print('Iter', reduction, sum(shift.values()))
                                        reduction -= sum(shift.values())
                                        reduction = round(reduction, 10)
                                        if counter > 2:
                                            break
                                        counter += 1

                                    max_cost_seqIDs = [k for k, v in sorted(matching['driver'][driverID]['sequence_cost'].items(), key=lambda item: item[1][1]) if demand[k] >= 1e-9]
                                    seq_flows = [demand[k] for k in max_cost_seqIDs]
                                    residual = np.cumsum(seq_flows) - reduction
                                    index = np.where(residual > 0)[0]
                                    #print(residual, index, demand, seq_flow_gap, max_cost_seqIDs)
                                    if len(index) and index[0] > 0:
                                        reduce_seqs = max_cost_seqIDs[:index[0]]
                                        demand.update({seqID: 0 for seqID in reduce_seqs})
                                        demand[max_cost_seqIDs[index[0]]] = residual[index[0]]
                                    else:
                                        demand[max_cost_seqIDs[0]] = residual[0]
                                else: # Reduce all by the same ratio
                                    demand.update({seqID: round(demand[seqID] * demands['RD'][driverID[0]][driverID[1]] / previous_total_flow, 10)
                                                   for seqID in pushed_seqIDs if seqID in matching_assignment or seqID == 0 or demand[seqID] >= 1e-9})

                    remaining_demand = demands['RD'][driverID[0]][driverID[1]] - sum(demand.values())
                    demand[0] += remaining_demand
                else:
                    if demands['RD'][driverID[0]][driverID[1]] > previous_total_flow:
                        demand = {seqID: 0 for seqID in list(matching['driver'][driverID]['sequence_flow'])}
                        demand.update({seqID: round(matching['driver'][driverID]['sequence_flow'][seqID]
                                         * demands['RD'][driverID[0]][driverID[1]]/previous_total_flow, 10)
                                                for seqID in pushed_seqIDs if seqID in matching_assignment or seqID == 0 or matching['driver'][driverID]['sequence_flow'][seqID] >= 1e-9})

                    else:
                        demand = {seqID: matching['driver'][driverID]['sequence_flow'][seqID]
                                  for seqID in list(matching['driver'][driverID]['sequence_flow'])}
                        if round(demands['RD'][driverID[0]][driverID[1]], 10) == 0:
                            demand = {seqID: 0 for seqID in list(matching['driver'][driverID]['sequence_flow'])}
                        else:
                            reduction = round(previous_total_flow - demands['RD'][driverID[0]][driverID[1]], 10)

                            if reduction >= 1e-9:
                                min_cost = min([v[0] for k, v in matching['driver'][driverID]['sequence_cost'].items()
                                                if matching_assignment.get(k, 0) >= 1e-9 or
                                                matching['driver'][driverID]['sequence_flow'][k] >= 1e-9 or k == 0])
                                # Gap can be zero, but it should still be considered
                                seq_flow_gap = {k: v[0] - min_cost for k, v in
                                                matching['driver'][driverID]['sequence_cost'].items() if
                                                matching['driver'][driverID]['sequence_flow'][k] >= 1e-9}
                                # If zero, means all flows are on the cheapest already, reduction has to be done on the cheapest as well
                                if sum(seq_flow_gap.values()) > 0:
                                    counter = 0

                                    while reduction >= 1e-6:
                                        shift = {seqID: round(min(demand[seqID], reduction * seq_flow_gap[seqID] / sum(
                                            seq_flow_gap.values())), 10) for seqID in seq_flow_gap}
                                        demand.update({seqID: demand[seqID] - shift[seqID] for seqID in seq_flow_gap})
                                        print('Iter', reduction, sum(shift.values()))
                                        reduction -= sum(shift.values())
                                        reduction = round(reduction, 10)
                                        if counter > 2:
                                            break
                                        counter += 1

                                    max_cost_seqIDs = [k for k, v in
                                                       sorted(matching['driver'][driverID]['sequence_cost'].items(),
                                                              key=lambda item: item[1][1]) if demand[k] >= 1e-9]
                                    seq_flows = [demand[k] for k in max_cost_seqIDs]
                                    residual = np.cumsum(seq_flows) - reduction
                                    index = np.where(residual > 0)[0]
                                    if len(index) and index[0] > 0:
                                        reduce_seqs = max_cost_seqIDs[:index[0]]
                                        demand.update({seqID: 0 for seqID in reduce_seqs})
                                        demand[max_cost_seqIDs[index[0]]] = residual[index[0]]
                                    else:
                                        demand[max_cost_seqIDs[0]] = residual[0]
                                else:  # Reduce all by the same ratio
                                    demand.update({seqID: round(
                                        demand[seqID] * demands['RD'][driverID[0]][driverID[1]] / previous_total_flow,
                                        10)
                                                   for seqID in pushed_seqIDs if
                                                   seqID in matching_assignment or seqID == 0 or demand[seqID] >= 1e-9})

                    remaining_demand = demands['RD'][driverID[0]][driverID[1]] - sum(demand.values())
                    demand[0] += remaining_demand
            else:
                pushed_seqIDs = [0]
                demand = {pushed_seqIDs[0]: demands['RD'][driverID[0]][driverID[1]]}

        for seqID in pushed_seqIDs:
            if driverID == (19, 17):
                print('To push', driverID, seqID, demand[seqID])
            matching = seq_flow_update(matching=matching,
                                              driverID=driverID,
                                              driver_seq_ID=seqID,
                                              x=demand[seqID])

            seq = matching['driver'][driverID]['driver_sequence'][seqID]

            if driverID == (19, 17):
                print('push seq', seq)


            # Initialization by simultaneously assigning demands on the minimal cost sequence and route
            for taskID in range(len(seq)):
                bush_ori = seq[taskID][0]
                bush_destination = seq[taskID][1]
                if bush_ori == bush_destination:  # No need to do anything, skip
                    continue
                classID = matching['driver'][driverID]['segment_classID'][seqID][taskID]
                bushID = (classID, bush_ori)
                groupID = (taskID, driverID, seqID)

                # Initialization by assigning all demands to the shortest path (which also corresponds to the cheapest sequence)
                bush_demand = {bush_ori: {bush_destination: demand[seqID]}}

                # This function indeed assigns flows on the shortest (cheapest) route
                bush, network = bush_flow_push(ori=bush_ori,
                                                network=network,
                                                demand=bush_demand,
                                                bush=bushes[bushID],
                                                network_identifier=network_md5,
                                                classID=classID,
                                                groupID=groupID,
                                                rectify=rectify)
                bushes[bushID] = bush

                if driverID == (19, 17):
                    print('node_group_incident', bushes[bushID]['node_group_incident'], bush['associate_group_list'])

                classID_split = classID.split('_')
                # Only copy if this is RD flow with onboard passenger
                if len(classID_split) > 1 and classID_split[1] != 'DA' and int(classID_split[1]) > 0:  # Copy to passenger flows
                    passenger_classID = classID.replace('RD', 'RP')
                    passenger_bushID = (passenger_classID, seq[taskID][0])
                    passenger_groupIDs = [(taskID, passenger[0], passenger[1])  # not sure yet, should the taskID for the driver/passenger
                                          for passenger in matching['driver'][driverID]['segment_mapping'][seqID][taskID]]

                    # The bush structure of the driver and passe ger are the same, copy shortest/longest_pred to passenger
                    bushes[passenger_bushID]['merges']['shortest_pred'] \
                        = bushes[bushID]['merges']['shortest_pred']
                    bushes[passenger_bushID]['merges']['longest_pred'] \
                        = bushes[bushID]['merges']['longest_pred']

                    tmp_passenger_bushFlow = {
                        passenger_groupID: bushes[bushID]['bushFlow'][groupID] for passenger_groupID in passenger_groupIDs
                    }
                    # Enforcing coupling, copy driver flows to passengers
                    bushes[passenger_bushID]['bushFlow'].update(tmp_passenger_bushFlow)

                    existing_groups = set(bushes[passenger_bushID]['associate_group_list'])
                    add_groups = set(passenger_groupIDs) - existing_groups
                    bushes[passenger_bushID]['associate_group_list'] += list(add_groups)
                    group_index = [bushes[passenger_bushID]['associate_group_list'].index(passenger_groupID) for
                                   passenger_groupID in passenger_groupIDs]

                    driver_group_index = bushes[bushID]['associate_group_list'].index(groupID)
                    driver_node_group_incident = bushes[bushID]['node_group_incident'][
                        bushes[bushID]['node_group_incident'][:, 1] == driver_group_index]

                    passenger_node_group_incident = bushes[passenger_bushID]['node_group_incident'][
                        ~np.isin(bushes[passenger_bushID]['node_group_incident'][:, 1], group_index)]
                    passenger_node_group_incident = np.vstack((passenger_node_group_incident,
                                                              np.array([np.tile(driver_node_group_incident[:, 0],
                                                                                len(group_index)),
                                                                        np.repeat(group_index,
                                                                                  driver_node_group_incident.shape[
                                                                                      0])]).T))

                    bushes[passenger_bushID]['node_group_incident'] = passenger_node_group_incident

                    tmp_passenger_nodeFlow = {
                        passenger_groupID: bushes[bushID]['nodeFlow'][groupID] for passenger_groupID in passenger_groupIDs
                    }
                    # Enforcing coupling, copy driver flows to passengers
                    bushes[passenger_bushID]['nodeFlow'].update(tmp_passenger_nodeFlow)

                    tmp_passenger_nodeFlow = {
                        passenger_groupID: bushes[bushID]['merges']['approach_flows'][groupID]
                        for passenger_groupID in passenger_groupIDs
                    }
                    # Enforcing coupling, copy driver flows to passengers
                    bushes[passenger_bushID]['merges']['approach_flows'].update(tmp_passenger_nodeFlow)

                    # Sync the identifier, since the bush structure is the same
                    bushes[passenger_bushID]['network_identifier'] = bushes[bushID]['network_identifier']

    for ori in demands['RP']:
        for des in demands['RP'][ori]:  # The RP_PT flow should be override each time
            matching['passenger'][(ori, des)]['sequence_flow'][0] = 0
            if matching['passenger'][(ori, des)]['sequences'][0][1] != 'PT':
                raise IndexError('Error in RP_PT index! It should be the last sequence!')
            else:
                matching['passenger'][(ori, des)]['sequence_flow'][0] \
                    = round(demands['RP'][ori][des] - sum( matching['passenger'][(ori, des)]['sequence_flow'].values()), 10)

    return bushes, matching, network

def sequence_flow_shift(matching, matching_assignment, bushes, driverID, network, inner_iter, outer_iter, prev_sequence_cost, step_size=1):  # only flows between a divergent and merge nodes need to be shifted


    found_shift = False

    carried_flow_seqID = [seqID for seqID, seq_flow in matching['driver'][driverID]['sequence_flow'].items() if round(seq_flow, 10) > 1e-9]
    max_cost_seqIDs = [k for k, v in sorted(matching['driver'][driverID]['sequence_cost'].items(), key=lambda item: item[1][1], reverse=True) if k in carried_flow_seqID]
    min_cost_seqIDs = [k for k, v in sorted(matching['driver'][driverID]['sequence_cost'].items(), key=lambda item: item[1][0])
                       if k == 0
                       or (k in matching_assignment and matching_assignment[k]>1e-9)
                       or (k in carried_flow_seqID)]

    dx_aux = 0
    longest_seq_links = []
    shortest_seq_links = []
    LP_bush_demand_shift_count = []
    SP_bush_demand_shift_count = []

    prev_sequence_cost_data = prev_sequence_cost.get(inner_iter - 2, None)

    max_cost_seqID = None
    min_cost_seqID = None

    for max_cost_seqID in max_cost_seqIDs:
        #print('Running max', max_cost_seqID, matching['driver'][driverID]['sequence_cost'][max_cost_seqID], matching_assignment.get(max_cost_seqID, 0))
        # For shifting from RD seq, there is only one iteration, since there is always positive gradient
        # When the max_cost_seqID corresponds to the most costly sequence (i.e., index 0),
        # its generalized cost might not be the maximum, therefore we could try to improve the next costly seq
        segment_occupancy_sum = 0

        min_segment_flow = np.inf

        max_seq = matching['driver'][driverID]['driver_sequence'][max_cost_seqID]
        longest_seq_links = []
        LP_bush_demand_shift_count = {}

        on_board_passengers = []
        gradient = 0

        for taskID in range(len(max_seq)):
            bush_ori = max_seq[taskID][0]
            bush_destination = max_seq[taskID][1]
            if bush_ori == bush_destination:  # No need to do anything, skip
                continue

            classID = matching['driver'][driverID]['segment_classID'][max_cost_seqID][taskID]
            bushID = (classID, bush_ori)
            groupID = (taskID, driverID, max_cost_seqID)

            try:
                longest_segment_route = reconstruct_path(came_from=bushes[bushID]['merges']['longest_pred'][groupID],
                                                         start=bush_ori,
                                                         goal=bush_destination)
            except Exception:
                print('Max flow', max_cost_seqID, matching['driver'][driverID]['sequence_flow'][max_cost_seqID])
                print('node_group_incident', bushes[bushID]['node_group_incident'])
                print('bush_topology', bushes[bushID]['bush_topology'])
                print('pred', bushes[bushID]['pred'])
                print('SeqFlow', np.array(list(matching['driver'][driverID]['sequence_flow'].values())))
                print('bushFlow', bushes[bushID]['bushFlow'][groupID])
                print('nodeFlow', bushes[bushID]['nodeFlow'][groupID])
                print('longest_pred', bushes[bushID]['merges']['longest_pred'])
                print('longest_pred_group', bushID, groupID, bushes[bushID]['merges']['longest_pred'][groupID])
                groupID_index = bushes[bushID]['associate_group_list'].index(groupID)
                print('node_group_incident', groupID_index, bushes[bushID]['node_group_incident'][bushes[bushID]['node_group_incident'][:, 1] == groupID_index])
                raise Exception

            if bushID not in LP_bush_demand_shift_count:
                LP_bush_demand_shift_count[(bushID, groupID, bush_destination)] = 1
            else:
                LP_bush_demand_shift_count[(bushID, groupID, bush_destination)] += 1

            longest_segment_route_links = list(zip(longest_segment_route[:-1], longest_segment_route[1:]))
            longest_seq_links += [(bushID, groupID, link) for link in longest_segment_route_links]

            min_longest_route_bush_flow = min([bushes[bushID]['bushFlow'][groupID][link]
                                               for link in longest_segment_route_links])

            if min_longest_route_bush_flow < min_segment_flow:
                min_segment_flow = min_longest_route_bush_flow

            segment_on_board_passengers = matching['driver'][driverID]['segment_mapping'][max_cost_seqID][taskID]
            on_board_passengers += segment_on_board_passengers
            segment_occupancy_sum += (1 + len(segment_on_board_passengers))

        exceed_current = False
        for min_cost_seqID in min_cost_seqIDs:
            if exceed_current:
                break
            if min_cost_seqID == max_cost_seqID:
                exceed_current = True
            #print('Running min', min_cost_seqID, min_cost_seqIDs, matching['driver'][driverID]['sequence_cost'][min_cost_seqID], matching_assignment.get(min_cost_seqID, 0))
            # One iteration for RD to any sequence, since the gradient_driver > 0
            # Since for shifting from DA to RD, some passengers are involved, but the multipliers are not known in advanced
            # Therefore, the selection of the min sequence (for the group) is not trivial
            # for better convergence for shifting from DA to RD, we iterate through sequences (ascending in cost)
            gradient_driver = matching['driver'][driverID]['sequence_cost'][max_cost_seqID][1] - matching['driver'][driverID]['sequence_cost'][min_cost_seqID][0]
            prev_gradient_driver = None
            if prev_sequence_cost_data != None:
                if prev_sequence_cost_data['driver'][driverID][max_cost_seqID][1] > -np.inf:
                    prev_gradient_driver = prev_sequence_cost_data['driver'][driverID][max_cost_seqID][1] - \
                                           prev_sequence_cost_data['driver'][driverID][min_cost_seqID][0]
                else:
                    prev_gradient_driver = prev_sequence_cost_data['driver'][driverID][max_cost_seqID][0] - \
                                           prev_sequence_cost_data['driver'][driverID][min_cost_seqID][0]

            min_seq = matching['driver'][driverID]['driver_sequence'][min_cost_seqID]
            shortest_seq_links = []
            SP_bush_demand_shift_count = {}
            shortest_gradient_passenger = set()

            for taskID in range(len(min_seq)):
                bush_ori = min_seq[taskID][0]
                bush_destination = min_seq[taskID][1]
                if bush_ori == bush_destination:  # No need to do anything, skip
                    continue
                classID = matching['driver'][driverID]['segment_classID'][min_cost_seqID][taskID]
                bushID = (classID, bush_ori)
                groupID = (taskID, driverID, min_cost_seqID)
                shortest_segment_route = reconstruct_path(came_from=bushes[bushID]['merges']['shortest_pred'],
                                                         start=bush_ori,
                                                         goal=bush_destination)

                shortest_matched_passengers = matching['driver'][driverID]['segment_mapping'][min_cost_seqID][taskID]
                if bushID not in SP_bush_demand_shift_count:
                    SP_bush_demand_shift_count[(bushID, groupID, bush_destination)] = 1
                else:
                    SP_bush_demand_shift_count[(bushID, groupID, bush_destination)] += 1

                shortest_segment_route_links = list(zip(shortest_segment_route[:-1], shortest_segment_route[1:]))
                shortest_seq_links += [(bushID, groupID, link) for link in shortest_segment_route_links]

                #segment_occupancy_sum += 1  # For the driver
                if len(shortest_matched_passengers):  # Shifting passengers from reserve to the matching options
                    for passenger_seq in shortest_matched_passengers:
                        shortest_gradient_passenger.add(tuple(passenger_seq))
                        #segment_occupancy_sum += 1  # For the passengers

            coupling_multipliers = 0
            prev_coupling_multipliers = 0

            if prev_sequence_cost_data != None and outer_iter == 0:
                gradient = (2 * (gradient_driver + coupling_multipliers) - (
                            prev_gradient_driver + prev_coupling_multipliers)) / segment_occupancy_sum
                # print('Gradient components1', gradient, gradient_driver, coupling_multipliers, prev_gradient_driver,
                #       prev_coupling_multipliers, segment_occupancy_sum)
            else:
                gradient = (gradient_driver + coupling_multipliers) / segment_occupancy_sum
                #print('Gradient components2', gradient, gradient_driver, coupling_multipliers, segment_occupancy_sum)

            if max_cost_seqID == 0 and gradient > 0:
                shortest_gradient_passenger_list = Counter([item[0] for item in shortest_gradient_passenger])
                shortest_gradient_passenger_list = [matching['passenger'][passengerID]['sequence_flow'][0]/count for passengerID, count in shortest_gradient_passenger_list.items()]
                gradient = min(gradient, min(shortest_gradient_passenger_list))

            if gradient > 0:
                found_shift = True
                break

        dx_aux = min(min_segment_flow, gradient)

        # if prev_sequence_cost_data != None:
        #     print('Auxiliary dx', dx_aux, 'min_segment_flow', min_segment_flow, 'gradient', gradient,
        #           'gradient_components', gradient_driver, prev_gradient_driver, segment_occupancy_sum, 'coupling_multipliers', coupling_multipliers, 'From',
        #           max_cost_seqID, 'To', min_cost_seqID, found_shift)
        # else:
        #     print('Auxiliary dx', dx_aux, 'min_segment_flow', min_segment_flow, 'gradient', gradient,
        #           'gradient_components', gradient_driver, None, segment_occupancy_sum,
        #           'coupling_multipliers', coupling_multipliers, 'From',
        #           max_cost_seqID, 'To', min_cost_seqID, found_shift)
        if found_shift:
            break

    dx = max(0, dx_aux)
    #print('Shift: dx', dx)
    # where to shift flows: longest_seq_links, shortest_seq_links
    # how much: gradient
    return longest_seq_links, shortest_seq_links, dx, \
           LP_bush_demand_shift_count, SP_bush_demand_shift_count, min_cost_seqID, max_cost_seqID, prev_sequence_cost, gradient

def bush_flow_shift(links, dx, demand_shift_counts, sign, bushes, network, matching):
    # Update bushFlow and approach_flow
    #print('Applying flows {} {}'.format(sign, dx), links)
    positive_shift_links = {}
    for link in links:
        if (sign * dx) == np.inf:
            raise ValueError('6. Inf flow in {} {} update {} from {}'.format(link[0][0], link,
                                                                          (sign * dx),
                                                                          network['edges'][link[2]]['flow']))
        network['edges'][link[2]]['flow'] = round(network['edges'][link[2]]['flow'] + (sign * dx), 10)
        network['edges'][link[2]]['classFlow'][link[0][0]] = round(network['edges'][link[2]]['classFlow'][link[0][0]] + (sign * dx), 10)
        # groupID = link[1]
        if link[2] not in bushes[link[0]]['bushFlow'][link[1]]:
            bushes[link[0]]['bushFlow'][link[1]][link[2]] = 0

        if link[1] == (2, (11, 20), 22557):
            print(link, 'bushFlow', bushes[link[0]]['bushFlow'][link[1]][link[2]], (sign * dx))
            if link[1] in bushes[link[0]]['merges']['approach_flows'] and link[2][1] in bushes[link[0]]['merges']['approach_flows'][link[1]] and link[2][0] in bushes[link[0]]['merges']['approach_flows'][link[1]][link[2][1]]:
                print('approach_flows', bushes[link[0]]['merges']['approach_flows'][link[1]][link[2][1]][link[2][0]])

        if sign > 0:
            if link[0] not in positive_shift_links:
                positive_shift_links[link[0]] = []
            positive_shift_links[link[0]].append(link[2])

        bushes[link[0]]['bushFlow'][link[1]][link[2]] = round(bushes[link[0]]['bushFlow'][link[1]][link[2]] + (sign * dx), 10)
        if sign > 0 and link[2][1] not in bushes[link[0]]['merges']['approach_flows'][link[1]]:
            bushes[link[0]]['merges']['approach_flows'][link[1]][link[2][1]] = {}

        if sign > 0 and link[2][0] not in bushes[link[0]]['merges']['approach_flows'][link[1]][link[2][1]]:
            bushes[link[0]]['merges']['approach_flows'][link[1]][link[2][1]][link[2][0]] = 0

        try:
            bushes[link[0]]['merges']['approach_flows'][link[1]][link[2][1]][link[2][0]] = round(bushes[link[0]]['merges']['approach_flows'][link[1]][link[2][1]][link[2][0]] + (sign * dx), 10)
        except Exception as err:
            raise Exception('Error in bush flow shift sign {}, bushID{}, groupID {}, link {}, dx {}'.format(sign, link[0], link[1], link[2], dx))

        # Avoid floating error
        if bushes[link[0]]['merges']['approach_flows'][link[1]][link[2][1]][link[2][0]] < 0 \
                and abs(bushes[link[0]]['merges']['approach_flows'][link[1]][link[2][1]][link[2][0]]) < 1e-6:
            bushes[link[0]]['merges']['approach_flows'][link[1]][link[2][1]][link[2][0]] = 0
        elif bushes[link[0]]['merges']['approach_flows'][link[1]][link[2][1]][link[2][0]] < 0:
            raise ValueError('Negative approach_flows! {} {} {} {}'.format(link, sign, dx, bushes[link[0]]['merges']['approach_flows'][link[1]][link[2][1]][link[2][0]]))
        if bushes[link[0]]['bushFlow'][link[1]][link[2]] < 0 \
                and abs(bushes[link[0]]['bushFlow'][link[1]][link[2]]) < 1e-6:
            bushes[link[0]]['bushFlow'][link[1]][link[2]] = 0
        elif bushes[link[0]]['bushFlow'][link[1]][link[2]] < 0:
            raise ValueError('Negative bush flow! {} {}'.format(dx, bushes[link[0]]['bushFlow'][link[1]][link[2]]))
        if network['edges'][link[2]]['flow'] < 0 and abs(network['edges'][link[2]]['flow']) < 1e-6:
            network['edges'][link[2]]['flow'] = 0
        elif network['edges'][link[2]]['flow'] < 0:
            raise ValueError('Negative network flow! {} {}'.format(dx,  network['edges'][link[2]]['flow']))
        if network['edges'][link[2]]['classFlow'][link[0][0]] < 0 \
                and abs(network['edges'][link[2]]['classFlow'][link[0][0]]) < 1e-6:
            network['edges'][link[2]]['classFlow'][link[0][0]] = 0
        elif network['edges'][link[2]]['classFlow'][link[0][0]] < 0:
            raise ValueError('Negative network class flow! {} {} {}'.format(dx, network['edges'][link[2]]['classFlow'][link[0][0]], link[0][0]))

    for bush_info in list(demand_shift_counts):

        if bush_info[1] in bushes[bush_info[0]]['nodeFlow'] \
                and bush_info[2] in bushes[bush_info[0]]['nodeFlow'][bush_info[1]]:
            d = round(bushes[bush_info[0]]['nodeFlow'][bush_info[1]][bush_info[2]] + (sign * dx) * demand_shift_counts[bush_info], 10)
        else:
            d = round((sign * dx) * demand_shift_counts[bush_info], 10)

        d2 = round(sum([bushes[bush_info[0]]['bushFlow'][bush_info[1]][(pred_node, bush_info[2])]
                                          for pred_node in network['backward_star'][bush_info[2]]
                                          if (pred_node, bush_info[2]) in bushes[bush_info[0]]['bushFlow'][bush_info[1]]]), 10)

        if abs(d2-d) > 1e-8:
            with open(log_name, 'a') as f:
                f.write('\n  Gap in node flow {} {}'.format(d, d2))
        demand = {bush_info[0][1]: {bush_info[2]: d2}}

        # Just for re-pushing nodeFlows
        bush, network = bush_flow_push(ori=bush_info[0][1],
                                       network=network,
                                       demand=demand,
                                       bush=bushes[bush_info[0]],
                                       classID=bush_info[0][0],
                                       groupID=bush_info[1],
                                       flow_shift=True)
        bushes[bush_info[0]] = bush
        classID_split = bush_info[0][0].split('_')
        # Only copy if this is RD flow with onboard passenger
        if len(classID_split) > 1 and classID_split[1] != 'DA' and int(classID_split[1]) > 0:  # Copy to passenger flows
            passenger_classID = bush_info[0][0].replace('RD', 'RP')
            passenger_bushID = (passenger_classID, bush_info[0][1])
            passenger_groupIDs = [(bush_info[1][0], passenger[0], passenger[1])
                                  # not sure yet, should the taskID for the driver/passenger
                                  for passenger in
                                  matching['driver'][bush_info[1][1]]['segment_mapping'][bush_info[1][2]][bush_info[1][0]]]
            tmp_passenger_bushFlow = {
                passenger_groupID: bushes[bush_info[0]]['bushFlow'][bush_info[1]] for passenger_groupID in passenger_groupIDs
            }
            # Enforcing coupling, copy driver flows to passengers
            bushes[passenger_bushID]['bushFlow'].update(tmp_passenger_bushFlow)

            existing_groups = set(bushes[passenger_bushID]['associate_group_list'])
            add_groups = set(passenger_groupIDs) - existing_groups
            bushes[passenger_bushID]['associate_group_list'] += list(add_groups)
            group_index = [bushes[passenger_bushID]['associate_group_list'].index(passenger_groupID) for passenger_groupID in passenger_groupIDs]

            driver_group_index = bushes[bush_info[0]]['associate_group_list'].index(bush_info[1])
            driver_node_group_incident = bushes[bush_info[0]]['node_group_incident'][bushes[bush_info[0]]['node_group_incident'][:, 1] == driver_group_index]

            try:
                passenger_node_group_incident = bushes[passenger_bushID]['node_group_incident'][~np.isin(bushes[passenger_bushID]['node_group_incident'][:, 1], group_index)]
            except Exception:
                print(bushes[passenger_bushID]['node_group_incident'], passenger_groupIDs)
                raise Exception

            passenger_node_group_incident = np.vstack((passenger_node_group_incident,
                                                      np.array([np.tile(driver_node_group_incident[:,0], len(group_index)),
                                                      np.repeat(group_index, driver_node_group_incident.shape[0])]).T))

            bushes[passenger_bushID]['node_group_incident'] = passenger_node_group_incident

            tmp_passenger_nodeFlow = {
                passenger_groupID: bushes[bush_info[0]]['nodeFlow'][bush_info[1]] for passenger_groupID in passenger_groupIDs
            }
            # Enforcing coupling, copy driver flows to passengers
            bushes[passenger_bushID]['nodeFlow'].update(tmp_passenger_nodeFlow)

            tmp_passenger_nodeFlow = {
                passenger_groupID: bushes[bush_info[0]]['merges']['approach_flows'][bush_info[1]]
                for passenger_groupID in passenger_groupIDs
            }
            # Enforcing coupling, copy driver flows to passengers
            bushes[passenger_bushID]['merges']['approach_flows'].update(tmp_passenger_nodeFlow)

            if sign > 0:
                additional_pred = [link for link in positive_shift_links[bush_info[0]] if
                                   link[0] not in bushes[passenger_bushID]['pred'][link[1]]]
                # Due to the acyclic driver bush, the additional pred should still maintain acyclic
                # The acyclic property will certainly be preserved in the next bush structure update

                for link in additional_pred:
                    with_flow_pred = [link[0]]
                    if len(bushes[passenger_bushID]['pred'][link[1]]) >= 1:
                        bush_flow_group_list = set([bushes[passenger_bushID]['associate_group_list'][int(group_index)] for group_index in
                                                    bushes[passenger_bushID]['node_group_incident'][1:, 1]])
                        for pred_node in bushes[passenger_bushID]['pred'][link[1]]:
                            link_flow = sum([bushes[passenger_bushID]['bushFlow'][groupID][(pred_node, link[1])]
                                         for groupID in bush_flow_group_list
                                         if (pred_node, link[1]) in bushes[passenger_bushID]['bushFlow'][groupID]])
                            print('Checked', (pred_node, link[1]), link_flow)
                            if link_flow > 0:
                                with_flow_pred.append(pred_node)
                    with_flow_pred = list(set(with_flow_pred))
                    bushes[passenger_bushID]['pred'][link[1]] = with_flow_pred

                    with_flow_pred_2 = []
                    if len(bushes[passenger_bushID]['pred'][link[0]]) > 1:
                        bush_flow_group_list = set(
                            [bushes[passenger_bushID]['associate_group_list'][int(group_index)] for group_index in
                             bushes[passenger_bushID]['node_group_incident'][1:, 1]])
                        for pred_node in bushes[passenger_bushID]['pred'][link[0]]:
                            link_flow = sum([bushes[passenger_bushID]['bushFlow'][groupID][(pred_node, link[0])]
                                         for groupID in bush_flow_group_list
                                         if (pred_node, link[0]) in bushes[passenger_bushID]['bushFlow'][groupID]])
                            print('Checked 2', (pred_node, link[0]), link_flow)
                            if link_flow > 0:
                                with_flow_pred_2.append(pred_node)
                        bushes[passenger_bushID]['pred'][link[0]] = with_flow_pred_2

                if len(additional_pred): # Re-ordering and check for acyclic
                    try:
                        bushes[passenger_bushID]['bush_topology'] = topological_ordering(bush_info[0][1], network, bushes[passenger_bushID]['pred'])
                    except Exception:
                        print('bushID', passenger_bushID)
                        print('pred', bushes[passenger_bushID]['pred'])
                        print('new_links', additional_pred)
                        #print('flow links', bush['bushFlow'])
                        raise Exception

    return bushes, matching, network

def flow_push_worker(longest_seq_links, shortest_seq_links, dx, LP_bush_demand_shift_count,
                     SP_bush_demand_shift_count, min_cost_seqID, max_cost_seqID,
                     bushes, network, matching, driverID):
    bushes, matching, network = bush_flow_shift(links=longest_seq_links,
                                       dx=dx,
                                       demand_shift_counts=LP_bush_demand_shift_count,
                                       sign=-1,
                                       bushes=bushes,
                                       network=network,
                                       matching=matching)

    bushes, matching, network = bush_flow_shift(links=shortest_seq_links,
                                       dx=dx,
                                       demand_shift_counts=SP_bush_demand_shift_count,
                                       sign=1,
                                       bushes=bushes,
                                       network=network,
                                       matching=matching)

    # print('MIN SEQ', min_cost_seqID)
    matching = seq_flow_update(matching=matching,
                                      driverID=driverID,
                                      driver_seq_ID=min_cost_seqID,
                                      dx=dx,
                                      sign=1)

    matching= seq_flow_update(matching=matching,
                                      driverID=driverID,
                                      driver_seq_ID=max_cost_seqID,
                                      dx=dx,
                                      sign=-1)

    return bushes, matching, network

def group_flow_update(bushes, network, demands, matching, matching_assignment, multiplier_penalty, inner_iter, outer_iter, prev_sequence_cost, prev_dx_norm=np.inf, group_beta=1, step_regularize=False, bush_norm_gap={}, prev_betas= {}):
    # Compute the cost for each sequence for the drivers and passengers from underlying bushes

    # Do this at initialization
    updated = 0
    total_dx = []
    dx_norm = prev_dx_norm
    beta_updated = False

    #if inner_iter <= 200:
    #    group_beta = group_beta + 0.001
    if outer_iter == 0:
        if inner_iter >= 500:
            group_beta = group_beta + 0.001

    betas = []
    for driverID in matching['driver']:
        # For initialization

        # No RD demand, continue to next
        if driverID[0] not in demands['RD'] or driverID[1] not in demands['RD'][driverID[0]] or demands['RD'][driverID[0]][driverID[1]] == 0:
            continue

        # Update matching and bushes
        a = time.time()
        #print('\nStart RS group bush scan ... {} {}'.format(driverID, demands['RD'][driverID[0]][driverID[1]]))
        count = 0
        local_bush_index = [bushID for bushID in matching['driver'][driverID]['associated_bushes'] if 'RP' in bushID[0] and 'PT' not in bushID[0] and 'RD' not in bushID[0]]
        count += len(local_bush_index)
        bushes = scan_bushes(bushes=bushes, network=network, bushIDs=local_bush_index, partial_update=list(matching['driver'][driverID]['driver_sequence']))

        local_bush_index = [bushID for bushID in matching['driver'][driverID]['associated_bushes'] if 'RD' in bushID[0]]
        count += len(local_bush_index)
        # TO DO: to include also partial update for the driver bush groups for better performance
        bushes = scan_bushes(bushes=bushes, network=network, bushIDs=local_bush_index)
        #print('Scan {} bushes in {}'.format(count, time.time()-a))
        matching, bushes = passenger_seq_cost_update(bushes, matching, init=False, multiplier_penalty=multiplier_penalty,
                                                     matching_assignment=matching_assignment,network=network,
                                                     passenger_list=matching['driver'][driverID]['associated_passengers'])
        matching, bushes = driver_seq_cost_update(bushes, matching, init=False, multiplier_penalty=multiplier_penalty,
                                                  matching_assignment=matching_assignment,
                                                  network=network, driver_list=[driverID])


        longest_seq_links, \
        shortest_seq_links, \
        dx, \
        LP_bush_demand_shift_count, \
        SP_bush_demand_shift_count,\
        min_cost_seqID, \
        max_cost_seqID, \
        prev_sequence_cost, gradient = sequence_flow_shift(matching, matching_assignment, bushes, driverID, network, inner_iter, outer_iter, prev_sequence_cost, step_size=1)

        a = time.time()

        # Avoid floating error
        dx = round(dx, 10)

        if dx < 0:
            warnings.warn('Negative dx {}'.format(dx))
            updated = 1
        elif dx > 0:
            #print('Shifted', dx, longest_seq_links, shortest_seq_links,)
            updated = 1

            if driverID not in prev_betas:
                prev_betas[driverID] = 1

            total_dx.append(max(0, dx))
            if outer_iter == 0:
                if gradient <= 100 * dx:
                    if driverID in bush_norm_gap and bush_norm_gap[driverID][1] >= bush_norm_gap[driverID][0]:
                        prev_betas[driverID] += 0.001
                    dx = (1 / prev_betas[driverID]) * dx
                else:
                    dx = dx
            elif gradient <= 100 * dx:
                if driverID in bush_norm_gap and bush_norm_gap[driverID][1] >= bush_norm_gap[driverID][0]:
                    if matching['driver'][driverID]['sequence_flow'][min_cost_seqID] + (1 / prev_betas[driverID]) * dx >= matching_assignment.get(min_cost_seqID, 0):
                        prev_betas[driverID] += 0.1
                        if inner_iter >= 800:
                            prev_betas[driverID] += 0.05 * (outer_iter - 1)

                        if inner_iter >= 1200:
                            prev_betas[driverID] += 0.5 * (outer_iter - 1)
                    else:
                        prev_betas[driverID] += 0.01

                dx = (1 / prev_betas[driverID]) * dx
            elif dx >= 1e-4:
                dx *= 0.5

            #print('Applying betas', prev_betas[driverID], gradient, dx, bush_norm_gap.get(driverID, None))
            bushes, matching, network = flow_push_worker(longest_seq_links, shortest_seq_links, dx,
                                                       LP_bush_demand_shift_count,
                                                       SP_bush_demand_shift_count, min_cost_seqID, max_cost_seqID,
                                                       bushes, network, matching, driverID)

            for ori in demands['RP']:
                for des in demands['RP'][ori]:  # The RP_PT flow should be override each time
                    matching['passenger'][(ori, des)]['sequence_flow'][0] = 0
                    if matching['passenger'][(ori, des)]['sequences'][0][1] != 'PT':
                        raise IndexError('Error in RP_PT index! It should be the last sequence!')
                    else:
                        matching['passenger'][(ori, des)]['sequence_flow'][0] \
                            = round(demands['RP'][ori][des] - sum(matching['passenger'][(ori, des)]['sequence_flow'].values()), 10)

            network = network_cost_update(network=network)
            #print('After Shifted', network['edges'][(2, 6)]['cost'])

        print('Group flow update in {}'.format(time.time()-a))


    if outer_iter > 0:
        dx_norm = np.linalg.norm(total_dx) / len(total_dx)

        #group_beta = np.mean(betas)
        #group_beta += 0.005

        #if dx_norm >= prev_dx_norm:
        #    if inner_iter <= 200:
        #        group_beta = max(group_beta - 0.045, group_beta + 0.001)


    return bushes, matching, network, updated, group_beta, dx_norm, step_regularize, prev_betas


def bush_init_worker(network, demands, bushID, network_identifier, class_names):
    pred, cost_so_far = dijkstra_search_bheap_all(start=bushID[1],
                                                  pred_dict=network['forward_star'],
                                                  edges=network['edges'],
                                                  non_passing_nodes=network['non_passing_nodes'],
                                                  classCost=bushID[0])
    if len(set(network['nodes']) - set(list(pred)) - set(network['non_passing_nodes'])) != 0:
        raise ValueError('Cannot find initial connected bush!')

    # At initialization, the shortest path tree is acyclic, therefore, each node has only one predecessor
    pred = {node: [pred[node]] for node in pred}

    bush_topology = topological_ordering(bushID[1], network, pred)

    bush = {
            #'network': network,
            'LPcost': {},
            'LP_generalized_cost': {},
            'associate_group_list': [],
            'node_group_incident': np.array([[np.nan, np.nan]]),
            'SPcost': cost_so_far,
            'SP_generalized_cost': {},
            'SPcost_carried': {},
            'bushFlow': {},
            'nodeFlow': {},
            'pred': pred,
            'bush_topology': bush_topology,
            'merges': {'shortest_pred': {},
                       'longest_pred': {},
                       'approach_flows': {},
                       'shiftable_links': {}  # Only for DA
                       },
            'network_identifier': network_identifier,
            }
    return bush


def bush_initialization(network, demands, matching, matching_assignment, multiplier_penalty, class_names, all_zones, bush_index, outer_iter,
                        bushes={}, DA_rectify=False, RS_rectify=False, flow_push=True):

    # For identify if rectify is needed
    network_md5 = hashlib.md5(pickle.dumps(network['edges'])).hexdigest()
    demands_md5 = hashlib.md5(pickle.dumps(demands)).hexdigest()

    # Default: create bushes from scratch for each physical origin node and for each class
    local_bush_index = bush_index['DA'] + bush_index['RD'] + bush_index['PT'] + bush_index['RP'][0]
    generate_bushID = set(local_bush_index) - set(list(bushes))
    print('\n Creating initial bushes...')

    for bushID in generate_bushID:
        bushes[bushID] = bush_init_worker(network, demands, bushID, network_md5, class_names)

    # Initial feasible solution for the RP, by enforcing coupling of RD and RP routes and flows (processed later)
    # The actual shortest path for RP will be computed on flow-shift, and influence the projected gradients
    RP_bush_index = set(bush_index['RP'][1]) - set(list(bushes))
    for bushID in RP_bush_index:
        corresponding_driver_bushID = (bushID[0].replace('RP', 'RD'), bushID[1])
        bush = {
            # 'network': network,
            'LP_generalized_cost': {},
            'associate_group_list': [],
            'node_group_incident': np.array([[np.nan, np.nan]]),
            'SP_generalized_cost': {},
            'LPcost': {},
            'SPcost': {},
            'SPcost_carried': {},
            'bushFlow': {},
            'nodeFlow': {},
            'pred': bushes[corresponding_driver_bushID]['pred'],
            'bush_topology': bushes[corresponding_driver_bushID]['bush_topology'],
            'merges': {'shortest_pred': {},
                       'longest_pred': {},
                       'approach_flows': {},
                       'shiftable_links': {}
                       },
            'network_identifier': bushes[corresponding_driver_bushID]['network_identifier'],
        }
        # Initializing the passenger bush with the same pred as the driver
        # The bush scan indeed updates the LPcost and SPcost with repsect to passenger classes
        bush = bush_scan(start=bushID[1],
                         bush=bush,
                         edges=network['edges'],
                         flow_ignore_threshold=flow_ignore_threshold,
                         classCost=bushID[0],
                         LPRule=LPRule)
        bushes[bushID] = bush

    RP_PT_bush_index = set(bush_index['RP'][0]) - set(list(bushes))
    for bushID in RP_PT_bush_index:
        # Initializing the passenger bush with the same pred as the driver
        # The bush scan indeed updates the LPcost and SPcost with repsect to passenger classes
        bushes[bushID] = bush_scan(start=bushID[1],
                         bush=bushes[bushID],
                         edges=network['edges'],
                         flow_ignore_threshold=flow_ignore_threshold,
                         classCost=bushID[0],
                         LPRule=LPRule)

    bushes['md5'] = (network_md5, demands_md5)
    print('-> Created initial bushes')


    if flow_push:
        print('\n Pushing initial flows...')
        if len(matching):
            bushes, matching, network = group_flow_push(bushes, network, demands, network_md5, class_names, matching,
                                               matching_assignment, multiplier_penalty, outer_iter=outer_iter, init=True, rectify=RS_rectify)

        # DA flow handling
        for bushID in bushes:
            if bushID[0] == 'DA' and bushID != 'md5':
                #if sum(demands['DA'][bushID[1]].values()) > 0:
                bush, network = bush_flow_push(ori=bushID[1],
                                                network=network,
                                                demand=demands[bushID[0]],
                                                bush=bushes[bushID],
                                                network_identifier=network_md5,
                                                groupID='DA',
                                                DA=True,
                                                rectify=DA_rectify)
                bushes[bushID] = bush
        print('-> Pushed initial flows')


        # After creating the bushes, update the link flows in the network

        a = time.time()
        network = network_cost_update(network=network)
        print('After', network['edges'][(1, 2)])
        print('-> Updated network link flows and costs in {}'.format(time.time()-a))


    return bushes, network, matching

def bush_structure_update_worker(bushID, network, network_identifier, bushes, scan=True):
    # if the network in the bush is outdated, update it with latest network
    bush = bushes[bushID]
    if bush['network_identifier'] != network_identifier:
        #bush['network'] = network
        bush['network_identifier'] = network_identifier

    # Update LPcost (longest_pred) and SPcost (shortest_pred) in the bush, after flow loading, before expansion
    classID = bushID[0].split('_')
    passenger_classID = None

    a = time.time()
    if scan:
        if classID[0] == 'RD' and classID[1] != 'DA' and int(classID[1]) > 0:
            passenger_classID = bushID[0].replace('RD', 'RP')
            passenger_bushID = (passenger_classID, bushID[1])
            passenger_PT_bushID = ('RP_PT', bushID[1])

            bush = bush_scan(start=bushID[1],
                             bush=bush,
                             edges=network['edges'],
                             flow_ignore_threshold=flow_ignore_threshold,
                             classCost=bushID[0],
                             LPRule=LPRule,
                             bush_RP=bushes[passenger_bushID],
                             RP_classID=passenger_classID,
                             bush_RP_PT=bushes[passenger_PT_bushID])
        else:
            bush = bush_scan(start=bushID[1],
                             bush=bush,
                             edges=network['edges'],
                             flow_ignore_threshold=flow_ignore_threshold,
                             classCost=bushID[0],
                             LPRule=LPRule)

    a = time.time()
    new_links = []
    P1_links = []
    P2_links = []
    group_link_flows = {}
    updated = False
    group_time = 0
    for link in network['edges']:
        b = time.time()

        check_list = bush['node_group_incident'][bush['node_group_incident'][:, 0] == link[0], 1].tolist()
        check_list += bush['node_group_incident'][bush['node_group_incident'][:, 0] == link[1], 1].tolist()
        group_list = set([bush['associate_group_list'][int(group_index)] for group_index in check_list])
        group_link_flows[link] = sum([bush['bushFlow'][groupID][link] for groupID in group_list if link in bush['bushFlow'][groupID]])

        group_time += time.time() - b
        if (link in group_link_flows and group_link_flows[link] >= 1e-9) or link[1] == bushID[1] \
                or (link[0] in network['non_passing_nodes'] and link[0] != bushID[1]):
            #if (bushID == ('DA', 8) or bushID == ('RD_DA', 8)):
            #    print('Skip link', link, link in group_link_flows, group_link_flows[link], group_link_flows[link] > 0, link[1] == bushID[1], link[0] in network['non_passing_nodes'], link[0] != bushID[1])
            continue  # this link has already been included in the bush

        multipliers = 0
        if passenger_classID:
            multipliers = bushes[(passenger_classID, bushID[1])]['SPcost'][link[0]] \
                          + network['edges'][link]['classCost'][passenger_classID] \
                          - min(bushes[(passenger_classID, bushID[1])]['SPcost'][link[1]],
                                bushes[('RP_PT', bushID[1])]['SPcost'][link[1]])
            multipliers *= int(passenger_classID.split('_')[1])

        # otherwise, check if this link provides a shortcut, based on a stricter condition (P1 and P2 in Nie 2010)

        if bush['SPcost'][link[0]] + network['edges'][link]['classCost'][bushID[0]] < bush['SPcost'][link[1]]:
            #print('Inside', bushID, link, bush['SPcost'][link[0]] + network['edges'][link]['classCost'][bushID[0]], bush['SPcost'][link[1]])
            P1_links.append(link)

        b = time.time()
        if len(group_list):
            P2_add = 1
        else:
            P2_add = 0
        for groupID in group_list:
            if bush['LPcost'][groupID][link[1]] > 0:
                if bush['LPcost'][groupID][link[0]] > 0 and bush['LPcost'][groupID][link[0]] + network['edges'][link]['classCost'][bushID[0]] < bush['LPcost'][groupID][link[1]]:
                    P2_add *= 1
                else:
                    P2_add *= 0

        if P2_add:
            P2_links.append(link)  # Nie (2010) P2, stricter than Bar-Gera (2002), i.e., potentially less links

        group_time += time.time() - b

    a = time.time()

    # In case no new link is found, and P1 and P2 are not empty, P2 links are used to avoid breakdown
    new_links += list(set(P1_links).intersection(set(P2_links)))
    new_links = set(new_links)

    # In case for specific class, there is no demand, update the bush with shortest path, important for modal costs
    if sum(group_link_flows.values()) < 1e-9:
        new_links = set(P1_links)

    if len(new_links):
        updated = True
        #print('Added new links', bushID, new_links, [[[pred_node for pred_node in bush['pred'][link[1]]], link[1]] for link in new_links])

    if LPRule == 'LONGEST_BUSH_PATH' and len(new_links) == 0 and len(P1_links) > 0 and len(P2_links) > 0:
        new_links = P2_links
        warnings.warn('P2 links are directly added for bush of origin {} to avoid breakdown!'.format(bushID[1]))

    # Keep all links in the bush with positive flows
    for link in group_link_flows:
        if group_link_flows[link] >= 1e-9:
            new_links.add(link)

    if len(new_links) > 0:
        # Update pred merges
        bush_pred = {bushID[1]: [None]}
        #bush['pred'] = {bushID[1]: [None]}  # recompute the pred
        for link in new_links:
            # Initialize the next_node in bush['pred'] for all nodes
            if link[1] not in bush_pred:
                bush_pred[link[1]] = []
            bush_pred[link[1]].append(link[0])

        #print(' >> Bush structure update: 3.update links in {}'.format(time.time() - a))
        a = time.time()

        # To maintain fully connectivity (i.e., there exists at least 1 route from root to all nodes)
        shortest_links = []
        for node in set(network['nodes']) - set(list(bush_pred)):
            try:
                shortest_path = reconstruct_path(bush['merges']['shortest_pred'], bushID[1], node)
                shortest_links += list(zip(shortest_path[:-1], shortest_path[1:]))
            except Exception:
                print('TO handle error!', bushID)
                #print(bush)
                print('bush_topology',bush['bush_topology'])
                print('pred', bush['pred'])
                print('bush_pred', bush_pred)
                raise Exception
        shortest_links = set(shortest_links)

        for link in shortest_links:
            # Initialize the next_node in bush['pred'] for all nodes
            if link[1] not in bush_pred:
                bush_pred[link[1]] = [link[0]]

        a = time.time()

        bush['pred'] = bush_pred

        try:
            bush['bush_topology'] = topological_ordering(bushID[1], network, bush['pred'])
        except Exception:
            print('bushID', bushID)
            print('pred', bush['pred'])
            print('new_links', new_links)
            print('flow links', group_link_flows)
            print('link rule', P1_links, P2_links)
            print('shortest_links', shortest_links)
            raise Exception

    if set(list(bush['pred'])) != set(network['nodes']) - set(network['non_passing_nodes']):
        raise ValueError('Updated bush pred not reaching all nodes {} should be {}'.format(set(list(bush['pred'])), set(network['nodes']) - set(network['non_passing_nodes'])))
    return (bushID, bush, updated)


def update_bush_structures(bushes, network, demands, class_names, all_zones, bush_index, darp_solved):
    # Assume a segmented transit system (e.g., metro), in which the travel times are assumed constant

    a = time.time()
    bush_structure_updated = False
    network_identifier = hashlib.md5(pickle.dumps(network['edges'])).hexdigest()
    if network_identifier != bushes['md5'][0]:
        prev_network_identifier = network_identifier
        network = network_cost_update(network=network) # for consistency, update network costs with current flows
        network_identifier = hashlib.md5(pickle.dumps(network['edges'])).hexdigest()

    if darp_solved:
        for bushID in bush_index['RP'][1]:
            bushID, bush, updated = bush_structure_update_worker(network=network,
                                                          bushID=bushID,
                                                          #bush=bushes[bushID],
                                                          network_identifier=network_identifier,
                                                          bushes=bushes)
            bushes[bushID] = bush
            if updated:
                bush_structure_updated = True
            # If update RP-PT costs
            for bushID in bush_index['RP'][0]:
                bushID, bush, updated = bush_structure_update_worker(network=network,
                                                              bushID=bushID,
                                                              #bush=bushes[bushID],
                                                              network_identifier=network_identifier,
                                                              bushes=bushes)
                bushes[bushID] = bush
                if updated:
                    bush_structure_updated = True

        for bushID in bush_index['RD']:
            bushID, bush, updated = bush_structure_update_worker(network=network,
                                                                 bushID=bushID,
                                                                 # bush=bushes[bushID],
                                                                 network_identifier=network_identifier,
                                                                 bushes=bushes)
            bushes[bushID] = bush
            if updated:
                bush_structure_updated = True

    for bushID in bush_index['DA']:

        bushID, bush, updated = bush_structure_update_worker(network=network,
                                                      bushID=bushID,
                                                      #bush=bushes[bushID],
                                                      network_identifier=network_identifier,
                                                      bushes=bushes)
        bushes[bushID] = bush
        if updated:
            bush_structure_updated = True


    # Update the network has been updated, update the identifier for all the bushes
    bushes['md5'] = (network_identifier, bushes['md5'][1])
    print('-> Updated bush structures in {}'.format(time.time() - a))

    return bushes, bush_structure_updated

def find_shiftable_links(ori, bush):  # only flows between a divergent and merge nodes need to be shifted
    bush['merges']['shiftable_links'] = {}
    backtrack_bush_topology = bush['bush_topology'].copy()
    backtrack_bush_topology.reverse()

    for i in range(len(backtrack_bush_topology)):
        merge_node = backtrack_bush_topology[i]
        if len(bush['pred'][merge_node]) > 1: # this is an actual merge node
            if 'DA' not in bush['merges']['longest_pred'] or merge_node not in bush['merges']['longest_pred']['DA']:
                continue  # when there is no longest used (or shortest) path incidents the merge_node, continue
            if merge_node not in bush['merges']['shortest_pred']:
                warnings.warn('Merge node in pred list, but not (used) on the shortest path!')

            shortest_path = reconstruct_path(bush['merges']['shortest_pred'], ori, merge_node)
            longest_path = reconstruct_path(bush['merges']['longest_pred']['DA'], ori, merge_node)

            common_divergent_nodes = set(shortest_path).intersection(set(longest_path)) - {merge_node}

            SP_links = shortest_path[max([shortest_path.index(node) for node in common_divergent_nodes]):]
            SP_links = list(zip(SP_links[:-1], SP_links[1:]))


            LP_links = longest_path[max([longest_path.index(node) for node in common_divergent_nodes]):]
            LP_links = list(zip(LP_links[:-1], LP_links[1:]))
            #print('Merge node', ori, merge_node, SP_links, LP_links)

            bush['merges']['shiftable_links'][merge_node] = {'SP_links': SP_links, 'LP_links': LP_links}

    return bush

def update_bush_flows(bushes, network, demands, inner_iteration, bush_index, run_DA=True):
    """
    This function should be performed in a sequential manner for better convergence performances
    :param bushes:
    :param network:
    :param demands:
    :return:
    """
    #print('\n Updating bush flows ...')
    updated = []
    inner_gap = []
    time_updated = False
    bushWeightedGap = 0
    total_DA = 0
    for bushID in bush_index['DA']:
        # Update LPcost (longest_pred) and SPcost (shortest_pred) in the bush

        if run_DA:
            bushes[bushID] = bush_scan(start=bushID[1],
                                       bush=bushes[bushID],
                                       edges=network['edges'],
                                       flow_ignore_threshold=flow_ignore_threshold,
                                       classCost=bushID[0],
                                       LPRule=LPRule)

            bushes[bushID] = find_shiftable_links(ori=bushID[1], bush=bushes[bushID])

        # Heuristic to stop processing current bush if the gap is relatively small, and move to next bush
        #a = time.time()
        max_bush_gap = []
        bushSPTT = 0
        bushExcess = 0
        for des in demands[bushID[0]][bushID[1]]:
            if demands[bushID[0]][bushID[1]][des] > 0:
                try:
                    bushSPTT += demands[bushID[0]][bushID[1]][des] * bushes[bushID]['SPcost'][des]
                    bushWeightedGap += demands[bushID[0]][bushID[1]][des] * abs(bushes[bushID]['LPcost']['DA'][des] - bushes[bushID]['SPcost'][des])
                    total_DA += demands[bushID[0]][bushID[1]][des]
                except Exception:
                    raise Exception("{} {} {}".format(bushSPTT, des in demands[bushID[0]][bushID[1]], des in bushes[bushID]['SPcost']))

        nodes = list(bushes[bushID]['SPcost'])

        max_bush_gap += [abs(bushes[bushID]['LPcost']['DA'][node] - bushes[bushID]['SPcost'][node])
                         for node in nodes if 'DA' in bushes[bushID]['LPcost'] if bushes[bushID]['LPcost']['DA'][node] != -np.inf]
        bushExcess += sum([bushes[bushID]['bushFlow']['DA'][link]
                           * (bushes[bushID]['SPcost'][link[0]]
                              + network['edges'][link]['classCost'][bushID[0]]
                              - bushes[bushID]['SPcost'][link[1]]) for link in bushes[bushID]['bushFlow']['DA']])


        # If either condition is satisfied, moved to next bush
        if max(max_bush_gap, default=0) < thresholdGap or (bushExcess/bushSPTT) < thresholdAEC:
            continue

        # Additional flag for stopping the inner iterations

        inner_gap.append(max(max_bush_gap, default=0))

        # Update bush flow
        #a = time.time()
        if run_DA:
            updated.append(bushID)
            bushes[bushID], network, updated_links = bush_flow_push(ori=bushID[1],
                                                                    network=network,
                                                                    demand=demands[bushID[0]],
                                                                    bush=bushes[bushID],
                                                                    flow_shift=True,
                                                                    classID=bushID[0],
                                                                    groupID=bushID[0],
                                                                    DA=True,
                                                                    inner_iteration=inner_iteration)
            network = network_cost_update(network=network, updated_links=updated_links)
            time_updated = True

    if not time_updated:
        network = network_cost_update(network=network)

    if len(updated):
        updated = 1
    else:
        updated = 0
    return bushes, network, updated, bushWeightedGap, bushWeightedGap

def scan_bushes(bushes, network, bushIDs, partial_update=None):
    for bushID in bushIDs:
        classID = bushID[0].split('_')

        if classID[0] == 'RD' and classID[1] != 'DA' and int(classID[1]) > 0:
            passenger_classID = bushID[0].replace('RD', 'RP')
            passenger_bushID = (passenger_classID, bushID[1])
            passenger_PT_bushID = ('RP_PT', bushID[1])
            bushes[bushID] = bush_scan(start=bushID[1],
                             bush=bushes[bushID],
                             edges=network['edges'],
                             flow_ignore_threshold=flow_ignore_threshold,
                             classCost=bushID[0],
                             LPRule=LPRule,
                             bush_RP=bushes[passenger_bushID],
                             RP_classID=passenger_classID,
                             bush_RP_PT=bushes[passenger_PT_bushID])
        else:
            bushes[bushID] = bush_scan(start=bushID[1],
                             bush=bushes[bushID],
                             edges=network['edges'],
                             flow_ignore_threshold=flow_ignore_threshold,
                             classCost=bushID[0],
                             LPRule=LPRule,
                             partial_update=partial_update)

    return bushes


def algorithmBush():

    bushes = {}

    matching = {}
    group_beta = 1
    group_dx_norm = np.inf

    print('\n** Importing network **')
    class_names = ['DA', 'RD_DA', 'RD_0', 'RD_1', 'RD_2', 'RP_PT', 'RP_1', 'RP_2', 'PT']
    edges = {}
    edges_list = [(1,2), (1,3), (2,3), (2,4), (3,4),
                  (4,5), (4,6), (5,6), (5,7), (6,7),
                  (7,8), (7,9), (8,9), (8,10), (9,10),
                  (10,11), (10, 12), (11, 12), (11,13), (12,13),
                  (13, 14), (13,15), (14,15), (14, 16), (15,16)]
    for edge in edges_list:
        edges[edge[0], edge[1]] = {'Capacity': 10000,
                                    'ff_tt': 5,
                                    'alpha': 0.15,
                                    'beta': 4,
                                    'flow': 0,
                                    'cost': 5,
                                    'classToll': {'DA':20, 'RD_DA':20, 'RD_0':20, 'RD_1':10, 'RD_2':10, 'RP_PT':15, 'RP_1':10, 'RP_2':10, 'PT':15},
                                    'classFlow': {},
                                    'classCost': {},
                                    }
        edges[edge[1], edge[0]] = {'Capacity': 10000,
                                   'ff_tt': 5,
                                   'alpha': 0.15,
                                   'beta': 4,
                                   'flow': 0,
                                   'cost': 5,
                                   'classToll': {'DA': 20, 'RD_DA': 20, 'RD_0': 20, 'RD_1': 10, 'RD_2': 10, 'RP_PT': 15,
                                                 'RP_1': 10, 'RP_2': 10, 'PT': 15},
                                   'classFlow': {},
                                   'classCost': {},
                                   }

    edges, nodes, forward_star, backward_star = network_import_postprocessing(edges, class_names)

    network = {'edges': edges,
               'nodes': nodes,
               'forward_star': forward_star,
               'backward_star': backward_star,
               'non_passing_nodes': [],
               'Initialized': True
               }

    inner_gap = 999999
    prev_inner_gap = np.inf
    DA_inner_gap = 999999
    DAbushWeightedGap = 999999
    RSbushWeightedGap = 999999
    bush_structure_updated = True
    darp_solved = False
    matching_assignment = {}
    bush_norm_gap = {}
    prev_betas = {}
    multiplier_penalty = {'driver': {},
                          'passenger': {}, 'init': True}


    demands = {
        'DA': {},
        'RD': {1: {16: 40000}},
        'RP': {4: {10: 20000}, 7:{13: 20000}},
        'PT': {}
    }

    all_zones = [[ori, *destinations.keys()] for ori, destinations in demands['RP'].items()]
    all_zones = set(functools.reduce(operator.iconcat, all_zones, []))

    bush_index = {
        'DA': sorted(list(set([('DA', ori) for ori in demands['DA']]))),
        'RD': sorted(list(set([('RD_DA', ori) for ori in demands['RD']]))),
        'RP': [sorted(list(set([('RP_PT', zone) for zone in all_zones]))), set()],
        'PT': sorted(list(set([('PT', ori) for ori in demands['PT']]))),
    }

    print(bush_index)

    DA_rectify = False
    RS_rectify = False

    step_regularize = False

    import time

    prev_sequence_cost = {}

    for inner_iter in range(100):

        print('\n<<< Iteration {} >>>'.format(inner_iter))

        if not darp_solved:
            matching = {'driver': {}, 'passenger': {}, 'init': True}

            matching['driver'][(1, 16)] = {
                'driver_sequence': {
                    0: [(1, 16)],
                    1: [(1, 4), (4, 10), (10, 16)],
                    2: [(1, 7), (7, 13), (13, 16)],
                    3: [(1, 4), (4, 4), (4, 10), (10, 10), (10, 16)],
                    4: [(1, 4), (4, 10), (10, 4), (4, 10), (10, 16)],
                    5: [(1, 7), (7, 7), (7, 13), (13, 13), (13, 16)],
                    6: [(1, 7), (7, 13), (13, 7), (7, 13), (13, 16)],
                    7: [(1, 7), (7, 13), (13, 4), (4, 10), (10, 16)],
                    8: [(1, 4), (4, 10), (10, 7), (7, 13), (13, 16)],
                    9: [(1, 7), (7, 4), (4, 10), (10, 13), (13, 16)],
                    10: [(1, 7), (7, 4), (4, 13), (13, 10), (10, 16)],
                    11: [(1, 4), (4, 7), (7, 10), (10, 13), (13, 16)],
                    12: [(1, 4), (4, 7), (7, 13), (13, 10), (10, 16)]
                },
                'segment_classID': {
                    0: {            # 0: [(1, 16)],
                        0: 'RD_DA'
                    },
                    1: {            #1: [(1, 4), (4, 10), (10, 16)],
                        0: 'RD_0',
                        1: 'RD_1',
                        2: 'RD_0'
                    },
                    2: {            #2: [(1, 7), (7, 13), (13, 16)],
                        0: 'RD_0',
                        1: 'RD_1',
                        2: 'RD_0'
                    },
                    3: {            #3: [(1, 4), (4, 4), (4, 10), (10, 10), (10, 16)],
                        0: 'RD_0',
                        1: 'RD_1',
                        2: 'RD_2',
                        3: 'RD_1',
                        4: 'RD_0',
                    },
                    4: {            #4: [(1, 4), (4, 10), (10, 4), (4, 10), (10, 16)],
                        0: 'RD_0',
                        1: 'RD_1',
                        2: 'RD_0',
                        3: 'RD_1',
                        4: 'RD_0',
                    },
                    5: {            #5: [(1, 7), (7, 7), (7, 13), (13, 13), (13, 16)],
                        0: 'RD_0',
                        1: 'RD_1',
                        2: 'RD_2',
                        3: 'RD_1',
                        4: 'RD_0',
                    },
                    6: {            #6: [(1, 7), (7, 13), (13, 7), (7, 13), (13, 16)],
                        0: 'RD_0',
                        1: 'RD_1',
                        2: 'RD_0',
                        3: 'RD_1',
                        4: 'RD_0',
                    },
                    7: {            #7: [(1, 7), (7, 13), (13, 4), (4, 10), (10, 16)],
                        0: 'RD_0',
                        1: 'RD_1',
                        2: 'RD_0',
                        3: 'RD_1',
                        4: 'RD_0',
                    },
                    8: {            #8: [(1, 4), (4, 10), (10, 7), (7, 13), (13, 16)],
                        0: 'RD_0',
                        1: 'RD_1',
                        2: 'RD_0',
                        3: 'RD_1',
                        4: 'RD_0',
                    },
                    9: {            #9: [(1, 7), (7, 4), (4, 10), (10, 13), (13, 16)],
                        0: 'RD_0',
                        1: 'RD_1',
                        2: 'RD_2',
                        3: 'RD_1',
                        4: 'RD_0'
                    },
                    10: {           #10: [(1, 7), (7, 4), (4, 13), (13, 10), (10, 16)],
                        0: 'RD_0',
                        1: 'RD_1',
                        2: 'RD_2',
                        3: 'RD_1',
                        4: 'RD_0'
                    },
                    11: {           #11: [(1, 4), (4, 7), (7, 10), (10, 13), (13, 16)],
                        0: 'RD_0',
                        1: 'RD_1',
                        2: 'RD_2',
                        3: 'RD_1',
                        4: 'RD_0'
                    },
                    12: {           #12: [(1, 4), (4, 7), (7, 13), (13, 10), (10, 16)]
                        0: 'RD_0',
                        1: 'RD_1',
                        2: 'RD_2',
                        3: 'RD_1',
                        4: 'RD_0'
                    }
                },
                'segment_mapping': {#Mapping with onboard RP sequence
                    0: {
                        0: []
                    },
                    1: {  # 1: [(1, 4), (4, 10), (10, 16)],
                        0: [],
                        1: [((4, 10), 101)],
                        2: []
                    },
                    2: {  # 2: [(1, 7), (7, 13), (13, 16)],
                        0: [],
                        1: [((7, 13), 201)],
                        2: []
                    },
                    3: {  # 3: [(1, 4), (4, 4), (4, 10), (10, 10), (10, 16)],
                        0: [],
                        1: [((4, 10), 301)],
                        2: [((4, 10), 301), ((4, 10), 302)],
                        3: [((4, 10), 302)],
                        4: [],
                    },
                    4: {  # 4: [(1, 4), (4, 10), (10, 4), (4, 10), (10, 16)],
                        0: [],
                        1: [((4, 10), 401)],
                        2: [],
                        3: [((4, 10), 402)],
                        4: [],
                    },
                    5: {  # 5: [(1, 7), (7, 7), (7, 13), (13, 13), (13, 16)],
                        0: [],
                        1: [((7, 13), 501)],
                        2: [((7, 13), 501), ((7, 13), 502)],
                        3: [((7, 13), 502)],
                        4: [],
                    },
                    6: {  # 6: [(1, 7), (7, 13), (13, 7), (7, 13), (13, 16)],
                        0: [],
                        1: [((7, 13), 601)],
                        2: [],
                        3: [((7, 13), 602)],
                        4: [],
                    },
                    7: {  # 7: [(1, 7), (7, 13), (13, 4), (4, 10), (10, 16)],
                        0: [],
                        1: [((7, 13), 701)],
                        2: [],
                        3: [((4, 10), 702)],
                        4: [],
                    },
                    8: {  # 8: [(1, 4), (4, 10), (10, 7), (7, 13), (13, 16)],
                        0: [],
                        1: [((4, 10), 801)],
                        2: [],
                        3: [((7, 13), 802)],
                        4: [],
                    },
                    9: {  # 9: [(1, 7), (7, 4), (4, 10), (10, 13), (13, 16)],
                        0: [],
                        1: [((7, 13), 901)],
                        2: [((7, 13), 901), ((4, 10), 902)],
                        3: [((7, 13), 901)],
                        4: []
                    },
                    10: {  # 10: [(1, 7), (7, 4), (4, 13), (13, 10), (10, 16)],
                        0: [],
                        1: [((7, 13), 1001)],
                        2: [((7, 13), 1001), ((4, 10), 1002)],
                        3: [((4, 10), 1002)],
                        4: []
                    },
                    11: {  # 11: [(1, 4), (4, 7), (7, 10), (10, 13), (13, 16)],
                        0: [],
                        1: [((4, 10), 1101)],
                        2: [((4, 10), 1101), ((7, 13), 1102)],
                        3: [((7, 13), 1102)],
                        4: []
                    },
                    12: {  # 12: [(1, 4), (4, 7), (7, 13), (13, 10), (10, 16)]
                        0: [],
                        1: [((4, 10), 1201)],
                        2: [((4, 10), 1201), ((7, 13), 1202)],
                        3: [((4, 10), 1201)],
                        4: []
                    }
                },
                'sequence_cost': {
                    0: None,
                    1: None,
                    2: None,
                    3: None,
                    4: None,
                    5: None,
                    6: None,
                    7: None,
                    8: None,
                    9: None,
                    10: None,
                    11: None,
                    12: None
                },
                'sequence_flow': {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0,
                    9: 0,
                    10: 0,
                    11: 0,
                    12: 0,
                },
                'associated_bushes': {
                    ('RD_DA', 1),
                    ('RD_0', 1), ('RD_0', 10), ('RD_0', 13),
                    ('RD_1', 4), ('RD_1', 7), ('RD_1', 10), ('RD_1', 13),
                    ('RD_2', 4), ('RD_2', 7),
                    ('RP_1', 4), ('RP_1', 7), ('RP_1', 10), ('RP_1', 13),
                    ('RP_2', 4), ('RP_2', 7),
                },
                'associated_passengers': {
                    (4, 10), (7, 13)
                }
            }

            for driverID in matching['driver']:
                for seqID in matching['driver'][driverID]['segment_mapping']:
                    if seqID == 0:
                        continue
                    for taskID in sorted(matching['driver'][driverID]['segment_mapping'][seqID]):
                        for passenger_info in matching['driver'][driverID]['segment_mapping'][seqID][taskID]:

                            passengerID = passenger_info[0]
                            passenger_seqID = passenger_info[1]

                            if passengerID not in matching['passenger']:
                                matching['passenger'][passengerID] = {
                                    'sequences': {
                                        0: [[passengerID], 'PT', None, None, None]
                                    },
                                    'sequence_cost': {
                                        0: None
                                    },
                                    'sequence_flow': {
                                        0: 0
                                    }
                                }
                            if passenger_seqID not in matching['passenger'][passengerID]['sequences']:
                                matching['passenger'][passengerID]['sequences'][passenger_seqID] = [[], driverID, seqID, taskID, taskID]
                                matching['passenger'][passengerID]['sequence_cost'][passenger_seqID] = None
                                matching['passenger'][passengerID]['sequence_flow'][passenger_seqID] = 0

                            matching['passenger'][passengerID]['sequences'][passenger_seqID][0].append(matching['driver'][driverID]['driver_sequence'][seqID][taskID])
                            matching['passenger'][passengerID]['sequences'][passenger_seqID][-1] = taskID + 1

            matching['RD_bushIDs'] = set([bushID for driverID in matching['driver'] for bushID in
                                          matching['driver'][driverID]['associated_bushes'] if 'RD' in bushID[0]])
            matching['RP_bushIDs'] = set([bushID for driverID in matching['driver'] for bushID in
                                          matching['driver'][driverID]['associated_bushes'] if 'RP' in bushID[0]])

            darp_solved = True

            bush_index['RD'] = sorted(list(set(bush_index['RD']).union(matching['RD_bushIDs'])))
            bush_index['RP'][1] = sorted(list(matching['RP_bushIDs']))

            matching_assignment = {seqID: 20000 for driverID in matching['driver'] for seqID in matching['driver'][driverID]['driver_sequence'] if seqID != 0}

            bushes, network, matching = bush_initialization(network, demands, matching, matching_assignment,
                                                            multiplier_penalty, class_names,
                                                            all_zones, bush_index, outer_iter=1, bushes=bushes,
                                                            DA_rectify=DA_rectify, RS_rectify=RS_rectify, flow_push=False)
            print('Solved DARP sequences ...')

            # Obtain initial costs of the matching sequences
            print('Initial matching sequence update ...')
            matching, bushes = passenger_seq_cost_update(bushes, matching, True, multiplier_penalty, matching_assignment, network)
            matching, bushes = driver_seq_cost_update(bushes, matching, True, multiplier_penalty, matching_assignment, network)

            print(' -> Updated initial matching sequence ...')

            count = []
            for driverID in matching['driver']:
                count.append((driverID, len(matching['driver'][driverID]['driver_sequence'])))
            #print(count)

            # Create bushes for DA, RD, and onboard_RP
            bushes, network, matching = bush_initialization(network, demands, matching, matching_assignment, multiplier_penalty, class_names,
                                                            all_zones, bush_index, outer_iter=1, bushes=bushes, DA_rectify=DA_rectify, RS_rectify=RS_rectify)

        print('Start bush structure update ...')
        bushes, bush_structure_updated = update_bush_structures(bushes=bushes,
                                                                network=network,
                                                                demands=demands,
                                                                class_names=class_names,
                                                                all_zones=all_zones,
                                                                bush_index=bush_index,
                                                                darp_solved=darp_solved)


        print('-> Updated bush structure')
        # Solve the RMP, with flows sequentially updated, and link costs are only updated once per iteration
        # Consequently, only the gradient is affected, but the links to be updated have been defined
        updated = 0
        run_DA = True
        run_RS = True


        # Sync costs for each matching sequence, required before flow shifting and computing normalized gap
        bushes, network, updated_DA, DA_inner_gap, DAbushWeightedGap = update_bush_flows(bushes=bushes,
                                                                                         network=network,
                                                                                         demands=demands,
                                                                                         inner_iteration=1,
                                                                                         bush_index=bush_index,
                                                                                         run_DA=run_DA)
        print('-> Updated DA bush flows')
        updated += updated_DA

        if len(matching) and run_RS:
            # It has to be done only after driver and passenger seq_cost_update
            bushes, matching, network, updated_RS, group_beta, group_dx_norm, step_regularize, prev_betas = group_flow_update(bushes=bushes,
                                                                      network=network,
                                                                      demands=demands,
                                                                      matching=matching,
                                                                      matching_assignment=matching_assignment,
                                                                      multiplier_penalty=multiplier_penalty,
                                                                      inner_iter=inner_iter,
                                                                      outer_iter=1,
                                                                      prev_sequence_cost=prev_sequence_cost,
                                                                                                     group_beta=group_beta,
                                                                                                     prev_dx_norm=group_dx_norm,
                                                                                                                  step_regularize=step_regularize, bush_norm_gap=bush_norm_gap, prev_betas=prev_betas)

            updated += updated_RS

        # Update sequence cost and bushes for the mode choice
        print('Scanning bushes')
        a = time.time()
        #bush_index = [(classID, ori) for classID in class_names if 'RP' in classID and classID != 'RP_PT' for ori in all_zones]
        bushes = scan_bushes(bushes=bushes, network=network, bushIDs=bush_index['RP'][1])

        #bush_index = [(classID, ori) for classID in class_names if 'RP' not in classID and classID != 'PT' for ori in all_zones]
        bushes = scan_bushes(bushes=bushes, network=network, bushIDs=bush_index['DA'] + bush_index['RD'])
        print(' -> Scanned bushes in {}'.format(time.time()-a))

        if len(matching):
            # Sync costs for each matching sequence, required before flow shifting and computing normalized gap
            matching, bushes = passenger_seq_cost_update(bushes, matching, init=False, multiplier_penalty=multiplier_penalty, matching_assignment=matching_assignment, network=network)
            matching, bushes = driver_seq_cost_update(bushes, matching, init=False, multiplier_penalty=multiplier_penalty, matching_assignment=matching_assignment, network=network)
            inner_gap, RSbushWeightedGap, bush_norm_gap = normalized_gap(matching, matching_assignment, bush_norm_gap, log_name=None)
            print('Normalized gap:', inner_gap)
            driver = {driverID: matching['driver'][driverID]['sequence_cost'].copy() for driverID in matching['driver']}
            passenger = {passengerID: matching['passenger'][passengerID]['sequence_cost'].copy() for passengerID in matching['passenger']}
            prev_sequence_cost[inner_iter] = {'driver': driver, 'passenger': passenger}
            if inner_iter - 2 in prev_sequence_cost:
                del prev_sequence_cost[inner_iter - 2]

        if inner_gap <= 1e-8:
            normalized_gap(matching, matching_assignment, bush_norm_gap, log_name=None, log=True)
            print(matching['driver'][(1, 16)]['sequence_flow'])
            print({link: network['edges'][link]['flow'] for link in network['edges']})
            break


if __name__ == "__main__":
    start_time = time.time()
    algorithmBush()
    print('Runtime:', time.time()-start_time)
