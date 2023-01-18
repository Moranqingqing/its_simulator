from flow.core.params import InFlows
import numpy as np
import logging

logger = logging.getLogger(__name__)


def generate_gaussian_inflow(spawn_edges, mapping):
    """
    Generates gaussion inflow of traffic.
    Current nested for loop order is: loop over time and then loop over edges.
    Flow does not work if this loop order is reversed (first edges then time), and no vehicles show up.

    Args:
        spawn_edges (list): All the edges that should be spawing vehicles.
        mapping (function): Function that maps edges to schedule: [(t0, t1, mu, sigma), (t1, t2, mu, sigma) ... ]

    Returns:
        flow.core.params.InFlows: Inflows object.
    """
    inflows = InFlows()
    schedules = []

    for edge in spawn_edges:
        schedules.append(mapping(edge))
    for j, edge in enumerate(spawn_edges):
        for i, (begin, end, _, _) in enumerate(schedules[j]):
            _, _, mu, sigma = schedules[j][i]

            begin = 1 if begin < 1 else begin
            prob = mu if sigma == 0 else np.random.normal(mu, sigma)
            prob = np.clip(prob, 1e-8, 1.0)

            logger.info(f'Inflow probability: edge={edge} prob={prob} begin={begin} end={end}')

            inflows.add(
                veh_type='human',
                edge=edge,
                probability=prob,
                depart_lane='free',
                depart_speed=30,
                begin=begin,
                end=end,
            )

    inflows.sort(lambda inflow: inflow['begin'])
    return inflows


def generate_platoon_inflow(spawn_edges, mapping):
    """
    Generates platoon inflow of traffic.
    If inflow has Period X then equally spaced vehicles are inserted at interval of X seconds.
    For more refer to flow.core.params.InFlows.

    Even high values for Period, spawn atleast one vehicle at the defined begin time.
    So Period value of -1 is used to denote no-flow condition, and very low probability is used to execute it.

    Inflows need to be sorted by begin time, otherwise Sumo ignores them.

    Args:
        spawn_edges (list): All the edges that should be spawing vehicles.
        mapping (function): Function that maps edges to schedule: [(t0, t1, period), (t1, t2, period) ... ]

    Returns:
        flow.core.params.InFlows: Inflows object.
    """
    inflows = InFlows()
    schedules = []

    for edge in spawn_edges:
        schedules.append(mapping(edge))

    for i, edge in enumerate(spawn_edges):
        for j, (begin, end, _) in enumerate(schedules[i]):
            _, _, period = schedules[i][j]

            begin = 1 if begin < 1 else begin
            kwargs = {'probability': 1e-8} if period == -1 else {'period': period}
            logger.info(f'Inflow Period: edge={edge} {kwargs} begin={begin} end={end}')

            inflows.add(
                veh_type='human',
                edge=edge,
                depart_lane='free',
                depart_speed=30,
                begin=begin,
                end=end,
                **kwargs
            )

    inflows.sort(lambda inflow: inflow['begin'])
    return inflows


def generate_poisson_inflow(spawn_edges, mapping):
    """
    Generates traffic by independent Poisson processes
    on the specified edges

    Args:
        spawn_edges (list): List of edges that spawn vehicles
        mapping (function): Function that maps edges to a schedule of arrival times
                            at that edge (t0, t1, t2, ...) where tn is a float
    Returns:
        flow.core.params.InFlows: InFlows object.
    """
    inflows = InFlows()
    arrivals = []

    for edge in spawn_edges:
        for t in mapping(edge):
            arrivals.append( (t, edge) )

    arrivals.sort(key=lambda arrival: arrival[0])    # Sort by arrival time

    for t, edge in arrivals:
        inflows.add(
            edge=edge,
            veh_type='human',
            begin=t,
            number=1,
            probability=1,
            depart_lane='free',
            depart_speed=30,
        )

    return inflows
