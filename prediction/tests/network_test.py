from src.network.sumo_network import SUMONetwork


def test_1():
    net = SUMONetwork('tests/test_cases/network_1.xml')
    assert net.link_ids == ['A', 'B']
    assert net.downstream['A']['straight'] == ['B']
    assert not net.downstream['A']['left']
    assert not net.downstream['A']['right']

    assert net.upstream['B']['straight'] == ['A']
    assert not net.upstream['B']['left']
    assert not net.upstream['B']['right']


def test_2():
    net = SUMONetwork('tests/test_cases/network_2.xml')
    assert net.link_ids == ['C']
    assert not net.downstream['C']['straight']
    assert not net.downstream['C']['left']
    assert not net.downstream['C']['right']

    assert not net.upstream['C']['straight']
    assert not net.upstream['C']['left']
    assert not net.upstream['C']['right']


def test_3():
    net = SUMONetwork('tests/test_cases/network_3.xml')
    assert net.link_ids == ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    assert net.downstream['A']['straight'] == ['B', 'H']
    assert net.downstream['A']['left'] == ['C']
    assert net.downstream['A']['right'] == ['D']
    assert net.upstream['A']['straight'] == ['E']
    assert net.upstream['A']['left'] == ['F']
    assert net.upstream['A']['right'] == ['G']

    assert net.upstream['B']['straight'] == ['A']
    assert net.upstream['C']['left'] == ['A']
    assert net.upstream['D']['right'] == ['A']
    assert net.downstream['E']['straight'] == ['A']
    assert net.downstream['F']['left'] == ['A']
    assert net.downstream['G']['right'] == ['A']
