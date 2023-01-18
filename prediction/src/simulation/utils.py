import os

def quoted_str(s):
    if type(s) == float:
        return "%.6f" % s
    elif type(s) != str:
        return str(s)
    elif '"' in s or ' ' in s:
        if os.name == "nt":
            return '"' + s.replace('"', '\\"') + '"'
        else:
            return "'%s'" % s
    else:
        return s

def getRelative(options, path):
    result = []
    dirname = path
    ld = len(dirname)
    for o in options:
        if isinstance(o, str) and o[:ld] == dirname:
            remove = o[:ld]
            result.append(o.replace(remove, ''))
        else:
            result.append(o)
    return result

def get_gates(net):

    entrance = []
    for node, in_degree in dict(net.G.in_degree).items():
        if in_degree == 0:
            node1, node2, attr = list(net.G.out_edges(node, data=True))[0]
            entrance.append(attr['linkID'])

    exit = []
    for node, out_degree in dict(net.G.out_degree).items():
        if out_degree == 0:
            node1, node2, attr = list(net.G.in_edges(node, data=True))[0]
            exit.append(attr['linkID'])

    return entrance, exit
