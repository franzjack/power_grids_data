import pandas as pd
import io
import re
import pg_utils as sc
import old_code.pg_ham_comp as hc
import warnings
warnings.filterwarnings("ignore")


def load_pglib_opf(path):
    """
    Load an pglib_opf instance as a pandapower network.
    """

    with open(path, "r") as fi:
        tables = {}
        flag = False

        for line in fi.read().splitlines():

            # Start of each table
            if line.startswith("%% "):
                flag = True
                name = line.lstrip("%% ")
                tables[name] = []

            # End of each table
            elif line.startswith("];"):
                flag = False

            # Process the contents of each table
            elif flag:

                # Header data
                if line.startswith("% "):
                    headers = remove_empty(line[1:].split(""))
                    tables[name].append(headers)

                # Row data
                elif not line.startswith("mpc."):
                    data = remove_empty(re.split("\s+|;|%+", line))
                    tables[name].append(data)

    # Parse each table and transform into a dataframe
    bus_data = parse_bus_data(tables["bus data"])
    gen_data = parse_generator_data(
        tables["generator data"], tables["generator cost data"]
    )
    branch_data = parse_branch_data(tables["branch data"])
    bus_data['bustype'] = 2
    genlist = gen_data['bus'].tolist()
    bus_data['bustype'].loc[(bus_data['Gs'] != 0) | (bus_data['Bs'] !=0)] = 3
    bus_data['bustype'].loc[bus_data['bus_i'].isin(genlist)] = 1
    branch_data = infer_link_type(bus_data, branch_data)
    
    return [bus_data, gen_data, branch_data]

def infer_link_type(bus_data, dfnetwork_T):
    dfnetwork = dfnetwork_T.copy(deep=True)
    dfnetwork['from_type'] = 2
    dfnetwork['to_type'] = 2
    genlist = bus_data['bus_i'].loc[bus_data['bustype'] == 1].tolist()
    shuntlist = bus_data['bus_i'].loc[bus_data['bustype'] == 3].tolist()
    
    dfnetwork['from_type'].loc[dfnetwork['fbus'].isin(shuntlist)] = 3
    dfnetwork['from_type'].loc[dfnetwork['fbus'].isin(genlist)] = 1
    
    dfnetwork['to_type'].loc[dfnetwork['tbus'].isin(genlist)] = 1
    dfnetwork['to_type'].loc[dfnetwork['tbus'].isin(shuntlist)] = 3
    dfnetwork['link_type'] = dfnetwork['from_type'] + dfnetwork['to_type']
    dfnetwork['link_type'].loc[(dfnetwork['from_type'] == 1) & (dfnetwork['to_type'] == 1)] = 1
    dfnetwork['link_type'].loc[(dfnetwork['from_type'] == 1) & (dfnetwork['to_type'] == 2)]= 2
    dfnetwork['link_type'].loc[(dfnetwork['from_type'] == 2) & (dfnetwork['to_type'] == 1)] = 2
    dfnetwork['link_type'].loc[(dfnetwork['from_type'] == 1) & (dfnetwork['to_type'] == 3)] = 3
    dfnetwork['link_type'].loc[(dfnetwork['from_type'] == 3) & (dfnetwork['to_type'] == 1)] = 3
    return(dfnetwork)


def to_networkx(bus_data, gen_data, branch_data):
    """
    Makes a networkx instance from the passed-in data.
    """
    import networkx as nx

    G = nx.from_pandas_edgelist(
        branch_data,
        source="fbus",
        target="tbus",
        edge_attr=True,
        create_using=nx.MultiDiGraph,
        edge_key="line_idx",
    )

    busdict = bus_data.set_index("bus_i").T.to_dict()

    # We add an extra attribute 'generator_data' dict to each bus
    # which is empty of there is no generator
   # gendict = gen_data.set_index("bus").T.to_dict()
   # for key, data in busdict.items():
    #    data["generator_data"] = gendict.get(key, {})

    nx.set_node_attributes(G, busdict)

    return G


# --------------------------------Utils------------------------------------


def remove_empty(L):
    return list(filter(None, L))


def parse_bus_data(rows):
    return infer_types(pd.DataFrame(columns=rows[0], data=rows[1:]))


def parse_generator_data(generator_rows, generator_cost_rows):
    # Some data instances contain generator types as a comment, i.e., ";NG"
    # but this is not contained in the header.
    has_gen_type = len(generator_rows[0]) + 1 == len(generator_rows[1])
    headers = generator_rows[0] + (["generator_type"] if has_gen_type else [])

    # We determine how many cost coefficients are given
    # and create the corresponding headers
    n = int(generator_cost_rows[1][3])
    headers += generator_cost_rows[0][:4] + [f"c({n-1-i})" for i in range(n)]

    # Concatenate the generator and generator cost rows
    # but ignore the generator type from generator cost is it is there
    data_rows = [
        generator_rows[i]
        + (generator_cost_rows[i][:-1] if has_gen_type else generator_cost_rows[i])
        for i in range(1, len(generator_rows))
    ]

    return infer_types(pd.DataFrame(columns=headers, data=data_rows, dtype=object))


def parse_branch_data(rows):
    df = pd.DataFrame(columns=rows[0], data=rows[1:])
    df["line_idx"] = df.index
    return infer_types(df)


def infer_types(df):
    """
    # HACK: Infer the correct types from the dataframe. Currently, the provided
    pandas methods (df.convert_types, df.infer_types) do not work. The workaround
    is to convert to csv, then read as csv again.
    """
    return pd.read_csv(io.StringIO(df.to_csv(index=False)))


def pow_parser(name):
    bus_data, gen_data, branch_data = load_pglib_opf("Data/pglib_opf_case"+name+".m")
    bus_data['index1'] = bus_data.index
    bus_i = bus_data['bus_i'].to_numpy()
    bus_idx = bus_data['index1'].to_numpy()
    buslist = bus_data['bustype'].to_numpy()
    flist1 = branch_data['fbus'].to_numpy()
    tolist1 = branch_data['tbus'].to_numpy()
    flist, tolist = sc.bus_remap(bus_idx,bus_i,tolist1,flist1)
    truemat = sc.matrix_from_branch(flist,tolist,len(buslist))
    permutation = sc.bus_index(buslist)
    ordmat = sc.reorder_rows(truemat, permutation)
    k1,k2,k3,q1,q2,q3 = hc.avg_degreetype(truemat,buslist)
    ordlist = sc.ordered_buslist(q1,q2,q3)
    countlist= [q1,q2,q3]
    return(ordmat,ordlist,buslist,countlist)