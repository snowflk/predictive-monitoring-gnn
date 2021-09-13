import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.util import constants

CSV_FILE = './data/bpi12_complete.csv'

log_csv = pd.read_csv(CSV_FILE, sep=',')
log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
log_csv = log_csv.sort_values('startTime')
params = {
    constants.PARAMETER_CONSTANT_CASEID_KEY: "case",
    constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "event",
    constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "startTime"
}
event_log = log_converter.apply(log_csv, parameters=params, variant=log_converter.Variants.TO_EVENT_LOG)

tree = inductive_miner.apply_tree(event_log, parameters=params)
from pm4py.objects.conversion.process_tree import converter as pt_converter

#gviz = pt_visualizer.apply(tree)
#pt_visualizer.view(gviz)
net, im, fm = pt_converter.apply(tree)
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.visualization.transition_system import visualizer as ts_visualizer

ts = reachability_graph.construct_reachability_graph(net, im)
gviz = ts_visualizer.apply(ts, parameters={ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
ts_visualizer.save(gviz, "net.svg")