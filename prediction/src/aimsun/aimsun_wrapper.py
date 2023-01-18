"""
Created on Mon Jun 15 12:13:56 2020

@author: islam
"""
import sys
import os
executableName = os.path.basename(sys.executable)
if executableName != 'aconsole.exe':
    print 'Must run using Aimsun\n'\
          'Example: aconsole.exe --script aimsun_wrapper.py <args>'
    sys.exit()
else:
    from PyANGBasic import *
    from PyANGKernel import *
    from PyANGConsole import *
    from PyQt5.QtCore import *
import argparse
import random
import time

argParser = argparse.ArgumentParser(
	description = 'Run multiple microsimulation replications\n'
                  'Example: aconsole.exe --script aimsun_wrapper.py <args>')
argParser.add_argument('ang_file',
                        help='Aimsun model')
argParser.add_argument('repl_id',
                        help='Replication(s) ID(s) separated by commas')
argParser.add_argument('-d', "--demand_rand", action='store_true',
                        help='Randomize the demand')
args = argParser.parse_args()

console = ANGConsole()
assert console, 'Cannot open Aimsun'
console.open(args.ang_file)
model = console.getModel()
assert model, 'Cannot open the simulation model'

try:
    replication_ids = map(int, args.repl_id.split(','))
except:
    print 'Invalid Replication ID'
    sys.exit()

# Run all replications
for replication_id in replication_ids:
    replication = model.getCatalog().find(replication_id)
    if replication.isA("GKReplication"):
        # Randomize demand (if needed)
        if args.demand_rand:
            demand = replication.getExperiment().getScenario().getDemand()
            mul_factor_old = demand.getFactor()
            mul_factor = random.random()*100 + 50
            demand.setFactor(str(mul_factor))        
        GKSystem.getSystem().executeAction("execute", replication, [], 
                                           time.strftime("%Y%m%d-%H%M%S"))
        if args.demand_rand:
            demand.setFactor(mul_factor_old)