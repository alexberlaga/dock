import subprocess
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument(
#         "--toid",
#         action="store_true",
#         help="Make a peptoid, not a peptide"
#     )    
# args = parser.parse_args()

commands = ["gmx_mpi", "pdb2gmx", "-f", "prot.pdb", "-o", "prot_lig.gro", "-p", "topol.top", "-ff", "charmm36-feb2021", "-water", "spce", "-ter"]

process = subprocess.Popen(commands, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

process.communicate('0\n0\n3\n4\n')