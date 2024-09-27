# Machine-Learned Docking: Current Progress

This repository contains preliminary work on a project aimed at improving machine-learned protein-ligand docking predictions using molecular dynamics data. **This is a work-in-progress and not intended for immediate use.**

This repository contains:
1. Scripts for running docking simulations on a protein (PDB ID or FASTA) and a ligand (SMILES or peptide FASTA).  
2. Scripts for running volume-restricted (funnel or ellipsoid) metadynamics simulations of a protein and a ligand starting in a docked configuration. The major axis is selected by minimizing the cosine similarity between a unit vector along that axis and the vectors from the ligand's COM to the protein's atoms.
3. A notebook for analyzing the thermodynamics of funnel metadynamics simulations.   
4. Analysis of docked configurations and confidence scores -- a notebook that projects all protein-ligand pairs into a VAE latent space that has an additional loss term that places similar confidence scores closer together and different confidence scores farther away.

## Important Files 

1. dock.sh and run_chai.py: getting a bound configuration of a protein and a ligand using Chai Discovery's new structure prediction model (https://github.com/chaidiscovery/chai-lab)
2. get_embeddings.py and collect_embeddings.py: modifications of Chai Discovery's model that return protein/ligand embeddings.   
3. chai_results/token_vae.ipynb: VAE projecting a large set of peptides and peptoids bound to the SH3 domain (a well-known protein that binds to a large set of peptides and peptoids, especially those that are proline-rich).   
4. just_md.sh: scrpt for running the molecular dynamics using funnel metadynamics.
5. funnel_md/md_template/3_funnel/make_funnel.py and funnel_md/md_template/3_funnel/make_ellipsoid.py: automated parameterization of volume-restricted metadynamics.
6. funnel_md/multi_fes.ipynb: notebook that analyzes and visualizes a variety of simulations.   
   

