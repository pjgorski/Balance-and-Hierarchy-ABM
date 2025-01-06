This folder contains analysis of the results obtained in the simulations of the agent-based model. In this folder, you will find jupyter notebook files and figures (pdfs). Some of the figures were placed in the folders figs and maps.  

Analysis below does not require environment defined to run the simulations. More recent python is acceptable. One might need to install additional Python packages. 

# List of files and analysis description:

* Analysis concerning complete graph simulations:
	* anal-agent-based-reps.ipynb - analysis of triad density deviations for different parameters
	* anal-agent-based-reps-old.ipynb - same as above 
	* anal-change-ph.ipynb - phase transition with p_{ST} as the control parameter
	* anal-change-q2.ipynb - phase transition with q as the control parameter
	* anal-change-q.ipynb - same as above with older way of determining quasi-stationary state
	* anal-complete-network-triad-vs-agent-based.ipynb - comparison of triad density deviations for triad-based and agent-based dynamics
	* anal-quasi-new-n.ipynb - dependence of the simulations on the system size for large q values
	* analysis-sepa-thorough-fig0_2.ipynb - analysis of separatrix
	* anal-quasi-sepa-panels.ipynb - creating panel b of Fig.3 of the paper. Discontinuous phase transition with p_{SBT} as the control parameter with the inset showing separatrix.  
* Analysis concerning Epinions dataset:
	* anal-epinions-triad-rhoinits2.ipynb - first analysis of Epinions
	* anal-epinions-triad-rhoinits3.ipynb - analysis with extended data by incorporating longer simulations
	* anal-epinions-triad-rhoinits4.ipynb - most recent analysis with more longer simulations included
* Analysis concerning WikiElections dataset:
	* anal-real-triad-sims21.ipynb - first analysis of WikiElections
	* anal-real-triad-sims-rhoinit.ipynb - improved analysis of WikiElections with added different starting points. 
	* anal-wiki-triad-rhoinits3.ipynb - repeated and longer simulations on WikiElections.
	* anal-wiki-triad-rhoinits4.ipynb - improved function of obtaining QS level, proper grouping of the results.
* Analysis concerning Slashdot dataset:
	* anal-slash-triad-rhoinits3.ipynb
* Analysis concerning Spanish dataset:
	* anal-spanish-schools-classes-controldown3.ipynb
* Combining all the results (generate Fig. 4)
	* anal-summary-rhoinits3.ipynb
* Analysis concerning other datasets:
	* anal-bitcoin-alpha-reps.ipynb
	* anal-bitcoin-alpha-triad-rhoinits.ipynb
	* anal-bitcoin-otc-triad-rhoinits.ipynb
	* anal-sampson-triad-rhoinits2.ipynb
* Other analysis:
	* anal-real-triad-stop-analysis-wiki.ipynb - analysis how many steps should be made to obtain proper quasistationary values and how big errors are when the simulations are shorter. 
	* anal-real-triad-sims-eth-wiki-100M-cleaned.ipynb - among others, some decription how bar plots are created, see #Actual plotting barplots section
* Files used in processing the ABM simulation resuts:
	* analyze_sims_spanish_classes_script3.py
	* analyze_sims_spanish_classes_script4.py
	* analyze_sims_spanish_classes_script52.py
	* analyze_sims_spanish_classes_script5.py
	* analyze_sims_spanish_classes_script_grouping.py
	* analyze_sims_spanish_classes_script.py
	* analyze_simulations_funs.py
	* analyze_simulations_script2.py
	* analyze_simulations_script.py
	* analyze_simulations_spanish_script2.py
	* analyze_simulations_spanish_script3.py
	* analyze_simulations_spanish_script4.py
	* analyze_simulations_spanish_script52.py
	* analyze_simulations_spanish_script5.py
	* analyze_simulations_spanish_script_grouping.py
	* analyze_simulations_spanish_script.py
* Results of analysis:
	* figs - folder with figures from dataset and complete graph analysis
	* maps - folder with maps when grid searching space of parameters. 
	* stop_analysis2_err_df.h5
	* stop_analysis3_err_df.h5
	* stop_analysis_err_df.h5
	* wiki_pb_1.pdf
	* wiki-s1000-triads.h5
* Results of analysis of high school dataset:
	* spanish_ave_best2.h5
	* spanish_ave_best3.h5
	* spanish_ave_best.h5
	* spanish-schools-classes-bestfits10-2.csv
	* spanish-schools-classes-bestfits10-3.csv
	* spanish-schools-classes-bestfits10.csv
	* spanish-schools-classes-bestfits20-2.csv
	* spanish-schools-classes-bestfits20.csv
	* spanish-schools-classes-bestfits-all-2.csv
	* spanish-schools-classes-bestfits-all.csv
	* spanish-schools-classes-bestfits-errors_2.csv
	* spanish-schools-classes-bestfits-errors.csv
	* spanish-schools-classes-bestfits-rel-2.csv
	* spanish-schools-classes-bestfits-rel.csv
	* spanish-schools-classes-bestfits-rho-2.csv
	* spanish-schools-classes-bestfits-rho.csv
	* spanish_schools_closest_to_best.pkl

