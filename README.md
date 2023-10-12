# optimal-intervention
Optimal_intervention_definition files contain the function definitions for computing herd immunity, final size etc. The analytical.py file uses analytical formulae coded in python, for two groups to determine the intervention. The numerical.py file uses an optimiser from scipy to determine the intervention. 

script_execution_two_groups.py is a script that would allow you to set the parameters of the model, specify the cost function, select either the analytical or numerical solver, and it will return the final size without intervention and with optimal intervention.

script_execution_any_num_of_groups.py lets you do the same for an arbitrary number of groups but uses the numerical solver.

The heatmap file produces the figure given in the appendix of the paper.

script_execution_any_num_of_groups_real_matrix.py computes the optimal intervention using contact matrix from Wallinga 2007 (http://academic.oup.com/aje/article/164/10/936/162511/Using-Data-on-Social-Contacts-to-Estimate) and CFRs of various diseases.

The ipython notebook is work in progress and is not related to the current paper.
