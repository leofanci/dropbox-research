# dropbox-research

This exercise simulates user behavior in the cloud service, exploring the best decisions regarding file deletion, upgrading to a premium service, and sending referrals for additional storage. It calculates the value function iteration and recovers the true parameters by minimizing the likelihood function based on simulated data. The process is repeated for three users with heterogeneous parameters (theta) and is run in parallel.

### File Descriptions

- **functions.py**: python file containing all the functions used for the estimation process.
- **powell_multi.ipynb**: jupyter notebook that imports the functions and performs the parallel estimations.
- **optimization_pow_m.csv**: dataset containing the estimation results for three users. It consists of 300 rows (3 users, with 100 estimations each for bootstrapping purposes).
- **results_pow.ipynb**: jupyter notebook that plots the results.
