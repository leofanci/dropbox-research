# dropbox-research

This exercise simulates user behavior in the cloud service, exploring the best decisions regarding file deletion, upgrading to a premium service, and sending referrals for additional storage. It calculates the value function iteration and recovers the true parameters by minimizing the likelihood function based on simulated data. The process is repeated for three users with heterogeneous parameters (theta) and is run in parallel.

### File Descriptions

- `functions.py`: python file containing all the functions used for the estimation process.
- `powell_multi.ipynb`: jupyter notebook that imports the functions and performs the parallel estimations.
- `optimization_pow_m.csv`: dataset containing the estimation results for three users. It consists of 300 rows (3 users, with 100 estimations each for bootstrapping purposes).
- `results_pow.ipynb`: jupyter notebook that plots the results.

### Usage

Run `optimization_pow.ipynb`, which saves the checkpoint for each user in parallel after each bootstrap iteration. Each bootstrap iteration overwrites the previous one. After all 100 observations have been completed, the last checkpoint is deleted, and a `.csv` file is created for a specific user. A combined `.csv` file for all users is also created. 

The main function, `params_iteration`, takes two inputs:
- `user_vec`: A vector of user names to identify separate users.
- `theta_vec`: A vector of heterogeneous parameters. Each parameter is associated with a value in `user_vec`.

The code takes approximately 17 hours to run.

Once the running process is completed, a dataframe containing all the bootstrapping iterations for the optimization of each user is created. This will produce the `optimization_powell_m.csv` file, which can be used to run the codes in `results_pow.ipynp`. Note that `user_vec` and `theta_vec` also need to be set in `results_pow.ipynp`.

You can modify the problem settings by changing certain values in the `params_iteration` function found in `functions.py`. Modifiable aspects include:
- The specifics of the problem, such as the state and decision variable grids as well as the shocks.
- Homogeneous parameters.
- Other main settings, such as price `P`, number of bootstrap iterations, data points generated in the data generation function (currently set at `N=1000`) etc.


