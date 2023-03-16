## Overview
This folder contains the implementation of the 
FALKON and StreaMRAK regression functions

### Falkon class
This class generates the Falkon trainer and prediction interface that allows training of a model and prediction
    with a trained model

    ## Variables
        - config: solver configurations
        - bw: bandwidth of kernel model
        - lms: the landmarks (Nystrom sub-samples)
        - nlm: number of landmarks
        - model:
            - regression model (bw, lm, coef)

        - kernel_obj: kernel used in regression model
        - lmfactor: number of landmarks is selected as nlm = lmfactor*sqrt(num training samples)
        - solver: instance of FalkonSolver class
        - max_ram_usage: max ram usage in Bytes

    ## Public methods
        # train
            - trains model on training data
        # predict
            - predicts with recieved data
        # add_trained_model
            - Adds a trained falkon model to the model variable
        # set_bw
            - sets the bandwidth of the kernel
        # append_specific_landmarks
            - if specific landmarks should be added
        # select_random_landmarks
            - selects landmarks uniformly from training data
        # clear
            -clears model

    ## Private methods
        # split_in_batches
        # solve_in_batches
        # train_with_timeit
        # predict_with_timeit

    ## Get functions
        # get_model
    """

### Lpkrr class
Base class for the Streamrak algorithm. Laplacian pyramid (LP) formulation of kernel ridge regression (KRR).
    Uses FALKON as a bases solver. This class generates the Lpkrr trainer and prediction interface
    that allows training of a model and prediction with a trained model

        ## Variables
        - config: solver configurations
        - pred_dict: dictionary pred_dict['lvl'] containing the prediction results at a given level
        - res_dict: dictionary res_dict['lvl'] containing the residuals at a given level on which the next level
        regress on
        - model:
            - dictionary containing the regression model at each level model['lvl'] = (bw, lm, coef)

        - kernel_obj: kernel used in regression model
        - solver: instance of FalkonSolver class
        - max_ram_usage: max ram usage in Bytes

    ## Public methods
        # train
            - trains model on training data
        # predict
            - predicts with recieved data
        # add_trained_model
            - Adds a trained falkon model to the model variable
        # set_bw
            - sets the bandwidth of the kernel
        # append_specific_landmarks
            - if specific landmarks should be added
        # select_random_landmarks
            - selects landmarks uniformly from training data
        # clear
            -clears model

    ## Private methods
        # split_in_batches
        # solve_in_batches
        # train_with_timeit
        # predict_with_timeit

    
### StreaMRAK class
 Streamrak class, inherits from Lpkrr builds an epsilon cover using the cover-tree algorithm from which its
    selects suitable landmarks for each level in the Laplacian pyramid in Lpkrr. Uses FALKON as a bases solver.
    This class generates the Streamrak trainer and prediction interface that allows training of a model and prediction
    with a trained model

    ## Variables
        - config: solver configurations
        - lmfactor: number of landmarks is selected as nlm = lmfactor*sqrt(num training samples)
        - variables inherited from Lpkrr

    ## Public methods
        # train
            - trains model on training data
        # predict
            - predicts with recieved data
        # clear
            -clears model

    ## Private methods
        # build_cover_tree
        # select_landmarks
        # train_with_timeit
        # predict_with_timeit

