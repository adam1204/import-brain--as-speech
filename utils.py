from sklearn.model_selection import train_test_split

def train_valid_test_split(trials, labels, train_size, test_size):
        """
        Split the input data into 3 different set: train, validation, test.
        The ratio between the sets is:   train:validation:test <-> train_size:validation_size:test_size
        
        Input
        -----
        trials: list
            The EEG data by trials.
        labels: list
            The labels by trials.
        train_size: int
            The train set's ratio.
        test_size: int
            The test set's ratio.

        Returns
        -------
        X_train: list
            Training set EEG trials.
        X_valid: list
            Validation set EEG trials.
        X_test: list
            Test set EEG trials.
        Y_train: list
            Training set labels.
        Y_valid: list
            Validation set labels.
        Y_test: list
            Test set labels.
        """
        X = trials
        Y = labels
        X_train, X_rem, Y_train, Y_rem = train_test_split(X, Y, train_size=train_size)
        X_valid, X_test, Y_valid, Y_test = train_test_split(X_rem, Y_rem, test_size= test_size/(1-train_size))
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test