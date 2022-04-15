function [Err, Acc]=Classifier(Features, Labels, FeatureRows, Classifier_Type, CV_K)

    % This Function is Responsible of Classifying and Bringing Back and Err and Acc
    % Feature Rows: A Vector indicating the Selected Features
    % Classifier Type Options: 1.knn    2.lda   3.tree
    % K fold Cross Val, K = ?

    if (nargin == 3)                    % Default Classifier
        Classifier_Type = 'knn';
        CV_K = 10;
    end

    % Validate Feature Vector
    if (sum(find(FeatureRows > size(Features, 2))) ~= 0) || (numel(FeatureRows) > size(Features, 2))
        error('Invalid Feature Vector...!');
    end

    Classifier_Type = lower(Classifier_Type);

    confmat = 0;            % Empty Confusion Matrix

    cat_num = numel(unique(Labels));

    % Feature Filtering Due to Optimization
    myFeatures = Features(:, FeatureRows);

    % Number of Available Data for Each Class
    DataNum = numel(Labels);
    TestData_num = round(DataNum / CV_K);

    % Have the Process Done for k Time
    for  k = 1:CV_K

        % Shuffle Data and Divide Test and Train!
        shuffle_idx = randperm(DataNum);

        trian_idx = shuffle_idx(1:end-TestData_num);
        test_idx = shuffle_idx(end-TestData_num + 1:end);

        TrainData  = myFeatures(trian_idx, :);
        TrainLabel = Labels(trian_idx);

        TestData  = myFeatures(test_idx, :);
        TestLabel = Labels(test_idx);

        if strcmp(Classifier_Type, 'knn')
            % Train Model Using KNN
            Model = fitcknn(TrainData, TrainLabel, 'Distance',...
                'cityblock', 'NumNeighbors', 5);

            Out = predict(Model, TestData);             % Validate Model

        elseif strcmp(Classifier_Type, 'lda')
            % Train Model Using LDA
            Model = fitcdiscr(TrainData, TrainLabel, 'discrimType' ...
                , 'diaglinear', 'Gamma', 0.00023215, 'Delta', 0.0022196);

            Out  = predict(Model, TestData);             % Validate Model

        elseif strcmp(Classifier_Type, 'tree')
            % Train Model Using Decision Tree
            Model = fitctree(TrainData, TrainLabel, 'MinLeafSize', 4,...
                'Surrogate','on');

            Out  = predict(Model, TestData);             % Validate Model
        end

        confmat = confmat + confusionmat(TestLabel, Out);
    end

    % Create Empty Accuracy Vector
    Acc = zeros(1, cat_num);

    for i = 1: numel(Acc)
        Acc(i) = confmat(i, i)/ sum(confmat(i, :));
    end

    Err = 1/prod(Acc);                  % Error Fuction #1
    % Err = 1/sum(Acc);                   % Error Fuction #2

    % TotalAcc = sum(diag(confmat))/sum(confmat, "all");
    % Err = 1/TotalAcc;                   % Error Fuction #3
    % Err = 1-TotalAcc;                   % Error Fuction #4

end