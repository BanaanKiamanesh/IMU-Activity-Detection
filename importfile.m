function [features, labels] = importfile(filename_x, filename_y, startRow, endRow)

    % Import Train Data Like:
    %   [features, labels] = importfile('Data\X_train.txt','Data\y_train.txt', 1, 7352);
    %           or
    %   [features, labels] = importfile('Data\X_train.txt','Data\y_train.txt');

    % Import Test Data Like:
    %   [features, labels] = importfile('Data\X_test.txt','Data\y_test.txt', 1, 2947);
    %           or
    %   [features, labels] = importfile('Data\X_test.txt','Data\y_test.txt');

    % Input Check
    if nargin<=2
        startRow = 1;
        endRow = inf;
    end

    %% Import X_Train
    % Format for each line of text:
    formatSpec = ['%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%' ...
        '16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f' ...
        '%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16' ...
        'f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%1' ...
        '6f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%' ...
        '16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f' ...
        '%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16' ...
        'f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%1' ...
        '6f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%' ...
        '16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f' ...
        '%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16' ...
        'f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%1' ...
        '6f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%' ...
        '16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f' ...
        '%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16' ...
        'f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%1' ...
        '6f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%' ...
        '16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f' ...
        '%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16' ...
        'f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%1' ...
        '6f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%' ...
        '16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f' ...
        '%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16' ...
        'f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%1' ...
        '6f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%' ...
        '16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f' ...
        '%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16' ...
        'f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%1' ...
        '6f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%' ...
        '16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f' ...
        '%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16' ...
        'f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%1' ...
        '6f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%' ...
        '16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%f%[^\n\r]'];

    % Open the text file.
    fileID = fopen(filename_x,'r');

    % Read columns of data according to the format.
    dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, ...
        'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string', ...
        'EmptyValue', NaN, 'HeaderLines', startRow(1)-1, ...
        'ReturnOnError', false, 'EndOfLine', '\r\n');

    for block=2:length(startRow)

        frewind(fileID);
        dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, ...
            'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string', 'EmptyValue', ...
            NaN, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');

        for col=1:length(dataArray)
            dataArray{col} = [dataArray{col};dataArrayBlock{col}];
        end
    end

    % Close the text file.
    fclose(fileID);

    % Create output variable
    x = [dataArray{1:end-1}];

    %% Import Y_Train

    opts = delimitedTextImportOptions("NumVariables", 1);       % Set up the Import Options and import the data

    opts.DataLines = [startRow, endRow];                        % Specify range and delimiter
    opts.Delimiter = ",";

    opts.VariableNames = "VarName1";                            % Specify column names and types
    opts.VariableTypes = "double";

    opts.ImportErrorRule = "omitrow";                           % Specify file level properties
    opts.MissingRule = "omitrow";
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";

    y = readtable(filename_y, opts);                              % Import the data
    y = table2array(y);                                         % Convert to output type

    features = x;
    labels = y;
end