function Sol = DiscreteBBO(CostFunction, HabitationNum, MaxIt, VarNum, Range)

    % Discrete Bio-Geography-Based Optimizer for Collective Feature Selection

    %%%%%%%%%%%%%%%%%%%%%%%%%%% BBO Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%
    Range = sort(Range);
    Range2 = Range(2);

    % Migration Rates
    Mu = linspace(1, 0, HabitationNum);        % Emmigration Rates
    Lambda = 1-Mu;                             % Immigration Rates

    Alpha = 0.9;                               % Transferation Rate
    MutationRate = 0.05;                       % Mutation Rate

    %%%%%%%%%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample Habitat
    Habitat.SIV = [];
    Habitat.HSI = [];
    Habitat.Acc = [];
    Counter = 0;

    % Create Habitats Array
    hbt = repmat(Habitat, HabitationNum, 1);

    % Initialize Habitats
    parfor i = 1:HabitationNum
        hbt(i).SIV = randi(Range, [1 VarNum]);
        [hbt(i).HSI, hbt(i).Acc] = CostFunction(sort(unique(hbt(i).SIV)));
    end

    % Sort Population
    [~,  SortOrder] = sort([hbt.HSI]);
    hbt = hbt(SortOrder);

    % Best Solution Ever Found
    BestSol = hbt(1);

    % Array to Hold Best Costs
    BestHSI = zeros(MaxIt, 1);

    %%%%%%%%%%%%%%%%%%%%%%%%% BBO Main Loop %%%%%%%%%%%%%%%%%%%%%%%%%%%

    for it = 1:MaxIt

        newhbt = hbt;
        parfor i = 1:HabitationNum
            for k = 1:VarNum
                % Migration
                if rand <= Lambda(i)
                    % Emmigration Probabilities
                    EP = Mu;
                    EP(i) = 0;
                    EP = EP/sum(EP);

                    % Select Source Habitat
                    j = RouletteWheelSelection(EP);

                    % Migration
                    newhbt(i).SIV(k) = ceil(hbt(i).SIV(k) + Alpha*(hbt(j).SIV(k)-hbt(i).SIV(k)));

                    if newhbt(i).SIV(k)>Range2
                        newhbt(i).SIV(k) = Range2;
                    end
                end

                % Mutation
                if rand <= MutationRate
                    newhbt(i).SIV(k) = Mutate(newhbt(i).SIV(k), Range);
                end
            end

            % Evaluation
            [newhbt(i).HSI, newhbt(i).Acc] = CostFunction(sort(unique(newhbt(i).SIV)));
        end

        % Sort New Population
        [~,  SortOrder] = sort([newhbt.HSI]);
        newhbt = newhbt(SortOrder);

        % Merge New and Old Populations
        hbt = [hbt; newhbt];

        % Sort Population
        [~,  SortOrder] = sort([hbt.HSI]);
        hbt = hbt(SortOrder);

        % Select Next Iteration Population(Truncate)
        hbt = hbt(1: HabitationNum);

        % Update Best Solution Ever Found
        BestSol = hbt(1);

        % Store Best HSI Ever Found
        BestHSI(it) = BestSol.HSI;

        % Count Stall Iterations to Stop (Stall Iteration)
        if  it>1 && (BestHSI(it) ==  BestHSI(it-1))
            Counter = Counter+1;
        end

        % End the Loop if There is no Change in Last 15 Iterations
        if Counter == 15
            break;
        end

    end

    % Return Best Solution Ever Found
    Sol = BestSol;
    Sol.SIV = sort(unique(Sol.SIV));
    Sol.MeanAcc = mean(Sol.Acc);
end

function z = Mutate(x, Range)

    % Function to Mutate the Population With Discrete Amounts In Given Range

    % Create a Vector For the Existence Range
    Range = Range(1):Range(2);
    % Remove the Current Value
    Range(Range == x) = [];
    % Take a Random number in the Range
    z = randsample(Range, 1);

end

function j = RouletteWheelSelection(P)

    % Function for Probablistic RouletteWheel Selection
    r = rand;
    C = cumsum(P);
    j = find(r <= C, 1, 'first');

end