function eao = get_eao(result_sequences, year)
%get_eao - Description
%
% Syntax: eao = get_eao()

% Input:
% - result_sequences (cell): A cell array of valid sequence descriptor structures.

addpath('path_to/vot-toolkit/'); toolkit_path; % Make sure that VOT toolkit is in the path

pwd = ['path_to/vot-toolkit/', 'vot-workspace', year]  % year is a str (can not be a number)
[gt_sequences, experiments] = workspace_load(pwd);

experiment_sequences = convert_sequences(gt_sequences, experiments{1}.converter);
weights = ones(numel(experiment_sequences), 1);

failures = {};
segments = {}; %overlap
practicals = {};
sources = [];


for s = 1:numel(experiment_sequences)
    trajectory = result_sequences{s};
    % for python api
    trajectory = cellfun(@cell2mat, trajectory, 'UniformOutput', 0);
    trajectory = reshape(trajectory, [], 1);
    % for python api
    sequence = experiment_sequences{s};
    [~, frames] = estimate_accuracy(trajectory, sequence);
    [~, sub_failures] = estimate_failures(trajectory, sequence);
    practical = get_frame_value(sequence, 'practical');

    failures{end+1} = sub_failures(sub_failures <= sequence.length);
    segments{end+1} = frames;
    sources(end+1) = s;

     if isempty(practical)
        practicals{end+1} = zeros(sequence.length, 1);
    else
        practicals{end+1} = practical;
    end
end

maxlen = max(cellfun(@(x) x.length, experiment_sequences, 'UniformOutput', true));
lengths = 1:maxlen;

expected_overlaps = zeros(numel(lengths), 1);
practical_difference = zeros(numel(lengths), 1);
occurences = hist(sources, max(sources));
fragments_count = sum(cellfun(@(x) numel(x) + 1, failures, 'UniformOutput', true));
fragments_length = max(lengths);

%%---------------------------------------------------------
sequences = experiment_sequences;
sequence_weights = weights(sources(:));
frequency = occurences(sources(:));
sequence_weights = sequence_weights(:) ./ frequency(:);

fragments = nan(fragments_count, fragments_length);
fpractical = nan(fragments_count, fragments_length);
fweights = nan(fragments_count, 1);
f = 1;
tag = 'all';
skipping = experiments{1}.parameters.skip_initialize;
for i = 1:numel(segments)
    % calculate number of failures and their positions in the trajectory
    F = numel(failures{i});
    if F > 0
        % add first part of the trajectory to the fragment list
        points = failures{i}' + skipping;
        points = [1, points(points <= numel(segments{i}))];

        for j = 1:numel(points)-1;
            o = segments{i}(points(j):points(j+1)); o(isnan(o)) = 0;
            fragments(f, :) = 0;
            fragments(f, 1:min(numel(o), fragments_length)) = o;

            o = practicals{i}(points(j):points(j+1)); o(isnan(o)) = 0;
            fpractical(f, :) = 0;
            fpractical(f, 1:min(numel(o), fragments_length)) = o;

            w = numel(sequence_query_tag(sequences{sources(i)}, tag, points(j):(points(j+1)))) ...
                / (points(j+1) - points(j) + 1);

            fweights(f) = sequence_weights(i) * w;

            f = f + 1;
        end;

        o = segments{i}(points(end):end); o(isnan(o)) = 0;
        fragments(f, 1:min(numel(o), fragments_length)) = o;
        o = practicals{i}(points(end):end); o(isnan(o)) = 0;
        fpractical(f, 1:min(numel(o), fragments_length)) = o;

        w = numel(sequence_query_tag(sequences{sources(i)}, tag, points(end):length(segments{i}))) ...
            / (sequences{sources(i)}.length - points(end) + 1);

        fweights(f) = sequence_weights(i) * w;

        f = f + 1;
    else
    % process also last part of the trajectory - segment without failure
        if numel(segments{i}) >= fragments_length
            % tracker did not fail on this sequence and it is longer than
            % observed interval
            fragments(f, :) = segments{i}(1:fragments_length);
            fpractical(f, :) = practicals{i}(1:fragments_length);

            w = numel(sequence_query_tag(sequences{sources(i)}, tag, 1:fragments_length)) ...
                / fragments_length;
        else
            fragments(f, 1:numel(segments{i})) = segments{i};
            fpractical(f, 1:numel(practicals{i})) = practicals{i};

            w = numel(sequence_query_tag(sequences{sources(i)}, tag)) ...
                / sequences{sources(i)}.length;
        end

        fweights(f) = sequence_weights(i) * w;
        f = f + 1;
    end
end

for e = 1:size(expected_overlaps, 1)
    len = lengths(e);
    % do not calculate for Ns == 1: overlap on first frame is always NaN
    if len == 1
        expected_overlaps(e, 1) = 1;
        continue;
    end

    usable = ~isnan(fragments(:, len));

    if ~any(usable)
        continue;
    end;

    % for each len get a single number - average overlap
    expected_overlaps(e, 1) = sum(mean(fragments(usable, 2:len), 2) .* fweights(usable)) ./ sum(fweights(usable));
    practical_difference(e, 1) = sum(mean(fpractical(usable, 2:len), 2) .* fweights(usable)) ./ sum(fweights(usable));

end

evaluated_lengths = lengths;
    
[~, peak, low, high] = estimate_evaluation_interval(experiment_sequences, get_global_variable('eao_range_threshold', 0.5));
weights = ones(numel(evaluated_lengths(:)), 1);
weights(:) = 0;
weights(low:high) = 1;

curves = {expected_overlaps};
valid =  cellfun(@(x) numel(x) > 0, curves, 'UniformOutput', true)';
eao = cellfun(@(x) sum(x(~isnan(x(:, 1)), 1) .* weights(~isnan(x(:, 1)))) / sum(weights(~isnan(x(:, 1)))), curves(valid), 'UniformOutput', true);

end

%%----------------------------------------
function [gmm, peak, low, high] = estimate_evaluation_interval(sequences, threshold)

sequence_lengths = cellfun(@(x) x.length, sequences, 'UniformOutput', true);
model = gmm_estimate(sequence_lengths(:)'); % estimate the pdf by KDE

% tabulate the GMM from zero to max length
x = 1:max(sequence_lengths) ;
p = gmm_evaluate(model, x) ;
p = p / sum(p);
gmm.x = x;
gmm.p = p;

[low, high] = find_range(p, threshold) ;
[~, peak] = max(p);

end

%%----------------------------------------
function [low, high] = find_range(p, density)

% find maximum on the KDE
[~, x_max] = max(p);
low = x_max ;
high = x_max ;

for i = 0:length(p)
    x_lo_tmp = low - 1 ;
    x_hi_tmp = high + 1 ;

    sw_lo = 0 ; sw_hi = 0 ; % boundary indicator
    % clip
    if x_lo_tmp <= 0 , x_lo_tmp = 1 ;  sw_lo = 1 ; end
    if x_hi_tmp >= length(p), x_hi_tmp = length(p); sw_hi = 1; end

    % increase left or right boundary
    if sw_lo==1 && sw_hi==1
        low = x_lo_tmp ;
        high = x_hi_tmp ;
        break ;
    elseif sw_lo==0 && sw_hi==0
        if p(x_lo_tmp) > p(x_hi_tmp)
            low = x_lo_tmp ;
        else
            high = x_hi_tmp ;
        end
    else
        if sw_lo==0, low = x_lo_tmp ; else high = x_hi_tmp ; end
    end

    % check the integral under the range
    s_p = sum(p(low:high)) ;
    if s_p >= density
        return ;
    end
end

end
