function [Miu, Sigma, log_transition_probability] = HMM(mfcc_cell, state)
num = length(mfcc_cell);
label_cell = cell(1, num);
log_transition_probability = ones(state, state)*(-Inf);
% initialized 5 equal size labels
for j = 1:num
    mfcc = mfcc_cell{j};
    [len, D] = size(mfcc);
    label = ones(1, len) * state;
    for k = 1:state-1
        label(floor(len*(k-1)/state)+1 : floor(len*k/state)) = k;
    end
    label_cell{j} = label;
end

sum_label_prev = Inf;
Miu = zeros(state, D);
Sigma = zeros(D*state, D);
while true
    for i = 1:state
        label_mfcc = [];
        count = 0;
        for j = 1:num
            mfcc = mfcc_cell{j};
            label = label_cell{j};
            mfcc_label_i = mfcc(label==i+1,:);
            if ~isempty(mfcc_label_i)
                count = count + 1;
            end
            label_mfcc = [label_mfcc; mfcc(label==i,:)];
        end
        label_mfcc_len = size(label_mfcc,1);
        
        log_transition_probability(i,i) = log((label_mfcc_len-count)/label_mfcc_len);
        if i < state
            log_transition_probability(i+1,i) = log(count/label_mfcc_len);
        end
        Miu(i,:) = sum(label_mfcc) / label_mfcc_len;
        shift = label_mfcc - repmat(Miu(i,:), label_mfcc_len, 1);
        Sigma((i-1)*D+1:i*D,:) = shift'*shift/label_mfcc_len;
    end
    log_transition_probability(state, state) = 0;
    sum_label = 0;
    for i = 1:num
        mfcc = mfcc_cell{i};
        label_cell{i} = DTW_HMM(Miu, Sigma, mfcc, log_transition_probability);
        sum_label = sum_label + sum(label_cell{i});
    end
    fprintf('%f\n',sum_label_prev - sum_label);
    if (abs(sum_label_prev - sum_label) < 1)
        break;
    end
    sum_label_prev = sum_label;
end
    