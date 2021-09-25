function best_path = viterbi_search_backpointer(mfcc, Miu, Sigma, log_transition_probability, state,...
            noise_Miu, noise_Sigma, noise_log_transition_probability, noise_state)
[mfcc_len, D] = size(mfcc);
total_state = size(Miu,1);
word_num = total_state/state;
Miu = [Miu; noise_Miu];
Sigma = [Sigma; noise_Sigma];
I = diag(ones(1, D));
log_transition_probability((1:word_num)*state, state) = log(0.9);
noise_log_transition_probability(noise_state, noise_state) = log(0.9);
iterative_transition_prob = ones(word_num+1, 1)*log(0.01/word_num);
iterative_transition_prob(word_num+1) = log(0.09);
prev_score = ones(total_state+noise_state, 1)*(-Inf);
current_score = zeros(total_state+noise_state, 1);
for j = 1:state:total_state+noise_state
    shift = mfcc(1,:) - Miu(j,:);
    inv_Sigma = inv(Sigma((j-1)*D+1:j*D,:,:).*I);
    prev_score(j) = -(D/2)*log(2*pi)+0.5*log(det(inv_Sigma))-0.5*sum(shift*inv_Sigma.*shift,2);
end

path_cell = cell(total_state+noise_state, 1);
for i = 1:word_num
    path_cell{(i-1)*state+1} = int2str(i-1);
end

for i = 2:mfcc_len
    iterative_edge_score = zeros(word_num+1, 1);
    iterative_edge_score(1:word_num) = prev_score((1:word_num)*state) + iterative_transition_prob(1:word_num);
    iterative_edge_score(word_num+1) = prev_score(end) + iterative_transition_prob(end);
    max_iterative_edge_score = max(iterative_edge_score) - 20; % 20~50是跳转到另一个HMM的惩罚�?
    max_index = find(iterative_edge_score == max_iterative_edge_score + 20) - 1;
    for j = 1:word_num
        temp = repmat(prev_score((j-1)*state+1:j*state)', state, 1) + log_transition_probability((j-1)*state+1:j*state,:);
        current_score((j-1)*state+1:j*state) = max(temp,[],2);
        for k = 1:state
            max_node = find(temp(k,:) == current_score((j-1)*state+k));
            path_cell{(j-1)*state+k} = path_cell{(j-1)*state+max_node(1)};
        end
        if max_iterative_edge_score > current_score((j-1)*state+1)
            current_score((j-1)*state+1) = max_iterative_edge_score;
            if max_index<10
                path_cell{(j-1)*state+1} = [path_cell{(max_index+1)*state}, int2str(j-1)];
            else
                path_cell{(j-1)*state+1} = [path_cell{end}, int2str(j-1)];
            end
        end
    end
    temp = repmat(prev_score(total_state+1:end)', noise_state, 1) + noise_log_transition_probability;
    current_score(total_state+1:end) = max(temp, [], 2);
    if max_iterative_edge_score > current_score(end)
        current_score(end) = max_iterative_edge_score;
        if max_index<10
            path_cell{total_state+1} = path_cell{(max_index+1)*state};
        end
    end
    
    for j = 1:total_state+noise_state
        shift = mfcc(i,:) - Miu(j,:);
        inv_Sigma = inv(Sigma((j-1)*D+1:D*j,:,:).*I);
        log_prob = -(D/2)*log(2*pi)+0.5*log(det(inv_Sigma))-0.5*sum(shift*inv_Sigma.*shift,2);
        current_score(j) = current_score(j) + log_prob;
    end
    prev_score = current_score;
end
index = find(current_score == max(current_score));
best_path = path_cell{index};