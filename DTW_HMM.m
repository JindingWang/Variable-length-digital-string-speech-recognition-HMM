function label = DTW_HMM(Miu, Sigma, mfcc, log_transition_probability)
[state, D] = size(Miu);
mfcc_row = size(mfcc, 1);
label = ones(1,mfcc_row);
log_gussian_prob = zeros(mfcc_row, state);
I = diag(ones(1, D));
for i = 1:state
    shift = mfcc - repmat(Miu(i,:), mfcc_row, 1);
    inv_Sigma = inv(Sigma((i-1)*D+1:i*D,:).*I);
    log_gussian_prob(:,i) = -(D/2)*log(2*pi)+0.5*log(det(inv_Sigma))-0.5*sum(shift*inv_Sigma.*shift,2);
end

current_state = 1;
for i = 1:mfcc_row
    prob = log_gussian_prob(i,:) + log_transition_probability(:,current_state(1))';
    current_state = find(prob == max(prob));
    label(i) = current_state(1);
end
label(1:2) = 1;